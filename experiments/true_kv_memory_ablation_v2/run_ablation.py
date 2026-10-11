"""Nine-configuration paired True-KV GPU memory ablation (no streaming eviction).

F0: one-shot Full KV
F1/F2/F3: no-eviction chunked Full KV with chunks 128/256/512
K0: legacy hook saliency, one-shot, post-prefill Layerwise True KV
K1: deferred saliency, one-shot, post-prefill Layerwise True KV
K2/K3/K4: deferred saliency, chunks 128/256/512, post-prefill True KV
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch

from kiaomni.repro.benchmark import load_model_and_tokenizer
from kiaomni.repro.controlled_benchmark import DEFAULT_TASKS, make_case, score_output
from kiaomni.repro.kv_policy_qwen25 import layerwise_masks
from kiaomni.repro.qwen25_kv_runtime import (
    cache_bytes, cache_lengths, cuda_sync, greedy_decode, prime_cache,
)
from memory_runtime import MemoryPhases, prefill_path, deferred_saliency

VARIANTS = {
    "F0": (None, False, False),
    "F1": (128, False, False),
    "F2": (256, False, False),
    "F3": (512, False, False),
    "K0": (None, True, False),
    "K1": (None, False, True),
    "K2": (128, False, True),
    "K3": (256, False, True),
    "K4": (512, False, True),
}
BUDGETS = ("B98", "B128", "B256", "B512", "r0.0625", "r0.125", "r0.25")
PROFILES = {
    "smoke": {"contexts": (2048,), "tasks": ("single_needle", "multi_query"),
              "repeats": 1, "budgets": ("B128", "B256", "r0.125"), "max_new": 32},
    "pilot": {"contexts": (4096,), "tasks": tuple(DEFAULT_TASKS),
              "repeats": 1, "budgets": BUDGETS, "max_new": 128},
    "final": {"contexts": (4096, 8192), "tasks": tuple(DEFAULT_TASKS),
              "repeats": 2, "budgets": BUDGETS, "max_new": 256},
}


def budget_schedule(labels: tuple[str, ...], context: int) -> list[dict]:
    groups = {}
    for label in labels:
        value = (int(round(context * float(label[1:]))) if label.startswith("r")
                 else int(label[1:]) if label.startswith("B") else None)
        if value is None or value <= 0:
            raise ValueError("Invalid budget: " + label)
        value = min(value, context)
        if value < 48:
            raise ValueError("Budget below 16 sink + 32 recent tokens")
        groups.setdefault(value, []).append(label)
    return [{"tokens": k, "aliases": aliases} for k, aliases in sorted(groups.items())]


def drop_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cuda_sync()


def mask_digest(masks) -> str:
    array = np.stack(masks).astype(np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def run_one(model, tokenizer, ids, case, *,
            variant: str, budget: int | None, aliases: list[str],
            bridge_ids: list[int], max_new: int, window: int) -> dict:
    chunk, original_sal, deferred = VARIANTS[variant]
    compressed = variant.startswith("K")
    if compressed != (budget is not None):
        raise ValueError("Budget must be set for K variants only")
    phases = MemoryPhases()
    stage = "prefill"
    started = time.perf_counter()
    pf = None
    bridge = None
    decoded = None
    try:
        phases.start(stage)
        pf = prefill_path(
            model, ids, chunk_size=chunk, original_saliency=original_sal,
            deferred_query=deferred, window=window,
        )
        phases.finish()
        total_length = int(ids.shape[-1])
        full_kv_bytes = int(cache_bytes(pf["cache"]))
        if any(v != total_length for v in cache_lengths(pf["cache"])):
            raise RuntimeError("Full cache must be retained until post-prefill")

        selected_digest = None
        selected_tokens = total_length
        post_bytes = full_kv_bytes
        if compressed:
            stage = "saliency"
            phases.start(stage)
            saliencies = (pf["layer_saliencies"] if original_sal else
                          deferred_saliency(model, pf["cache"], pf["queries"], total_length))
            phases.finish()

            stage = "selection"
            phases.start(stage)
            masks = layerwise_masks(
                saliencies, budget, n_sink=16, recency=32, sigma=4.0,
            )
            selected_digest = mask_digest(masks)
            phases.finish()

            stage = "compaction"
            phases.start(stage)
            from kiaomni.repro.qwen25_kv_runtime import compact_cache_layerwise
            compact_cache_layerwise(pf["cache"], masks)
            phases.finish()
            post_bytes = int(cache_bytes(pf["cache"]))
            selected_tokens = budget
            if post_bytes >= full_kv_bytes:
                raise RuntimeError("Cache compaction did not reduce physical KV bytes")
            if any(x != budget for x in cache_lengths(pf["cache"])):
                raise RuntimeError("Compaction violated exact per-layer budget")

        stage = "bridge"
        phases.start(stage)
        bridge = prime_cache(
            model, cache=pf["cache"], forced_token_ids=bridge_ids,
            start_position=total_length,
        )
        phases.finish()
        stage = "decode"
        phases.start(stage)
        decoded = greedy_decode(
            model, tokenizer, initial_logits=bridge["logits"],
            cache=bridge["cache"], start_position=bridge["next_position"],
            max_new_tokens=max_new,
        )
        phases.finish()
        score = score_output(decoded["text"], case.expected_answers)
        elapsed = time.perf_counter() - started
        peaks = phases.overall()
        return {
            "status": "ok",
            "case_id": case.case_id, "task": case.task,
            "seed": case.metadata["seed"], "context_tokens": total_length,
            "method": variant, "chunk_size": chunk,
            "budget_tokens": budget, "budget_aliases": aliases,
            "physical_kv_compression": compressed,
            "kv_before_bytes": full_kv_bytes, "kv_after_bytes": post_bytes,
            "kv_reduction_ratio": full_kv_bytes / post_bytes,
            "mask_sha256": selected_digest,
            "kept_tokens": selected_tokens,
            "original_hooks_in_prefill": bool(original_sal),
            "deferred_saliency": bool(deferred),
            "all_correct": bool(score["all_correct"]),
            "answer_recall": float(score["answer_recall"]),
            "expected_answers": list(case.expected_answers),
            "output_text": decoded["text"],
            "generated_tokens": decoded["generated_tokens"],
            "tokens_per_second": decoded["tokens_per_second"],
            "greedy_self_ppl": decoded["greedy_self_ppl"],
            "prefill_seconds": pf["elapsed_seconds"],
            "ttft_seconds": sum(
                p["seconds"] for name, p in phases.phases.items()
                if name != "decode"
            ),
            "total_seconds": elapsed,
            "phase_memory": phases.phases,
            **peaks,
        }
    except BaseException as exc:
        return {
            "status": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
            "error_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
            "case_id": case.case_id, "task": case.task,
            "seed": case.metadata["seed"], "context_tokens": len(case.input_ids),
            "method": variant, "chunk_size": chunk, "budget_tokens": budget,
            "budget_aliases": aliases,
            "completed_phases": phases.phases,
        }
    finally:
        del pf, bridge, decoded
        drop_gpu_cache()


def preflight(model, tokenizer, out: Path, *, window: int) -> bool:
    """GPU check: one-shot vs no-eviction chunking, and original vs deferred scores."""
    case = make_case(tokenizer, "multi_key", 1024, 20261004)
    ids = torch.tensor([case.input_ids], device="cuda:0", dtype=torch.long)
    results = {"status": "FAIL", "context": 1024, "chunked": {}, "saliency": {}}
    full = None
    old = None
    new = None
    try:
        full = prefill_path(
            model, ids, chunk_size=None, original_saliency=False,
            deferred_query=False, window=window
        )
        baseline = torch.argmax(full["logits"], dim=-1)
        for chunk in (128, 256, 512):
            candidate = prefill_path(
                model, ids, chunk_size=chunk, original_saliency=False,
                deferred_query=False, window=window
            )
            lengths = cache_lengths(candidate["cache"])
            results["chunked"][str(chunk)] = {
                "same_cache_lengths": lengths == cache_lengths(full["cache"]),
                "same_next_argmax": bool(torch.equal(
                    torch.argmax(candidate["logits"], dim=-1), baseline)),
                "logits_max_abs_diff": float(
                    (candidate["logits"].float() - full["logits"].float())
                    .abs().max().item()),
            }
            del candidate
            drop_gpu_cache()
        old = prefill_path(
            model, ids, chunk_size=None, original_saliency=True,
            deferred_query=False, window=window
        )
        new = prefill_path(
            model, ids, chunk_size=None, original_saliency=False,
            deferred_query=True, window=window
        )
        score_old = old["layer_saliencies"]
        score_new = deferred_saliency(model, new["cache"], new["queries"], 1024)
        tests = {}
        for b in (128, 256, 512):
            m_old = layerwise_masks(score_old, b)
            m_new = layerwise_masks(score_new, b)
            tests[str(b)] = all(np.array_equal(a, b) for a, b in zip(m_old, m_new))
        results["saliency"] = {
            "max_abs_diff": float(np.max(np.abs(score_old - score_new))),
            "same_positions_by_budget": tests,
        }
        results["status"] = ("PASS" if
            all(x["same_cache_lengths"] and x["same_next_argmax"]
                for x in results["chunked"].values()) and
            all(tests.values()) else "FAIL")
    except BaseException as exc:
        results["error"] = str(exc)
        results["traceback"] = traceback.format_exc(limit=12)
    finally:
        del full, old, new, ids
        drop_gpu_cache()
    (out / "preflight.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results["status"] == "PASS"


def save_summary(rows: list[dict], out: Path):
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return
    keys = ["context_tokens", "method", "budget_tokens"]
    grouped = {}
    for row in rows:
        k = tuple(row.get(x) for x in keys)
        grouped.setdefault(k, []).append(row)
    columns = keys + [
        "n", "successful", "errors_or_oom", "accuracy", "mean_recall",
        "peak_allocated_gib", "peak_reserved_gib", "kv_reduction_ratio",
        "ttft_seconds", "total_seconds", "tokens_per_second",
    ]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for key, entries in sorted(grouped.items(), key=lambda x: str(x[0])):
            valid = [r for r in entries if r["status"] == "ok"]
            row = dict(zip(keys, key))
            row.update(n=len(entries), successful=len(valid),
                       errors_or_oom=len(entries) - len(valid))
            for source, target in [
                ("all_correct", "accuracy"), ("answer_recall", "mean_recall"),
                ("peak_allocated_gib", "peak_allocated_gib"),
                ("peak_reserved_gib", "peak_reserved_gib"),
                ("kv_reduction_ratio", "kv_reduction_ratio"),
                ("ttft_seconds", "ttft_seconds"),
                ("total_seconds", "total_seconds"),
                ("tokens_per_second", "tokens_per_second"),
            ]:
                values = [float(r[source]) for r in valid if r.get(source) is not None]
                row[target] = sum(values) / len(values) if values else None
            writer.writerow(row)

    control = {(r["case_id"], r["method"]): r for r in ok}
    paired = []
    for r in ok:
        if r["method"] in ("F0", "F1", "F2", "F3"):
            continue
        fc = control.get((r["case_id"], "F0"))
        if fc is None:
            continue
        paired.append({
            "case_id": r["case_id"], "method": r["method"],
            "budget_tokens": r["budget_tokens"], "context_tokens": r["context_tokens"],
            "fc_correct": fc["all_correct"], "candidate_correct": r["all_correct"],
            "regression": fc["all_correct"] and not r["all_correct"],
            "rescue": not fc["all_correct"] and r["all_correct"],
            "delta_peak_gib": r["peak_allocated_gib"] - fc["peak_allocated_gib"],
            "delta_recall": r["answer_recall"] - fc["answer_recall"],
        })
    if paired:
        with (out / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(paired[0]))
            w.writeheader()
            w.writerows(paired)

    # Matched K0/K1 mask hashes expose invalid saliency-isolation assumptions.
    by_key = {(r["case_id"], r["method"], r["budget_tokens"]): r for r in ok}
    mask_equivalence = []
    for r in ok:
        if r["method"] not in ("K1", "K2", "K3", "K4"):
            continue
        old = by_key.get((r["case_id"], "K0", r["budget_tokens"]))
        if old:
            mask_equivalence.append({
                "case_id": r["case_id"], "method": r["method"],
                "budget_tokens": r["budget_tokens"],
                "same_legacy_masks": old["mask_sha256"] == r["mask_sha256"],
            })
    if mask_equivalence:
        with (out / "mask_equivalence.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mask_equivalence[0]))
            w.writeheader()
            w.writerows(mask_equivalence)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=PROFILES, default="smoke")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--contexts", nargs="*", type=int)
    p.add_argument("--methods", nargs="+", choices=VARIANTS,
                   default=list(VARIANTS))
    p.add_argument("--budgets", nargs="+", default=None)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--case-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU is required; no fake CPU GPU results")
    cfg = PROFILES[args.mode]
    contexts = tuple(args.contexts) if args.contexts else cfg["contexts"]
    if not contexts or any(x < 1024 for x in contexts):
        raise ValueError("Contexts must be at least 1024 tokens")
    tasks = tuple(cfg["tasks"])
    if args.case_limit is not None:
        if args.case_limit < 1:
            raise ValueError("case-limit must be positive")
        tasks = tasks[:args.case_limit]
    repeats = args.repeat if args.repeat is not None else cfg["repeats"]
    if repeats < 1:
        raise ValueError("repeat must be positive")
    specs = tuple(args.budgets) if args.budgets else cfg["budgets"]
    schedules = {ctx: budget_schedule(specs, ctx) for ctx in contexts}
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / "runs.jsonl").exists():
        raise RuntimeError("Output already contains runs: use --resume or a new directory")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    protocol = {
        "experiment": "true_kv_memory_ablation_v2", "mode": args.mode,
        "methods": args.methods, "contexts": contexts, "budgets": schedules,
        "tasks": tasks, "repeats": repeats, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens or cfg["max_new"],
        "window": args.window, "model": "Qwen/Qwen2.5-7B-Instruct",
        "weight_quant": "NF4 4-bit", "attention": "sdpa",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "python": platform.python_version(), "git_commit": commit,
        "no_eviction_during_prefill": True,
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["sha256"] = digest
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol["sha256"] != digest:
            raise RuntimeError("Protocol mismatch. Do not mix or overwrite runs")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    model, tokenizer, info = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    (out / "model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    if not args.skip_preflight:
        if not preflight(model, tokenizer, out, window=args.window):
            raise RuntimeError("Preflight failed; do not promote or run final ablation")
    bridge_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not bridge_ids:
        raise RuntimeError("Missing bridge token")

    rows_path = out / "runs.jsonl"
    rows = ([json.loads(line) for line in rows_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
            if rows_path.exists() else [])
    done = {(r["case_id"], r["method"], r.get("budget_tokens"))
            for r in rows if r["status"] == "ok"}
    cases_done = 0
    with rows_path.open("a", encoding="utf-8") as fh:
        for ctx in contexts:
            for rep in range(repeats):
                for task_idx, task in enumerate(tasks):
                    seed = args.seed + ctx * 10 + rep * 1000 + task_idx * 37
                    case = make_case(tokenizer, task, ctx, seed)
                    ids = torch.tensor(
                        [case.input_ids], dtype=torch.long, device="cuda:0")
                    for variant in args.methods:
                        budgets = schedules[ctx] if variant.startswith("K") else [
                            {"tokens": None, "aliases": []}]
                        for setting in budgets:
                            budget = setting["tokens"]
                            key = (case.case_id, variant, budget)
                            if key in done:
                                continue
                            print("[run]", ctx, task, variant, budget, flush=True)
                            row = run_one(
                                model, tokenizer, ids, case,
                                variant=variant, budget=budget,
                                aliases=setting["aliases"],
                                bridge_ids=bridge_ids,
                                max_new=args.max_new_tokens or cfg["max_new"],
                                window=args.window,
                            )
                            rows.append(row)
                            fh.write(json.dumps(row, sort_keys=True) + "\n")
                            fh.flush()
                            os.fsync(fh.fileno())
                            print("[result]", row["status"],
                                  row.get("peak_allocated_gib"),
                                  row.get("all_correct"),
                                  row.get("error_stage"), flush=True)
                    del ids
                    drop_gpu_cache()
                    cases_done += 1
                    save_summary(rows, out)

    status = "COMPLETED" if all(r["status"] == "ok" for r in rows) else "COMPLETED_WITH_ERRORS"
    manifest = {
        "status": status, "protocol_sha256": digest,
        "expected_cases": len(contexts) * repeats * len(tasks),
        "cases_processed": cases_done, "rows": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "oom": sum(r["status"] == "oom" for r in rows),
        "errors": sum(r["status"] == "error" for r in rows),
        "raw_runs": str(rows_path), "scientific_status": "UNVALIDATED_UNTIL_GPU_RUN",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0 if status == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
