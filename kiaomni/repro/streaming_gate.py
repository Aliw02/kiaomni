"""Strict paired quality gate for experimental streaming KV eviction.

Streaming is opt-in and never replaces full_kv or true_kv. A streaming
candidate is marked SKIP if it fails a paired quality gate, returns an
error, or does not produce genuine physical KV reduction.
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
import sys
import time
import traceback

import numpy as np
import torch

from .benchmark import load_model_and_tokenizer
from .controlled_benchmark import DEFAULT_TASKS, make_case, score_output
from .kv_policy_qwen25 import layerwise_masks
from .qwen25_kv_runtime import (
    cache_bytes, cleanup_cuda, compact_cache_layerwise,
    cuda_sync, greedy_decode, peak_memory_gb, prefill, prime_cache,
    reset_peak_memory,
)
from .streaming_qwen25_runtime import (
    real_no_eviction_equivalence, streaming_prefill,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def budget_table(context: int, labels: list[str]) -> list[dict]:
    by_size: dict[int, dict] = {}
    for label in labels:
        if label.startswith("r"):
            count = round(context * float(label[1:]))
        elif label.startswith("B"):
            count = int(label[1:])
        else:
            raise ValueError(f"Invalid budget label {label!r}; use B256 or r0.125")
        count = min(int(count), context)
        if count <= 0:
            raise ValueError("Budget must be positive")
        item = by_size.setdefault(count, {"budget": count, "labels": []})
        item["labels"].append(label)
    return list(by_size.values())


def trial(model, tokenizer, ids, case, *, method: str, budget: int,
          chunk_size: int, multiplier: int, max_new_tokens: int,
          bridge_ids: list[int]) -> dict:
    reset_peak_memory()
    t = time.perf_counter()
    kept_bytes = None
    peak_kv = None
    trigger = None

    if method == "full_kv":
        pf = prefill(model, ids, collect_saliency=False)
        before = cache_bytes(pf["cache"])
        kept_bytes = before
        cache = pf["cache"]
        stage_seconds = float(pf["elapsed_seconds"])
        physical = False
    elif method == "true_kv":
        pf = prefill(model, ids, collect_saliency=True)
        before = cache_bytes(pf["cache"])
        masks = layerwise_masks(pf["layer_saliencies"], budget)
        c = compact_cache_layerwise(pf["cache"], masks)
        kept_bytes = int(c["after_bytes"])
        cache = pf["cache"]
        stage_seconds = time.perf_counter() - t
        physical = kept_bytes < before
    else:
        if method not in {"stream_persistent_layerwise", "stream_persistent_global"}:
            raise ValueError(f"Unknown method {method}")
        if multiplier < 1:
            raise ValueError("trigger multiplier cannot be below 1")
        variant = method.removeprefix("stream_")
        trigger = int(multiplier * budget)
        pf = streaming_prefill(
            model, ids, budget=budget, chunk_size=chunk_size,
            variant=variant, eviction_trigger_tokens=trigger,
        )
        before = None
        kept_bytes = int(pf["final_kv_bytes"])
        peak_kv = int(pf["peak_kv_bytes"])
        cache = pf["cache"]
        stage_seconds = float(pf["total_seconds"])
        physical = bool(int(ids.shape[1]) > budget and pf["eviction_events"] > 0)
        if int(ids.shape[1]) > budget and not physical:
            raise RuntimeError("Streaming performed no eviction despite budget < context")

    prime = prime_cache(
        model, cache=cache,
        forced_token_ids=bridge_ids,
        start_position=int(ids.shape[1]),
    )
    decoded = greedy_decode(
        model, tokenizer,
        initial_logits=prime["logits"], cache=prime["cache"],
        start_position=prime["next_position"],
        max_new_tokens=max_new_tokens,
    )
    cuda_sync()
    elapsed = time.perf_counter() - t
    score = score_output(decoded["text"], case.expected_answers)
    result = {
        "case_id": case.case_id,
        "task": case.task,
        "method": method,
        "budget_tokens": None if method == "full_kv" else budget,
        "trigger_multiplier": multiplier if method.startswith("stream_") else None,
        "trigger_tokens": trigger,
        "status": "ok",
        "all_correct": bool(score["all_correct"]),
        "answer_recall": float(score["answer_recall"]),
        "expected": list(case.expected_answers),
        "output_text": decoded["text"],
        "physical_kv_compression": physical,
        "kv_before_bytes": before,
        "kv_after_bytes": kept_bytes,
        "prefill_peak_kv_bytes": peak_kv,
        "prefill_seconds": stage_seconds,
        "ttft_seconds": stage_seconds + float(prime["elapsed_seconds"]),
        "total_seconds": elapsed,
        "tokens_per_second": decoded["tokens_per_second"],
        "generated_tokens": decoded["generated_tokens"],
        **peak_memory_gb(),
    }
    del pf, cache, prime, decoded
    return result


def judge(rows: list[dict], equivalence: dict, context: int) -> dict:
    """Fail closed: require all FullKV-correct cases preserved, and no
    regression relative to matched post-prefill true-KV controls."""
    if not (equivalence.get("same_cache_lengths") and
            equivalence.get("same_next_token_argmax")):
        return {"decision": "SKIP", "reason": "no_eviction_chunk_equivalence_failed"}

    reference = {r["case_id"]: r for r in rows if r["method"] == "full_kv"}
    controls = {
        (r["case_id"], r["budget_tokens"]): r
        for r in rows if r["method"] == "true_kv"
    }
    result = {}
    for method in sorted({r["method"] for r in rows if r["method"].startswith("stream_")}):
        for multiplier in sorted({r["trigger_multiplier"] for r in rows
                                  if r["method"] == method}):
            for budget in sorted({r["budget_tokens"] for r in rows
                                  if r["method"] == method and r["trigger_multiplier"] == multiplier}):
                group = [r for r in rows if r["method"] == method
                         and r["budget_tokens"] == budget and
                         r["trigger_multiplier"] == multiplier]
                regressions, expected_ok, errors, compressed = 0, 0, 0, 0
                paired = []
                for row in group:
                    fc = reference.get(row["case_id"])
                    kv = controls.get((row["case_id"], budget))
                    if not fc or not kv or fc.get("status") != "ok" or kv.get("status") != "ok":
                        errors += 1
                        continue
                    if row.get("status") != "ok":
                        errors += 1
                        continue
                    protected = bool(fc["all_correct"] or kv["all_correct"])
                    expected_ok += int(protected)
                    if protected and not row["all_correct"]:
                        regressions += 1
                    compressed += int(row.get("physical_kv_compression") and
                                      row["kv_after_bytes"] < fc["kv_after_bytes"])
                    paired.append({
                        "case_id": row["case_id"],
                        "full_correct": fc["all_correct"],
                        "true_kv_correct": kv["all_correct"],
                        "stream_correct": row["all_correct"],
                    })
                ok = (len(group) >= 2 and len(group) == len(reference) and
                      errors == 0 and regressions == 0 and
                      compressed == len(group) and expected_ok > 0)
                key = f"{method}:B{budget}:trigger_x{multiplier}"
                result[key] = {
                    "decision": "PASS_CANDIDATE" if ok else "SKIP",
                    "cases": len(group),
                    "paired": len(paired),
                    "protected_correct": expected_ok,
                    "regressions": regressions,
                    "errors_or_missing_controls": errors,
                    "physical_reduction_cases": compressed,
                    "reason": "paired_quality_and_physical_reduction" if ok
                    else "did_not_meet_strict_paired_quality_gate",
                    "paired_outcomes": paired,
                }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in streaming quality repair gate")
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="/kaggle/working/kiaomni_streaming_gate")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--budgets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=[
        "stream_persistent_layerwise", "stream_persistent_global"])
    parser.add_argument("--trigger-multipliers", nargs="+", type=int, default=[1, 2])
    opts = parser.parse_args()

    if opts.context < 256:
        parser.error("context must be >= 256")
    allowed = {"stream_persistent_layerwise", "stream_persistent_global"}
    if not set(opts.methods).issubset(allowed):
        parser.error(f"methods must be among {sorted(allowed)}")
    if any(v < 1 for v in opts.trigger_multipliers):
        parser.error("trigger multipliers must be >= 1")

    labels = opts.budgets or (
        ["B256"] if opts.profile == "smoke" else
        ["B98", "B128", "B256", "B512", "r0.0625", "r0.125", "r0.25"]
    )
    table = budget_table(opts.context, labels)
    if min(x["budget"] for x in table) < 48:
        parser.error("budget below default protected sink+recency count 48")
    tasks = ["single_needle", "multi_key", "multi_query"] if opts.profile == "smoke" else list(DEFAULT_TASKS)
    repeats = 1 if opts.profile == "smoke" else 2
    out = Path(opts.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(f"Required transformers==4.57.6; got {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is required. This test has not run on CPU.")

    protocol = {
        "profile": opts.profile, "model_id": opts.model_id,
        "seed": opts.seed, "context": opts.context, "chunk_size": opts.chunk_size,
        "max_new_tokens": opts.max_new_tokens, "budgets": table,
        "methods": opts.methods, "trigger_multipliers": opts.trigger_multipliers,
        "tasks": tasks, "repeats": repeats, "transformers": transformers.__version__,
        "torch": str(torch.__version__), "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda": torch.version.cuda,
    }
    protocol["sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    model, tokenizer, model_info = load_model_and_tokenizer(opts.model_id)
    protocol["model_revision"] = getattr(model.config, "_commit_hash", None)
    (out / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    first_case = make_case(tokenizer, tasks[0], min(1024, opts.context), opts.seed)
    ids = torch.tensor([first_case.input_ids], dtype=torch.long, device="cuda:0")
    eq = real_no_eviction_equivalence(model, ids, chunk_size=opts.chunk_size)
    (out / "equivalence.json").write_text(json.dumps(eq, indent=2), encoding="utf-8")
    del ids
    cleanup_cuda()

    bridge_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not bridge_ids:
        raise RuntimeError("Tokenizer returned empty bridge")
    rows: list[dict] = []
    runs_file = out / "streaming_runs.jsonl"
    with runs_file.open("w", encoding="utf-8") as fp:
        for task_idx, task in enumerate(tasks):
            for rep in range(repeats):
                seed = opts.seed + task_idx * 31 + rep * 1009
                case = make_case(tokenizer, task, opts.context, seed)
                ids = torch.tensor([case.input_ids], dtype=torch.long, device="cuda:0")
                specs = [("full_kv", None, 1)]
                for budget in table:
                    specs.append(("true_kv", budget["budget"], 1))
                    for method in opts.methods:
                        for multiplier in opts.trigger_multipliers:
                            specs.append((method, budget["budget"], multiplier))
                for method, budget, multiplier in specs:
                    print(f"[run] {case.case_id} {method} B{budget} trigger_x{multiplier}", flush=True)
                    try:
                        record = trial(
                            model, tokenizer, ids, case, method=method,
                            budget=budget if budget is not None else opts.context,
                            chunk_size=opts.chunk_size, multiplier=multiplier,
                            max_new_tokens=opts.max_new_tokens,
                            bridge_ids=bridge_ids,
                        )
                    except Exception as exc:
                        record = {
                            "case_id": case.case_id, "task": task, "method": method,
                            "budget_tokens": budget, "trigger_multiplier":
                            multiplier if method.startswith("stream_") else None,
                            "status": "error", "error_type": type(exc).__name__,
                            "error": str(exc), "traceback": traceback.format_exc(limit=12),
                        }
                    rows.append(record)
                    fp.write(json.dumps(record, sort_keys=True) + "\n")
                    fp.flush()
                    print(json.dumps({
                        k: record.get(k) for k in [
                            "status", "method", "all_correct",
                            "kv_after_bytes", "peak_allocated_gb",
                            "ttft_seconds", "tokens_per_second"]}), flush=True)
                    gc.collect()
                    cleanup_cuda()
                del ids
                cleanup_cuda()

    verdict = judge(rows, eq, opts.context)
    if isinstance(verdict, dict) and "decision" in verdict:
        verdict = {"all": verdict}
    passing = [name for name, item in verdict.items()
               if item["decision"] == "PASS_CANDIDATE"]
    skipped = [name for name, item in verdict.items()
               if item["decision"] != "PASS_CANDIDATE"]
    status = ("PASS_CANDIDATE" if passing and not skipped else
              "PARTIAL_CANDIDATE" if passing else "SKIP")
    summary = {
        "status": status, "passing_variants": passing,
        "skipped_variants": skipped, "equivalence": eq,
        "gate": verdict, "records": len(rows),
        "successful": sum(x.get("status") == "ok" for x in rows),
        "errors": sum(x.get("status") != "ok" for x in rows),
        "notes": "GPU validation only. PASS_CANDIDATE is not broad deployment approval.",
    }
    (out / "streaming_gate_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("[streaming gate]", json.dumps(summary, indent=2), flush=True)
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
