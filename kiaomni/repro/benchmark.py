from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from .controlled_benchmark import DEFAULT_TASKS, make_case, score_output
from .kv_policy_qwen25 import global_mask, layerwise_masks
from .qwen25_kv_runtime import (
    cache_bytes,
    cleanup_cuda,
    compact_cache_global,
    compact_cache_layerwise,
    cuda_sync,
    greedy_decode,
    peak_memory_gb,
    prefill,
    prime_cache,
    reset_peak_memory,
    saliency_only,
    validate_qwen25_model,
)


MODEL_ID_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

MODE_CONFIGS = {
    "smoke": {
        "context_lengths": [2048],
        "budget_specs": ["r0.125"],
        "tasks": ["single_needle", "multi_query"],
        "repeats": 1,
        "max_new_tokens": 24,
    },
    "pilot": {
        "context_lengths": [4096],
        "budget_specs": [256, 512, "r0.125", "r0.25"],
        "tasks": list(DEFAULT_TASKS),
        "repeats": 1,
        "max_new_tokens": 128,
    },
    "final": {
        "context_lengths": [4096, 8192],
        "budget_specs": [98, 128, 256, 512, "r0.0625", "r0.125", "r0.25"],
        "tasks": list(DEFAULT_TASKS),
        "repeats": 2,
        "max_new_tokens": 256,
    },
}

METHODS = ("full_kv", "prompt_selection", "kv_global", "kv_layerwise")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="smoke")
    p.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    p.add_argument("--output-dir", default="results/kv_cache_compression_v1/kaggle_qwen25_7b")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--n-sink", type=int, default=16)
    p.add_argument("--recency", type=int, default=32)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def resolve_budgets(specs, context_length: int):
    out = []
    seen = set()
    for spec in specs:
        if isinstance(spec, str) and spec.startswith("r"):
            ratio = float(spec[1:])
            budget = int(round(context_length * ratio))
            label = spec
        else:
            budget = int(spec)
            label = f"B{budget}"
        budget = min(budget, context_length)
        if budget <= 0 or budget in seen:
            continue
        seen.add(budget)
        out.append((label, budget))
    return out


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_key(record: dict):
    return (
        record.get("case_id"),
        record.get("method"),
        record.get("budget_tokens"),
    )


def load_model_and_tokenizer(model_id: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Kaggle Qwen2.5-7B experiment")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    info = validate_qwen25_model(model)
    return model, tokenizer, info


def base_record(case, method: str, budget_label, budget_tokens, model_id: str, mode: str):
    return {
        "case_id": case.case_id,
        "task": case.task,
        "context_tokens": int(len(case.input_ids)),
        "case_seed": int(case.metadata["seed"]),
        "fact_depths": case.metadata["fact_depths"],
        "expected_answers": list(case.expected_answers),
        "method": method,
        "budget_label": budget_label,
        "budget_tokens": budget_tokens,
        "model_id": model_id,
        "mode": mode,
        "status": "ok",
    }


def score_and_finish(record: dict, decoded: dict, case) -> dict:
    score = score_output(decoded["text"], case.expected_answers)
    record.update({
        "output_text": decoded["text"],
        "generated_tokens": int(decoded["generated_tokens"]),
        "decode_seconds": float(decoded["decode_seconds"]),
        "tokens_per_second": decoded["tokens_per_second"],
        "greedy_self_ppl": float(decoded["greedy_self_ppl"]),
        "all_correct": bool(score["all_correct"]),
        "answer_recall": float(score["answer_recall"]),
        "answer_hits": score["hits"],
    })
    return record


def run_full(model, tokenizer, ids, case, *, bridge_ids, args):
    record = base_record(case, "full_kv", None, None, args.model_id, args.mode)
    reset_peak_memory()
    total_started = time.perf_counter()

    pf = prefill(model, ids, collect_saliency=False, window_size=args.window_size)
    kv = cache_bytes(pf["cache"])
    prime = prime_cache(
        model,
        cache=pf["cache"],
        forced_token_ids=bridge_ids,
        start_position=ids.shape[1],
    )
    decoded = greedy_decode(
        model,
        tokenizer,
        initial_logits=prime["logits"],
        cache=prime["cache"],
        start_position=prime["next_position"],
        max_new_tokens=args.max_new_tokens_effective,
    )
    cuda_sync()
    total_elapsed = time.perf_counter() - total_started

    record.update({
        "original_tokens": int(ids.shape[1]),
        "kept_tokens": int(ids.shape[1]),
        "prompt_compression": False,
        "physical_kv_compression": False,
        "reference_full_kv_bytes": int(kv),
        "kv_before_bytes": int(kv),
        "kv_after_bytes": int(kv),
        "kv_reduction_ratio": 1.0,
        "prefill_seconds": float(pf["elapsed_seconds"]),
        "saliency_seconds": 0.0,
        "selection_seconds": 0.0,
        "compaction_seconds": 0.0,
        "bridge_seconds": float(prime["elapsed_seconds"]),
        "ttft_seconds": float(pf["elapsed_seconds"] + prime["elapsed_seconds"]),
        "total_seconds": float(total_elapsed),
    })
    record.update(peak_memory_gb())
    record = score_and_finish(record, decoded, case)
    del pf, prime, decoded
    cleanup_cuda()
    return record


def run_prompt_selection(model, tokenizer, ids, case, *, budget_label, budget, bridge_ids, full_ref_bytes, args):
    record = base_record(case, "prompt_selection", budget_label, budget, args.model_id, args.mode)
    reset_peak_memory()
    total_started = time.perf_counter()

    sal = saliency_only(model, ids, window_size=args.window_size)
    select_started = time.perf_counter()
    keep = global_mask(
        sal["layer_saliencies"],
        budget,
        n_sink=args.n_sink,
        recency=args.recency,
        sigma=args.sigma,
    )
    selection_seconds = time.perf_counter() - select_started
    keep_t = torch.as_tensor(keep, dtype=torch.long, device=ids.device)
    pruned = ids.index_select(1, keep_t)

    pf = prefill(model, pruned, collect_saliency=False, window_size=args.window_size)
    pruned_kv = cache_bytes(pf["cache"])
    prime = prime_cache(
        model,
        cache=pf["cache"],
        forced_token_ids=bridge_ids,
        start_position=pruned.shape[1],
    )
    decoded = greedy_decode(
        model,
        tokenizer,
        initial_logits=prime["logits"],
        cache=prime["cache"],
        start_position=prime["next_position"],
        max_new_tokens=args.max_new_tokens_effective,
    )
    cuda_sync()
    total_elapsed = time.perf_counter() - total_started

    record.update({
        "original_tokens": int(ids.shape[1]),
        "kept_tokens": int(pruned.shape[1]),
        "prompt_compression": True,
        "physical_kv_compression": False,
        "reference_full_kv_bytes": int(full_ref_bytes),
        "kv_before_bytes": None,
        "kv_after_bytes": int(pruned_kv),
        "kv_reduction_ratio": float(full_ref_bytes / pruned_kv),
        "prefill_seconds": float(pf["elapsed_seconds"]),
        "saliency_seconds": float(sal["elapsed_seconds"]),
        "selection_seconds": float(selection_seconds),
        "compaction_seconds": 0.0,
        "bridge_seconds": float(prime["elapsed_seconds"]),
        "ttft_seconds": float(
            sal["elapsed_seconds"] + selection_seconds + pf["elapsed_seconds"] + prime["elapsed_seconds"]
        ),
        "total_seconds": float(total_elapsed),
    })
    record.update(peak_memory_gb())
    record = score_and_finish(record, decoded, case)
    del sal, keep, keep_t, pruned, pf, prime, decoded
    cleanup_cuda()
    return record


def run_true_kv(model, tokenizer, ids, case, *, variant, budget_label, budget, bridge_ids, full_ref_bytes, args):
    method = "kv_global" if variant == "global" else "kv_layerwise"
    record = base_record(case, method, budget_label, budget, args.model_id, args.mode)
    reset_peak_memory()
    total_started = time.perf_counter()

    pf = prefill(model, ids, collect_saliency=True, window_size=args.window_size)

    select_started = time.perf_counter()
    if variant == "global":
        keep = global_mask(
            pf["layer_saliencies"],
            budget,
            n_sink=args.n_sink,
            recency=args.recency,
            sigma=args.sigma,
        )
    else:
        keep = layerwise_masks(
            pf["layer_saliencies"],
            budget,
            n_sink=args.n_sink,
            recency=args.recency,
            sigma=args.sigma,
        )
    selection_seconds = time.perf_counter() - select_started

    cuda_sync()
    compact_started = time.perf_counter()
    if variant == "global":
        compact = compact_cache_global(pf["cache"], keep)
    else:
        compact = compact_cache_layerwise(pf["cache"], keep)
    cuda_sync()
    compaction_seconds = time.perf_counter() - compact_started

    prime = prime_cache(
        model,
        cache=pf["cache"],
        forced_token_ids=bridge_ids,
        start_position=ids.shape[1],
    )
    decoded = greedy_decode(
        model,
        tokenizer,
        initial_logits=prime["logits"],
        cache=prime["cache"],
        start_position=prime["next_position"],
        max_new_tokens=args.max_new_tokens_effective,
    )
    cuda_sync()
    total_elapsed = time.perf_counter() - total_started

    post_bytes = int(compact["after_bytes"])
    before_bytes = int(compact["before_bytes"])
    record.update({
        "original_tokens": int(ids.shape[1]),
        "kept_tokens": int(budget),
        "prompt_compression": False,
        "physical_kv_compression": True,
        "reference_full_kv_bytes": int(full_ref_bytes),
        "kv_before_bytes": before_bytes,
        "kv_after_bytes": post_bytes,
        "kv_reduction_ratio": float(before_bytes / post_bytes),
        "prefill_seconds": float(pf["elapsed_seconds"]),
        "saliency_seconds": None,
        "selection_seconds": float(selection_seconds),
        "compaction_seconds": float(compaction_seconds),
        "bridge_seconds": float(prime["elapsed_seconds"]),
        "ttft_seconds": float(
            pf["elapsed_seconds"] + selection_seconds + compaction_seconds + prime["elapsed_seconds"]
        ),
        "total_seconds": float(total_elapsed),
        "cache_lengths_after_compaction": compact["lengths"],
    })
    record.update(peak_memory_gb())
    record = score_and_finish(record, decoded, case)
    del pf, keep, compact, prime, decoded
    cleanup_cuda()
    return record


def summarize(records: list[dict], out_dir: Path):
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return

    df = pd.DataFrame(ok)
    numeric = [
        "all_correct",
        "answer_recall",
        "kv_reduction_ratio",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "ttft_seconds",
        "total_seconds",
        "tokens_per_second",
        "greedy_self_ppl",
    ]
    agg = (
        df.groupby(["context_tokens", "method", "budget_label"], dropna=False)[numeric]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"all_correct": "accuracy"})
    )
    counts = (
        df.groupby(["context_tokens", "method", "budget_label"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    summary = counts.merge(agg, on=["context_tokens", "method", "budget_label"], how="left")
    summary.to_csv(out_dir / "summary.csv", index=False)

    full = {
        r["case_id"]: r for r in ok
        if r["method"] == "full_kv"
    }
    paired_rows = []
    for r in ok:
        if r["method"] == "full_kv" or r["case_id"] not in full:
            continue
        f = full[r["case_id"]]
        paired_rows.append({
            "case_id": r["case_id"],
            "context_tokens": r["context_tokens"],
            "method": r["method"],
            "budget_label": r["budget_label"],
            "budget_tokens": r["budget_tokens"],
            "full_correct": bool(f["all_correct"]),
            "compressed_correct": bool(r["all_correct"]),
            "full_recall": float(f["answer_recall"]),
            "compressed_recall": float(r["answer_recall"]),
            "recall_delta": float(r["answer_recall"] - f["answer_recall"]),
            "preserved": bool(f["all_correct"] and r["all_correct"]),
            "regression": bool(f["all_correct"] and not r["all_correct"]),
            "rescue": bool((not f["all_correct"]) and r["all_correct"]),
            "both_wrong": bool((not f["all_correct"]) and (not r["all_correct"])),
        })
    if paired_rows:
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(out_dir / "paired_cases.csv", index=False)
        paired_summary = (
            paired.groupby(["context_tokens", "method", "budget_label"])
            .agg(
                n=("case_id", "count"),
                preserved=("preserved", "sum"),
                regressions=("regression", "sum"),
                rescues=("rescue", "sum"),
                both_wrong=("both_wrong", "sum"),
                mean_recall_delta=("recall_delta", "mean"),
            )
            .reset_index()
        )
        paired_summary.to_csv(out_dir / "paired_summary.csv", index=False)


def main():
    args = parse_args()
    cfg = dict(MODE_CONFIGS[args.mode])
    args.max_new_tokens_effective = int(args.max_new_tokens or cfg["max_new_tokens"])

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / f"{args.mode}_runs.jsonl"

    protocol = {
        "experiment": "kiaomni_true_kv_qwen25_7b_v1",
        "mode": args.mode,
        "model_id": args.model_id,
        "context_lengths": cfg["context_lengths"],
        "budget_specs": cfg["budget_specs"],
        "tasks": cfg["tasks"],
        "repeats": cfg["repeats"],
        "methods": args.methods,
        "window_size": args.window_size,
        "n_sink": args.n_sink,
        "recency": args.recency,
        "sigma": args.sigma,
        "max_new_tokens": args.max_new_tokens_effective,
        "seed": args.seed,
        "bridge_text": "\n",
        "transformers_expected": "4.57.6",
        "quantization": "NF4 4-bit weights, FP16 compute",
        "attention_backend": "sdpa",
        "batch_size": 1,
        "quality_note": "First free answer token is predicted only after a forced post-compression bridge token.",
        "dataset_note": "Deterministic controlled long-context benchmark; RULER-style tasks but not official RULER.",
    }
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode("utf-8")
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    protocol["sha256"] = protocol_hash
    json_dump(out_dir / f"{args.mode}_protocol.json", protocol)
    print(f"[protocol] sha256={protocol_hash}")

    if not args.resume and runs_path.exists():
        runs_path.unlink()
    existing = load_existing(runs_path) if args.resume else []
    done = {run_key(r) for r in existing if r.get("status") == "ok"}
    records = list(existing)

    model, tokenizer, model_info = load_model_and_tokenizer(args.model_id)
    json_dump(out_dir / "model_info.json", model_info)
    print("[model]", json.dumps(model_info, indent=2))

    bridge_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not bridge_ids:
        raise RuntimeError("tokenizer produced no bridge token ids")
    print(f"[bridge] token_ids={bridge_ids}")

    device = torch.device("cuda:0")
    full_refs = {
        r["case_id"]: int(r["kv_after_bytes"])
        for r in records
        if r.get("status") == "ok" and r.get("method") == "full_kv"
    }

    for context_length in cfg["context_lengths"]:
        budgets = resolve_budgets(cfg["budget_specs"], context_length)
        for repeat_idx in range(int(cfg["repeats"])):
            for task_idx, task in enumerate(cfg["tasks"]):
                seed = int(args.seed + context_length * 10 + repeat_idx * 1000 + task_idx * 37)
                case = make_case(tokenizer, task, context_length, seed)
                ids = torch.tensor([case.input_ids], dtype=torch.long, device=device)

                if "full_kv" in args.methods and (case.case_id, "full_kv", None) not in done:
                    try:
                        print(f"\n[run] {case.case_id} full_kv")
                        rec = run_full(model, tokenizer, ids, case, bridge_ids=bridge_ids, args=args)
                    except Exception as exc:
                        rec = base_record(case, "full_kv", None, None, args.model_id, args.mode)
                        rec.update({
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=20),
                        })
                        cleanup_cuda()
                    append_jsonl(runs_path, rec)
                    records.append(rec)
                    if rec.get("status") == "ok":
                        done.add(run_key(rec))
                        full_refs[case.case_id] = int(rec["kv_after_bytes"])
                    print(json.dumps({k: rec.get(k) for k in (
                        "status", "method", "all_correct", "answer_recall",
                        "kv_reduction_ratio", "peak_allocated_gb", "ttft_seconds"
                    )}, indent=2))

                full_ref = full_refs.get(case.case_id)
                if full_ref is None:
                    print(f"[skip] compressed methods require successful full_kv for {case.case_id}")
                    del ids
                    cleanup_cuda()
                    continue

                for budget_label, budget in budgets:
                    if budget < args.n_sink + args.recency:
                        print(f"[skip] {budget_label}={budget} below protected minimum")
                        continue

                    compressed_jobs = []
                    if "prompt_selection" in args.methods:
                        compressed_jobs.append(("prompt_selection", None))
                    if "kv_global" in args.methods:
                        compressed_jobs.append(("kv_global", "global"))
                    if "kv_layerwise" in args.methods:
                        compressed_jobs.append(("kv_layerwise", "layerwise"))

                    for method, variant in compressed_jobs:
                        key = (case.case_id, method, budget)
                        if key in done:
                            continue
                        try:
                            print(f"[run] {case.case_id} {method} {budget_label}={budget}")
                            if method == "prompt_selection":
                                rec = run_prompt_selection(
                                    model, tokenizer, ids, case,
                                    budget_label=budget_label,
                                    budget=budget,
                                    bridge_ids=bridge_ids,
                                    full_ref_bytes=full_ref,
                                    args=args,
                                )
                            else:
                                rec = run_true_kv(
                                    model, tokenizer, ids, case,
                                    variant=variant,
                                    budget_label=budget_label,
                                    budget=budget,
                                    bridge_ids=bridge_ids,
                                    full_ref_bytes=full_ref,
                                    args=args,
                                )
                        except Exception as exc:
                            rec = base_record(case, method, budget_label, budget, args.model_id, args.mode)
                            rec.update({
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(limit=20),
                            })
                            cleanup_cuda()

                        append_jsonl(runs_path, rec)
                        records.append(rec)
                        if rec.get("status") == "ok":
                            done.add(run_key(rec))
                        print(json.dumps({k: rec.get(k) for k in (
                            "status", "method", "budget_label", "all_correct", "answer_recall",
                            "kv_reduction_ratio", "peak_allocated_gb", "ttft_seconds"
                        )}, indent=2))

                del ids
                cleanup_cuda()
                summarize(records, out_dir)

    summarize(records, out_dir)
    final_manifest = {
        "protocol_sha256": protocol_hash,
        "mode": args.mode,
        "records": len(records),
        "successful": sum(r.get("status") == "ok" for r in records),
        "errors": sum(r.get("status") != "ok" for r in records),
        "runs_file": str(runs_path),
        "summary_file": str(out_dir / "summary.csv"),
        "paired_summary_file": str(out_dir / "paired_summary.csv"),
    }
    json_dump(out_dir / f"{args.mode}_manifest.json", final_manifest)
    print("\n[done]", json.dumps(final_manifest, indent=2))


if __name__ == "__main__":
    main()
