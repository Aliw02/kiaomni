from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from controlled_benchmark import make_case
from qwen25_kv_runtime import (
    cleanup_cuda,
    greedy_decode,
    peak_memory_gb,
    prime_cache,
    reset_peak_memory,
)
from rescore_strict import strict_score
from run_kaggle_qwen25 import (
    MODEL_ID_DEFAULT,
    base_record,
    load_model_and_tokenizer,
    score_and_finish,
    summarize,
)
from streaming_qwen25_runtime import (
    real_no_eviction_equivalence,
    streaming_prefill,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-runs", required=True)
    p.add_argument(
        "--output-dir",
        default="/kaggle/working/kiaomni_kv_results/streaming_supplement",
    )
    p.add_argument("--model-id", default=None)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["global", "layerwise"],
        default=["global", "layerwise"],
    )
    p.add_argument("--limit-cases", type=int, default=None)
    p.add_argument("--limit-budgets", type=int, default=None)
    p.add_argument("--context-lengths", nargs="+", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-equivalence", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_reference_protocol(reference_runs: Path) -> dict:
    candidates = sorted(reference_runs.parent.glob("*_protocol.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no *_protocol.json next to {reference_runs}"
        )
    # Prefer the protocol with the same prefix as <prefix>_runs.jsonl.
    prefix = reference_runs.name.removesuffix("_runs.jsonl")
    preferred = reference_runs.parent / f"{prefix}_protocol.json"
    path = preferred if preferred.exists() else candidates[0]
    return json.loads(path.read_text(encoding="utf-8"))


def strict_rescore_reference(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        if r.get("status") == "ok":
            strict = strict_score(
                r.get("output_text", ""),
                list(r.get("expected_answers", [])),
            )
            r["legacy_all_correct"] = r.get("all_correct")
            r["legacy_answer_recall"] = r.get("answer_recall")
            r["all_correct"] = bool(strict["strict_all_correct"])
            r["answer_recall"] = float(strict["strict_answer_recall"])
            r["answer_hits"] = strict["strict_hits"]
            r["strict_first_line"] = strict["strict_first_line"]
        out.append(r)
    return out


def reference_cases(rows: list[dict]) -> list[dict]:
    # Include failed FullKV rows as well. This is required for contexts that
    # FullKV cannot materialize on the target GPU (for example 8K on T4) but
    # streaming may still be able to process.
    cases = [
        r
        for r in rows
        if r.get("method") == "full_kv"
    ]
    cases.sort(
        key=lambda r: (
            int(r["context_tokens"]),
            str(r["task"]),
            str(r["case_id"]),
        )
    )
    return cases


def reference_budgets(rows: list[dict], protocol: dict) -> dict[int, list[tuple[str, int]]]:
    # Prefer budgets actually present in successful reference rows. For a
    # context where FullKV OOM prevented compressed jobs from running, rebuild
    # the same budget schedule from the frozen protocol.
    per_context: dict[int, dict[int, str]] = {}
    contexts = sorted({int(r["context_tokens"]) for r in rows if r.get("method") == "full_kv"})
    for row in rows:
        if row.get("status") != "ok":
            continue
        budget = row.get("budget_tokens")
        label = row.get("budget_label")
        if budget is None or label is None:
            continue
        ctx = int(row["context_tokens"])
        per_context.setdefault(ctx, {})
        per_context[ctx].setdefault(int(budget), str(label))

    specs = list(protocol.get("budget_specs", []))
    for ctx in contexts:
        if per_context.get(ctx):
            continue
        seen: set[int] = set()
        rebuilt: dict[int, str] = {}
        for spec in specs:
            if isinstance(spec, str) and spec.startswith("r"):
                ratio = float(spec[1:])
                budget = int(round(ctx * ratio))
                label = spec
            else:
                budget = int(spec)
                label = f"B{budget}"
            budget = min(budget, ctx)
            if budget <= 0 or budget in seen:
                continue
            seen.add(budget)
            rebuilt[budget] = label
        per_context[ctx] = rebuilt

    return {
        ctx: [(label_by_budget[budget], budget) for budget in sorted(label_by_budget)]
        for ctx, label_by_budget in per_context.items()
    }


def full_reference_bytes(rows: list[dict]) -> tuple[dict[str, int], float]:
    # Ground byte-per-token scaling in an actual successful FullKV run, then
    # extrapolate linearly for FullKV-OOM contexts. Raw KV storage is linear in
    # sequence length for fixed model/cache dtype.
    successful = [
        r for r in rows
        if r.get("status") == "ok"
        and r.get("method") == "full_kv"
        and r.get("kv_after_bytes") is not None
    ]
    if not successful:
        raise RuntimeError("need at least one successful FullKV row to calibrate KV bytes/token")
    calibration = successful[0]
    bytes_per_token = float(calibration["kv_after_bytes"]) / float(calibration["context_tokens"])

    out: dict[str, int] = {}
    for r in rows:
        if r.get("method") != "full_kv":
            continue
        if r.get("status") == "ok" and r.get("kv_after_bytes") is not None:
            out[str(r["case_id"])] = int(r["kv_after_bytes"])
        else:
            out[str(r["case_id"])] = int(round(bytes_per_token * int(r["context_tokens"])))
    return out, bytes_per_token


def run_streaming_method(
    model,
    tokenizer,
    ids,
    case,
    *,
    variant: str,
    budget_label: str,
    budget: int,
    bridge_ids: list[int],
    full_ref_bytes: int,
    chunk_size: int,
    settings,
):
    method = (
        "stream_global"
        if variant == "global"
        else "stream_layerwise"
    )
    record = base_record(
        case,
        method,
        budget_label,
        budget,
        settings.model_id,
        "streaming_supplement",
    )

    reset_peak_memory()
    total_started = time.perf_counter()

    stream = streaming_prefill(
        model,
        ids,
        budget=budget,
        chunk_size=chunk_size,
        variant=variant,
        window_size=settings.window_size,
        n_sink=settings.n_sink,
        recency=settings.recency,
        sigma=settings.sigma,
    )

    prime = prime_cache(
        model,
        cache=stream["cache"],
        forced_token_ids=bridge_ids,
        start_position=ids.shape[1],
    )
    decoded = greedy_decode(
        model,
        tokenizer,
        initial_logits=prime["logits"],
        cache=prime["cache"],
        start_position=prime["next_position"],
        max_new_tokens=settings.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - total_started

    final_bytes = int(stream["final_kv_bytes"])
    peak_kv_bytes = int(stream["peak_kv_bytes"])

    record.update(
        {
            "original_tokens": int(ids.shape[1]),
            "kept_tokens": int(budget),
            "prompt_compression": False,
            "physical_kv_compression": True,
            "streaming_kv_eviction": True,
            "streaming_chunk_size": int(chunk_size),
            "reference_full_kv_bytes": int(full_ref_bytes),
            "kv_before_bytes": None,
            "kv_after_bytes": final_bytes,
            "kv_reduction_ratio": float(
                full_ref_bytes / final_bytes
            ),
            "prefill_peak_kv_bytes": peak_kv_bytes,
            "prefill_peak_kv_reduction_ratio": float(
                full_ref_bytes / peak_kv_bytes
            ),
            "eviction_events": int(stream["eviction_events"]),
            "prefill_seconds": float(
                stream["model_forward_seconds"]
            ),
            "streaming_prefill_total_seconds": float(
                stream["total_seconds"]
            ),
            "selection_seconds": float(
                stream["selection_seconds"]
            ),
            "compaction_seconds": float(
                stream["compaction_seconds"]
            ),
            "bridge_seconds": float(prime["elapsed_seconds"]),
            "ttft_seconds": float(
                stream["total_seconds"]
                + prime["elapsed_seconds"]
            ),
            "total_seconds": float(total_seconds),
            "cache_lengths_after_compaction": stream[
                "final_cache_lengths"
            ],
        }
    )
    record.update(peak_memory_gb())
    record = score_and_finish(
        record,
        decoded,
        case,
    )

    del stream, prime, decoded
    cleanup_cuda()
    return record


def main():
    args = parse_args()
    reference_path = Path(args.reference_runs).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_raw = load_jsonl(reference_path)
    reference_strict = strict_rescore_reference(reference_raw)
    protocol = read_reference_protocol(reference_path)

    model_id = (
        args.model_id
        or protocol.get("model_id")
        or MODEL_ID_DEFAULT
    )
    settings = SimpleNamespace(
        model_id=model_id,
        window_size=int(protocol.get("window_size", 32)),
        n_sink=int(protocol.get("n_sink", 16)),
        recency=int(protocol.get("recency", 32)),
        sigma=float(protocol.get("sigma", 4.0)),
        max_new_tokens=int(protocol.get("max_new_tokens", 40)),
    )

    cases = reference_cases(reference_strict)
    if args.context_lengths is not None:
        allowed_contexts = {int(v) for v in args.context_lengths}
        cases = [r for r in cases if int(r["context_tokens"]) in allowed_contexts]
    if args.limit_cases is not None:
        # Limit per context, not globally, so a two-context smoke can test both.
        limited = []
        counts: dict[int, int] = {}
        for r in cases:
            ctx = int(r["context_tokens"])
            if counts.get(ctx, 0) >= int(args.limit_cases):
                continue
            limited.append(r)
            counts[ctx] = counts.get(ctx, 0) + 1
        cases = limited

    budgets_by_context = reference_budgets(reference_strict, protocol)
    full_bytes, full_bytes_per_token = full_reference_bytes(reference_strict)

    supplement_protocol = {
        "experiment": "kiaomni_qwen25_streaming_supplement_v1",
        "reference_runs": str(reference_path),
        "reference_protocol_sha256": protocol.get("sha256"),
        "model_id": model_id,
        "chunk_size": int(args.chunk_size),
        "variants": list(args.variants),
        "window_size": settings.window_size,
        "n_sink": settings.n_sink,
        "recency": settings.recency,
        "sigma": settings.sigma,
        "max_new_tokens": settings.max_new_tokens,
        "case_count": len(cases),
        "context_lengths": sorted({int(r["context_tokens"]) for r in cases}),
        "full_kv_bytes_per_token_calibration": full_bytes_per_token,
        "scoring": "strict_first_non_empty_line",
        "comparison_methods": [
            "full_kv",
            "prompt_selection",
            "kv_global",
            "kv_layerwise",
            "stream_global",
            "stream_layerwise",
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            supplement_protocol,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    supplement_protocol["sha256"] = digest
    (
        out_dir / "streaming_protocol.json"
    ).write_text(
        json.dumps(
            supplement_protocol,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[protocol] sha256={digest}")

    runs_path = out_dir / "streaming_runs.jsonl"
    if not args.resume and runs_path.exists():
        runs_path.unlink()

    existing = load_jsonl(runs_path) if args.resume else []
    done = {
        (
            str(r.get("case_id")),
            str(r.get("method")),
            int(r.get("budget_tokens")),
        )
        for r in existing
        if r.get("status") == "ok"
        and r.get("budget_tokens") is not None
    }
    streaming_rows = list(existing)

    model, tokenizer, model_info = load_model_and_tokenizer(
        model_id
    )
    capability_rows = []
    for r in reference_strict:
        if r.get("method") != "full_kv":
            continue
        capability_rows.append({
            "case_id": r.get("case_id"),
            "context_tokens": r.get("context_tokens"),
            "task": r.get("task"),
            "full_kv_status": r.get("status"),
            "full_kv_error_type": r.get("error_type"),
            "full_kv_reference_bytes": full_bytes.get(str(r.get("case_id"))),
        })
    pd.DataFrame(capability_rows).to_csv(out_dir / "reference_capability.csv", index=False)
    (
        out_dir / "model_info.json"
    ).write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    bridge_ids = tokenizer.encode(
        "\n",
        add_special_tokens=False,
    )
    if not bridge_ids:
        raise RuntimeError("tokenizer produced no bridge token ids")

    device = torch.device("cuda:0")

    if not args.skip_equivalence:
        first = cases[0]
        eq_length = min(1024, int(first["context_tokens"]))
        eq_case = make_case(
            tokenizer,
            str(first["task"]),
            eq_length,
            int(first["case_seed"]),
        )
        eq_ids = torch.tensor(
            [eq_case.input_ids],
            dtype=torch.long,
            device=device,
        )
        print(
            f"[equivalence] context={eq_length} "
            f"chunk={args.chunk_size}"
        )
        equivalence = real_no_eviction_equivalence(
            model,
            eq_ids,
            chunk_size=int(args.chunk_size),
            window_size=settings.window_size,
        )
        (
            out_dir / "streaming_equivalence.json"
        ).write_text(
            json.dumps(equivalence, indent=2),
            encoding="utf-8",
        )
        print(
            "[equivalence]",
            json.dumps(equivalence, indent=2),
        )
        del eq_ids
        cleanup_cuda()

    for case_ref in cases:
        context = int(case_ref["context_tokens"])
        case = make_case(
            tokenizer,
            str(case_ref["task"]),
            context,
            int(case_ref["case_seed"]),
        )
        if case.case_id != str(case_ref["case_id"]):
            raise RuntimeError(
                "deterministic case reconstruction mismatch: "
                f"{case.case_id} != {case_ref['case_id']}"
            )

        ids = torch.tensor(
            [case.input_ids],
            dtype=torch.long,
            device=device,
        )
        case_budgets = list(
            budgets_by_context.get(context, [])
        )
        if args.limit_budgets is not None:
            case_budgets = case_budgets[
                : int(args.limit_budgets)
            ]

        if case.case_id not in full_bytes:
            raise RuntimeError(
                f"missing FullKV byte reference for {case.case_id}"
            )

        for budget_label, budget in case_budgets:
            for variant in args.variants:
                method = (
                    "stream_global"
                    if variant == "global"
                    else "stream_layerwise"
                )
                key = (
                    case.case_id,
                    method,
                    int(budget),
                )
                if key in done:
                    continue

                print(
                    f"[run] {case.case_id} "
                    f"{method} {budget_label}={budget} "
                    f"chunk={args.chunk_size}"
                )
                try:
                    row = run_streaming_method(
                        model,
                        tokenizer,
                        ids,
                        case,
                        variant=variant,
                        budget_label=budget_label,
                        budget=int(budget),
                        bridge_ids=bridge_ids,
                        full_ref_bytes=full_bytes[
                            case.case_id
                        ],
                        chunk_size=int(args.chunk_size),
                        settings=settings,
                    )
                except Exception as exc:
                    row = base_record(
                        case,
                        method,
                        budget_label,
                        int(budget),
                        model_id,
                        "streaming_supplement",
                    )
                    row.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(
                                limit=30
                            ),
                        }
                    )
                    cleanup_cuda()

                append_jsonl(runs_path, row)
                streaming_rows.append(row)
                if row.get("status") == "ok":
                    done.add(key)

                print(
                    json.dumps(
                        {
                            k: row.get(k)
                            for k in (
                                "status",
                                "method",
                                "budget_label",
                                "all_correct",
                                "answer_recall",
                                "kv_reduction_ratio",
                                "prefill_peak_kv_reduction_ratio",
                                "peak_allocated_gb",
                                "ttft_seconds",
                                "tokens_per_second",
                            )
                        },
                        indent=2,
                    )
                )

        del ids
        cleanup_cuda()

    # Re-score streaming rows too, then combine them with the already-computed
    # strict reference rows. This makes the final table directly comparable
    # even when the reference Final used the legacy anywhere-in-output scorer.
    streaming_strict = strict_rescore_reference(
        streaming_rows
    )
    combined = reference_strict + streaming_strict

    combined_path = out_dir / "combined_runs_strict.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(
                json.dumps(row, sort_keys=True)
                + "\n"
            )

    summarize(combined, out_dir)

    manifest = {
        "protocol_sha256": digest,
        "reference_records": len(reference_strict),
        "streaming_records": len(streaming_strict),
        "streaming_successful": sum(
            r.get("status") == "ok"
            for r in streaming_strict
        ),
        "streaming_errors": sum(
            r.get("status") != "ok"
            for r in streaming_strict
        ),
        "combined_runs": str(combined_path),
        "summary": str(out_dir / "summary.csv"),
        "paired_summary": str(
            out_dir / "paired_summary.csv"
        ),
    }
    (
        out_dir / "streaming_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("\n[done]", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
