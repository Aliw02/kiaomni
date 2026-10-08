"""Small paired benchmark for experimental Streaming V2 vs frozen controls.

Five deterministic task families per selected context. No result is
pre-populated; every reported score and plot is recomputed from raw outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import time
import traceback

import pandas as pd
import torch

from .benchmark import (
    load_model_and_tokenizer, run_full, run_prompt_selection,
    run_true_kv, base_record, score_and_finish
)
from .controlled_benchmark import DEFAULT_TASKS, make_case
from .qwen25_kv_runtime import (
    cleanup_cuda, reset_peak_memory, peak_memory_gb,
    prime_cache, greedy_decode, cuda_sync
)
from .streaming_v2_runtime import streaming_v2_prefill

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TASKS = tuple(DEFAULT_TASKS[:5])


def budgets_for_context(specs: list[str], context: int) -> list[dict]:
    groups: dict[int, dict] = {}
    for value in specs:
        if value.startswith("r"):
            tokens = round(context * float(value[1:]))
        elif value.startswith("B"):
            tokens = int(value[1:])
        else:
            raise ValueError(f"Budget must be Bnnn or r0.x, got {value!r}")
        tokens = min(int(tokens), context)
        if tokens < 48:
            raise ValueError(f"Budget {tokens} violates sink/recency minimum 48")
        entry = groups.setdefault(tokens, {"tokens": tokens, "labels": []})
        entry["labels"].append(value)
    return list(groups.values())


def estimate_full_cache_bytes(model, ctx: int) -> int:
    cfg = model.config
    return (
        int(ctx) * int(cfg.num_hidden_layers)
        * int(cfg.num_key_value_heads)
        * int(cfg.head_dim) * 2 * 2
    )


def run_v2(model, tokenizer, ids, case, *, budget: int,
           budget_label: str, full_ref_bytes: int, bridge_ids,
           options, policy: str, vault_enabled: bool):
    method = f"streaming_v2_{policy}_{'vault' if vault_enabled else 'no_vault'}"
    record = base_record(
        case, method, budget_label, budget, options.model_id, "v2_pilot"
    )
    reset_peak_memory()
    started = time.perf_counter()

    pf = streaming_v2_prefill(
        model, tokenizer, ids, budget=budget, chunk_size=options.chunk_size,
        trigger_multiplier=options.trigger_multiplier, policy=policy,
        vault_tokens=options.vault_tokens, use_vault=vault_enabled,
        memory_guard_mb=options.memory_guard_mb,
    )
    prime = prime_cache(
        model, cache=pf["cache"], forced_token_ids=bridge_ids,
        start_position=ids.shape[1],
    )
    decoded = greedy_decode(
        model, tokenizer,
        initial_logits=prime["logits"], cache=prime["cache"],
        start_position=prime["next_position"],
        max_new_tokens=options.max_new_tokens_effective,
    )
    cuda_sync()
    elapsed = time.perf_counter() - started
    final_bytes = int(pf["final_kv_bytes"])
    record.update({
        "original_tokens": int(ids.shape[1]), "kept_tokens": budget,
        "prompt_compression": False,
        "physical_kv_compression": final_bytes < full_ref_bytes,
        "streaming_kv_eviction": True,
        "reference_full_kv_bytes": full_ref_bytes,
        "kv_before_bytes": None, "kv_after_bytes": final_bytes,
        "kv_reduction_ratio": float(full_ref_bytes / final_bytes),
        "prefill_peak_kv_bytes": int(pf["peak_kv_bytes"]),
        "prefill_peak_kv_reduction_ratio":
            float(full_ref_bytes / max(1, pf["peak_kv_bytes"])),
        "eviction_events": int(pf["eviction_events"]),
        "eviction_log": pf["eviction_log"],
        "evidence_retained": int(pf["selected_evidence_positions"]),
        "prefill_seconds": pf["model_forward_seconds"],
        "saliency_seconds": pf["saliency_seconds"],
        "selection_seconds": pf["selection_seconds"],
        "compaction_seconds": pf["compaction_seconds"],
        "ttft_seconds": float(pf["total_seconds"] + prime["elapsed_seconds"]),
        "bridge_seconds": float(prime["elapsed_seconds"]),
        "total_seconds": float(elapsed),
        "policy": policy, "vault_enabled": vault_enabled,
        "vault_tokens": int(pf["vault_tokens"]),
        "trigger_tokens": int(pf["trigger_tokens"]),
    })
    record.update(peak_memory_gb())
    record = score_and_finish(record, decoded, case)
    del pf, prime, decoded
    cleanup_cuda()
    return record


def summary_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    reference = {
        (r["context_tokens"], r["case_id"]): r
        for r in rows if r["method"] == "full_kv"
    }
    data = []
    # Case-bucket grouping avoids pairing a 2K example with a 4K control.
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        group = (
            row["context_tokens"], row["method"],
            str(row.get("budget_label") or "FULL"),
        )
        groups.setdefault(group, []).append(row)

    for (ctx, method, budget), group in groups.items():
        valid = [r for r in group if r["status"] == "ok"]
        paired = [
            (r, reference[(ctx, r["case_id"])])
            for r in valid if (ctx, r["case_id"]) in reference
            and reference[(ctx, r["case_id"])]["status"] == "ok"
        ]
        regressions = sum(
            bool(fc["all_correct"]) and not bool(r["all_correct"])
            for r, fc in paired
        )
        rescues = sum(
            not bool(fc["all_correct"]) and bool(r["all_correct"])
            for r, fc in paired
        )
        preserved = sum(
            bool(fc["all_correct"]) and bool(r["all_correct"])
            for r, fc in paired
        )
        data.append({
            "context_tokens": ctx, "method": method, "budget": budget,
            "n": len(group), "successful": len(valid),
            "errors": len(group) - len(valid),
            "oom": sum("outofmemory" in str(r.get("error_type", "")).lower()
                       or "out of memory" in str(r.get("error", "")).lower()
                       for r in group),
            "accuracy": round(
                sum(bool(r["all_correct"]) for r in valid)/len(valid), 4
            ) if valid else None,
            "mean_recall": round(
                sum(float(r["answer_recall"]) for r in valid)/len(valid), 4
            ) if valid else None,
            "paired_n": len(paired), "preserved": preserved,
            "regressions": regressions, "rescues": rescues,
            "peak_vram_gb": round(
                sum(r["peak_allocated_gb"] for r in valid)/len(valid), 4
            ) if valid else None,
            "mean_ttft_s": round(
                sum(r["ttft_seconds"] for r in valid)/len(valid), 4
            ) if valid else None,
            "mean_total_s": round(
                sum(r["total_seconds"] for r in valid)/len(valid), 4
            ) if valid else None,
            "mean_tokens_s": round(
                sum(r["tokens_per_second"] for r in valid)/len(valid), 4
            ) if valid else None,
            "mean_kv_reduction": round(
                sum(r["kv_reduction_ratio"] for r in valid)/len(valid), 4
            ) if valid else None,
            "mean_peak_prefill_kv_reduction": round(
                sum(r.get("prefill_peak_kv_reduction_ratio", 1.0) for r in valid)
                / len(valid), 4
            ) if valid else None,
            "mean_ppl": round(
                sum(r["greedy_self_ppl"] for r in valid)/len(valid), 4
            ) if valid else None,
        })
    return pd.DataFrame(data).sort_values(
        ["context_tokens", "budget", "method"]
    ).reset_index(drop=True)


def create_plots(df: pd.DataFrame, dest: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True, exist_ok=True)
    usable = df[df["successful"] > 0].copy()
    if usable.empty:
        return
    usable["label"] = usable["method"] + " | " + usable["budget"]
    charts = [
        ("accuracy", "Accuracy (paired task sets)", "accuracy"),
        ("peak_vram_gb", "Peak GPU allocation (GB)", "peak_vram"),
        ("mean_ttft_s", "Time to first generated token (s)", "ttft"),
        ("mean_kv_reduction", "Final KV storage reduction (x)", "kv_reduction"),
    ]
    for col, title, filename in charts:
        pivot = usable.pivot_table(
            index="context_tokens", columns="label", values=col, aggfunc="mean"
        )
        ax = pivot.plot(marker="o", figsize=(12, 6))
        ax.set_title(title)
        ax.set_xlabel("Context length (tokens)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(dest / (filename + ".png"), dpi=160, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[2048, 4096, 8192])
    parser.add_argument("--budgets", nargs="+", default=["B256", "r0.125"])
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--trigger-multiplier", type=float, default=2.0)
    parser.add_argument("--vault-tokens", type=int, default=100)
    parser.add_argument("--memory-guard-mb", type=int, default=900)
    parser.add_argument("--policies", nargs="+", default=["gaussian"],
                        choices=["gaussian", "s8"])
    parser.add_argument("--compare-no-vault", action="store_true")
    parser.add_argument("--output-dir",
                        default="/kaggle/working/kiaomni_v2_pilot")
    parser.add_argument("--fresh", action="store_true")
    options = parser.parse_args()

    import transformers
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(f"Requires transformers==4.57.6, got {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not 1 <= options.cases <= len(DEFAULT_TASKS):
        raise ValueError(f"cases must be 1..{len(DEFAULT_TASKS)}")
    if not options.contexts or any(x < 512 for x in options.contexts):
        raise ValueError("contexts must all be >=512")
    if options.vault_tokens < 0:
        raise ValueError("vault_tokens cannot be negative")
    if options.trigger_multiplier < 1.0:
        raise ValueError("trigger_multiplier must be >= 1")

    tasks = list(DEFAULT_TASKS[:options.cases])
    budgets_by_context = {
        context: budgets_for_context(options.budgets, context)
        for context in options.contexts
    }
    out = Path(options.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_commit = None
    protocol = {
        "model_id": options.model_id, "contexts": options.contexts,
        "tasks": tasks, "case_seeds": [
            options.seed + i * 31 for i in range(options.cases)
        ],
        "budgets": budgets_by_context, "max_new_tokens": options.max_new_tokens,
        "chunk_size": options.chunk_size,
        "trigger_multiplier": options.trigger_multiplier,
        "vault_tokens": options.vault_tokens, "policies": options.policies,
        "compare_no_vault": options.compare_no_vault,
        "memory_guard_mb": options.memory_guard_mb, "model_quant": "NF4-4bit",
        "transformers": transformers.__version__, "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "git_commit": git_commit,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    protocol["sha256"] = protocol_hash
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and not options.fresh:
        previous = json.loads(protocol_path.read_text())
        if previous["sha256"] != protocol_hash:
            raise RuntimeError("Resume protocol mismatch; choose another output dir or --fresh")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    model, tokenizer, model_info = load_model_and_tokenizer(options.model_id)
    (out / "model_info.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )
    bridge_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not bridge_ids:
        raise ValueError("Empty bridge token")
    opt = SimpleNamespace(
        model_id=options.model_id, mode="v2_pilot",
        window_size=32, n_sink=16, recency=32, sigma=4.0,
        max_new_tokens_effective=options.max_new_tokens,
        chunk_size=options.chunk_size,
        trigger_multiplier=options.trigger_multiplier,
        vault_tokens=options.vault_tokens,
        memory_guard_mb=options.memory_guard_mb,
    )
    run_file = out / "runs.jsonl"
    if options.fresh and run_file.exists():
        run_file.unlink()
    rows = [
        json.loads(x) for x in run_file.read_text().splitlines() if x.strip()
    ] if run_file.exists() else []
    done = {
        (r["case_id"], r["method"], r.get("budget_tokens"))
        for r in rows if r.get("status") == "ok"
    }
    with run_file.open("a", encoding="utf-8") as writer:
        for ctx in options.contexts:
            for task_index, task in enumerate(tasks):
                seed = options.seed + task_index * 31
                case = make_case(tokenizer, task, ctx, seed)
                ids = torch.tensor(
                    [case.input_ids], dtype=torch.long, device="cuda:0"
                )
                base = [("full_kv", None, None, None)]
                for b in budgets_by_context[ctx]:
                    label = ",".join(b["labels"])
                    tokens = b["tokens"]
                    base.extend([
                        ("prompt_selection", tokens, label, None),
                        ("kv_layerwise", tokens, label, None),
                    ])
                    for policy in options.policies:
                        base.append((
                            f"streaming_v2_{policy}_vault", tokens, label,
                            (policy, True)
                        ))
                        if options.compare_no_vault:
                            base.append((
                                f"streaming_v2_{policy}_no_vault", tokens, label,
                                (policy, False)
                            ))
                for method, budget, label, v2 in base:
                    key = (case.case_id, method, budget)
                    if key in done:
                        print(f"[resume] {case.case_id} {method} {label}", flush=True)
                        continue
                    print(f"[run] {case.case_id} {method} {label}", flush=True)
                    try:
                        full = next(
                            (r for r in rows if r["case_id"] == case.case_id
                             and r["method"] == "full_kv"
                             and r["status"] == "ok"), None
                        )
                        full_bytes = (
                            full["kv_after_bytes"] if full else
                            estimate_full_cache_bytes(model, ctx)
                        )
                        if method == "full_kv":
                            record = run_full(
                                model, tokenizer, ids, case,
                                bridge_ids=bridge_ids, args=opt
                            )
                        elif method == "prompt_selection":
                            record = run_prompt_selection(
                                model, tokenizer, ids, case,
                                budget_label=label, budget=budget,
                                full_ref_bytes=full_bytes, bridge_ids=bridge_ids,
                                args=opt
                            )
                        elif method == "kv_layerwise":
                            record = run_true_kv(
                                model, tokenizer, ids, case,
                                variant="layerwise", budget_label=label,
                                budget=budget, full_ref_bytes=full_bytes,
                                bridge_ids=bridge_ids, args=opt
                            )
                        else:
                            policy, vault_on = v2
                            record = run_v2(
                                model, tokenizer, ids, case,
                                budget=budget, budget_label=label,
                                full_ref_bytes=full_bytes,
                                bridge_ids=bridge_ids, options=opt,
                                policy=policy, vault_enabled=vault_on,
                            )
                    except Exception as exc:
                        record = base_record(
                            case, method, label, budget,
                            options.model_id, "v2_pilot"
                        )
                        record.update({
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=12),
                        })
                    rows.append(record)
                    writer.write(json.dumps(record, sort_keys=True) + "\n")
                    writer.flush()
                    if record["status"] == "ok":
                        done.add(key)
                    print(json.dumps({
                        "status": record["status"],
                        "all_correct": record.get("all_correct"),
                        "peak_allocated_gb": record.get("peak_allocated_gb"),
                        "tokens_per_second": record.get("tokens_per_second"),
                        "error": record.get("error"),
                    }), flush=True)
                    cleanup_cuda()
                del ids
                cleanup_cuda()

    # Retain the latest attempt per exact case/method/budget for resumed runs.
    latest = {}
    for row in rows:
        latest[(row["case_id"], row["method"], row.get("budget_tokens"))] = row
    rows = list(latest.values())
    table = summary_table(rows)
    table.to_csv(out / "comparison.csv", index=False)
    create_plots(table, out / "plots")
    (out / "manifest.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash,
        "expected_cases": options.cases * len(options.contexts),
        "total_rows": len(rows),
        "successful": sum(r["status"] == "ok" for r in rows),
        "errors": sum(r["status"] != "ok" for r in rows),
        "comparison": str(out / "comparison.csv"),
        "raw_runs": str(run_file),
        "streaming_status": "EXPERIMENTAL_NOT_VALIDATED",
    }, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print("[done]", out)


if __name__ == "__main__":
    main()
