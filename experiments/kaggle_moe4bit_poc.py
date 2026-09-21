"""
Kaggle 16GB POC: KiaOmni + adaptive low-jitter MoE routing.

Default target:
    cyankiwi/LFM2.5-8B-A1B-AWQ-INT4

The checkpoint is a modern 4-bit AWQ MoE small enough to leave substantial
headroom on a single 16GB GPU. The experiment keeps model weights frozen and
runs four controlled arms:

A. vanilla model
B. adaptive route stability only
C. KiaOmni only
D. KiaOmni + adaptive route stability

The output JSON records quality preservation, needle retrieval, routing
stability, latency, throughput, and peak allocated VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import (
    apply_kiaomni,
    apply_moe_route_stability,
    remove_kiaomni,
    remove_moe_route_stability,
)


DEFAULT_MODEL = "cyankiwi/LFM2.5-8B-A1B-AWQ-INT4"
DEFAULT_OUT = "results/moe_route_stability_poc.json"


SHORT_CASES = [
    {
        "name": "math",
        "prompt": "Reply with only the answer. What is 37 * 19?",
        "max_new_tokens": 12,
    },
    {
        "name": "science",
        "prompt": (
            "Reply in one short sentence. Why does pressure rise when a sealed "
            "rigid container of gas is heated?"
        ),
        "max_new_tokens": 28,
    },
    {
        "name": "logic",
        "prompt": (
            "Reply with only the final number. A box has 12 red balls and twice "
            "as many blue balls. Five blue balls are removed. How many balls remain?"
        ),
        "max_new_tokens": 16,
    },
    {
        "name": "code",
        "prompt": (
            "Write only one Python expression that returns the largest value "
            "from a list named values."
        ),
        "max_new_tokens": 20,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=int, default=768)
    parser.add_argument("--alpha-max", type=float, default=0.10)
    parser.add_argument("--long-tokens", type=int, default=1800)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_name": props.name,
        "total_vram_gb": props.total_memory / (1024**3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def first_cuda_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    return next(model.parameters()).device


def build_needle_case(
    tokenizer,
    *,
    name: str,
    key: str,
    fraction: float,
    target_tokens: int,
) -> dict[str, Any]:
    filler = (
        "Field note: the river level was normal, the instruments were calibrated, "
        "and no unusual event was recorded during this observation. "
    )
    record = f"CRITICAL RECORD: the access code is {key}. "
    question = (
        "\nQuestion: According to the CRITICAL RECORD, what is the access code? "
        "Reply with only the code."
    )

    repeats = max(20, target_tokens // 18)
    parts = [filler] * repeats
    insert_at = min(len(parts) - 1, max(0, int(len(parts) * fraction)))
    parts.insert(insert_at, record)
    prompt = "".join(parts) + question

    # Grow until the requested scale is reached. Exact equality is unnecessary;
    # the measured prompt token count is recorded in the artifact.
    while len(tokenizer(prompt, add_special_tokens=False).input_ids) < target_tokens:
        parts.append(filler)
        prompt = "".join(parts) + question

    return {
        "name": name,
        "prompt": prompt,
        "expected": key,
        "max_new_tokens": 16,
        "kind": "needle",
    }


def build_cases(tokenizer, target_tokens: int) -> list[dict[str, Any]]:
    cases = [dict(case, kind="short") for case in SHORT_CASES]
    cases.extend(
        [
            build_needle_case(
                tokenizer,
                name="needle_early",
                key="KIA-2718",
                fraction=0.12,
                target_tokens=target_tokens,
            ),
            build_needle_case(
                tokenizer,
                name="needle_middle",
                key="KIA-3141",
                fraction=0.50,
                target_tokens=target_tokens,
            ),
            build_needle_case(
                tokenizer,
                name="needle_late",
                key="KIA-1618",
                fraction=0.82,
                target_tokens=target_tokens,
            ),
        ]
    )
    return cases


def longest_common_prefix_ratio(a: list[int], b: list[int]) -> float:
    if not a:
        return 1.0 if not b else 0.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), 1)


@torch.inference_mode()
def generate_one(model, tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    device = first_cuda_device(model)
    encoded = tokenizer(
        case["prompt"],
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": case["max_new_tokens"],
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    output = model.generate(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    new_ids = output[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    expected = case.get("expected")
    passed = None if expected is None else expected.lower() in text.lower()

    return {
        "name": case["name"],
        "kind": case["kind"],
        "prompt_tokens": int(input_ids.shape[1]),
        "new_tokens": len(new_ids),
        "new_token_ids": new_ids,
        "text": text,
        "expected": expected,
        "exact_contains_expected": passed,
        "elapsed_s": elapsed,
        "tokens_per_s": len(new_ids) / max(elapsed, 1e-9),
        "peak_allocated_vram_gb": peak_gb,
    }


def configure_arm(
    model,
    *,
    arm: str,
    budget: int,
    alpha_max: float,
    verbose: bool,
):
    remove_kiaomni(model)
    remove_moe_route_stability(model)

    controller = None
    if arm in {"route_only", "kiaomni_plus_route"}:
        controller = apply_moe_route_stability(
            model,
            alpha_max=alpha_max,
            verbose=verbose,
        )
        controller.reset_metrics()

    if arm in {"kiaomni_only", "kiaomni_plus_route"}:
        apply_kiaomni(
            model,
            policy="kiaomni_s8",
            budget=budget,
            verbose=verbose,
        )

    return controller


def aggregate_arm(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    route_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    preservation = []
    needle = []
    for row in rows:
        base = baseline[row["name"]]
        ratio = longest_common_prefix_ratio(
            base["new_token_ids"],
            row["new_token_ids"],
        )
        row["baseline_token_lcp_ratio"] = ratio
        preservation.append(ratio)
        if row["kind"] == "needle" and row["exact_contains_expected"] is not None:
            needle.append(bool(row["exact_contains_expected"]))

    return {
        "mean_baseline_token_lcp_ratio": sum(preservation) / max(len(preservation), 1),
        "needle_pass_rate": sum(needle) / max(len(needle), 1) if needle else None,
        "mean_tokens_per_s": sum(r["tokens_per_s"] for r in rows) / max(len(rows), 1),
        "max_peak_allocated_vram_gb": max(
            (r["peak_allocated_vram_gb"] for r in rows),
            default=0.0,
        ),
        "router": route_metrics,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=" * 88)
    print("KiaOmni + Adaptive MoE Route Stability POC")
    print(f"Model: {args.model}")
    print(f"Budget: {args.budget} | alpha_max: {args.alpha_max}")
    print("=" * 88)

    if not torch.cuda.is_available():
        raise RuntimeError("This POC is intended for a CUDA GPU (Kaggle T4/P100 class).")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    load_allocated = torch.cuda.memory_allocated() / (1024**3)
    print(f"Loaded model VRAM allocation: {load_allocated:.2f} GB")
    if load_allocated > 14.5:
        raise RuntimeError(
            "Model left too little headroom on a 16GB card. "
            f"Allocated immediately after load: {load_allocated:.2f} GB"
        )

    cases = build_cases(tokenizer, args.long_tokens)
    arms = [
        "baseline",
        "route_only",
        "kiaomni_only",
        "kiaomni_plus_route",
    ]

    artifact: dict[str, Any] = {
        "experiment": "KIAOMNI_MOE_ROUTE_STABILITY_POC_V1",
        "model": args.model,
        "quantization_target": "4-bit AWQ",
        "weights_frozen": True,
        "budget": args.budget,
        "alpha_max": args.alpha_max,
        "environment": {
            **gpu_metadata(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model_loaded_allocated_vram_gb": load_allocated,
        },
        "arms": {},
    }

    baseline_index: dict[str, dict[str, Any]] = {}

    try:
        for arm in arms:
            print(f"\n--- {arm} ---")
            controller = configure_arm(
                model,
                arm=arm,
                budget=args.budget,
                alpha_max=args.alpha_max,
                verbose=args.verbose,
            )
            rows = []
            for case in cases:
                row = generate_one(model, tokenizer, case)
                rows.append(row)
                print(
                    f"{case['name']:<16} prompt={row['prompt_tokens']:>5} "
                    f"new={row['new_tokens']:>3} "
                    f"tok/s={row['tokens_per_s']:.2f} "
                    f"peak={row['peak_allocated_vram_gb']:.2f}GB "
                    f"text={row['text'][:80]!r}"
                )

            if arm == "baseline":
                baseline_index = {r["name"]: r for r in rows}

            route_metrics = controller.snapshot() if controller is not None else None
            aggregate = aggregate_arm(rows, baseline_index or {r["name"]: r for r in rows}, route_metrics)
            artifact["arms"][arm] = {
                "aggregate": aggregate,
                "cases": rows,
            }
    finally:
        remove_kiaomni(model)
        remove_moe_route_stability(model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("POC SUMMARY")
    for arm in arms:
        agg = artifact["arms"][arm]["aggregate"]
        router = agg["router"]
        route_text = ""
        if router:
            route_text = (
                f" raw_switch={router['raw_top1_transition_rate']:.3f}"
                f" stable_switch={router['stable_top1_transition_rate']:.3f}"
                f" intervention={router['intervention_rate']:.3f}"
                f" mean_alpha={router['mean_alpha']:.4f}"
            )
        print(
            f"{arm:<20} preserve={agg['mean_baseline_token_lcp_ratio']:.3f} "
            f"needle={agg['needle_pass_rate']} "
            f"tok/s={agg['mean_tokens_per_s']:.2f} "
            f"peak={agg['max_peak_allocated_vram_gb']:.2f}GB"
            f"{route_text}"
        )
    print(f"Saved: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
