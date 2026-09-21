"""
Kaggle single-T4 POC: KiaOmni + adaptive low-jitter MoE routing.

Default target:
    facebook/MobileMoE-M-SFT

The checkpoint is a 2026 instruction-tuned 2.8B-total MoE that fits directly
in FP16 on a single 16GB-class T4, avoiding quantization-loader confounds. The experiment keeps model weights frozen and
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


DEFAULT_MODEL = "facebook/MobileMoE-M-SFT"
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
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Optional primary KiaOmni budget; prepended to --budgets for compatibility.",
    )
    parser.add_argument(
        "--budgets",
        default="512,256,128,98",
        help="Comma-separated KiaOmni budget sweep.",
    )
    parser.add_argument("--alpha-max", type=float, default=0.10)
    parser.add_argument(
        "--router-score-func",
        choices=("sigmoid", "softmax"),
        default="sigmoid",
        help="Router score semantics. MobileMoE uses normalized sigmoid Top-K routing.",
    )
    parser.add_argument("--long-tokens", type=int, default=3200)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument(
        "--router-preflight-only",
        action="store_true",
        help="Validate router instrumentation and exit before the benchmark.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "total_vram_gb": props.total_memory / (1024**3),
            }
        )
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def input_device(model) -> torch.device:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is not None and weight.device.type != "meta":
        return weight.device
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    return next(model.parameters()).device


def render_prompt(tokenizer, prompt: str) -> tuple[str, bool]:
    """Render an instruction-tuned chat prompt when a chat template is available."""
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, True
    return prompt, False


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
    rendered_prompt, uses_chat_template = render_prompt(tokenizer, prompt)
    while len(
        tokenizer(
            rendered_prompt,
            add_special_tokens=not uses_chat_template,
        ).input_ids
    ) < target_tokens:
        parts.append(filler)
        prompt = "".join(parts) + question
        rendered_prompt, uses_chat_template = render_prompt(tokenizer, prompt)

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
    device = input_device(model)
    rendered_prompt, uses_chat_template = render_prompt(tokenizer, case["prompt"])
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=not uses_chat_template,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for gpu_idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gpu_idx)
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
    if hasattr(model, "_kia_last_compression"):
        delattr(model, "_kia_last_compression")

    output = model.generate(**kwargs)
    compression = getattr(model, "_kia_last_compression", None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    new_ids = output[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    peak_by_gpu = (
        {
            str(gpu_idx): torch.cuda.max_memory_allocated(gpu_idx) / (1024**3)
            for gpu_idx in range(torch.cuda.device_count())
        }
        if torch.cuda.is_available()
        else {}
    )
    peak_gb = max(peak_by_gpu.values(), default=0.0)

    expected = case.get("expected")
    passed = None if expected is None else expected.lower() in text.lower()

    return {
        "name": case["name"],
        "kind": case["kind"],
        "prompt_tokens": int(input_ids.shape[1]),
        "used_chat_template": uses_chat_template,
        "compression": compression,
        "new_tokens": len(new_ids),
        "new_token_ids": new_ids,
        "text": text,
        "expected": expected,
        "exact_contains_expected": passed,
        "elapsed_s": elapsed,
        "tokens_per_s": len(new_ids) / max(elapsed, 1e-9),
        "peak_allocated_vram_gb": peak_gb,
        "peak_allocated_vram_by_gpu_gb": peak_by_gpu,
    }


def configure_arm(
    model,
    *,
    arm: str,
    budget: int,
    alpha_max: float,
    score_func: str,
    verbose: bool,
):
    # Do not call remove_kiaomni on a never-patched model: some model runtimes
    # may carry an Accelerate-installed instance-level generate wrapper that
    # must remain intact for the true baseline.
    if hasattr(model, "_kia_arch_info"):
        remove_kiaomni(model)
    remove_moe_route_stability(model)

    controller = None
    if arm in {"route_only", "kiaomni_plus_route"}:
        controller = apply_moe_route_stability(
            model,
            alpha_max=alpha_max,
            score_func=score_func,
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


@torch.inference_mode()
def validate_router_instrumentation(model, tokenizer, score_func: str) -> dict[str, Any]:
    """Fail fast unless real MobileMoE router tokens reach the controller."""
    controller = apply_moe_route_stability(
        model,
        alpha_max=0.0,
        score_func=score_func,
        verbose=False,
    )
    try:
        rendered, uses_chat_template = render_prompt(
            tokenizer,
            "Reply with one word: ready",
        )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=not uses_chat_template,
        )
        device = input_device(model)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        _ = model(**encoded, use_cache=False)
        snapshot = controller.snapshot()
    finally:
        remove_moe_route_stability(model)

    print(
        "Router preflight: "
        f"tokens={snapshot['tokens_observed']} "
        f"hooks={snapshot['hook_calls']} "
        f"functional={snapshot['functional_router_calls']} "
        f"structured={snapshot['structured_outputs']} "
        f"shape_fixups={snapshot['shape_reconciliations']} "
        f"score_func={snapshot['score_func']}"
    )
    if snapshot["tokens_observed"] <= 0:
        raise RuntimeError(
            "Router instrumentation preflight failed: discovered routers but "
            "observed zero routed tokens. Refusing to run jitter benchmark."
        )
    return snapshot


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
    budgets = []
    for raw in args.budgets.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value < 1:
            raise ValueError("All KiaOmni budgets must be positive integers.")
        if value not in budgets:
            budgets.append(value)
    if args.budget is not None:
        budgets = [args.budget] + [b for b in budgets if b != args.budget]
    if not budgets:
        raise ValueError("At least one KiaOmni budget is required.")

    print(
        f"Budgets: {budgets} | alpha_max: {args.alpha_max} "
        f"| router_score_func: {args.router_score_func}"
    )
    print(f"Long-context target: {args.long_tokens}+ tokens")
    print("=" * 88)

    if not torch.cuda.is_available():
        raise RuntimeError("This POC is intended for a CUDA GPU (Kaggle T4/P100 class).")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU detected.")

    # MobileMoE-M-SFT is only 2.8B total parameters. Load the official SFT
    # checkpoint directly in FP16 on one T4. This deliberately removes
    # AWQ/GPTQ/bitsandbytes from the POC so quantization/runtime adapters
    # cannot confound the routing experiment.
    # Follow Meta's documented MobileMoE runtime on Transformers 4.57.x.
    # Keep the 2.8B FP16 checkpoint on GPU0 only so latency/routing metrics are
    # not contaminated by cross-GPU transfers. Transformers 5.x is avoided
    # because its meta-device construction breaks this custom RoPE init.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()

    load_allocated_by_gpu = {
        str(gpu_idx): torch.cuda.memory_allocated(gpu_idx) / (1024**3)
        for gpu_idx in range(torch.cuda.device_count())
    }
    print(f"HF device map: {getattr(model, 'hf_device_map', None)}")
    print(f"Model parameter device: {next(model.parameters()).device}")
    print(f"Loaded model VRAM allocation by GPU: {load_allocated_by_gpu}")
    gpu0_allocated = load_allocated_by_gpu.get("0", 0.0)
    if gpu0_allocated > 12.5:
        raise RuntimeError(
            "The FP16 MobileMoE checkpoint left too little runtime headroom "
            f"on a 15 GB T4. GPU0 allocated after load: {gpu0_allocated:.2f} GB"
        )

    router_preflight = validate_router_instrumentation(
        model,
        tokenizer,
        args.router_score_func,
    )
    if args.router_preflight_only:
        print(json.dumps(router_preflight, indent=2))
        return

    cases = build_cases(tokenizer, args.long_tokens)

    # Baseline and route-only do not depend on the KiaOmni budget, so run them
    # once. Sweep the compression budgets only for the KiaOmni arms.
    run_specs: list[tuple[str, int | None]] = [
        ("baseline", None),
        ("route_only", None),
    ]
    for budget in budgets:
        run_specs.extend(
            [
                ("kiaomni_only", budget),
                ("kiaomni_plus_route", budget),
            ]
        )

    artifact: dict[str, Any] = {
        "experiment": "KIAOMNI_MOE_ROUTE_STABILITY_POC_V1",
        "model": args.model,
        "precision_target": "FP16 on single T4 (no quantization runtime)",
        "weights_frozen": True,
        "budgets": budgets,
        "long_tokens_target": args.long_tokens,
        "alpha_max": args.alpha_max,
        "router_score_func": args.router_score_func,
        "router_preflight": router_preflight,
        "environment": {
            **gpu_metadata(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model_loaded_allocated_vram_by_gpu_gb": load_allocated_by_gpu,
            "hf_device_map": getattr(model, "hf_device_map", None),
        },
        "arms": {},
    }

    baseline_index: dict[str, dict[str, Any]] = {}
    run_keys: list[str] = []

    try:
        for arm, budget in run_specs:
            run_key = arm if budget is None else f"{arm}_b{budget}"
            run_keys.append(run_key)
            budget_text = "" if budget is None else f" | budget={budget}"
            print(f"\n--- {arm}{budget_text} ---")
            controller = configure_arm(
                model,
                arm=arm,
                budget=budget if budget is not None else budgets[0],
                alpha_max=args.alpha_max,
                score_func=args.router_score_func,
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
                    + (
                        f"kept={row['compression']['kept_tokens']} "
                        if row.get("compression") is not None
                        else ""
                    )
                    + f"text={row['text'][:80]!r}"
                )

            if arm == "baseline":
                baseline_index = {r["name"]: r for r in rows}

            route_metrics = controller.snapshot() if controller is not None else None
            aggregate = aggregate_arm(
                rows,
                baseline_index or {r["name"]: r for r in rows},
                route_metrics,
            )
            artifact["arms"][run_key] = {
                "arm": arm,
                "budget": budget,
                "aggregate": aggregate,
                "cases": rows,
            }
    finally:
        if hasattr(model, "_kia_arch_info"):
            remove_kiaomni(model)
        remove_moe_route_stability(model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("POC SUMMARY")
    for run_key in run_keys:
        agg = artifact["arms"][run_key]["aggregate"]
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
            f"{run_key:<28} preserve={agg['mean_baseline_token_lcp_ratio']:.3f} "
            f"needle={agg['needle_pass_rate']} "
            f"tok/s={agg['mean_tokens_per_s']:.2f} "
            f"peak={agg['max_peak_allocated_vram_gb']:.2f}GB"
            f"{route_text}"
        )
    print(f"Saved: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
