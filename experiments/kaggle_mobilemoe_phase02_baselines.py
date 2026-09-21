"""
Phase 02: MobileMoE 4K multi-needle + trusted baseline benchmark.

Methods
-------
FullContext       : no compression.
KiaOmni           : KiaOmni sigma=8 prompt-side saliency compression.
BlockSal          : KiaOmni block-wise prompt selector (our method).
SnapKV            : NVIDIA kvpress 0.5.5 SnapKVPress, reference defaults.
StreamingLLM      : NVIDIA kvpress 0.5.5 StreamingLLMPress wrapped with
                    KeyRerotationPress as required for paper-faithful RoPE
                    behavior.

The benchmark refuses to run external baseline scores unless the validation
gate proves that kvpress can compress MobileMoE to the exact requested
per-layer KV length at every budget.

Kaggle target: one T4, FP16 MobileMoE, <=4096 rendered prompt tokens.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import apply_kiaomni, remove_kiaomni
from kiaomni.adapters import ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter
from kiaomni.blocksal import BLOCK_SIZE_DEFAULT, select_blocksal_keep


DEFAULT_MODEL = "facebook/MobileMoE-M-SFT"
DEFAULT_OUTPUT = (
    "results/kiaomni_moe_model_lab/"
    "phase_02_mobilemoe_multineedle_baselines/phase02_results.json"
)
EXPECTED_KVPRESS_VERSION = "0.5.5"
DEFAULT_BUDGETS = (512, 256, 128, 98)
SNAPKV_WINDOW = 64
SNAPKV_KERNEL = 5
STREAMING_SINK = 4
KIA_POLICY = "kiaomni_s8"

METHODS = ("FullContext", "KiaOmni", "BlockSal", "SnapKV", "StreamingLLM")
TASKS = ("single", "multi", "hard_multi", "reason")

_SUBJECTS = (
    "The northern depot",
    "A regional audit",
    "The maintenance group",
    "The archive office",
    "Field engineers",
    "The logistics unit",
    "A visiting committee",
    "The harbor desk",
    "The pilot team",
    "Local observers",
    "The training division",
    "A follow-up review",
)
_VERBS = (
    "recorded",
    "checked",
    "confirmed",
    "reviewed",
    "catalogued",
    "reported",
    "inspected",
    "summarized",
)
_OBJECTS = (
    "routine calibration of the measurement rigs",
    "stable humidity readings throughout the building",
    "a minor update to the ventilation schedule",
    "the relocation of storage containers near gate three",
    "a backlog of paperwork from the previous quarter",
    "normal afternoon traffic around the loading area",
    "the replacement of worn signage in the east corridor",
    "unchanged energy consumption during the inspection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budgets", default="512,256,128,98")
    parser.add_argument("--target-prompt-tokens", type=int, default=3900)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_budgets(raw: str) -> list[int]:
    budgets: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 48:
            raise ValueError(
                "Budgets below 48 conflict with KiaOmni/BlockSal sink+recency protection."
            )
        if value not in budgets:
            budgets.append(value)
    if not budgets:
        raise ValueError("At least one budget is required.")
    return budgets


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
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device
    return next(model.parameters()).device


def render_prompt(tokenizer, prompt: str) -> tuple[str, bool]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, True
    return prompt, False


def encode_prompt(tokenizer, prompt: str, *, device=None) -> dict[str, Any]:
    rendered, uses_chat_template = render_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=not uses_chat_template,
    )
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return {
        "encoded": encoded,
        "rendered": rendered,
        "uses_chat_template": uses_chat_template,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
    }


def count_prompt_tokens(tokenizer, prompt: str) -> int:
    return encode_prompt(tokenizer, prompt)["prompt_tokens"]


def filler_sentence(rng: random.Random, idx: int) -> str:
    return (
        f"Log {idx:04d}: {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
        f"{rng.choice(_OBJECTS)}. "
    )


def compose_document(fillers: list[str], records: list[tuple[float, str]]) -> str:
    out = list(fillers)
    for depth, sentence in sorted(records, key=lambda item: item[0], reverse=True):
        insert_at = min(len(out), max(0, int(round(depth * len(out)))))
        out.insert(insert_at, sentence + " ")
    return "".join(out)


def fit_case_to_context(
    tokenizer,
    *,
    question: str,
    records: list[tuple[float, str]],
    seed: int,
    target_prompt_tokens: int,
    max_prompt_tokens: int,
) -> tuple[str, int]:
    if target_prompt_tokens > max_prompt_tokens:
        raise ValueError("target_prompt_tokens cannot exceed max_prompt_tokens")

    rng = random.Random(seed)
    pool = [filler_sentence(rng, idx) for idx in range(1, 1401)]

    def build(n_fillers: int) -> tuple[str, int]:
        document = compose_document(pool[:n_fillers], records)
        prompt = f"Document:\n{document}\n\nQuestion: {question}"
        return prompt, count_prompt_tokens(tokenizer, prompt)

    base_prompt, base_tokens = build(0)
    if base_tokens > target_prompt_tokens:
        raise RuntimeError(
            f"Case instructions alone use {base_tokens} tokens, above target "
            f"{target_prompt_tokens}."
        )

    lo, hi = 0, len(pool)
    best_prompt, best_tokens = base_prompt, base_tokens
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt, tokens = build(mid)
        if tokens <= target_prompt_tokens:
            best_prompt, best_tokens = prompt, tokens
            lo = mid + 1
        else:
            hi = mid - 1

    if best_tokens > max_prompt_tokens:
        raise RuntimeError(
            f"Rendered prompt exceeds hard cap: {best_tokens} > {max_prompt_tokens}"
        )
    return best_prompt, best_tokens


def _unique_number(rng: random.Random, used: set[int], lo: int, hi: int) -> int:
    while True:
        value = rng.randint(lo, hi)
        if value not in used:
            used.add(value)
            return value


def make_single_case(
    tokenizer,
    index: int,
    *,
    seed: int,
    target_prompt_tokens: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 1000 + index)
    code = str(rng.randint(100000, 999999))
    dossier = f"S-{index + 1:02d}"
    depth = (0.15, 0.35, 0.55, 0.75, 0.88)[index % 5]
    records = [
        (
            depth,
            f"Registry record for dossier {dossier}: access code = {code}.",
        )
    ]
    question = (
        f"What is the access code for dossier {dossier}? "
        "Reply with the number only."
    )
    prompt, prompt_tokens = fit_case_to_context(
        tokenizer,
        question=question,
        records=records,
        seed=seed + 1100 + index,
        target_prompt_tokens=target_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    return {
        "name": f"single_{index:02d}",
        "task": "single",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "expected_values": [code],
        "forbidden_values": [],
        "max_new_tokens": 20,
        "metadata": {"dossier": dossier, "depth": depth},
    }


def make_multi_case(
    tokenizer,
    index: int,
    *,
    seed: int,
    target_prompt_tokens: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 2000 + index)
    code = str(rng.randint(100000, 999999))
    locker = str(rng.randint(100, 999))
    codename = rng.choice(
        ("FALCON", "GRANITE", "MERIDIAN", "COBALT", "JUNIPER", "VERTEX", "QUARTZ")
    )
    dossier = f"M-{index + 1:02d}"
    records = [
        (0.18, f"Dossier {dossier}: access code = {code}."),
        (0.51, f"Dossier {dossier}: spare-key locker = {locker}."),
        (0.83, f"Dossier {dossier}: liaison codename = {codename}."),
    ]
    question = (
        f"For dossier {dossier}, return the access code, locker number, and "
        "liaison codename in that order."
    )
    prompt, prompt_tokens = fit_case_to_context(
        tokenizer,
        question=question,
        records=records,
        seed=seed + 2100 + index,
        target_prompt_tokens=target_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    return {
        "name": f"multi_{index:02d}",
        "task": "multi",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "expected_values": [code, locker, codename],
        "forbidden_values": [],
        "max_new_tokens": 48,
        "metadata": {"dossier": dossier, "depths": [0.18, 0.51, 0.83]},
    }


def make_hard_multi_case(
    tokenizer,
    index: int,
    *,
    seed: int,
    target_prompt_tokens: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 3000 + index)
    used_numbers: set[int] = set()
    target_id = f"TARGET-{index + 1:02d}"
    code = str(_unique_number(rng, used_numbers, 100000, 999999))
    locker = str(_unique_number(rng, used_numbers, 100, 999))
    codenames = ["EMBER", "NOVA", "ATLAS", "ORBIT", "ONYX", "HELIOS", "LYNX", "VEGA"]
    codename = codenames[index % len(codenames)]

    records: list[tuple[float, str]] = [
        (0.16, f"Dossier {target_id}: access code = {code}."),
        (0.53, f"Dossier {target_id}: spare-key locker = {locker}."),
        (0.86, f"Dossier {target_id}: liaison codename = {codename}."),
    ]
    forbidden: list[str] = []

    distractor_depths = (0.08, 0.24, 0.39, 0.62, 0.74, 0.92)
    for d_idx, base_depth in enumerate(distractor_depths):
        other_id = f"DECOY-{index + 1:02d}-{d_idx + 1}"
        d_code = str(_unique_number(rng, used_numbers, 100000, 999999))
        d_locker = str(_unique_number(rng, used_numbers, 100, 999))
        d_name = codenames[(index + d_idx + 1) % len(codenames)]
        forbidden.extend([d_code, d_locker, d_name])
        records.extend(
            [
                (
                    min(0.96, base_depth),
                    f"Dossier {other_id}: access code = {d_code}.",
                ),
                (
                    min(0.97, base_depth + 0.025),
                    f"Dossier {other_id}: spare-key locker = {d_locker}.",
                ),
                (
                    min(0.98, base_depth + 0.05),
                    f"Dossier {other_id}: liaison codename = {d_name}.",
                ),
            ]
        )

    question = (
        f"Use only dossier {target_id}. Return its access code, spare-key locker, "
        "and liaison codename in that order. Ignore every other dossier."
    )
    prompt, prompt_tokens = fit_case_to_context(
        tokenizer,
        question=question,
        records=records,
        seed=seed + 3100 + index,
        target_prompt_tokens=target_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    return {
        "name": f"hard_multi_{index:02d}",
        "task": "hard_multi",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "expected_values": [code, locker, codename],
        "forbidden_values": forbidden,
        "max_new_tokens": 56,
        "metadata": {
            "dossier": target_id,
            "target_depths": [0.16, 0.53, 0.86],
            "distractor_dossiers": len(distractor_depths),
        },
    }


def make_reason_case(
    tokenizer,
    index: int,
    *,
    seed: int,
    target_prompt_tokens: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 4000 + index)
    target = str(rng.randint(10000, 99999))
    distractor = str(rng.randint(10000, 99999))
    while distractor == target:
        distractor = str(rng.randint(10000, 99999))

    names = [f"V{rng.randint(10, 99)}{suffix}" for suffix in "ABCDXY"]
    a, b, c, d, x, y = names
    records = [
        (0.13, f"Assignment: variable {a} = {target}."),
        (0.37, f"Assignment: variable {b} = value of {a}."),
        (0.59, f"Assignment: variable {c} = value of {b}."),
        (0.84, f"Assignment: variable {d} = value of {c}."),
        (0.27, f"Assignment: variable {x} = {distractor}."),
        (0.71, f"Assignment: variable {y} = value of {x}."),
    ]
    question = (
        f"Track the assignments. What is the final value of variable {d}? "
        "Reply with the number only."
    )
    prompt, prompt_tokens = fit_case_to_context(
        tokenizer,
        question=question,
        records=records,
        seed=seed + 4100 + index,
        target_prompt_tokens=target_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    return {
        "name": f"reason_{index:02d}",
        "task": "reason",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "expected_values": [target],
        "forbidden_values": [distractor],
        "max_new_tokens": 24,
        "metadata": {
            "chain_length": 4,
            "target_variable": d,
            "distractor_variable": y,
        },
    }


def build_cases(
    tokenizer,
    *,
    samples_per_task: int,
    seed: int,
    target_prompt_tokens: int,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    makers = (
        make_single_case,
        make_multi_case,
        make_hard_multi_case,
        make_reason_case,
    )
    cases: list[dict[str, Any]] = []
    for maker in makers:
        for index in range(samples_per_task):
            case = maker(
                tokenizer,
                index,
                seed=seed,
                target_prompt_tokens=target_prompt_tokens,
                max_prompt_tokens=max_prompt_tokens,
            )
            if case["prompt_tokens"] > max_prompt_tokens:
                raise RuntimeError(
                    f"{case['name']} exceeded context cap: "
                    f"{case['prompt_tokens']} > {max_prompt_tokens}"
                )
            cases.append(case)
    return cases


def _contains_value(text: str, value: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def score_answer(case: dict[str, Any], text: str) -> dict[str, Any]:
    expected = case["expected_values"]
    forbidden = case.get("forbidden_values", [])
    target_hits = [value for value in expected if _contains_value(text, value)]
    forbidden_hits = [value for value in forbidden if _contains_value(text, value)]
    score = len(target_hits) / max(len(expected), 1)
    exact_pass = len(target_hits) == len(expected) and not forbidden_hits
    return {
        "score": score,
        "exact_pass": exact_pass,
        "target_hits": target_hits,
        "target_total": len(expected),
        "forbidden_hits": forbidden_hits,
    }


def ratio_for_budget(prompt_len: int, budget: int) -> float:
    if budget >= prompt_len:
        return 0.0
    ratio = 1.0 - (budget + 0.5) / prompt_len
    ratio = min(max(ratio, 0.0), 0.999999)
    kept = max(1, int(prompt_len * (1.0 - ratio)))
    if kept != budget:
        raise RuntimeError(
            f"Exact kvpress ratio failed: L={prompt_len} B={budget} -> kept={kept}"
        )
    return ratio


def reset_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    for gpu_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(gpu_idx)
    torch.cuda.synchronize()


def peak_memory() -> tuple[float, dict[str, float]]:
    if not torch.cuda.is_available():
        return 0.0, {}
    per_gpu = {
        str(gpu_idx): torch.cuda.max_memory_allocated(gpu_idx) / (1024**3)
        for gpu_idx in range(torch.cuda.device_count())
    }
    return max(per_gpu.values(), default=0.0), per_gpu


def generate_raw(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    max_new_tokens: int,
    press=None,
) -> dict[str, Any]:
    reset_peak_memory()
    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    if press is None:
        output = model.generate(**kwargs)
    else:
        with press(model):
            output = model.generate(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_gb, peak_by_gpu = peak_memory()
    new_ids = output[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "new_tokens": len(new_ids),
        "elapsed_s": elapsed,
        "tokens_per_s": len(new_ids) / max(elapsed, 1e-9),
        "peak_allocated_vram_gb": peak_gb,
        "peak_allocated_vram_by_gpu_gb": peak_by_gpu,
    }


def build_blocksal_adapter(model) -> SaliencyAdapter:
    probe = ArchitectureProbe.probe(model, force=True)
    adapter = SaliencyAdapter(probe)
    if hasattr(model, "_kia_arch_info"):
        delattr(model, "_kia_arch_info")
    return adapter


def run_full_context(model, tokenizer, encoded, case) -> dict[str, Any]:
    result = generate_raw(
        model,
        tokenizer,
        encoded["input_ids"],
        encoded.get("attention_mask"),
        max_new_tokens=case["max_new_tokens"],
    )
    result["compression"] = {
        "kind": "none",
        "original_tokens": int(encoded["input_ids"].shape[1]),
        "retained_tokens": int(encoded["input_ids"].shape[1]),
    }
    return result


def run_kiaomni(model, tokenizer, encoded, case, budget: int) -> dict[str, Any]:
    apply_kiaomni(model, policy=KIA_POLICY, budget=budget, verbose=False)
    try:
        result = generate_raw(
            model,
            tokenizer,
            encoded["input_ids"],
            encoded.get("attention_mask"),
            max_new_tokens=case["max_new_tokens"],
        )
        telemetry = getattr(model, "_kia_last_compression", None)
    finally:
        remove_kiaomni(model)

    if telemetry is None:
        raise RuntimeError("KiaOmni did not emit compression telemetry.")
    retained = int(telemetry["kept_tokens"])
    expected = min(budget, int(encoded["input_ids"].shape[1]))
    if retained != expected:
        raise RuntimeError(
            f"KiaOmni exact-budget failure: retained={retained} expected={expected}"
        )
    result["compression"] = {
        "kind": "prompt_side",
        "policy": KIA_POLICY,
        **telemetry,
        "retained_tokens": retained,
    }
    return result


def run_blocksal(
    model,
    tokenizer,
    encoded,
    case,
    budget: int,
    saliency_adapter: SaliencyAdapter,
) -> dict[str, Any]:
    input_ids = encoded["input_ids"]
    L = int(input_ids.shape[1])
    reset_peak_memory()
    started = time.perf_counter()

    sal_batch = saliency_adapter.extract(input_ids, model)
    selection = select_blocksal_keep(
        sal_batch[0],
        budget=budget,
        L=L,
        block_size=BLOCK_SIZE_DEFAULT,
    )
    keep_t = torch.as_tensor(
        selection.keep_indices,
        device=input_ids.device,
        dtype=torch.long,
    )
    pruned = input_ids[:, keep_t]
    pruned_mask = torch.ones_like(pruned)

    kwargs: dict[str, Any] = {
        "input_ids": pruned,
        "attention_mask": pruned_mask,
        "max_new_tokens": case["max_new_tokens"],
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output = model.generate(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_gb, peak_by_gpu = peak_memory()

    new_ids = output[0, pruned.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "new_tokens": len(new_ids),
        "elapsed_s": elapsed,
        "tokens_per_s": len(new_ids) / max(elapsed, 1e-9),
        "peak_allocated_vram_gb": peak_gb,
        "peak_allocated_vram_by_gpu_gb": peak_by_gpu,
        "compression": {
            "kind": "prompt_side_block",
            "original_tokens": L,
            "retained_tokens": int(len(selection.keep_indices)),
            "budget": budget,
            "block_size": selection.block_size,
            "partial_boundary_block": selection.partial_boundary_block,
            "protected_tokens": selection.protected_tokens,
        },
    }


def load_kvpress():
    from kvpress import KeyRerotationPress, SnapKVPress, StreamingLLMPress

    version = importlib.metadata.version("kvpress")
    return {
        "version": version,
        "SnapKVPress": SnapKVPress,
        "StreamingLLMPress": StreamingLLMPress,
        "KeyRerotationPress": KeyRerotationPress,
    }


def make_press(kvpress_api: dict[str, Any], method: str, prompt_len: int, budget: int):
    ratio = ratio_for_budget(prompt_len, budget)
    if method == "SnapKV":
        return kvpress_api["SnapKVPress"](
            compression_ratio=ratio,
            window_size=SNAPKV_WINDOW,
            kernel_size=SNAPKV_KERNEL,
        )
    if method == "StreamingLLM":
        base = kvpress_api["StreamingLLMPress"](
            compression_ratio=ratio,
            n_sink=STREAMING_SINK,
        )
        return kvpress_api["KeyRerotationPress"](press=base)
    raise KeyError(method)


def run_external_press(
    model,
    tokenizer,
    encoded,
    case,
    *,
    method: str,
    budget: int,
    kvpress_api: dict[str, Any],
) -> dict[str, Any]:
    L = int(encoded["input_ids"].shape[1])
    press = make_press(kvpress_api, method, L, budget)
    result = generate_raw(
        model,
        tokenizer,
        encoded["input_ids"],
        encoded.get("attention_mask"),
        max_new_tokens=case["max_new_tokens"],
        press=press,
    )
    result["compression"] = {
        "kind": "kv_cache",
        "original_tokens": L,
        "retained_tokens_target_per_layer": min(budget, L),
        "budget": budget,
        "kvpress_compression_ratio": float(press.compression_ratio),
    }
    return result


def extract_cache_lengths(cache) -> list[int]:
    lengths: list[int] = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if torch.is_tensor(keys) and keys.ndim >= 3:
                lengths.append(int(keys.shape[-2]))
        return lengths

    if isinstance(cache, (tuple, list)):
        for item in cache:
            if isinstance(item, (tuple, list)) and item and torch.is_tensor(item[0]):
                keys = item[0]
                if keys.ndim >= 3:
                    lengths.append(int(keys.shape[-2]))
    return lengths


def attention_modules(model) -> list[Any]:
    base = getattr(model, "model", None)
    if base is None:
        return []
    language_model = (
        getattr(base, "language_model", None)
        if hasattr(base, "language_model")
        else base
    )
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return []
    return [
        layer.self_attn
        for layer in layers
        if hasattr(layer, "self_attn")
    ]


@torch.inference_mode()
def validate_external_baseline(
    model,
    tokenizer,
    *,
    method: str,
    budgets: list[int],
    kvpress_api: dict[str, Any],
    validation_prompt: str,
) -> dict[str, Any]:
    version = kvpress_api["version"]
    report: dict[str, Any] = {
        "method": method,
        "kvpress_version": version,
        "expected_kvpress_version": EXPECTED_KVPRESS_VERSION,
        "version_ok": version == EXPECTED_KVPRESS_VERSION,
        "budgets": {},
        "valid": False,
    }
    if not report["version_ok"]:
        report["error"] = (
            f"kvpress version mismatch: installed={version}, "
            f"expected={EXPECTED_KVPRESS_VERSION}"
        )
        return report

    device = input_device(model)
    packed = encode_prompt(tokenizer, validation_prompt, device=device)
    encoded = packed["encoded"]
    L = int(encoded["input_ids"].shape[1])
    report["validation_prompt_tokens"] = L
    if L <= max(budgets):
        report["error"] = (
            f"validation prompt is too short: L={L}, max budget={max(budgets)}"
        )
        return report

    attn_modules = attention_modules(model)
    hooks_before = sum(len(module._forward_hooks) for module in attn_modules)
    all_ok = True

    for budget in budgets:
        entry: dict[str, Any] = {"budget": budget}
        try:
            press = make_press(kvpress_api, method, L, budget)
            expected = min(budget, L)
            entry["compression_ratio"] = float(press.compression_ratio)
            entry["expected_kept"] = expected

            with press(model):
                outputs = model(**encoded, use_cache=True)
            cache = getattr(outputs, "past_key_values", None)
            lengths = extract_cache_lengths(cache)
            entry["layer_cache_lengths"] = lengths
            entry["observed_layers"] = len(lengths)
            entry["exact_budget"] = bool(lengths) and all(
                length == expected for length in lengths
            )
            entry["valid"] = entry["exact_budget"]
            if not entry["valid"]:
                all_ok = False
        except Exception as exc:
            entry["valid"] = False
            entry["error_type"] = type(exc).__name__
            entry["error"] = str(exc)
            all_ok = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        report["budgets"][str(budget)] = entry

    hooks_after = sum(len(module._forward_hooks) for module in attn_modules)
    report["forward_hooks_before"] = hooks_before
    report["forward_hooks_after"] = hooks_after
    report["hooks_restored"] = hooks_before == hooks_after
    report["valid"] = all_ok and report["hooks_restored"]
    if not report["hooks_restored"]:
        report["error"] = "kvpress context manager left forward hooks installed."
    return report


def build_validation_prompt(tokenizer, *, seed: int) -> str:
    rng = random.Random(seed + 9000)
    parts = [filler_sentence(rng, idx) for idx in range(1, 220)]
    prompt = (
        "Document:\n"
        + "".join(parts)
        + "\n\nQuestion: Reply with one word: ready"
    )
    while count_prompt_tokens(tokenizer, prompt) <= 700:
        start = len(parts) + 1
        parts.extend(filler_sentence(rng, idx) for idx in range(start, start + 40))
        prompt = (
            "Document:\n"
            + "".join(parts)
            + "\n\nQuestion: Reply with one word: ready"
        )
    return prompt


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (
            row["method"]
            if row["budget"] is None
            else f"{row['method']}_b{row['budget']}"
        )
        bucket = grouped.setdefault(
            key,
            {
                "method": row["method"],
                "budget": row["budget"],
                "rows": [],
            },
        )
        bucket["rows"].append(row)

    out: dict[str, Any] = {}
    for key, bucket in grouped.items():
        values = bucket.pop("rows")
        by_task: dict[str, Any] = {}
        for task in TASKS:
            task_rows = [row for row in values if row["task"] == task]
            if not task_rows:
                continue
            by_task[task] = {
                "mean_score": sum(row["score"]["score"] for row in task_rows)
                / len(task_rows),
                "exact_pass_rate": sum(
                    1 for row in task_rows if row["score"]["exact_pass"]
                )
                / len(task_rows),
                "n": len(task_rows),
            }
        out[key] = {
            **bucket,
            "mean_score": sum(row["score"]["score"] for row in values) / len(values),
            "exact_pass_rate": sum(
                1 for row in values if row["score"]["exact_pass"]
            )
            / len(values),
            "mean_tokens_per_s": sum(row["tokens_per_s"] for row in values)
            / len(values),
            "max_peak_allocated_vram_gb": max(
                row["peak_allocated_vram_gb"] for row in values
            ),
            "tasks": by_task,
        }
    return out


def method_metadata(kvpress_version: str | None) -> dict[str, Any]:
    return {
        "FullContext": {
            "owner": "control",
            "compression_kind": "none",
        },
        "KiaOmni": {
            "owner": "ours",
            "compression_kind": "prompt_side",
            "policy": KIA_POLICY,
        },
        "BlockSal": {
            "owner": "ours",
            "compression_kind": "prompt_side_block",
            "block_size": BLOCK_SIZE_DEFAULT,
            "block_score": "mean raw KiaOmni saliency",
            "exact_budget_boundary": "top-saliency tokens from one partial boundary block",
        },
        "SnapKV": {
            "owner": "external",
            "compression_kind": "kv_cache",
            "implementation": "NVIDIA kvpress SnapKVPress",
            "kvpress_version": kvpress_version,
            "source": (
                "https://github.com/NVIDIA/kvpress/blob/main/"
                "kvpress/presses/snapkv_press.py"
            ),
            "paper": "arXiv:2404.14469",
            "window_size": SNAPKV_WINDOW,
            "kernel_size": SNAPKV_KERNEL,
        },
        "StreamingLLM": {
            "owner": "external",
            "compression_kind": "kv_cache",
            "implementation": (
                "NVIDIA kvpress StreamingLLMPress + KeyRerotationPress"
            ),
            "kvpress_version": kvpress_version,
            "source": (
                "https://github.com/NVIDIA/kvpress/blob/main/"
                "kvpress/presses/streaming_llm_press.py"
            ),
            "rerotation_source": (
                "https://github.com/NVIDIA/kvpress/blob/main/"
                "kvpress/presses/key_rerotation_press.py"
            ),
            "paper": "arXiv:2309.17453",
            "n_sink": STREAMING_SINK,
        },
    }


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)

    if args.samples_per_task < 1:
        raise ValueError("samples-per-task must be >= 1")
    if args.target_prompt_tokens > args.max_prompt_tokens:
        raise ValueError("target-prompt-tokens must be <= max-prompt-tokens")
    if args.max_prompt_tokens > 4096:
        raise ValueError(
            "This Phase 02 Kaggle protocol is intentionally capped at 4096 tokens."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this Kaggle protocol.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=" * 96)
    print("KiaOmni x MoE Model Lab — Phase 02")
    print(f"Model: {args.model}")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Budgets: {budgets}")
    print(
        f"Rendered prompt target: {args.target_prompt_tokens} "
        f"| hard cap: {args.max_prompt_tokens}"
    )
    print(f"Samples/task: {args.samples_per_task}")
    print("=" * 96)

    model, tokenizer = load_model_and_tokenizer(args.model)
    device = input_device(model)
    print(f"Model parameter device: {next(model.parameters()).device}")
    print(f"HF device map: {getattr(model, 'hf_device_map', None)}")

    try:
        kvpress_api = load_kvpress()
        kvpress_version = kvpress_api["version"]
    except Exception as exc:
        kvpress_api = None
        kvpress_version = None
        kvpress_load_error = f"{type(exc).__name__}: {exc}"
    else:
        kvpress_load_error = None

    validation_prompt = build_validation_prompt(tokenizer, seed=args.seed)
    validation: dict[str, Any] = {
        "kvpress_load_error": kvpress_load_error,
        "external": {},
    }

    if kvpress_api is not None:
        for method in ("SnapKV", "StreamingLLM"):
            print(f"\nValidating {method} on MobileMoE...")
            report = validate_external_baseline(
                model,
                tokenizer,
                method=method,
                budgets=budgets,
                kvpress_api=kvpress_api,
                validation_prompt=validation_prompt,
            )
            validation["external"][method] = report
            print(json.dumps(report, indent=2))
    else:
        for method in ("SnapKV", "StreamingLLM"):
            validation["external"][method] = {
                "method": method,
                "valid": False,
                "error": kvpress_load_error,
            }

    validation_ok = all(
        validation["external"][method].get("valid", False)
        for method in ("SnapKV", "StreamingLLM")
    )
    validation["all_external_valid"] = validation_ok

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preflight_artifact = {
        "experiment": "KIAOMNI_MOE_PHASE02_MULTINEEDLE_BASELINES_V1",
        "stage": "validation",
        "model": args.model,
        "precision": "FP16",
        "budgets": budgets,
        "target_prompt_tokens": args.target_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "samples_per_task": args.samples_per_task,
        "methods": method_metadata(kvpress_version),
        "validation": validation,
        "environment": {
            **gpu_metadata(),
            "python": platform.python_version(),
            "transformers": importlib.metadata.version("transformers"),
            "kvpress": kvpress_version,
        },
    }

    if args.validation_only or not validation_ok:
        output_path.write_text(
            json.dumps(preflight_artifact, indent=2),
            encoding="utf-8",
        )
        if not validation_ok:
            raise RuntimeError(
                "External baseline validation failed. No benchmark scores were "
                "produced. Inspect the saved validation artifact before adapting "
                "kvpress to MobileMoE."
            )
        print(f"Validation PASS. Saved: {output_path}")
        return

    cases = build_cases(
        tokenizer,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        target_prompt_tokens=args.target_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    print("\nCase prompt lengths:")
    for case in cases:
        print(f"  {case['name']:<18} {case['prompt_tokens']} tokens")

    blocksal_adapter = build_blocksal_adapter(model)
    rows: list[dict[str, Any]] = []

    for case in cases:
        packed = encode_prompt(tokenizer, case["prompt"], device=device)
        encoded = packed["encoded"]
        actual_prompt_tokens = int(encoded["input_ids"].shape[1])
        if actual_prompt_tokens != case["prompt_tokens"]:
            raise RuntimeError(
                f"Prompt token count drift for {case['name']}: "
                f"build={case['prompt_tokens']} run={actual_prompt_tokens}"
            )
        if actual_prompt_tokens > args.max_prompt_tokens:
            raise RuntimeError(
                f"{case['name']} exceeds hard context cap: {actual_prompt_tokens}"
            )

        print(
            f"\n[{case['name']}] task={case['task']} "
            f"prompt={actual_prompt_tokens}"
        )

        full = run_full_context(model, tokenizer, encoded, case)
        full_score = score_answer(case, full["text"])
        full_row = {
            "case": case["name"],
            "task": case["task"],
            "method": "FullContext",
            "budget": None,
            "prompt_tokens": actual_prompt_tokens,
            "score": full_score,
            **full,
        }
        rows.append(full_row)
        print(
            f"  {'FullContext':<24} score={full_score['score']:.2f} "
            f"pass={full_score['exact_pass']} "
            f"tok/s={full['tokens_per_s']:.2f} "
            f"peak={full['peak_allocated_vram_gb']:.2f}GB "
            f"text={full['text'][:70]!r}"
        )

        for budget in budgets:
            for method in ("KiaOmni", "BlockSal", "SnapKV", "StreamingLLM"):
                if method == "KiaOmni":
                    result = run_kiaomni(
                        model,
                        tokenizer,
                        encoded,
                        case,
                        budget,
                    )
                elif method == "BlockSal":
                    result = run_blocksal(
                        model,
                        tokenizer,
                        encoded,
                        case,
                        budget,
                        blocksal_adapter,
                    )
                else:
                    result = run_external_press(
                        model,
                        tokenizer,
                        encoded,
                        case,
                        method=method,
                        budget=budget,
                        kvpress_api=kvpress_api,
                    )

                scored = score_answer(case, result["text"])
                row = {
                    "case": case["name"],
                    "task": case["task"],
                    "method": method,
                    "budget": budget,
                    "prompt_tokens": actual_prompt_tokens,
                    "score": scored,
                    **result,
                }
                rows.append(row)
                retained = result["compression"].get(
                    "retained_tokens",
                    result["compression"].get(
                        "retained_tokens_target_per_layer",
                        "?",
                    ),
                )
                print(
                    f"  {method + ' B=' + str(budget):<24} "
                    f"score={scored['score']:.2f} "
                    f"pass={scored['exact_pass']} "
                    f"kept={retained} "
                    f"tok/s={result['tokens_per_s']:.2f} "
                    f"peak={result['peak_allocated_vram_gb']:.2f}GB "
                    f"text={result['text'][:70]!r}"
                )

        gc.collect()
        torch.cuda.empty_cache()

    artifact = {
        **preflight_artifact,
        "stage": "complete",
        "tasks": list(TASKS),
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key != "prompt"
            }
            for case in cases
        ],
        "rows": rows,
        "aggregate": aggregate_rows(rows),
        "protocol_notes": [
            "All prompts are rendered to <=4096 tokens after the chat template.",
            "All compressed methods use the same retained-token target B.",
            "KiaOmni and BlockSal prune prompt positions before generation.",
            "SnapKV and StreamingLLM compress the per-layer KV cache at prefill.",
            "External scores are emitted only after exact-cache-length validation.",
            "BlockSal is an internal KiaOmni-family method, not an external baseline.",
            "StreamingLLM uses KeyRerotationPress because kvpress documents it as required for paper-faithful RoPE behavior.",
        ],
    }
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("PHASE 02 SUMMARY")
    for key, agg in artifact["aggregate"].items():
        print(
            f"{key:<28} score={agg['mean_score']:.3f} "
            f"pass={agg['exact_pass_rate']:.3f} "
            f"tok/s={agg['mean_tokens_per_s']:.2f} "
            f"peak={agg['max_peak_allocated_vram_gb']:.2f}GB"
        )
    print(f"Saved: {output_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
