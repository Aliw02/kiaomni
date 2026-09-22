from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import apply_kiaomni, remove_kiaomni
from kiaomni.adapters import ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter
from kiaomni.blocksal import BLOCK_SIZE_DEFAULT, select_blocksal_keep

MODEL_DEFAULT = "facebook/MobileMoE-M-SFT"
BUDGETS_DEFAULT = "512,256,128,98"
TARGET_TOKENS_DEFAULT = 3900
MAX_CONTEXT_DEFAULT = 4096
SEED_DEFAULT = 42
N_SINK = 16
RECENCY = 32
BLOCK_SIZE = BLOCK_SIZE_DEFAULT
SNAPKV_WINDOW = 64
SNAPKV_KERNEL = 5
KVPRESS_REF = "7331c23da9e6f1510d89ea651d0dea77a57b3252"
KVPRESS_REQUIRED = "0.5.5"
TRANSFORMERS_REQUIRED = "4.57.6"
OUTPUT_DEFAULT = (
    "results/kiaomni_moe_model_lab/phase_02_mobilemoe_multineedle_baselines/"
    "phase02_results.json"
)

SUBJECTS = [
    "The committee", "A regional survey", "The maintenance crew", "An early prototype",
    "The northern facility", "A visiting delegation", "The archive department", "Local observers",
    "The pilot program", "A follow-up study", "The logistics team", "An internal memo",
    "The harbor authority", "A quarterly audit", "The training division", "Field engineers",
]
VERBS = [
    "reported", "confirmed", "documented", "reviewed", "scheduled", "postponed",
    "evaluated", "inspected", "catalogued", "summarized", "approved", "recorded",
]
OBJECTS = [
    "minor adjustments to the ventilation schedule",
    "a gradual rise in afternoon foot traffic",
    "the relocation of two storage containers",
    "routine calibration of the measurement rigs",
    "an updated rotation plan for night shifts",
    "the replacement of worn signage near gate three",
    "consistent humidity readings across all halls",
    "a backlog of paperwork from the previous quarter",
    "slower than expected delivery of spare parts",
    "general satisfaction with the new canteen layout",
    "uneven wear on the loading dock surface",
    "stable energy consumption throughout the week",
]


@dataclass
class Case:
    task: str
    sample_id: int
    context: str
    question: str
    gold: list[str]
    distractors: list[str]
    info: str


@dataclass
class ValidationResult:
    method: str
    valid: bool
    status: str
    details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KiaOmni x MobileMoE Phase-02 baseline benchmark")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--budgets", default=BUDGETS_DEFAULT)
    p.add_argument("--target-tokens", type=int, default=TARGET_TOKENS_DEFAULT)
    p.add_argument("--max-context", type=int, default=MAX_CONTEXT_DEFAULT)
    p.add_argument("--samples-per-task", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output", default=OUTPUT_DEFAULT)
    p.add_argument("--skip-external", action="store_true")
    p.add_argument(
        "--validation-only",
        action="store_true",
        help="Run method compatibility/budget validation and exit before benchmark generation.",
    )
    return p.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def render_prompt(tokenizer, context: str, question: str) -> tuple[str, bool]:
    user = f"Document:\n{context}\n\nQuestion:\n{question}"
    if getattr(tokenizer, "chat_template", None):
        return (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            True,
        )
    return user, False


def encode_prompt(tokenizer, context: str, question: str) -> tuple[torch.Tensor, bool]:
    rendered, uses_chat = render_prompt(tokenizer, context, question)
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=not uses_chat,
    )
    return enc.input_ids, uses_chat


def filler_sentence(rng: random.Random) -> str:
    return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}."


def insert_at_depths(fillers: list[str], items: list[tuple[float, str]]) -> str:
    out = list(fillers)
    for depth, sentence in sorted(items, key=lambda x: x[0], reverse=True):
        out.insert(int(depth * len(out)), sentence)
    return " ".join(out)


def fit_case_to_context(
    tokenizer,
    question: str,
    items: list[tuple[float, str]],
    rng: random.Random,
    target_tokens: int,
    max_context: int,
) -> tuple[str, int]:
    if not (target_tokens <= max_context):
        raise ValueError("target_tokens must be <= max_context")

    filler_pool = [filler_sentence(rng) for _ in range(700)]
    lo, hi = 0, len(filler_pool)
    best_context = insert_at_depths([], items)
    best_len = encode_prompt(tokenizer, best_context, question)[0].shape[1]

    while lo <= hi:
        mid = (lo + hi) // 2
        ctx = insert_at_depths(filler_pool[:mid], items)
        plen = int(encode_prompt(tokenizer, ctx, question)[0].shape[1])
        if plen <= target_tokens:
            best_context, best_len = ctx, plen
            lo = mid + 1
        else:
            hi = mid - 1

    if best_len > max_context:
        raise RuntimeError(f"Prompt fitting failed: {best_len} > max_context={max_context}")
    if best_len < target_tokens - 96:
        raise RuntimeError(
            f"Prompt fitting underfilled context: {best_len} < {target_tokens - 96}"
        )
    return best_context, best_len


def make_single(tokenizer, i: int, seed: int, target: int, max_context: int) -> Case:
    rng = random.Random(seed + 100 + i)
    code = str(rng.randint(100000, 999999))
    depth = [0.10, 0.30, 0.50, 0.70, 0.90][i % 5]
    needle = f"Security register: vault {chr(65 + i % 26)} access code is {code}."
    question = "What is the vault access code in the document? Answer with the number only."
    context, plen = fit_case_to_context(
        tokenizer, question, [(depth, needle)], rng, target, max_context
    )
    return Case("single", i, context, question, [code], [], f"needle@{depth:.2f};prompt={plen}")


def make_multi(tokenizer, i: int, seed: int, target: int, max_context: int) -> Case:
    rng = random.Random(seed + 200 + i)
    code = str(rng.randint(100000, 999999))
    locker = str(rng.randint(100, 999))
    name = rng.choice(["FALCON", "GRANITE", "MERIDIAN", "COBALT", "JUNIPER", "VERTEX", "HARBOR", "QUARTZ"])
    items = [
        (0.18, f"Operations record: the access code is {code}."),
        (0.51, f"Equipment record: the spare keys are in locker {locker}."),
        (0.83, f"Personnel record: the liaison codename is {name}."),
    ]
    question = "Return the access code, locker number, and liaison codename. Include all three values."
    context, plen = fit_case_to_context(tokenizer, question, items, rng, target, max_context)
    return Case("multi", i, context, question, [code, locker, name], [], f"3 needles@18/51/83%;prompt={plen}")


def make_hard_multi(tokenizer, i: int, seed: int, target: int, max_context: int) -> Case:
    rng = random.Random(seed + 400 + i)
    projects = ["ORION", "ORBIT", "ORCHID", "OSPREY"]
    target_project = projects[i % len(projects)]
    code = str(rng.randint(100000, 999999))
    locker = str(rng.randint(100, 999))
    name = rng.choice(["SABLE", "QUARTZ", "EMBER", "NOVA", "CEDAR", "ONYX"])
    distract_code = str(rng.randint(100000, 999999))
    distract_locker = str(rng.randint(100, 999))
    distract_name = rng.choice([x for x in ["SABLE", "QUARTZ", "EMBER", "NOVA", "CEDAR", "ONYX"] if x != name])
    other_project = projects[(i + 1) % len(projects)]
    items = [
        (0.12, f"Registry entry: project {other_project}, revision 7, access code {distract_code}."),
        (0.22, f"Registry entry: project {target_project}, revision 7, access code {code}."),
        (0.39, f"Registry entry: project {other_project}, revision 7, locker {distract_locker}."),
        (0.55, f"Registry entry: project {target_project}, revision 7, locker {locker}."),
        (0.72, f"Registry entry: project {other_project}, revision 7, codename {distract_name}."),
        (0.86, f"Registry entry: project {target_project}, revision 7, codename {name}."),
    ]
    question = (
        f"For project {target_project} revision 7 only, return its access code, locker, and codename. "
        "Do not use values from other projects."
    )
    context, plen = fit_case_to_context(tokenizer, question, items, rng, target, max_context)
    return Case(
        "hard_multi", i, context, question, [code, locker, name],
        [distract_code, distract_locker, distract_name],
        f"matched distractors;prompt={plen}",
    )


def make_reason(tokenizer, i: int, seed: int, target: int, max_context: int) -> Case:
    rng = random.Random(seed + 300 + i)
    val = str(rng.randint(10000, 99999))
    dval = str(rng.randint(10000, 99999))
    while dval == val:
        dval = str(rng.randint(10000, 99999))
    names = [f"V{rng.randint(10,99)}{c}" for c in "ABCDXY"]
    a, b, c, d, x, y = names
    items = [
        (0.14, f"Assignment log: variable {a} was set to {val}."),
        (0.33, f"Assignment log: variable {b} was set to the value of {a}."),
        (0.57, f"Assignment log: variable {c} was set to the value of {b}."),
        (0.84, f"Assignment log: variable {d} was set to the value of {c}."),
        (0.25, f"Assignment log: variable {x} was set to {dval}."),
        (0.69, f"Assignment log: variable {y} was set to the value of {x}."),
    ]
    question = f"Track the assignments. What is the final value of variable {d}? Answer with the number only."
    context, plen = fit_case_to_context(tokenizer, question, items, rng, target, max_context)
    return Case("reason", i, context, question, [val], [dval], f"4-hop+distractor;prompt={plen}")


def build_cases(tokenizer, samples: int, seed: int, target: int, max_context: int) -> list[Case]:
    makers: list[Callable[..., Case]] = [make_single, make_multi, make_hard_multi, make_reason]
    cases: list[Case] = []
    for maker in makers:
        for i in range(samples):
            cases.append(maker(tokenizer, i, seed, target, max_context))
    return cases


def score_answer(case: Case, text: str) -> dict[str, Any]:
    up = text.upper()
    hits = [g for g in case.gold if re.search(rf"(?<![A-Z0-9]){re.escape(g.upper())}(?![A-Z0-9])", up)]
    distract_hits = [
        d for d in case.distractors
        if re.search(rf"(?<![A-Z0-9]){re.escape(d.upper())}(?![A-Z0-9])", up)
    ]
    recall = len(hits) / max(len(case.gold), 1)
    exact = len(hits) == len(case.gold) and not distract_hits
    return {
        "exact": bool(exact),
        "recall": float(recall),
        "gold_hits": hits,
        "distractor_hits": distract_hits,
    }


def clean_model(model) -> None:
    if getattr(model, "_phase02_kia_patch_active", False):
        remove_kiaomni(model)
        try:
            delattr(model, "_phase02_kia_patch_active")
        except AttributeError:
            pass
    elif hasattr(model, "_kia_arch_info"):
        # ArchitectureProbe also caches _kia_arch_info. Clearing that cache
        # must not delete/replace model.generate on an otherwise unpatched model.
        try:
            delattr(model, "_kia_arch_info")
        except AttributeError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def input_device(model) -> torch.device:
    return next(model.parameters()).device


def reset_peak_memory() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_memory_gb() -> dict[str, float]:
    return {
        str(i): torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        for i in range(torch.cuda.device_count())
    }


@torch.inference_mode()
def generate_full(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> dict[str, Any]:
    ids = input_ids.to(input_device(model))
    reset_peak_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out[0, ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    peaks = peak_memory_gb()
    return {
        "text": text,
        "new_tokens": int(new_ids.numel()),
        "elapsed_s": dt,
        "tokens_per_s": float(new_ids.numel() / max(dt, 1e-9)),
        "peak_vram_gb": max(peaks.values(), default=0.0),
        "peak_vram_by_gpu_gb": peaks,
    }


def blocksal_keep(
    saliency: np.ndarray,
    budget: int,
    seq_len: int,
    *,
    n_sink: int = N_SINK,
    recency: int = RECENCY,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Exact-budget adapter for our historical mean-saliency BlockSal ranking."""
    return select_blocksal_keep(
        np.asarray(saliency, dtype=np.float32).reshape(-1),
        budget=budget,
        L=seq_len,
        block_size=block_size,
        n_sink=n_sink,
        recency=recency,
    ).keep_indices


@torch.inference_mode()
def generate_blocksal(model, tokenizer, input_ids: torch.Tensor, budget: int, max_new_tokens: int) -> dict[str, Any]:
    clean_model(model)
    ids = input_ids.to(input_device(model))
    seq_len = ids.shape[1]

    # Architecture discovery is setup, like apply_kiaomni(), and is excluded
    # from per-sample latency. Saliency extraction/selection must be included.
    adapter = None
    if seq_len > budget:
        probe = ArchitectureProbe.probe(model)
        adapter = SaliencyAdapter(probe)

    reset_peak_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if seq_len <= budget:
        keep = np.arange(seq_len, dtype=np.int64)
        partial_boundary_block = False
        protected_tokens = min(seq_len, N_SINK + RECENCY)
    else:
        saliency = adapter.extract(ids, model)[0]
        selection = select_blocksal_keep(
            saliency,
            budget=budget,
            L=seq_len,
            block_size=BLOCK_SIZE,
            n_sink=N_SINK,
            recency=RECENCY,
        )
        keep = selection.keep_indices
        partial_boundary_block = selection.partial_boundary_block
        protected_tokens = selection.protected_tokens

    expected_kept = min(budget, seq_len)
    if len(keep) != expected_kept:
        raise RuntimeError(
            f"BlockSal exact-budget invariant failed: kept={len(keep)} "
            f"expected={expected_kept}"
        )

    keep_t = torch.as_tensor(keep, device=ids.device, dtype=torch.long)
    pruned = ids[:, keep_t]
    out = model.generate(
        pruned,
        attention_mask=torch.ones_like(pruned),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out[0, pruned.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    peaks = peak_memory_gb()
    return {
        "text": text,
        "new_tokens": int(new_ids.numel()),
        "elapsed_s": dt,
        "tokens_per_s": float(new_ids.numel() / max(dt, 1e-9)),
        "peak_vram_gb": max(peaks.values(), default=0.0),
        "peak_vram_by_gpu_gb": peaks,
        "compression": {
            "original_tokens": int(seq_len),
            "requested_budget": int(budget),
            "kept_tokens": int(len(keep)),
            "budget_delta": int(len(keep) - min(budget, seq_len)),
            "block_size": BLOCK_SIZE,
            "selector": "mean_saliency_block_ranking_exact_budget_boundary",
            "partial_boundary_block": bool(partial_boundary_block),
            "protected_tokens": int(protected_tokens),
        },
    }


@torch.inference_mode()
def generate_kiaomni(model, tokenizer, input_ids: torch.Tensor, budget: int, max_new_tokens: int) -> dict[str, Any]:
    clean_model(model)
    apply_kiaomni(model, policy="kiaomni_s8", budget=budget, verbose=False)
    model._phase02_kia_patch_active = True
    try:
        model._kia_last_compression = None
        result = generate_full(model, tokenizer, input_ids, max_new_tokens)
        result["compression"] = getattr(model, "_kia_last_compression", None)
        return result
    finally:
        remove_kiaomni(model)
        try:
            delattr(model, "_phase02_kia_patch_active")
        except AttributeError:
            pass


def ratio_for_budget(prompt_len: int, budget: int) -> float:
    if budget >= prompt_len:
        return 0.0
    ratio = 1.0 - (budget + 0.5) / prompt_len
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"Invalid ratio for prompt_len={prompt_len}, budget={budget}")
    kept = int(prompt_len * (1.0 - ratio))
    if kept != budget:
        raise AssertionError(f"Exact budget ratio failed: expected={budget}, got={kept}")
    return ratio


def make_press(method: str, prompt_len: int, budget: int):
    from kvpress import KeyRerotationPress, SnapKVPress, StreamingLLMPress

    ratio = ratio_for_budget(prompt_len, budget)
    if method == "snapkv":
        return SnapKVPress(
            compression_ratio=ratio,
            window_size=SNAPKV_WINDOW,
            kernel_size=SNAPKV_KERNEL,
        )
    if method == "streamingllm":
        # NVIDIA notes that full StreamingLLM paper parity requires key
        # rerotation after pruning so retained keys occupy continuous RoPE
        # positions. Keep the official wrapper instead of the scorer alone.
        return KeyRerotationPress(
            press=StreamingLLMPress(compression_ratio=ratio, n_sink=4)
        )
    raise KeyError(method)


@torch.inference_mode()
def generate_kvpress(
    method: str,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    budget: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    clean_model(model)
    ids = input_ids.to(input_device(model))
    press = make_press(method, ids.shape[1], budget)
    reset_peak_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with press(model):
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out[0, ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    peaks = peak_memory_gb()
    return {
        "text": text,
        "new_tokens": int(new_ids.numel()),
        "elapsed_s": dt,
        "tokens_per_s": float(new_ids.numel() / max(dt, 1e-9)),
        "peak_vram_gb": max(peaks.values(), default=0.0),
        "peak_vram_by_gpu_gb": peaks,
        "compression": {
            "original_tokens": int(ids.shape[1]),
            "requested_budget": int(budget),
            "expected_kept_tokens": int(budget),
            "compression_ratio": float(press.compression_ratio),
        },
    }


def cache_layer_lengths(past_key_values) -> list[int]:
    if past_key_values is None:
        return []

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        out: list[int] = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if torch.is_tensor(keys) and keys.ndim >= 3:
                out.append(int(keys.shape[-2]))
        if out:
            return out

    lengths: list[int] = []
    try:
        for item in past_key_values:
            key = item[0] if isinstance(item, (tuple, list)) else item
            if torch.is_tensor(key) and key.ndim >= 3:
                lengths.append(int(key.shape[-2]))
    except Exception:
        pass
    return lengths


def attention_modules(model) -> list[Any]:
    base = getattr(model, "model", None)
    if base is None:
        return []
    language_model = getattr(base, "language_model", base)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return []
    return [layer.self_attn for layer in layers if hasattr(layer, "self_attn")]


def forward_hook_count(model) -> int:
    return sum(len(module._forward_hooks) for module in attention_modules(model))


def validation_prompt(tokenizer, target_len: int = 768) -> torch.Tensor:
    text = " ".join(f"validation record {i} is ordinary filler." for i in range(300))
    q = "Repeat the word ready."
    ids, _ = encode_prompt(tokenizer, text, q)
    if ids.shape[1] < target_len:
        raise RuntimeError("Validation prompt unexpectedly short")
    return ids[:, :target_len]


@torch.inference_mode()
def validate_external_press(
    method: str,
    model,
    tokenizer,
    budgets: list[int],
) -> ValidationResult:
    details: dict[str, Any] = {
        "budgets": {},
        "all_budgets_exact": False,
        "hooks_restored": False,
    }
    try:
        clean_model(model)
        ids = validation_prompt(tokenizer).to(input_device(model))
        prompt_tokens = int(ids.shape[1])
        hooks_before = forward_hook_count(model)
        all_exact = True

        for budget in budgets:
            press = make_press(method, prompt_tokens, budget)
            entry: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "requested_budget": budget,
                "compression_ratio": float(press.compression_ratio),
            }
            try:
                with press(model):
                    out = model(ids, use_cache=True)
                lengths = cache_layer_lengths(getattr(out, "past_key_values", None))
                entry["layer_cache_lengths"] = lengths
                entry["observed_layers"] = len(lengths)
                entry["exact_budget"] = bool(lengths) and all(
                    length == budget for length in lengths
                )
                if not entry["exact_budget"]:
                    all_exact = False

                with press(model):
                    gen = model.generate(
                        ids,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                entry["generation_advanced"] = bool(gen.shape[1] > ids.shape[1])
                if not entry["generation_advanced"]:
                    all_exact = False
            except Exception as exc:
                entry["exact_budget"] = False
                entry["generation_advanced"] = False
                entry["error_type"] = type(exc).__name__
                entry["error"] = str(exc)
                all_exact = False
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            details["budgets"][str(budget)] = entry

        hooks_after = forward_hook_count(model)
        details["forward_hooks_before"] = hooks_before
        details["forward_hooks_after"] = hooks_after
        details["hooks_restored"] = hooks_before == hooks_after
        details["all_budgets_exact"] = all_exact
        valid = all_exact and details["hooks_restored"]
        return ValidationResult(
            method,
            valid,
            "VALIDATED" if valid else "VALIDATION_FAIL",
            details,
        )
    except Exception as exc:
        details["reason"] = type(exc).__name__
        details["error"] = str(exc)
        return ValidationResult(method, False, "VALIDATION_FAIL", details)
    finally:
        clean_model(model)


@torch.inference_mode()
def validate_kiaomni(
    model,
    tokenizer,
    budgets: list[int],
) -> ValidationResult:
    details: dict[str, Any] = {"budgets": {}}
    all_exact = True
    try:
        ids = validation_prompt(tokenizer).to(input_device(model))
        for budget in budgets:
            try:
                result = generate_kiaomni(model, tokenizer, ids, budget, 1)
                comp = result.get("compression") or {}
                kept = comp.get("kept_tokens")
                exact = kept == budget
                details["budgets"][str(budget)] = {
                    "prompt_tokens": int(ids.shape[1]),
                    "requested_budget": budget,
                    "actual_kept_tokens": kept,
                    "exact_budget": exact,
                    "compression": comp,
                }
                if not exact:
                    all_exact = False
            except Exception as exc:
                details["budgets"][str(budget)] = {
                    "requested_budget": budget,
                    "exact_budget": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                all_exact = False
        details["all_budgets_exact"] = all_exact
        return ValidationResult(
            "kiaomni_s8",
            all_exact,
            "VALIDATED" if all_exact else "VALIDATION_FAIL",
            details,
        )
    except Exception as exc:
        details["reason"] = type(exc).__name__
        details["error"] = str(exc)
        return ValidationResult("kiaomni_s8", False, "VALIDATION_FAIL", details)
    finally:
        clean_model(model)


def validate_blocksal(
    budgets: list[int],
    seq_len: int = 768,
) -> ValidationResult:
    details: dict[str, Any] = {
        "block_size": BLOCK_SIZE,
        "selector": "mean_saliency_block_ranking_exact_budget_boundary",
        "exact_budget": True,
        "budgets": {},
    }
    all_valid = True
    try:
        rng = np.random.RandomState(123)
        sal = rng.rand(seq_len).astype(np.float32)
        protected = set(range(N_SINK)) | set(range(seq_len - RECENCY, seq_len))

        for budget in budgets:
            selection = select_blocksal_keep(
                sal,
                budget=budget,
                L=seq_len,
                block_size=BLOCK_SIZE,
                n_sink=N_SINK,
                recency=RECENCY,
            )
            keep = selection.keep_indices
            kept_set = set(keep.tolist())
            protected_ok = protected.issubset(kept_set)
            exact = len(keep) == min(budget, seq_len)
            entry = {
                "requested_budget": budget,
                "actual_kept_tokens": len(keep),
                "budget_delta": len(keep) - min(budget, seq_len),
                "protected_tokens_present": protected_ok,
                "exact_budget": exact,
                "partial_boundary_block": selection.partial_boundary_block,
            }
            details["budgets"][str(budget)] = entry
            if not (protected_ok and exact):
                all_valid = False

        details["all_budgets_exact"] = all_valid
        return ValidationResult(
            "blocksal",
            all_valid,
            "VALIDATED" if all_valid else "VALIDATION_FAIL",
            details,
        )
    except Exception as exc:
        details["reason"] = type(exc).__name__
        details["error"] = str(exc)
        return ValidationResult("blocksal", False, "VALIDATION_FAIL", details)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "exact_accuracy": sum(float(r["score"]["exact"]) for r in rows) / len(rows),
        "mean_recall": sum(float(r["score"]["recall"]) for r in rows) / len(rows),
        "mean_tokens_per_s": sum(float(r["tokens_per_s"]) for r in rows) / len(rows),
        "max_peak_vram_gb": max(float(r["peak_vram_gb"]) for r in rows),
    }


def max_new_for_task(task: str) -> int:
    return 16 if task in {"single", "reason"} else 48


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    if not budgets or any(b < N_SINK + RECENCY for b in budgets):
        raise ValueError(f"All budgets must be >= {N_SINK + RECENCY}")
    if args.target_tokens > args.max_context:
        raise ValueError("--target-tokens cannot exceed --max-context")

    print("=" * 96)
    print("KiaOmni x MoE Model Lab — Phase 02: Multi-Needle + Baseline Validation")
    print(f"Model: {args.model}")
    print(f"Budgets: {budgets} | final prompt target <= {args.target_tokens} | hard max={args.max_context}")
    print("Methods: FullContext, KiaOmni-s8, BlockSal (ours), SnapKV, StreamingLLM")
    print("=" * 96)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    transformers_version = package_version("transformers")
    if transformers_version != TRANSFORMERS_REQUIRED:
        raise RuntimeError(
            f"Phase-02 requires transformers=={TRANSFORMERS_REQUIRED}; "
            f"found {transformers_version!r}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()

    environment = {
        "torch": torch.__version__,
        "transformers": transformers_version,
        "accelerate": package_version("accelerate"),
        "kvpress": package_version("kvpress"),
        "kvpress_pinned_ref": KVPRESS_REF,
        "python": os.sys.version,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "device_map": getattr(model, "hf_device_map", None),
    }

    validations: dict[str, dict[str, Any]] = {
        "full_context": ValidationResult("full_context", True, "VALIDATED", {"compression": False}).__dict__,
        "kiaomni_s8": validate_kiaomni(model, tokenizer, budgets).__dict__,
        "blocksal": validate_blocksal(budgets).__dict__,
    }
    if args.skip_external:
        validations["snapkv"] = ValidationResult("snapkv", False, "SKIPPED", {"reason": "--skip-external"}).__dict__
        validations["streamingllm"] = ValidationResult("streamingllm", False, "SKIPPED", {"reason": "--skip-external"}).__dict__
    else:
        kvpress_version = package_version("kvpress")
        if kvpress_version is None:
            validations["snapkv"] = ValidationResult(
                "snapkv", False, "VALIDATION_FAIL", {"reason": "kvpress_not_installed"}
            ).__dict__
            validations["streamingllm"] = ValidationResult(
                "streamingllm", False, "VALIDATION_FAIL", {"reason": "kvpress_not_installed"}
            ).__dict__
        elif kvpress_version != KVPRESS_REQUIRED:
            details = {
                "reason": "kvpress_version_mismatch",
                "required_version": KVPRESS_REQUIRED,
                "found_version": kvpress_version,
                "pinned_ref": KVPRESS_REF,
            }
            validations["snapkv"] = ValidationResult(
                "snapkv", False, "VALIDATION_FAIL", details
            ).__dict__
            validations["streamingllm"] = ValidationResult(
                "streamingllm", False, "VALIDATION_FAIL", details
            ).__dict__
        else:
            validations["snapkv"] = validate_external_press(
                "snapkv", model, tokenizer, budgets
            ).__dict__
            validations["streamingllm"] = validate_external_press(
                "streamingllm", model, tokenizer, budgets
            ).__dict__

    print("\nValidation gate:")
    for name, v in validations.items():
        print(f"  {name:<14} {v['status']:<16} {v['details']}")

    if args.validation_only:
        artifact = {
            "experiment": "KIAOMNI_MOE_MODEL_LAB_PHASE02_VALIDATION_ONLY_V1",
            "model": args.model,
            "mode": "validation_only",
            "budgets": budgets,
            "max_context_tokens": args.max_context,
            "target_final_prompt_tokens": args.target_tokens,
            "environment": environment,
            "validation_gate": validations,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"Validation artifact saved: {out_path}")
        return

    if not args.skip_external:
        invalid_external = [
            name for name in ("snapkv", "streamingllm")
            if not validations[name]["valid"]
        ]
        if invalid_external:
            artifact = {
                "experiment": "KIAOMNI_MOE_MODEL_LAB_PHASE02_VALIDATION_FAIL_V1",
                "model": args.model,
                "mode": "validation_fail",
                "budgets": budgets,
                "max_context_tokens": args.max_context,
                "target_final_prompt_tokens": args.target_tokens,
                "environment": environment,
                "validation_gate": validations,
                "invalid_external_methods": invalid_external,
            }
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            raise RuntimeError(
                "External baseline validation failed for "
                + ", ".join(invalid_external)
                + ". Refusing to produce comparison scores."
            )

    cases = build_cases(tokenizer, args.samples_per_task, args.seed, args.target_tokens, args.max_context)
    for case in cases:
        ids, _ = encode_prompt(tokenizer, case.context, case.question)
        if ids.shape[1] > args.max_context:
            raise AssertionError(f"{case.task}/{case.sample_id}: {ids.shape[1]} > max_context")

    rows: list[dict[str, Any]] = []

    def record(case: Case, method: str, budget: int | None, prompt_tokens: int, result: dict[str, Any]) -> None:
        sc = score_answer(case, result["text"])
        row = {
            "task": case.task,
            "sample_id": case.sample_id,
            "method": method,
            "budget": budget,
            "prompt_tokens": prompt_tokens,
            "gold": case.gold,
            "distractors": case.distractors,
            "info": case.info,
            "score": sc,
            **result,
        }
        rows.append(row)
        mark = "PASS" if sc["exact"] else f"FAIL recall={sc['recall']:.2f}"
        kept = (result.get("compression") or {}).get("kept_tokens")
        kept_text = f" kept={kept}" if kept is not None else ""
        print(
            f"  {method:<13} B={str(budget):>4} {mark:<18} "
            f"tok/s={result['tokens_per_s']:.2f} peak={result['peak_vram_gb']:.2f}GB{kept_text} "
            f"text={result['text'][:100]!r}"
        )

    for case in cases:
        ids, _ = encode_prompt(tokenizer, case.context, case.question)
        prompt_tokens = int(ids.shape[1])
        max_new = max_new_for_task(case.task)
        print("\n" + "-" * 96)
        print(f"{case.task} #{case.sample_id} | prompt={prompt_tokens} | gold={case.gold} | {case.info}")

        clean_model(model)
        record(case, "full_context", None, prompt_tokens, generate_full(model, tokenizer, ids, max_new))

        for budget in budgets:
            if validations["kiaomni_s8"]["valid"]:
                record(case, "kiaomni_s8", budget, prompt_tokens, generate_kiaomni(model, tokenizer, ids, budget, max_new))
            if validations["blocksal"]["valid"]:
                record(case, "blocksal", budget, prompt_tokens, generate_blocksal(model, tokenizer, ids, budget, max_new))
            if validations["snapkv"]["valid"]:
                record(case, "snapkv", budget, prompt_tokens, generate_kvpress("snapkv", model, tokenizer, ids, budget, max_new))
            if validations["streamingllm"]["valid"]:
                record(case, "streamingllm", budget, prompt_tokens, generate_kvpress("streamingllm", model, tokenizer, ids, budget, max_new))

    summary: dict[str, Any] = {}
    keys = sorted({(r["method"], r["budget"]) for r in rows}, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    for method, budget in keys:
        sub = [r for r in rows if r["method"] == method and r["budget"] == budget]
        name = method if budget is None else f"{method}_b{budget}"
        summary[name] = aggregate(sub)

    artifact = {
        "experiment": "KIAOMNI_MOE_MODEL_LAB_PHASE02_MULTINEEDLE_BASELINES_V1",
        "model": args.model,
        "weights_frozen": True,
        "precision": "FP16",
        "max_context_tokens": args.max_context,
        "target_final_prompt_tokens": args.target_tokens,
        "budgets": budgets,
        "samples_per_task": args.samples_per_task,
        "seed": args.seed,
        "tasks": ["single", "multi", "hard_multi", "reason"],
        "methods": {
            "full_context": {"class": "upper_bound", "owner": "baseline"},
            "kiaomni_s8": {"class": "our_method", "policy": "kiaomni_s8"},
            "blocksal": {
                "class": "our_method_internal_variant",
                "block_size": BLOCK_SIZE,
                "n_sink": N_SINK,
                "recency": RECENCY,
                "selector": "mean_saliency_block_ranking_exact_budget_boundary",
                "provenance": "our BlockSal design; canonical historical block ranking with exact-budget benchmark boundary adapter",
                "exact_budget": True,
                "ownership_note": "BlockSal is our internal method, not an external baseline.",
            },
            "snapkv": {
                "class": "external_baseline",
                "source": "NVIDIA/kvpress",
                "pinned_ref": KVPRESS_REF,
                "window_size": SNAPKV_WINDOW,
                "kernel_size": SNAPKV_KERNEL,
                "exact_budget_ratio": True,
            },
            "streamingllm": {
                "class": "external_baseline",
                "source": "NVIDIA/kvpress",
                "pinned_ref": KVPRESS_REF,
                "n_sink": 4,
                "key_rerotation": True,
                "implementation": "KeyRerotationPress(StreamingLLMPress)",
                "exact_budget_ratio": True,
            },
        },
        "environment": environment,
        "validation_gate": validations,
        "summary": summary,
        "rows": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("PHASE 02 SUMMARY")
    for name, metrics in summary.items():
        print(
            f"{name:<25} exact={metrics.get('exact_accuracy', 0):.3f} "
            f"recall={metrics.get('mean_recall', 0):.3f} "
            f"tok/s={metrics.get('mean_tokens_per_s', 0):.2f} "
            f"peak={metrics.get('max_peak_vram_gb', 0):.2f}GB"
        )
    print(f"Saved: {out_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
