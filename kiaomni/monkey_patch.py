"""
kiaomni/monkey_patch.py
=========================
Public entrypoint that attaches KiaOmni KV-cache eviction to any
HuggingFace causal LM via runtime introspection — no architecture
constants, no hardcoded module paths.

Quickstart
----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kiaomni import apply_kiaomni

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        attn_implementation="eager",   # required
        torch_dtype="auto",
    )
    apply_kiaomni(model, policy="kiaomni_s8", budget=256)

    out = model.generate(tok("...", return_tensors="pt").input_ids,
                         max_new_tokens=128)

Verified architectures
----------------------
    Llama 3 / 3.1, Mistral, Qwen2 / Qwen2.5, TinyLlama
    GPT-2, GPT-NeoX, Falcon

For anything else, the probe falls back to ``output_attentions=True``
extraction which is slower but model-agnostic.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np
import torch

from .adapters import ArchitectureProbe, KiaomniConfigError, ProbeResult
from .adapters.cache import CacheAdapter
from .adapters.saliency import SaliencyAdapter
from .policies import get_policy
from .utils import N_SINK_DEFAULT, RECENCY_DEFAULT, select_keep

logger = logging.getLogger(__name__)

_DEFAULT_BUDGET = 256


# ── Public API ───────────────────────────────────────────────────────────────

def apply_kiaomni(
    model,
    policy: str = "kiaomni_s8",
    budget: int = _DEFAULT_BUDGET,
    *,
    n_sink: int = N_SINK_DEFAULT,
    recency: int = RECENCY_DEFAULT,
    verbose: bool = False,
) -> ProbeResult:
    """
    Monkey-patch ``model.generate`` to apply KiaOmni KV eviction.

    Parameters
    ----------
    model    : any HuggingFace AutoModelForCausalLM loaded with
               ``attn_implementation="eager"``.
    policy   : key from ``kiaomni.policies.POLICY_REGISTRY``.
    budget   : number of KV positions to retain after eviction.
    n_sink   : how many initial tokens to always protect.
    recency  : how many trailing tokens to always protect.
    verbose  : print kept / total per call.

    Returns
    -------
    ProbeResult
        The probe result is also stored on ``model._kia_arch_info``.
    """
    if budget < n_sink + recency:
        raise ValueError(
            f"budget={budget} is smaller than n_sink+recency="
            f"{n_sink + recency}. Use a larger budget."
        )

    score_fn = get_policy(policy)
    probe = ArchitectureProbe.probe(model)
    saliency = SaliencyAdapter(probe)
    cache = CacheAdapter(probe)

    _orig = model.generate

    @functools.wraps(_orig)
    def _patched(input_ids: torch.Tensor, **kwargs):
        if input_ids.dim() != 2:
            return _orig(input_ids, **kwargs)
        B, L = input_ids.shape

        if L <= budget:
            return _orig(input_ids, **kwargs)

        # 1. Extract per-token saliency: (B, L)
        sal_batch = saliency.extract(input_ids, model)

        # 2. Score & select per batch row.
        keep_per_row = []
        for b in range(B):
            score = score_fn(sal_batch[b])
            keep = select_keep(
                score, budget, L, n_sink=n_sink, recency=recency
            )
            keep_per_row.append(keep)

        if verbose:
            kept = [len(k) for k in keep_per_row]
            print(f"[KiaOmni] {policy} budget={budget} kept={kept}/{L}")

        # 3. Build evicted KV cache.
        raw_kv = cache.gather(input_ids, model)
        pkv = cache.evict(raw_kv, keep_per_row, model)

        # 4. Resume generation from the cache.
        kwargs.setdefault("past_key_values", pkv)
        max_keep = max(len(k) for k in keep_per_row)

        # cache_position: where in the (compressed) cache the next K/V goes.
        # transformers>=4.45 indexes cache_position[-1] in prepare_inputs_for_generation;
        # if absent it derives from past_key_values, which fails for our custom cache.
        kwargs.setdefault(
            "cache_position",
            torch.arange(
                max_keep, max_keep + 1,
                device=input_ids.device, dtype=torch.long,
            ),
        )

        if probe.pos_encoding == "rope":
            # RoPE: cached K was rotated at its ORIGINAL positions, so the new
            # Q must also use the original position (L-1 for the last input token).
            pos_id = torch.tensor(
                [[L - 1]] * B,
                device=input_ids.device,
                dtype=torch.long,
            )
            kwargs.setdefault("position_ids", pos_id)
        # For learned / alibi / none, transformers handles positions internally.

        return _orig(input_ids[:, -1:], **kwargs)

    model.generate = _patched  # type: ignore[method-assign]
    return probe


def remove_kiaomni(model) -> None:
    """Undo a previous ``apply_kiaomni`` call."""
    orig = getattr(model.generate, "__wrapped__", None)
    if orig is not None:
        model.generate = orig
    if hasattr(model, "_kia_arch_info"):
        try:
            delattr(model, "_kia_arch_info")
        except AttributeError:
            pass


__all__ = ["apply_kiaomni", "remove_kiaomni", "KiaomniConfigError"]
