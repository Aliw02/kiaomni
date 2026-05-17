"""
kiaomni/monkey_patch.py
=========================
Public entrypoint that attaches KiaOmni KV-cache eviction to any
HuggingFace causal LM via runtime introspection.

Algorithm — prompt-side eviction
---------------------------------
KiaOmni runs ONE forward pass on the full prompt to extract per-token
saliency, then selects the top-`budget` token positions (always
protecting `n_sink` sinks and `recency` trailing tokens). The model is
then re-invoked on the kept-tokens subset as a fresh, shorter prompt.

This matches the paper-evaluated algorithm in
`notebook/kv_cache_benchmark/033_full_comparison.py` and has been
validated on Qwen2.5-7B, Mistral, BioMistral, Llama-3.1, and TinyLlama.

Why prompt-side, not cache-side?
--------------------------------
Cache-side eviction (gather KV, resume with `past_key_values`) requires
exact alignment of `cache_position`, `position_ids`, and RoPE rotation
positions — contracts that drift between transformers releases and break
across model families. Prompt-side eviction delegates all of that to
the model's own `generate`, which is the only path the HF maintainers
guarantee.

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
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import torch

from .adapters import ArchitectureProbe, KiaomniConfigError, ProbeResult
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
    Monkey-patch ``model.generate`` to apply KiaOmni prompt-side eviction.

    Parameters
    ----------
    model    : any HuggingFace AutoModelForCausalLM loaded with
               ``attn_implementation="eager"``.
    policy   : key from ``kiaomni.policies.POLICY_REGISTRY``.
    budget   : number of input-token positions to retain.
    n_sink   : how many initial tokens to always protect.
    recency  : how many trailing tokens to always protect.
    verbose  : print kept / total per call.

    Returns
    -------
    ProbeResult
        Also stored on ``model._kia_arch_info``.
    """
    if budget < n_sink + recency:
        raise ValueError(
            f"budget={budget} is smaller than n_sink+recency="
            f"{n_sink + recency}. Use a larger budget."
        )

    # Idempotency: if a previous apply_kiaomni left a wrapper on model.generate,
    # unwind it before installing a new one. Without this, repeated apply calls
    # stack wrappers; the innermost _orig stops resolving to the bound method
    # and the model's own Tensor input gets passed as `self` to generate().
    if hasattr(model, "_kia_arch_info") or hasattr(getattr(model, "generate", None), "__wrapped__"):
        remove_kiaomni(model)

    score_fn = get_policy(policy)
    probe = ArchitectureProbe.probe(model)
    saliency = SaliencyAdapter(probe)

    _orig = model.generate

    @functools.wraps(_orig)
    def _patched(input_ids: torch.Tensor, **kwargs):
        # Pass-through for non-2D inputs (e.g., inputs_embeds path).
        if input_ids.dim() != 2:
            return _orig(input_ids, **kwargs)
        B, L = input_ids.shape

        # No eviction needed — prompt fits in budget.
        if L <= budget:
            return _orig(input_ids, **kwargs)

        # 1. Extract per-token saliency over the full prompt: (B, L)
        sal_batch = saliency.extract(input_ids, model)

        # 2. Select top-budget positions per row, always protecting
        #    n_sink initial + recency trailing tokens.
        keep_per_row = [
            np.sort(
                select_keep(
                    score_fn(sal_batch[b]), budget, L,
                    n_sink=n_sink, recency=recency,
                )
            )
            for b in range(B)
        ]

        if verbose:
            kept = [len(k) for k in keep_per_row]
            print(f"[KiaOmni] {policy} budget={budget} kept={kept}/{L}")

        # 3. Slice input_ids by kept positions and run generate fresh.
        #    Single-row fast path (the common case for `model.generate`).
        if B == 1:
            keep_t = torch.as_tensor(
                keep_per_row[0], device=input_ids.device, dtype=torch.long
            )
            pruned = input_ids[:, keep_t]
            kwargs.setdefault("attention_mask", torch.ones_like(pruned))
            out = _orig(pruned, **kwargs)
        else:
            # Multi-row path: pad per-row keeps to the max length with the
            # pad token (or eos), and build an attention mask that zeroes
            # the pad slots so they don't pollute attention.
            max_keep = max(len(k) for k in keep_per_row)
            pad_id = _resolve_pad_token_id(model)
            pruned = torch.full(
                (B, max_keep), pad_id,
                device=input_ids.device, dtype=input_ids.dtype,
            )
            attn_mask = torch.zeros(
                (B, max_keep), device=input_ids.device, dtype=torch.long
            )
            for b, k in enumerate(keep_per_row):
                n = len(k)
                keep_t = torch.as_tensor(k, device=input_ids.device, dtype=torch.long)
                pruned[b, :n] = input_ids[b, keep_t]
                attn_mask[b, :n] = 1
            kwargs.setdefault("attention_mask", attn_mask)
            out = _orig(pruned, **kwargs)

        # 4. Restore HF generate contract: out[:, input_ids.shape[1]:] must
        # yield only the new tokens. Internally we shortened the prompt to
        # `pruned.shape[1]`, so we prepend the original prompt to the new
        # tokens, returning a tensor whose prefix matches the caller's input.
        if not isinstance(out, torch.Tensor):
            return out  # generate may return GenerateOutput dataclass — pass through
        pruned_len = pruned.shape[1]
        if out.shape[1] <= pruned_len:
            return out  # no new tokens generated (e.g., immediate EOS)
        new_tokens = out[:, pruned_len:]
        return torch.cat([input_ids, new_tokens], dim=1)

    model.generate = _patched  # type: ignore[method-assign]
    return probe


def remove_kiaomni(model) -> None:
    """Undo all previous ``apply_kiaomni`` calls, even if stacked.

    Walks the ``__wrapped__`` chain until it finds the original bound
    method, then restores it. Falls back to deleting the instance
    attribute so the class-level ``generate`` shows through.
    """
    # Unwind every wrapper level — handles repeated apply_kiaomni calls.
    current = getattr(model, "generate", None)
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "__wrapped__", None)
        if inner is None:
            break
        current = inner

    if current is not None:
        # If we found a non-wrapper that looks like the original bound
        # method, install it. Otherwise delete the instance attribute so
        # the descriptor-bound class method takes over.
        try:
            model.generate = current
        except Exception:
            try:
                delattr(model, "generate")
            except AttributeError:
                pass
    else:
        try:
            delattr(model, "generate")
        except AttributeError:
            pass

    if hasattr(model, "_kia_arch_info"):
        try:
            delattr(model, "_kia_arch_info")
        except AttributeError:
            pass


def _resolve_pad_token_id(model) -> int:
    """Best-effort pad token: model config → eos fallback → 0."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        pad = getattr(cfg, "pad_token_id", None)
        if pad is None:
            pad = getattr(cfg, "eos_token_id", None)
        if isinstance(pad, list) and pad:
            pad = pad[0]
        if isinstance(pad, int):
            return pad
    return 0


__all__ = ["apply_kiaomni", "remove_kiaomni", "KiaomniConfigError"]
