"""
kiaomni/loading.py
====================
Model loading with automatic attention-backend fallback.

Mirrors the loader used by every paper benchmark
(``experiments/033_full_comparison.py``): try the fastest available
attention implementation first and degrade gracefully —

    flash_attention_2  →  sdpa  →  eager

KiaOmni's saliency hooks sit on the ``q_proj``/``k_proj`` projection
layers and fire *before* the fused attention kernel, so every backend
in the chain is fully supported. Eager is only required by the
low-confidence ``output_attentions=True`` fallback inside the
``SaliencyAdapter``; it is the last resort here, never the default.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Default backend preference order — fastest first, eager as last resort.
ATTN_FALLBACK_CHAIN: Tuple[str, ...] = ("flash_attention_2", "sdpa", "eager")


def load_model(
    model_id: str,
    *,
    attn_chain: Sequence[str] = ATTN_FALLBACK_CHAIN,
    **from_pretrained_kwargs: Any,
):
    """Load ``AutoModelForCausalLM`` trying each attention backend in turn.

    Parameters
    ----------
    model_id              : HuggingFace model id or local path.
    attn_chain            : backend preference order; default
                            ``("flash_attention_2", "sdpa", "eager")``.
    from_pretrained_kwargs: forwarded verbatim to
                            ``AutoModelForCausalLM.from_pretrained``
                            (``quantization_config``, ``device_map``,
                            ``torch_dtype``, ...).

    Returns
    -------
    The loaded model, with ``attn_implementation`` set to the first
    backend in ``attn_chain`` that loads successfully.

    Raises
    ------
    The exception from the final backend if every backend fails
    (e.g. the model id does not exist or the download fails).
    """
    from transformers import AutoModelForCausalLM

    if not attn_chain:
        raise ValueError("attn_chain must contain at least one backend name")

    last_exc: BaseException | None = None
    for impl in attn_chain:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation=impl, **from_pretrained_kwargs
            )
            logger.info("kiaomni.load_model: attn_implementation=%s", impl)
            return model
        except Exception as exc:  # noqa: BLE001 — backend probing must be broad
            logger.info(
                "kiaomni.load_model: attn_implementation=%r unavailable (%s); "
                "trying next backend",
                impl,
                exc,
            )
            last_exc = exc
    assert last_exc is not None
    raise last_exc


__all__ = ["load_model", "ATTN_FALLBACK_CHAIN"]
