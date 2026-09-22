"""Faithful SnapKV compatibility adapter for MobileMoE.

This module adapts only MobileMoE's attention representation to NVIDIA
kvpress SnapKV. The SnapKV scoring rule, observation window, pooling,
GQA grouping, and Top-K KV pruning remain unchanged.

The adapter is intentionally narrow:
- reconstruct MobileMoE query states with its native q_proj + q_norm;
- normalize MobileMoE RoPE representation to the cos/sin form expected by
  kvpress when possible;
- delegate the rest of SnapKV semantics unchanged.

If MobileMoE exposes a RoPE representation that cannot be converted without
changing attention semantics, the adapter raises instead of approximating.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ADAPTER_NAME = "mobilemoe_snapkv_compat_v1"
ALGORITHM_MODIFIED = False


class MobileMoESnapKVCompatibilityError(RuntimeError):
    pass


def is_mobilemoe_module(module: nn.Module) -> bool:
    config = getattr(module, "config", None)
    model_type = getattr(config, "model_type", None)
    return str(model_type).lower() == "mobilemoe" or "mobilemoe" in type(module).__name__.lower()


def mobilemoe_prerope_query_states(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct MobileMoE pre-RoPE queries using native q_proj + q_norm."""
    if not hasattr(module, "q_proj"):
        raise MobileMoESnapKVCompatibilityError(
            f"{type(module).__name__} has no q_proj"
        )

    bsz, q_len, _ = hidden_states.shape
    num_heads = int(module.config.num_attention_heads)
    head_dim = int(module.head_dim)

    query_states = module.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


def _position_ids_from_kwargs(
    kwargs: dict[str, Any],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    position_ids = kwargs.get("position_ids")
    if torch.is_tensor(position_ids):
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        return position_ids

    cache_position = kwargs.get("cache_position")
    if torch.is_tensor(cache_position):
        if cache_position.ndim == 1:
            cache_position = cache_position.unsqueeze(0)
        return cache_position

    q_len = hidden_states.shape[1]
    return torch.arange(q_len, device=hidden_states.device).unsqueeze(0)


def describe_position_embeddings(value: Any) -> dict[str, Any]:
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [describe_position_embeddings(item) for item in value],
        }
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "is_complex": bool(torch.is_complex(value)),
        }
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def _extract_cos_sin(value: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        cos, sin = value
        if torch.is_tensor(cos) and torch.is_tensor(sin):
            return cos, sin
    return None


def resolve_mobilemoe_rope(
    module: nn.Module,
    hidden_states: torch.Tensor,
    kwargs: dict[str, Any],
) -> tuple[str, Any]:
    """Resolve the exact RoPE representation without changing its semantics."""
    position_embeddings = kwargs.get("position_embeddings")
    pair = _extract_cos_sin(position_embeddings)
    if pair is not None:
        return "kwargs_cos_sin", pair

    if torch.is_tensor(position_embeddings) and torch.is_complex(position_embeddings):
        return "kwargs_complex", position_embeddings

    rotary = getattr(module, "rotary_emb", None)
    if rotary is not None:
        position_ids = _position_ids_from_kwargs(kwargs, hidden_states)
        attempts = (
            lambda: rotary(hidden_states, position_ids),
            lambda: rotary(hidden_states, position_ids=position_ids),
            lambda: rotary(position_ids),
        )
        errors: list[str] = []
        for attempt in attempts:
            try:
                value = attempt()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            pair = _extract_cos_sin(value)
            if pair is not None:
                return "rotary_cos_sin", pair
            if torch.is_tensor(value) and torch.is_complex(value):
                return "rotary_complex", value
            errors.append(
                "unsupported rotary output "
                + str(describe_position_embeddings(value))
            )
    else:
        errors = ["module has no rotary_emb"]

    raise MobileMoESnapKVCompatibilityError(
        "Could not resolve MobileMoE RoPE without approximation. "
        f"position_embeddings={describe_position_embeddings(position_embeddings)}; "
        f"rotary_attempts={errors}"
    )


def _apply_cos_sin_rope(
    query_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    from transformers.models.llama.modeling_llama import rotate_half

    q_len = query_states.shape[-2]
    cos = cos[..., -q_len:, :]
    sin = sin[..., -q_len:, :]

    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    return (query_states * cos) + (rotate_half(query_states) * sin)


def _apply_complex_rope(
    query_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply standard complex RoPE exactly when MobileMoE exposes complex frequencies."""
    q_len = query_states.shape[-2]
    head_dim = query_states.shape[-1]
    if head_dim % 2 != 0:
        raise MobileMoESnapKVCompatibilityError(
            f"Complex RoPE requires even head_dim, got {head_dim}"
        )

    freqs = freqs_cis
    if freqs.shape[-2] >= q_len:
        freqs = freqs[..., -q_len:, :]

    q_float = query_states.float()
    q_complex = torch.view_as_complex(
        q_float.reshape(*q_float.shape[:-1], head_dim // 2, 2).contiguous()
    )

    while freqs.ndim < q_complex.ndim:
        freqs = freqs.unsqueeze(0)

    try:
        rotated = q_complex * freqs
    except RuntimeError as exc:
        raise MobileMoESnapKVCompatibilityError(
            "Complex RoPE shape mismatch: "
            f"query_complex={tuple(q_complex.shape)} freqs={tuple(freqs.shape)}"
        ) from exc

    return torch.view_as_real(rotated).flatten(-2).to(query_states.dtype)


def mobilemoe_window_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    keys: torch.Tensor,
    window_size: int,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Official SnapKV window-attention equation with MobileMoE representation adaptation."""
    from transformers.models.llama.modeling_llama import repeat_kv

    bsz, _, k_len, _ = keys.shape
    num_heads = int(module.config.num_attention_heads)
    head_dim = int(module.head_dim)
    num_key_value_groups = num_heads // int(module.config.num_key_value_heads)

    window_hidden = hidden_states[:, -window_size:]
    query_states = mobilemoe_prerope_query_states(module, window_hidden)

    rope_kind, rope_value = resolve_mobilemoe_rope(module, hidden_states, kwargs)
    if rope_kind.endswith("cos_sin"):
        cos, sin = rope_value
        query_states = _apply_cos_sin_rope(query_states, cos, sin)
    elif rope_kind.endswith("complex"):
        query_states = _apply_complex_rope(query_states, rope_value)
    else:
        raise MobileMoESnapKVCompatibilityError(
            f"Unsupported RoPE kind: {rope_kind}"
        )

    key_states = repeat_kv(keys, num_key_value_groups)
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)

    attention_mask = torch.ones_like(attn_weights) * float("-inf")
    attention_mask = torch.triu(
        attention_mask,
        diagonal=k_len - window_size + 1,
    )
    attn_weights += attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query_states.dtype)
    attn_weights = attn_weights[..., :-window_size]

    return attn_weights


def make_mobilemoe_snapkv_press(
    *,
    compression_ratio: float,
    window_size: int,
    kernel_size: int,
):
    """Create a faithful SnapKV press with MobileMoE representation adaptation."""
    from kvpress import SnapKVPress

    class MobileMoESnapKVPress(SnapKVPress):
        compatibility_adapter = ADAPTER_NAME
        algorithm_modified = ALGORITHM_MODIFIED

        def score(
            self,
            module: nn.Module,
            hidden_states: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attentions: torch.Tensor,
            kwargs,
        ) -> torch.Tensor:
            # This body intentionally mirrors NVIDIA kvpress SnapKVPress.score
            # at KVPRESS_REF. Only the no-attentions query/RoPE reconstruction
            # is routed through the MobileMoE representation adapter.
            bsz, num_key_value_heads, k_len, _ = keys.shape
            num_key_value_groups = (
                module.config.num_attention_heads // num_key_value_heads
            )

            assert hidden_states.shape[1] > self.window_size, (
                f"Query length {hidden_states.shape[1]} should be greater than "
                f"the window size {self.window_size}"
            )

            if attentions is not None:
                attn_weights = attentions[
                    ..., -self.window_size :, : -self.window_size
                ]
            else:
                attn_weights = mobilemoe_window_attention(
                    module,
                    hidden_states,
                    keys,
                    self.window_size,
                    kwargs,
                )

            scores = attn_weights.mean(dim=-2)
            scores = F.avg_pool1d(
                scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )

            scores = scores.view(
                bsz,
                num_key_value_heads,
                num_key_value_groups,
                k_len - self.window_size,
            )
            scores = scores.mean(2)

            scores = F.pad(
                scores,
                (0, self.window_size),
                value=scores.max().item() + 1,
            )
            return scores

    return MobileMoESnapKVPress(
        compression_ratio=compression_ratio,
        window_size=window_size,
        kernel_size=kernel_size,
    )


def adapter_provenance() -> dict[str, Any]:
    return {
        "adapter": ADAPTER_NAME,
        "algorithm_modified": ALGORITHM_MODIFIED,
        "scope": [
            "MobileMoE q_proj/q_norm query reconstruction",
            "MobileMoE RoPE representation normalization",
        ],
        "official_kvpress_ref": "7331c23da9e6f1510d89ea651d0dea77a57b3252",
        "official_snapkv_score_equation_preserved": True,
        "snapkv_scoring_changed": False,
        "snapkv_window_changed": False,
        "snapkv_pooling_changed": False,
        "snapkv_pruning_changed": False,
    }


__all__ = [
    "ADAPTER_NAME",
    "ALGORITHM_MODIFIED",
    "MobileMoESnapKVCompatibilityError",
    "adapter_provenance",
    "describe_position_embeddings",
    "is_mobilemoe_module",
    "make_mobilemoe_snapkv_press",
    "mobilemoe_prerope_query_states",
    "mobilemoe_window_attention",
    "resolve_mobilemoe_rope",
]
