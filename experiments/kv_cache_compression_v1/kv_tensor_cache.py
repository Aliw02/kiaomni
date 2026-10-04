"""Physical KV tensor compaction helpers for Gate-0 experiments."""

from __future__ import annotations

from typing import Iterable
import torch

LegacyKV = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def cache_nbytes(cache: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> int:
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)


def compact_legacy_kv(cache: LegacyKV, keep: torch.Tensor) -> LegacyKV:
    """Physically select the KV sequence axis (-2) for every layer."""
    compacted = []
    for key, value in cache:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("expected [batch, heads, seq_len, head_dim] KV tensors")
        compacted.append(
            (
                key.index_select(-2, keep.to(key.device)).contiguous(),
                value.index_select(-2, keep.to(value.device)).contiguous(),
            )
        )
    return tuple(compacted)
