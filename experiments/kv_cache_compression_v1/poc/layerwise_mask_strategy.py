from __future__ import annotations

import numpy as np
import torch

from kv_gaussian_policy import select_kv_positions
from toy_stacked_kv_model import CompactKVCache


def _to_numpy(saliency) -> np.ndarray:
    if isinstance(saliency, torch.Tensor):
        return saliency.detach().cpu().numpy().astype(np.float32)
    return np.asarray(saliency, dtype=np.float32)


def layerwise_keep_indices(
    saliencies,
    budget: int,
    *,
    n_sink: int = 4,
    recency: int = 8,
) -> list[np.ndarray]:
    """Emit an independent KV-position mask for each decoder layer."""
    return [
        select_kv_positions(_to_numpy(s), budget, n_sink=n_sink, recency=recency)
        for s in saliencies
    ]


def compact_layerwise(
    caches: list[CompactKVCache],
    keeps: list[np.ndarray],
) -> list[CompactKVCache]:
    """Apply each layer's independent mask while preserving equal per-layer budgets."""
    if len(caches) != len(keeps):
        raise ValueError("cache/keep layer count mismatch")
    return [
        cache.compact(torch.from_numpy(np.asarray(keep, dtype=np.int64)))
        for cache, keep in zip(caches, keeps)
    ]
