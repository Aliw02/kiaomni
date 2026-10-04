from __future__ import annotations

import numpy as np
import torch

from kv_gaussian_policy import select_kv_positions
from toy_stacked_kv_model import CompactKVCache


def _to_numpy(saliency) -> np.ndarray:
    if isinstance(saliency, torch.Tensor):
        return saliency.detach().cpu().numpy().astype(np.float32)
    return np.asarray(saliency, dtype=np.float32)


def global_keep_indices(
    saliencies,
    budget: int,
    *,
    n_sink: int = 4,
    recency: int = 8,
) -> np.ndarray:
    """Aggregate all layer saliencies and emit one shared KV-position mask."""
    matrix = np.stack([_to_numpy(s) for s in saliencies], axis=0)
    global_saliency = matrix.mean(axis=0)
    return select_kv_positions(global_saliency, budget, n_sink=n_sink, recency=recency)


def compact_global(caches: list[CompactKVCache], keep: np.ndarray) -> list[CompactKVCache]:
    """Apply the exact same position mask to every decoder-layer cache."""
    idx = torch.from_numpy(np.asarray(keep, dtype=np.int64))
    return [cache.compact(idx) for cache in caches]
