from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

KV_GAUSSIAN_SIGMA_DEFAULT = 4.0
KV_N_SINK_DEFAULT = 4
KV_RECENCY_DEFAULT = 8


def kv_gaussian_score(saliency: np.ndarray, sigma: float = KV_GAUSSIAN_SIGMA_DEFAULT) -> np.ndarray:
    """KV-specific copy/adaptation of KiaOmni Gaussian scoring.

    This file is intentionally independent from kiaomni/policies.py so the
    frozen prompt-selection implementation is never modified by KV POC work.
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    return gaussian_filter1d(np.log1p(saliency), sigma=sigma).astype(np.float32)


def select_kv_positions(
    saliency: np.ndarray,
    budget: int,
    *,
    n_sink: int = KV_N_SINK_DEFAULT,
    recency: int = KV_RECENCY_DEFAULT,
) -> np.ndarray:
    """Select absolute KV positions to keep, preserving sink + recent tokens."""
    saliency = np.asarray(saliency, dtype=np.float32)
    length = int(saliency.shape[0])
    if budget >= length:
        return np.arange(length, dtype=np.int64)
    score = kv_gaussian_score(saliency)
    protected = set(range(min(n_sink, length)))
    protected.update(range(max(0, length - recency), length))
    free = max(0, int(budget) - len(protected))
    candidates = np.asarray([i for i in range(length) if i not in protected], dtype=np.int64)
    if free > 0 and len(candidates):
        k = min(free, len(candidates))
        top_local = np.argpartition(-score[candidates], k - 1)[:k]
        protected.update(candidates[top_local].tolist())
    return np.asarray(sorted(protected), dtype=np.int64)
