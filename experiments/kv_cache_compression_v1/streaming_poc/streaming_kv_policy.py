from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


STREAMING_GAUSSIAN_SIGMA_DEFAULT = 4.0
STREAMING_N_SINK_DEFAULT = 4
STREAMING_RECENCY_DEFAULT = 8


def score_on_absolute_axis(
    saliency: np.ndarray,
    absolute_positions: np.ndarray,
    seen_length: int,
    sigma: float = STREAMING_GAUSSIAN_SIGMA_DEFAULT,
) -> np.ndarray:
    """Gaussian scoring without pretending compacted neighbors were adjacent.

    Saliency for currently retained entries is projected back onto the original
    absolute token axis, missing/evicted locations are zero, Gaussian smoothing
    happens on that original axis, then candidate scores are gathered back.
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    absolute_positions = np.asarray(absolute_positions, dtype=np.int64)
    dense = np.zeros(int(seen_length), dtype=np.float32)
    dense[absolute_positions] = np.maximum(saliency, 0.0)
    smooth = gaussian_filter1d(np.log1p(dense), sigma=sigma)
    return smooth[absolute_positions].astype(np.float32)


def select_streaming_entries(
    saliency: np.ndarray,
    absolute_positions: np.ndarray,
    budget: int,
    seen_length: int,
    *,
    n_sink: int = STREAMING_N_SINK_DEFAULT,
    recency: int = STREAMING_RECENCY_DEFAULT,
    sigma: float = STREAMING_GAUSSIAN_SIGMA_DEFAULT,
) -> np.ndarray:
    """Select local cache-entry indices while protecting absolute sink/recent positions."""
    absolute_positions = np.asarray(absolute_positions, dtype=np.int64)
    budget = min(int(budget), len(absolute_positions))
    if budget >= len(absolute_positions):
        return np.arange(len(absolute_positions), dtype=np.int64)

    scores = score_on_absolute_axis(
        saliency,
        absolute_positions,
        seen_length,
        sigma=sigma,
    )

    protected = {
        i
        for i, pos in enumerate(absolute_positions)
        if pos < n_sink or pos >= max(0, seen_length - recency)
    }
    if len(protected) > budget:
        raise ValueError(
            f"protected streaming set has {len(protected)} entries but budget={budget}"
        )

    free = budget - len(protected)
    candidates = np.asarray(
        [i for i in range(len(absolute_positions)) if i not in protected],
        dtype=np.int64,
    )
    if free > 0 and len(candidates):
        k = min(free, len(candidates))
        local = np.argpartition(-scores[candidates], k - 1)[:k]
        protected.update(int(v) for v in candidates[local])

    keep = np.asarray(sorted(protected), dtype=np.int64)
    if len(keep) != budget:
        raise RuntimeError(f"streaming selector returned {len(keep)} for budget={budget}")
    return keep
