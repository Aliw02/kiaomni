from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


STREAM_KV_GAUSSIAN_SIGMA_DEFAULT = 4.0
STREAM_KV_N_SINK_DEFAULT = 16
STREAM_KV_RECENCY_DEFAULT = 32


def streaming_gaussian_scores(
    saliency: np.ndarray,
    absolute_positions: np.ndarray,
    seen_length: int,
    *,
    sigma: float = STREAM_KV_GAUSSIAN_SIGMA_DEFAULT,
) -> np.ndarray:
    """Score surviving entries on the original absolute token axis.

    After streaming eviction, adjacent physical cache entries may be far apart
    in the source prompt. Smoothing over physical indices would falsely treat
    them as neighbors. Project to the absolute axis first, smooth there, then
    gather scores back to the current cache entries.
    """
    saliency = np.asarray(saliency, dtype=np.float32)
    absolute_positions = np.asarray(absolute_positions, dtype=np.int64)
    if saliency.ndim != 1 or absolute_positions.ndim != 1:
        raise ValueError("saliency and absolute_positions must both be 1-D")
    if len(saliency) != len(absolute_positions):
        raise ValueError("saliency/position length mismatch")

    dense = np.zeros(int(seen_length), dtype=np.float32)
    dense[absolute_positions] = np.maximum(saliency, 0.0)
    smoothed = gaussian_filter1d(np.log1p(dense), sigma=sigma)
    return smoothed[absolute_positions].astype(np.float32)


def select_streaming_positions(
    saliency: np.ndarray,
    absolute_positions: np.ndarray,
    budget: int,
    seen_length: int,
    *,
    n_sink: int = STREAM_KV_N_SINK_DEFAULT,
    recency: int = STREAM_KV_RECENCY_DEFAULT,
    sigma: float = STREAM_KV_GAUSSIAN_SIGMA_DEFAULT,
) -> np.ndarray:
    """Return local physical-cache indices to retain at this eviction event."""
    absolute_positions = np.asarray(absolute_positions, dtype=np.int64)
    budget = min(int(budget), len(absolute_positions))

    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget >= len(absolute_positions):
        return np.arange(len(absolute_positions), dtype=np.int64)

    scores = streaming_gaussian_scores(
        saliency,
        absolute_positions,
        seen_length,
        sigma=sigma,
    )

    protected = {
        idx
        for idx, absolute_pos in enumerate(absolute_positions)
        if absolute_pos < n_sink
        or absolute_pos >= max(0, int(seen_length) - recency)
    }
    if len(protected) > budget:
        raise ValueError(
            f"protected streaming entries={len(protected)} exceed budget={budget}"
        )

    free = budget - len(protected)
    candidates = np.asarray(
        [idx for idx in range(len(absolute_positions)) if idx not in protected],
        dtype=np.int64,
    )
    if free > 0 and len(candidates):
        k = min(free, len(candidates))
        top = np.argpartition(-scores[candidates], k - 1)[:k]
        protected.update(int(v) for v in candidates[top])

    keep = np.asarray(sorted(protected), dtype=np.int64)
    if len(keep) != budget:
        raise RuntimeError(
            f"streaming selector produced {len(keep)} entries for budget={budget}"
        )
    return keep
