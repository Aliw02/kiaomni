from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

KV_GAUSSIAN_SIGMA_DEFAULT = 4.0
KV_N_SINK_DEFAULT = 16
KV_RECENCY_DEFAULT = 32


def gaussian_score(
    saliency: np.ndarray,
    sigma: float = KV_GAUSSIAN_SIGMA_DEFAULT,
) -> np.ndarray:
    """Qwen2.5 true-KV copy of the frozen KiaOmni Gaussian scorer."""
    saliency = np.asarray(saliency, dtype=np.float32)
    if saliency.ndim != 1:
        raise ValueError(f"saliency must be 1-D, got {saliency.shape}")
    return gaussian_filter1d(np.log1p(np.maximum(saliency, 0.0)), sigma=sigma).astype(np.float32)


def select_positions(
    saliency: np.ndarray,
    budget: int,
    *,
    n_sink: int = KV_N_SINK_DEFAULT,
    recency: int = KV_RECENCY_DEFAULT,
    sigma: float = KV_GAUSSIAN_SIGMA_DEFAULT,
) -> np.ndarray:
    """Budget-exact position selection with sink and recency protection."""
    saliency = np.asarray(saliency, dtype=np.float32)
    length = int(saliency.shape[0])
    budget = min(int(budget), length)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget < min(length, n_sink + recency):
        raise ValueError(
            f"budget={budget} is smaller than protected sink+recency="
            f"{min(length, n_sink + recency)}"
        )
    if budget >= length:
        return np.arange(length, dtype=np.int64)

    score = gaussian_score(saliency, sigma=sigma)
    protected = set(range(min(n_sink, length)))
    protected.update(range(max(0, length - recency), length))

    free = max(0, budget - len(protected))
    candidates = np.asarray([i for i in range(length) if i not in protected], dtype=np.int64)
    if free > 0 and len(candidates):
        k = min(free, len(candidates))
        local = np.argpartition(-score[candidates], k - 1)[:k]
        protected.update(int(v) for v in candidates[local])

    keep = np.asarray(sorted(protected), dtype=np.int64)
    if len(keep) != budget:
        raise RuntimeError(f"selector produced {len(keep)} positions for budget={budget}")
    return keep


def global_mask(
    layer_saliencies: np.ndarray,
    budget: int,
    **kwargs,
) -> np.ndarray:
    """One shared mask from mean saliency across layers."""
    layer_saliencies = np.asarray(layer_saliencies, dtype=np.float32)
    if layer_saliencies.ndim != 2:
        raise ValueError(f"expected [layers, tokens], got {layer_saliencies.shape}")
    return select_positions(layer_saliencies.mean(axis=0), budget, **kwargs)


def layerwise_masks(
    layer_saliencies: np.ndarray,
    budget: int,
    **kwargs,
) -> list[np.ndarray]:
    """Independent budget-exact mask for each decoder layer."""
    layer_saliencies = np.asarray(layer_saliencies, dtype=np.float32)
    if layer_saliencies.ndim != 2:
        raise ValueError(f"expected [layers, tokens], got {layer_saliencies.shape}")
    return [select_positions(row, budget, **kwargs) for row in layer_saliencies]
