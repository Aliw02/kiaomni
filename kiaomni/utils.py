"""
kiaomni/utils.py
==================
Pure numpy helpers: smoothing kernels and the budget-aware top-K
selector. No torch / no model dependencies — easy to unit-test in
isolation against the published reference numbers.
"""

from __future__ import annotations

import numpy as np

N_SINK_DEFAULT  = 16
RECENCY_DEFAULT = 32


def boxcar(x: np.ndarray, sigma: int) -> np.ndarray:
    """O(N) prefix-sum boxcar smooth.

    Returns x unchanged (cast to float32) when ``sigma <= 0`` so that
    setting sigma=0 cleanly degrades to pointwise selection.
    """
    if sigma <= 0:
        return x.astype(np.float32)
    ps = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
    lo = np.maximum(0, np.arange(len(x)) - sigma)
    hi = np.minimum(len(x), np.arange(len(x)) + sigma + 1)
    return ((ps[hi] - ps[lo]) / (hi - lo)).astype(np.float32)


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth via scipy. Imported lazily so the rest of the
    package works without scipy installed (only needed for the
    kiaomni_gaussian policy)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x.astype(np.float32), sigma=sigma)


def select_keep(
    score: np.ndarray,
    budget: int,
    L: int,
    *,
    n_sink: int = N_SINK_DEFAULT,
    recency: int = RECENCY_DEFAULT,
    extra_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Choose which sequence positions to keep.

    Always protects the first ``n_sink`` tokens (attention sinks) and
    the last ``recency`` tokens. Any additional indices passed via
    ``extra_protect`` are also kept. The remaining budget is filled
    with the highest-scoring unprotected tokens.
    """
    prot: set[int] = set(range(min(n_sink, L))) | set(range(max(0, L - recency), L))
    if extra_protect is not None:
        prot.update(int(i) for i in extra_protect if 0 <= int(i) < L)

    free = max(0, budget - len(prot))
    cands = np.array([i for i in range(L) if i not in prot], dtype=np.int64)
    if free <= 0 or len(cands) == 0:
        return np.array(sorted(prot), dtype=np.int64)

    k = min(free, len(cands))
    top = np.argpartition(-score[cands], k - 1)[:k]
    keep_set = prot | set(cands[top].tolist())
    return np.array(sorted(keep_set), dtype=np.int64)


__all__ = ["boxcar", "gaussian", "select_keep", "N_SINK_DEFAULT", "RECENCY_DEFAULT"]
