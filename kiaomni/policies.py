"""
kiaomni/policies.py
=====================
Pluggable scoring functions that turn a saliency vector into a per-token
score for KV eviction selection.

Built-in policies are paper-grade winners on KiaOmni benchmark suite:

    "kiaomni_s8"        Boxcar smoothing (sigma=8) on log-saliency
                        — overall production winner across 32 experiments.
    "kiaomni_gaussian"  Gaussian smoothing (sigma=4) on log-saliency
                        — best on VectorTrace tasks.

External researchers can register their own policies via
``register_policy(name, fn)``.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .utils import boxcar, gaussian

ScoreFn = Callable[[np.ndarray], np.ndarray]


# Sigma defaults that reproduce the published benchmark numbers exactly.
SIGMA_BOXCAR_DEFAULT   = 8
SIGMA_GAUSSIAN_DEFAULT = 4.0


def _kiaomni_s8(sal: np.ndarray) -> np.ndarray:
    return boxcar(np.log1p(sal), sigma=SIGMA_BOXCAR_DEFAULT)


def _kiaomni_gaussian(sal: np.ndarray) -> np.ndarray:
    return gaussian(np.log1p(sal), sigma=SIGMA_GAUSSIAN_DEFAULT)


POLICY_REGISTRY: Dict[str, ScoreFn] = {
    "kiaomni_s8":       _kiaomni_s8,
    "kiaomni_gaussian": _kiaomni_gaussian,
}


def register_policy(name: str, fn: ScoreFn) -> None:
    """Register a new scoring policy.

    The function must accept a 1-D numpy array of raw saliencies and
    return a 1-D numpy array of scores of the same length. Higher
    scores mean "more important to keep".
    """
    if not callable(fn):
        raise TypeError(f"policy fn must be callable, got {type(fn).__name__}")
    POLICY_REGISTRY[name] = fn


def get_policy(name: str) -> ScoreFn:
    if name not in POLICY_REGISTRY:
        raise KeyError(
            f"Unknown policy {name!r}. Available: {sorted(POLICY_REGISTRY)}"
        )
    return POLICY_REGISTRY[name]


__all__ = [
    "POLICY_REGISTRY",
    "ScoreFn",
    "register_policy",
    "get_policy",
    "SIGMA_BOXCAR_DEFAULT",
    "SIGMA_GAUSSIAN_DEFAULT",
]
