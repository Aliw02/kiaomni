"""Isolated KV-cache Gaussian policy for the true-KV experimental line.

This module intentionally does not import or modify kiaomni.utils.gaussian.
It operates on torch tensors and is scoped to post-prefill KV experiments.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def kv_gaussian_smooth(scores: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a 1-D KV-position score vector with a normalized Gaussian kernel."""
    scores = scores.float()
    if sigma <= 0:
        return scores

    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=scores.device)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return F.conv1d(
        scores[None, None, :],
        kernel[None, None, :],
        padding=radius,
    )[0, 0]


def select_kv_positions(
    scores: torch.Tensor,
    budget: int,
    *,
    n_sink: int = 16,
    recency: int = 32,
) -> torch.Tensor:
    """Return sorted KV sequence positions while protecting sinks and recency."""
    length = int(scores.numel())
    if budget <= 0:
        raise ValueError("budget must be positive")
    budget = min(int(budget), length)

    protected = set(range(min(n_sink, length)))
    protected.update(range(max(0, length - recency), length))

    if len(protected) >= budget:
        # If protection exceeds the budget, prefer sinks first and then newest tokens.
        sinks = list(range(min(n_sink, length)))
        newest = list(range(length - 1, -1, -1))
        chosen = []
        seen = set()
        for idx in sinks + newest:
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)
            if len(chosen) == budget:
                break
        return torch.tensor(sorted(chosen), dtype=torch.long, device=scores.device)

    candidates = torch.tensor(
        [idx for idx in range(length) if idx not in protected],
        dtype=torch.long,
        device=scores.device,
    )
    free = budget - len(protected)
    if free > 0 and candidates.numel() > 0:
        top_local = torch.topk(scores.index_select(0, candidates), k=min(free, candidates.numel())).indices
        protected.update(candidates.index_select(0, top_local).tolist())

    return torch.tensor(sorted(protected), dtype=torch.long, device=scores.device)


def kv_norm_proxy_score(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Gate-0 plumbing score only; not a scientific KiaOmni importance estimator."""
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("expected [batch, heads, seq_len, head_dim] tensors")
    if key.shape != value.shape:
        raise ValueError("key/value shapes must match")
    return (
        key.float().pow(2).mean(dim=(0, 1, 3))
        + value.float().pow(2).mean(dim=(0, 1, 3))
    ).sqrt()
