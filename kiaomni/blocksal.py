"""BlockSal: KiaOmni block-wise saliency selector.

This module preserves the KiaOmni BlockSal idea: group evictable prompt
positions into fixed-size blocks, rank blocks by mean saliency, and retain
the most important regions while protecting sink and recency positions.

For fair head-to-head benchmarks, one final boundary block may be partially
retained so the requested token budget is matched exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import N_SINK_DEFAULT, RECENCY_DEFAULT


BLOCK_SIZE_DEFAULT = 8


@dataclass(frozen=True)
class BlockSalSelection:
    keep_indices: np.ndarray
    block_size: int
    partial_boundary_block: bool
    protected_tokens: int


def select_blocksal_keep(
    saliency: np.ndarray,
    budget: int,
    L: int,
    *,
    block_size: int = BLOCK_SIZE_DEFAULT,
    n_sink: int = N_SINK_DEFAULT,
    recency: int = RECENCY_DEFAULT,
) -> BlockSalSelection:
    """Select exactly ``budget`` prompt positions using BlockSal ranking."""
    saliency = np.asarray(saliency, dtype=np.float32).reshape(-1)
    if L < 1 or len(saliency) != L:
        raise ValueError(f"L={L} must match saliency length={len(saliency)}")
    if budget < 1:
        raise ValueError("budget must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    target = min(int(budget), int(L))
    protected = set(range(min(n_sink, L)))
    protected.update(range(max(0, L - recency), L))

    if target < len(protected):
        raise ValueError(
            f"budget={target} is smaller than protected token count={len(protected)}"
        )
    if target == L:
        return BlockSalSelection(
            keep_indices=np.arange(L, dtype=np.int64),
            block_size=block_size,
            partial_boundary_block=False,
            protected_tokens=len(protected),
        )

    free = np.asarray([i for i in range(L) if i not in protected], dtype=np.int64)
    remaining = target - len(protected)
    block_ids = free // block_size
    unique_blocks = np.unique(block_ids)
    block_scores = np.asarray(
        [
            float(np.mean(saliency[free[block_ids == block_id]]))
            for block_id in unique_blocks
        ],
        dtype=np.float32,
    )
    order = np.argsort(-block_scores, kind="stable")

    keep = set(protected)
    partial = False
    for pos in order:
        if remaining <= 0:
            break
        members = free[block_ids == unique_blocks[pos]]
        if len(members) <= remaining:
            keep.update(int(i) for i in members)
            remaining -= len(members)
            continue

        member_scores = saliency[members]
        top = np.argsort(-member_scores, kind="stable")[:remaining]
        keep.update(int(i) for i in members[top])
        remaining = 0
        partial = True

    keep_indices = np.asarray(sorted(keep), dtype=np.int64)
    if len(keep_indices) != target:
        raise RuntimeError(
            f"BlockSal exact-budget invariant failed: kept={len(keep_indices)} target={target}"
        )

    return BlockSalSelection(
        keep_indices=keep_indices,
        block_size=block_size,
        partial_boundary_block=partial,
        protected_tokens=len(protected),
    )


__all__ = [
    "BLOCK_SIZE_DEFAULT",
    "BlockSalSelection",
    "select_blocksal_keep",
]
