"""BlockSal: canonical KiaOmni block-wise saliency selector.

Canonical comparison semantics are taken from final_paper_data/033_full_comparison.py:
BLOCK_SIZE=16, sink/recency protection, mean saliency per block, and whole-block
eviction. Because eviction happens by whole blocks, the retained-token count can
land up to block_size-1 tokens below the nominal budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import N_SINK_DEFAULT, RECENCY_DEFAULT


BLOCK_SIZE_DEFAULT = 16


@dataclass(frozen=True)
class BlockSalSelection:
    keep_indices: np.ndarray
    block_size: int
    protected_tokens: int
    requested_budget: int
    actual_kept_tokens: int
    budget_delta: int


def select_blocksal_keep(
    saliency: np.ndarray,
    budget: int,
    L: int,
    *,
    block_size: int = BLOCK_SIZE_DEFAULT,
    n_sink: int = N_SINK_DEFAULT,
    recency: int = RECENCY_DEFAULT,
) -> BlockSalSelection:
    """Select prompt positions using canonical whole-block BlockSal semantics."""
    saliency = np.asarray(saliency, dtype=np.float32).reshape(-1)
    if L < 1 or len(saliency) != L:
        raise ValueError(f"L={L} must match saliency length={len(saliency)}")
    if budget < 1:
        raise ValueError("budget must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    target = min(int(budget), int(L))
    protected_mask = np.zeros(L, dtype=bool)
    protected_mask[: min(n_sink, L)] = True
    protected_mask[max(0, L - recency) :] = True
    protected = set(np.where(protected_mask)[0].tolist())

    if target < len(protected):
        raise ValueError(
            f"budget={target} is smaller than protected token count={len(protected)}"
        )
    if target >= L:
        keep = np.arange(L, dtype=np.int64)
        return BlockSalSelection(
            keep_indices=keep,
            block_size=block_size,
            protected_tokens=len(protected),
            requested_budget=target,
            actual_kept_tokens=L,
            budget_delta=L - target,
        )

    evict_idx = np.where(~protected_mask)[0]
    if evict_idx.size == 0:
        keep = np.arange(L, dtype=np.int64)
        return BlockSalSelection(
            keep_indices=keep,
            block_size=block_size,
            protected_tokens=len(protected),
            requested_budget=target,
            actual_kept_tokens=L,
            budget_delta=L - target,
        )

    block_ids = evict_idx // block_size
    unique_blocks = np.unique(block_ids)
    block_scores = np.asarray(
        [
            float(np.mean(saliency[evict_idx[block_ids == block_id]]))
            for block_id in unique_blocks
        ],
        dtype=np.float32,
    )

    order = np.argsort(block_scores, kind="stable")
    evicted = np.zeros(L, dtype=bool)
    target_evict = max(0, L - target)
    tokens_evicted = 0

    for pos in order:
        if tokens_evicted >= target_evict:
            break
        members = evict_idx[block_ids == unique_blocks[pos]]
        evicted[members] = True
        tokens_evicted += int(members.size)

    keep = np.where(~evicted)[0].astype(np.int64)
    actual = int(len(keep))
    if not (target - (block_size - 1) <= actual <= target):
        raise RuntimeError(
            f"BlockSal whole-block budget invariant failed: kept={actual}, "
            f"budget={target}, block_size={block_size}"
        )

    return BlockSalSelection(
        keep_indices=keep,
        block_size=block_size,
        protected_tokens=len(protected),
        requested_budget=target,
        actual_kept_tokens=actual,
        budget_delta=actual - target,
    )


__all__ = [
    "BLOCK_SIZE_DEFAULT",
    "BlockSalSelection",
    "select_blocksal_keep",
]
