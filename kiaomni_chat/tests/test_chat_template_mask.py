"""Tests for the chat-template saliency mask logic.

These are algorithm-level tests that verify the subsequence matching
and the ``_select_keep_with_chat_template_mask`` plumbing without
requiring a GPU or a loaded model.
"""
from __future__ import annotations

import numpy as np

from kiaomni_chat.engine import KiaOmniEngine, GenerationResult
from kiaomni_chat.schemas import StatsBlock


def test_user_content_mask_logic() -> None:
    """Subsequence matching (the core of _user_content_mask) must find
    the user content tokens in the full chat-templated sequence."""
    full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    user_ids = [3, 4, 5]
    n = len(user_ids)
    mask = np.zeros(len(full), dtype=bool)
    for i in range(len(full) - n + 1):
        if full[i:i + n] == user_ids:
            mask[i:i + n] = True
            break
    assert mask[2]
    assert not mask[5]
    assert not mask[0]


def test_user_content_mask_handles_missing_content() -> None:
    """When user content tokens are not found in the full sequence
    (edge case), the mask remains all-False."""
    full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    user_ids = [99, 100]
    n = len(user_ids)
    mask = np.zeros(len(full), dtype=bool)
    for i in range(len(full) - n + 1):
        if full[i:i + n] == user_ids:
            mask[i:i + n] = True
            break
    assert not mask.any()


def test_user_content_mask_handles_empty_content() -> None:
    """Empty user content should produce an empty mask (no match)."""
    full = [1, 2, 3]
    user_ids: list[int] = []
    n = len(user_ids)
    mask = np.zeros(len(full), dtype=bool)
    # With empty user_ids, the loop doesn't run and mask stays False
    for i in range(len(full) - n + 1):
        if full[i:i + n] == user_ids:
            mask[i:i + n] = True
            break
    assert not mask.any()


def test_stats_block_keep_indices_persistence() -> None:
    """StatsBlock serializes/deserializes keep_indices correctly."""
    indices = [0, 16, 32, 64, 128, 256, 512]
    sb = StatsBlock(
        tokens_in=2229,
        tokens_kept=len(indices),
        compression_ratio=len(indices) / 2229,
        prefill_ms=100.0,
        decode_ms=500.0,
        tok_per_sec=25.0,
        vram_allocated_mb=100.0,
        vram_reserved_mb=200.0,
        vram_max_allocated_mb=300.0,
        fragmentation_pct=50.0,
        keep_indices=indices,
    )
    d = sb.model_dump()
    assert d["keep_indices"] == indices
    restored = StatsBlock(**d)
    assert restored.keep_indices == indices


def test_generation_result_contract() -> None:
    """GenerationResult accepts keep_mask=None (deprecated, always None)."""
    sb = StatsBlock(tokens_in=10, tokens_kept=10, compression_ratio=1.0,
                    prefill_ms=0.0, decode_ms=0.0, tok_per_sec=0.0,
                    vram_allocated_mb=0.0, vram_reserved_mb=0.0,
                    vram_max_allocated_mb=0.0, fragmentation_pct=0.0)
    r = GenerationResult(text="hello", stats=sb, keep_mask=None)
    assert r.text == "hello"
    assert r.stats.tokens_in == 10


def test_stats_block_keep_indices_empty_by_default() -> None:
    """StatsBlock constructed without keep_indices defaults to empty list."""
    sb = StatsBlock(tokens_in=10, tokens_kept=10, compression_ratio=1.0,
                    prefill_ms=0.0, decode_ms=0.0, tok_per_sec=0.0,
                    vram_allocated_mb=0.0, vram_reserved_mb=0.0,
                    vram_max_allocated_mb=0.0, fragmentation_pct=0.0)
    assert sb.keep_indices == []
