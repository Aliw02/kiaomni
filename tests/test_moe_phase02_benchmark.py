from __future__ import annotations

import numpy as np

from experiments.kaggle_moe_phase02_multineedle_baselines import (
    BLOCK_SIZE,
    N_SINK,
    RECENCY,
    Case,
    blocksal_keep,
    ratio_for_budget,
    score_answer,
    validate_blocksal,
)


def test_ratio_for_budget_is_exact_for_phase02_grid():
    prompt_len = 3900
    for budget in (98, 128, 256, 512):
        ratio = ratio_for_budget(prompt_len, budget)
        assert int(prompt_len * (1.0 - ratio)) == budget


def test_blocksal_preserves_protected_tokens_and_whole_block_budget():
    seq_len = 3900
    budget = 98
    saliency = np.random.RandomState(7).rand(seq_len).astype(np.float32)

    keep = blocksal_keep(saliency, budget, seq_len)
    kept = set(keep.tolist())

    protected = set(range(N_SINK)) | set(range(seq_len - RECENCY, seq_len))
    assert protected.issubset(kept)
    assert budget - (BLOCK_SIZE - 1) <= len(keep) <= budget


def test_blocksal_whole_block_budget_holds_on_non_aligned_sequence():
    seq_len = 3911
    budget = 128
    saliency = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

    keep = blocksal_keep(saliency, budget, seq_len)

    assert budget - (BLOCK_SIZE - 1) <= len(keep) <= budget


def test_hard_multi_rejects_distractor_even_when_all_gold_values_are_present():
    case = Case(
        task="hard_multi",
        sample_id=0,
        context="",
        question="",
        gold=["123456", "321", "ONYX"],
        distractors=["654321", "777", "NOVA"],
        info="test",
    )

    clean = score_answer(case, "123456 321 ONYX")
    contaminated = score_answer(case, "123456 321 ONYX and NOVA")

    assert clean["exact"] is True
    assert clean["recall"] == 1.0
    assert contaminated["recall"] == 1.0
    assert contaminated["exact"] is False
    assert contaminated["distractor_hits"] == ["NOVA"]


def test_blocksal_canonical_block_size_is_16():
    assert BLOCK_SIZE == 16


def test_blocksal_validation_covers_full_phase02_budget_grid():
    budgets = [512, 256, 128, 98]
    result = validate_blocksal(budgets)

    assert result.valid is True
    assert set(result.details["budgets"]) == {str(b) for b in budgets}
    for budget in budgets:
        entry = result.details["budgets"][str(budget)]
        assert entry["protected_tokens_present"] is True
        assert entry["historical_whole_block_budget_ok"] is True
        assert budget - (BLOCK_SIZE - 1) <= entry["actual_kept_tokens"] <= budget
