from __future__ import annotations

import numpy as np
import pytest

from experiments.kaggle_moe_phase02_multineedle_baselines import (
    BLOCK_SIZE,
    N_SINK,
    RECENCY,
    Case,
    aggregate,
    blocksal_keep,
    paired_routing_vs_fullcontext,
    ratio_for_budget,
    score_answer,
    select_random_retention,
    select_recency_only,
    validate_blocksal,
    validate_subset_controls,
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


def test_aggregate_reports_full_context_conditioned_quality():
    rows = [
        {
            "score": {"exact": True, "recall": 1.0},
            "tokens_per_s": 2.0,
            "peak_vram_gb": 5.0,
            "full_context_exact": True,
        },
        {
            "score": {"exact": False, "recall": 0.0},
            "tokens_per_s": 2.0,
            "peak_vram_gb": 5.0,
            "full_context_exact": False,
        },
    ]

    metrics = aggregate(rows)

    assert metrics["exact_accuracy"] == 0.5
    assert metrics["full_context_eligible_n"] == 1
    assert metrics["full_context_eligibility_rate"] == 0.5
    assert metrics["conditional_exact_accuracy"] == 1.0
    assert metrics["conditional_mean_recall"] == 1.0



def test_recency_and_random_controls_match_exact_budget():
    seq_len = 3900
    for budget in (512, 256, 128, 98):
        recency = select_recency_only(seq_len, budget)
        random_keep = select_random_retention(
            seq_len,
            budget,
            seed=1234 + budget,
        )

        assert len(recency) == budget
        assert len(random_keep) == budget
        assert len(set(recency.tolist())) == budget
        assert len(set(random_keep.tolist())) == budget


def test_random_retention_is_deterministic_for_same_seed():
    a = select_random_retention(1000, 128, seed=77)
    b = select_random_retention(1000, 128, seed=77)
    np.testing.assert_array_equal(a, b)


def test_subset_control_validation_covers_grid():
    results = validate_subset_controls([512, 256, 128, 98])
    assert results["recency_only"]["valid"] is True
    assert results["random_retention"]["valid"] is True


def test_paired_routing_delta_signs_are_explicit():
    rows = [
        {
            "task": "single",
            "sample_id": 0,
            "method": "full_context",
            "budget": None,
            "routing": {
                "raw_top1_transition_rate": 0.50,
                "raw_topk_jaccard": 0.40,
                "stable_top1_transition_rate": 0.45,
            },
        },
        {
            "task": "single",
            "sample_id": 0,
            "method": "kiaomni_s8",
            "budget": 98,
            "routing": {
                "raw_top1_transition_rate": 0.30,
                "raw_topk_jaccard": 0.60,
                "stable_top1_transition_rate": 0.25,
            },
        },
    ]

    result = paired_routing_vs_fullcontext(rows)["kiaomni_s8_b98"]
    assert result["paired_n"] == 1
    assert result["raw_jitter_delta_vs_fullcontext"] == pytest.approx(-0.2)
    assert result["raw_continuity_delta_vs_fullcontext"] == pytest.approx(0.2)
    assert result["raw_topk_jaccard_delta_vs_fullcontext"] == pytest.approx(0.2)



def test_wilson_and_mcnemar_statistics_are_bounded():
    from experiments.kaggle_moe_phase02_multineedle_baselines import (
        mcnemar_exact_p,
        wilson_interval,
    )

    ci = wilson_interval(75, 100)
    assert ci is not None
    assert 0.0 <= ci["low"] <= 0.75 <= ci["high"] <= 1.0

    p = mcnemar_exact_p(8, 2)
    assert 0.0 <= p <= 1.0
