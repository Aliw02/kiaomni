from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "qwen3_30b_scale_gate.py"
spec = importlib.util.spec_from_file_location("qwen3_scale_gate", RUNNER)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_stage_plan_preserves_4x_8x_16x_pressure():
    final = mod.STAGE_PLANS["final"]
    assert final["retention_ratios"] == [0.25, 0.125, 0.0625]
    assert mod.PRIMARY_RATIO == 0.125


def test_smoke_exercises_real_benchmark_before_final():
    smoke = mod.STAGE_PLANS["smoke"]
    assert smoke["real_cases"] == 2
    assert smoke["real_min_tokens"] == 8192
    assert smoke["real_max_tokens"] == 12288


def test_protected_token_defaults_match_library():
    assert mod.N_SINK == 16
    assert mod.RECENCY == 32
    assert mod._budget_for_ratio(8192, 0.125) == 1024


def test_longbench_v2_scoring_uses_choice_letter():
    case = mod.EvalCase(
        case_id="x",
        source="longbench_v2",
        task="code/repository",
        context="ctx",
        question="q",
        gold=["C"],
        distractors=[],
    )
    assert mod.score_answer(case, "C")["success"] is True
    assert mod.score_answer(case, "The answer is C.")["success"] is True
    assert mod.score_answer(case, "B")["success"] is False


def test_synthetic_scoring_blocks_distractor_hits():
    case = mod.EvalCase(
        case_id="x",
        source="synthetic",
        task="hard_multi",
        context="ctx",
        question="q",
        gold=["111", "222"],
        distractors=["999"],
    )
    assert mod.score_answer(case, "111, 222")["success"] is True
    assert mod.score_answer(case, "111, 222, 999")["success"] is False


def test_recency_and_random_keep_exact_budget():
    length = 8192
    budget = 1024
    recency = mod._recency_keep(length, budget)
    random_keep = mod._random_keep(length, budget, seed=42)
    assert len(recency) == budget
    assert len(random_keep) == budget
    assert recency[0] == 0
    assert random_keep[0] == 0
