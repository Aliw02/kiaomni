from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "qwen3_30b_scale_gate.py"
spec = importlib.util.spec_from_file_location("qwen3_scale_gate", RUNNER)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_stage_plans_are_budget_valid():
    for cfg in mod.STAGE_PLANS.values():
        mod.validate_stage_config(cfg)


def test_stage_plan_preserves_compression_pressure():
    final = mod.STAGE_PLANS["final"]
    target = final["target_tokens"]
    ratios = [target / budget for budget in final["budgets"]]
    assert ratios == [4.0, 8.0, 16.0]


def test_rejects_budget_too_small_for_protected_tokens():
    cfg = dict(mod.STAGE_PLANS["preflight"])
    cfg["budgets"] = [mod.N_SINK + mod.RECENCY - 1]
    with pytest.raises(ValueError, match=r"n_sink\+recency"):
        mod.validate_stage_config(cfg)


def test_longbench_scoring_accepts_concise_gold_answer():
    case = mod.EvalCase(
        case_id="x",
        source="longbench",
        task="qasper",
        context="ctx",
        question="q",
        gold=["New York City", "NYC"],
        distractors=[],
    )
    assert mod.score_answer(case, "NYC")["success"] is True
    assert mod.score_answer(case, "Boston")["success"] is False


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
