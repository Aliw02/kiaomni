import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "kaggle_mobilemoe_phase02_baselines.py"
)
SPEC = importlib.util.spec_from_file_location("phase02_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
phase02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase02)


def test_ratio_for_budget_is_exact_for_phase02_budgets():
    for prompt_len in (700, 3900, 4096):
        for budget in (98, 128, 256, 512):
            ratio = phase02.ratio_for_budget(prompt_len, budget)
            kept = max(1, int(prompt_len * (1.0 - ratio)))
            assert kept == budget


def test_score_answer_multi_requires_all_targets():
    case = {
        "expected_values": ["2718", "314", "FALCON"],
        "forbidden_values": [],
    }
    full = phase02.score_answer(case, "2718, 314, FALCON")
    partial = phase02.score_answer(case, "2718 and FALCON")

    assert full["score"] == 1.0
    assert full["exact_pass"] is True
    assert partial["score"] == 2 / 3
    assert partial["exact_pass"] is False


def test_score_answer_rejects_forbidden_distractor():
    case = {
        "expected_values": ["2718"],
        "forbidden_values": ["9999"],
    }
    scored = phase02.score_answer(case, "The answer is 2718, not 9999.")

    assert scored["score"] == 1.0
    assert scored["exact_pass"] is False
    assert scored["forbidden_hits"] == ["9999"]


def test_phase02_method_set_excludes_adasnapkv_and_classifies_blocksal_as_ours():
    assert "AdaSnapKV" not in phase02.METHODS
    meta = phase02.method_metadata("0.5.5")
    assert meta["BlockSal"]["owner"] == "ours"
    assert meta["SnapKV"]["owner"] == "external"
    assert meta["StreamingLLM"]["owner"] == "external"
