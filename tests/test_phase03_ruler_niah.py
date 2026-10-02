from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "qwen3_30b_ruler_niah.py"
LAUNCHER = ROOT / "modal" / "qwen3_30b_ruler_niah_modal.py"
PROTOCOL = (
    ROOT
    / "results"
    / "kiaomni_moe_model_lab"
    / "phase_03_ruler_niah_v1"
    / "PROTOCOL_FREEZE.json"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("ruler_niah_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_files_parse():
    ast.parse(_source(RUNNER))
    ast.parse(_source(LAUNCHER))


def test_suite_is_frozen():
    mod = _load_runner()
    assert mod.RULER_TASKS == (
        "niah_single_2",
        "niah_multikey_1",
        "niah_multivalue",
        "niah_multiquery",
    )
    assert mod.RULER_LENGTHS == (8192, 16384)
    assert mod.SAMPLES_PER_GROUP == 5
    assert mod.PERCENTAGE_BUDGETS == (0.25, 0.125, 0.0625)
    assert mod.MAX_NEW_TOKENS == 64


def test_protocol_records_official_generation_provenance():
    data = json.loads(_source(PROTOCOL))
    assert data["data"]["official_repo"] == "NVIDIA/RULER"
    assert data["data"]["official_generation_commit"] == (
        "38da79d79519ef87aa46ae804f838e1eab7f86d7"
    )
    assert data["data"]["sha256_verification_required"] is True
    assert data["suite"]["total_cases"] == 40
    assert data["suite"]["no_post_result_reselection"] is True


def test_ruler_scoring_matches_string_match_all_semantics():
    mod = _load_runner()
    full = mod.score_ruler("The values are 1234567 and 7654321.", ["1234567", "7654321"])
    half = mod.score_ruler("Only 1234567 is present.", ["1234567", "7654321"])
    none = mod.score_ruler("No target values.", ["1234567", "7654321"])
    assert full["ruler_string_match_pct"] == 100.0
    assert full["correct"] is True
    assert half["ruler_string_match_pct"] == 50.0
    assert half["correct"] is False
    assert none["ruler_string_match_pct"] == 0.0


def test_survival_metrics_distinguish_partial_and_complete_evidence():
    mod = _load_runner()
    spans = [[2, 3], [8, 9]]
    keep = np.array([0, 1, 2, 3, 8], dtype=np.int64)
    out = mod.survival_metrics(spans, keep, full_prompt_len=10)
    assert out["complete_required_answers"] == 1
    assert out["required_answers_total"] == 2
    assert out["complete_required_answer_rate"] == 0.5
    assert out["required_answer_token_recall"] == 0.75
    assert out["all_required_answers_complete"] is False


def test_launcher_freezes_mirror_sha_checks_and_depth_strata():
    src = _source(LAUNCHER)
    assert 'RULER_DATASET_ID = "VenusChenyy/RULER_50"' in src
    assert "source_official_generation_manifest.json" in src
    assert "observed_sha != expected_sha" in src
    assert "_select_depth_stratified" in src
    for label in ("0-20", "20-40", "40-60", "60-80", "80-100"):
        assert label in src
    assert "answer_prefix_policy" in src


def test_ruler_runner_records_new_paper_metrics():
    src = _source(RUNNER)
    for metric in (
        "ruler_string_match_pct",
        "required_answer_token_recall",
        "complete_required_answer_rate",
        "all_required_answers_complete",
        "actual_retention_pct",
        "compression_ratio",
        "time_to_first_token_seconds",
        "decode_after_first_token_seconds",
        "top1_expert_agreement",
        "top8_set_jaccard",
        "worst_layer_top8_set_jaccard",
        "generation_peak_allocated_vram_gb",
        "pipeline_peak_allocated_vram_gb",
    ):
        assert metric in src


def test_preflight_is_small_and_final_is_full_suite():
    mod = _load_runner()
    assert mod.STAGES["preflight"]["max_cases"] == 1
    assert mod.STAGES["preflight"]["ratios"] == (0.25,)
    assert mod.STAGES["final"]["max_cases"] is None
    assert mod.STAGES["final"]["ratios"] == (0.25, 0.125, 0.0625)
