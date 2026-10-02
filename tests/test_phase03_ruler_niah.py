from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "qwen3_30b_ruler_niah.py"
LAUNCHER = ROOT / "modal" / "qwen3_30b_ruler_niah_modal.py"
PROTOCOL = ROOT / "results" / "kiaomni_moe_model_lab" / "phase_03_ruler_niah_v1" / "PROTOCOL_FREEZE.json"


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


def test_ruler_suite_is_frozen_before_gpu_results():
    data = json.loads(_source(PROTOCOL))
    assert data["ruler"]["official_generation_commit"] == "38da79d79519ef87aa46ae804f838e1eab7f86d7"
    assert data["ruler"]["context_lengths"] == [8192, 16384]
    assert data["ruler"]["samples_per_task_length"] == 5
    assert data["ruler"]["expected_cases"] == 40
    assert data["invariants"]["percentage_budgets"] == [0.25, 0.125, 0.0625]
    assert data["invariants"]["no_post_result_case_selection"] is True


def test_required_ruler_tasks_are_controlled_niah_tasks():
    mod = _load_runner()
    assert mod.RULER_TASKS == (
        "niah_single_2",
        "niah_multikey_1",
        "niah_multivalue",
        "niah_multiquery",
    )
    assert mod.RULER_LENGTHS == (8192, 16384)
    assert mod.SAMPLES_PER_GROUP == 5


def test_ruler_scoring_matches_all_reference_fraction_semantics():
    mod = _load_runner()
    out = mod.score_ruler("values are 111 and 222", ["111", "222"])
    assert out["correct"] is True
    assert out["ruler_string_match_fraction"] == 1.0
    out = mod.score_ruler("only 111 is here", ["111", "222"])
    assert out["correct"] is False
    assert out["ruler_string_match_fraction"] == 0.5


def test_survival_metric_separates_partial_and_complete_answers():
    import numpy as np
    mod = _load_runner()
    spans = [[10, 11], [20, 21]]
    out = mod.survival_metrics(spans, np.array([10, 11, 20]), 100)
    assert out["required_answer_token_recall"] == 0.75
    assert out["complete_required_answers"] == 1
    assert out["required_answers_total"] == 2
    assert out["all_required_answers_complete"] is False


def test_ruler_records_mechanism_routing_and_system_metrics():
    src = _source(RUNNER)
    for metric in (
        "required_answer_token_recall",
        "complete_required_answer_rate",
        "all_required_answers_complete",
        "ruler_string_match_pct",
        "gold_answer_ppl",
        "actual_retention_pct",
        "compression_ratio",
        "top1_expert_agreement",
        "top8_set_jaccard",
        "worst_layer_top8_set_jaccard",
        "time_to_first_token_seconds",
        "decode_after_first_token_seconds",
        "generation_peak_allocated_vram_gb",
        "pipeline_peak_allocated_vram_gb",
    ):
        assert metric in src


def test_modal_pins_source_integrity_and_depth_selection():
    src = _source(LAUNCHER)
    assert 'RULER_DATASET_ID = "VenusChenyy/RULER_50"' in src
    assert 'RULER_OFFICIAL_GENERATION_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"' in src
    assert "source_official_generation_manifest.json" in src
    assert 'RULER_MIRROR_ROWS_PER_GROUP = 50' in src
    assert 'filename = f"{task}/{length}.jsonl"' in src
    assert '"relative_path": filename' in src
    assert "relative_to(root)" not in src
    assert '"mirror_sha256": observed_sha' in src
    assert '"official_source_sha256": str(source_entry.get("sha256", ""))' in src
    assert "_select_depth_stratified" in src
    assert "10.0, 30.0, 50.0, 70.0, 90.0" in src
