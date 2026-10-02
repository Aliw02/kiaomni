from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "qwen3_30b_percentage_replay.py"
LAUNCHER = ROOT / "modal" / "qwen3_30b_percentage_replay_modal.py"
PROTOCOL = ROOT / "results" / "kiaomni_moe_model_lab" / "phase_03_percentage_replay_v1" / "PROTOCOL_FREEZE.json"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("percentage_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_files_parse():
    ast.parse(_source(RUNNER))
    ast.parse(_source(LAUNCHER))


def test_percentage_track_is_frozen_and_legacy_fixed_track_is_preserved_as_reference():
    mod = _load_runner()
    assert mod.PERCENTAGE_BUDGETS == (0.25, 0.125, 0.0625)
    assert mod.LEGACY_FIXED_BUDGETS == (512, 256, 128, 98)
    assert mod.POLICIES == ("kiaomni_s8", "kiaomni_gaussian")
    assert mod.N_SINK == 16
    assert mod.RECENCY == 32
    assert mod.MAX_NEW_TOKENS == 256
    assert mod.STAGES["final"]["real_cases"] == 27
    assert mod.STAGES["final"]["reason_cases"] == 0


def test_ratio_budget_matches_historical_semantics():
    mod = _load_runner()
    assert mod.budget_for_ratio(10000, 0.25) == 2500
    assert mod.budget_for_ratio(10000, 0.125) == 1250
    assert mod.budget_for_ratio(10000, 0.0625) == 625


def test_protocol_freezes_same_27_cases_and_no_reselection():
    data = json.loads(_source(PROTOCOL))
    assert data["invariants"]["exact_same_27_longbench_ids"] is True
    assert data["invariants"]["no_reselection"] is True
    assert data["invariants"]["ratios_do_not_replace_fixed_budgets"] is True
    assert data["dataset"]["expected_real_cases"] == 27
    assert "time_to_first_token_seconds" in data["required_metrics"]["systems"]
    assert "decode_after_first_token_seconds" in data["required_metrics"]["systems"]


def test_paper_company_metrics_are_recorded():
    src = _source(RUNNER)
    for metric in (
        "actual_retention_pct",
        "compression_ratio",
        "gold_answer_ppl",
        "accuracy_ci95_low",
        "accuracy_ci95_high",
        "fc_correct_preservation_rate",
        "regression_rate_on_fc_correct",
        "rescue_rate_on_fc_wrong",
        "mcnemar_exact_p",
        "top1_expert_agreement",
        "top8_set_jaccard",
        "worst_layer_top1_expert_agreement",
        "worst_layer_top8_set_jaccard",
        "generation_peak_allocated_vram_gb",
        "pipeline_peak_allocated_vram_gb",
        "time_to_first_token_seconds",
        "decode_after_first_token_seconds",
        "inference_path_elapsed_seconds",
        "output_tokens_per_second",
    ):
        assert metric in src


def test_final_uses_only_percentage_ratios_not_fixed_budgets():
    mod = _load_runner()
    assert mod.STAGES["final"]["ratios"] == (0.25, 0.125, 0.0625)
    src = _source(RUNNER)
    assert 'cfg["budgets"]' not in src
    assert "select_budget(" not in src


def test_modal_reuses_frozen_assets_and_writes_new_results_volume():
    src = _source(LAUNCHER)
    assert 'ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"' in src
    assert 'RESULTS_VOLUME_NAME = "kiaomni-qwen3-frontier-results"' in src
    assert "snapshot_download" not in src
    assert "qwen3_30b_percentage_replay.py" in src


def test_exact_mcnemar_is_symmetric():
    mod = _load_runner()
    assert mod._exact_mcnemar_p(5, 6) == mod._exact_mcnemar_p(6, 5)
    assert mod._exact_mcnemar_p(0, 0) == 1.0
