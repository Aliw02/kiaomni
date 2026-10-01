from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "qwen3_30b_adjudication_routing.py"
LAUNCHER = ROOT / "modal" / "qwen3_30b_adjudication_modal.py"
PROTOCOL = (
    ROOT
    / "results"
    / "kiaomni_moe_model_lab"
    / "phase_03_adjudication_routing_v1"
    / "PROTOCOL_FREEZE.json"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("adjudication_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_launcher():
    spec = importlib.util.spec_from_file_location("adjudication_modal_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_files_parse():
    ast.parse(_source(RUNNER))
    ast.parse(_source(LAUNCHER))


def test_fixed_budgets_and_policy_names_are_frozen():
    mod = _load_runner()
    assert mod.FIXED_BUDGETS == (512, 256, 128, 98)
    assert mod.POLICIES == ("kiaomni_s8", "kiaomni_gaussian")
    assert mod.N_SINK == 16
    assert mod.RECENCY == 32
    assert mod.MAX_NEW_TOKENS == 256


def test_protocol_does_not_replace_fixed_budgets():
    data = json.loads(_source(PROTOCOL))
    assert data["invariants"]["fixed_budgets"] == [512, 256, 128, 98]
    assert data["invariants"]["policies"] == ["kiaomni_s8", "kiaomni_gaussian"]
    assert data["invariants"]["ratios_do_not_replace_fixed_budgets"] is True
    assert data["invariants"]["no_policy_renaming"] is True
    assert data["routing"]["shadow_or_projected"] is False
    assert data["dataset"]["expected_real_cases"] == 27


def test_actual_routing_is_verified_at_experts_input():
    src = _source(RUNNER)
    assert "register_forward_hook" in src
    assert "register_forward_pre_hook" in src
    assert "Actual expert-dispatch count verification failed" in src
    assert "torch.bincount" in src
    assert "torch.equal(expected, observed)" in src
    assert "actual_dispatch_verified" in src
    assert "shadow" not in src.lower()


def test_required_metrics_are_recorded():
    src = _source(RUNNER)
    for metric in (
        "gold_answer_ppl",
        "gold_answer_nll",
        "generation_peak_allocated_vram_gb",
        "pipeline_peak_allocated_vram_gb",
        "output_tokens_per_second",
        "top1_expert_agreement",
        "top8_set_jaccard",
        "dispatch_weight_cosine",
        "expert_load_jsd",
        "hit_token_limit",
    ):
        assert metric in src


def test_mc_parser_allows_explanatory_text():
    mod = _load_runner()
    assert mod.parse_mc_answer("The correct answer is (B).") == ("B", "explicit")
    assert mod.parse_mc_answer("After checking, the correct answer is C because of the manual.") == (
        "C",
        "explicit",
    )
    assert mod.parse_mc_answer("D") == ("D", "bare")
    assert mod.parse_mc_answer("I am unsure between A and B") == (None, "unparsed")


def test_pairwise_summary_tracks_rescues_and_regressions():
    mod = _load_runner()
    rows = [
        {
            "case_id": "a",
            "source": "longbench_v2",
            "result": {"method": "full_context", "correct": True, "parsed_answer": "A", "hit_token_limit": False},
        },
        {
            "case_id": "a",
            "source": "longbench_v2",
            "result": {"method": "kiaomni_s8_B512", "correct": False, "parsed_answer": "B", "hit_token_limit": False},
        },
        {
            "case_id": "b",
            "source": "longbench_v2",
            "result": {"method": "full_context", "correct": False, "parsed_answer": "C", "hit_token_limit": False},
        },
        {
            "case_id": "b",
            "source": "longbench_v2",
            "result": {"method": "kiaomni_s8_B512", "correct": True, "parsed_answer": "D", "hit_token_limit": False},
        },
    ]
    out = mod.pairwise_summary(rows)[0]
    assert out["fc_correct_to_kia_wrong"] == 1
    assert out["fc_wrong_to_kia_correct"] == 1


def test_modal_launcher_imports_with_pinned_sdk():
    mod = _load_launcher()
    assert mod.RESULTS_VOLUME_NAME == "kiaomni-qwen3-adjudication-results"
    assert mod.STAGE_TIMEOUTS["preflight"] == 20 * 60
    assert mod.STAGE_TIMEOUTS["final"] == 110 * 60


def test_launcher_reuses_assets_without_deleting_or_redownloading():
    src = _source(LAUNCHER)
    assert 'ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"' in src
    assert "snapshot_download" not in src
    assert "Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=False)" in src
    assert "prepare_adjudication_index.remote()" in src


def test_routing_similarity_is_exact_for_identical_actual_dispatch():
    import numpy as np
    import torch

    mod = _load_runner()
    indices = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=torch.uint8,
    )
    weights = torch.tensor(
        [[0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04],
         [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]],
        dtype=torch.float16,
    )
    routes = {0: {"indices": indices, "weights": weights}}
    out = mod.compare_routes(
        routes,
        routes,
        np.array([0, 1], dtype=np.int64),
        full_prompt_len=2,
        compressed_prompt_len=2,
        num_experts=128,
    )
    assert out["top1_expert_agreement"] == 1.0
    assert out["top8_set_jaccard"] == 1.0
    assert abs(out["dispatch_weight_cosine"] - 1.0) < 1e-6
    assert abs(out["expert_load_jsd"]) < 1e-8
