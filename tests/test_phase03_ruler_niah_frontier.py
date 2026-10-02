from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "qwen3_30b_ruler_niah_frontier.py"
LAUNCHER = ROOT / "modal" / "qwen3_30b_ruler_niah_modal.py"
PROTOCOL = ROOT / "results" / "kiaomni_moe_model_lab" / "phase_03_ruler_niah_frontier_v1" / "PROTOCOL_FREEZE.json"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("ruler_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_files_parse():
    ast.parse(_source(RUNNER))
    ast.parse(_source(LAUNCHER))


def test_ruler_revision_and_fast_suite_are_frozen():
    mod = _load_runner()
    assert mod.RULER_REVISION == "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
    assert mod.RULER_TASKS == ("niah_single_1", "niah_multikey_2", "niah_multikey_3")
    assert mod.RULER_LENGTHS == (8192, 16384)
    assert mod.STAGES["final"]["samples_per_task"] == 4
    assert mod.STAGES["final"]["ratios"] == (0.25, 0.125, 0.0625)


def test_protocol_uses_official_pinned_ruler_generator():
    data = json.loads(_source(PROTOCOL))
    assert data["ruler"]["repo"] == "NVIDIA/RULER"
    assert data["ruler"]["source"] == "scripts/data/prepare.py"
    assert data["ruler"]["tokenizer_type"] == "hf"
    assert data["ruler"]["random_seed"] == 42


def test_needle_survival_and_actual_routing_metrics_are_recorded():
    src = _source(RUNNER)
    for metric in (
        "ruler_score",
        "gold_token_recall",
        "complete_reference_survival",
        "all_references_survived",
        "needle_depth_pct",
        "top1_expert_agreement",
        "top8_set_jaccard",
        "dispatch_weight_cosine",
        "expert_load_jsd",
        "gold_answer_ppl",
        "generation_peak_allocated_vram_gb",
        "pipeline_peak_allocated_vram_gb",
    ):
        assert metric in src


def test_ruler_string_match_all_matches_upstream_semantics_per_sample():
    mod = _load_runner()
    score, matched = mod.ruler_string_match_all(
        "Values are 123 and ABC.",
        ["123", "abc", "missing"],
    )
    assert matched == 2
    assert abs(score - (2 / 3)) < 1e-12


def test_launcher_clones_exact_ruler_revision_and_reuses_qwen_assets():
    src = _source(LAUNCHER)
    assert 'ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"' in src
    assert 'RESULTS_VOLUME_NAME = "kiaomni-qwen3-frontier-results"' in src
    assert 'https://github.com/NVIDIA/RULER.git' in src
    assert 'git", "checkout", RULER_REVISION' in src
    assert "--tokenizer_type" in src
    assert '"hf"' in src
