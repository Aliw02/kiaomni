from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "qwen3_30b_scale_gate.py"
LAUNCHER = ROOT / "modal" / "qwen3_30b_scale_gate_modal.py"
SALIENCY = ROOT / "kiaomni" / "adapters" / "saliency.py"
PROTOCOL = (
    ROOT
    / "results"
    / "kiaomni_moe_model_lab"
    / "phase_03_qwen3_30b_scale_gate"
    / "PROTOCOL_FREEZE.json"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("phase03_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase03_python_files_parse():
    ast.parse(_source(RUNNER))
    ast.parse(_source(LAUNCHER))
    ast.parse(_source(SALIENCY))


def test_protocol_is_claim_bounded_and_budget_capped():
    data = json.loads(_source(PROTOCOL))
    assert data["kiaomni"]["n_sink"] == 16
    assert data["kiaomni"]["recency"] == 32
    assert data["kiaomni"]["budgets"] == [2048, 1024, 512]
    assert data["cost_guard"]["one_pass_max_gpu_minutes"] == 195
    assert "does not claim real past_key_values" in data["claim_scope"]
    assert data["final_gate"]["16x_512_budget"] == "diagnostic stress condition only"


def test_runner_reuses_saliency_and_forbids_old_wrapper_path():
    src = _source(RUNNER)
    assert "SaliencyAdapter(probe_obj, offload_to_cpu=False)" in src
    assert "select_pruned_ids" in src
    assert "apply_kiaomni" not in src
    assert "CPU/disk offload is forbidden" in src
    assert "Non-finite saliency" in src
    assert "top_k_jaccard" in src


def test_launcher_pins_assets_and_has_hard_timeouts():
    src = _source(LAUNCHER)
    assert 'MODEL_REVISION = "b9b7053e66b5de60c03b1913dbc21e900ef7ded7"' in src
    assert 'DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"' in src
    assert '"preflight": 30 * 60' in src
    assert '"smoke": 45 * 60' in src
    assert '"final": 120 * 60' in src
    assert "max_containers=1" in src
    assert "with_options(" in src


def test_final_gate_pass_fail_and_inconclusive():
    mod = _load_runner()

    def summary(combined, real, synthetic, n=8):
        return {
            "methods": {
                "kiaomni_2048": {
                    "conditioned_n": n,
                    "full_context_conditioned_accuracy": combined,
                },
                "kiaomni_1024": {
                    "conditioned_n": n,
                    "full_context_conditioned_accuracy": combined,
                },
            },
            "by_source": {
                "longbench-v2": {
                    "kiaomni_1024": {
                        "conditioned_n": n,
                        "full_context_conditioned_accuracy": real,
                    }
                },
                "synthetic": {
                    "kiaomni_1024": {
                        "conditioned_n": n,
                        "full_context_conditioned_accuracy": synthetic,
                    }
                },
            },
        }

    assert mod.evaluate_gate("final", summary(0.95, 0.80, 0.90))["status"] == "PASS"
    assert mod.evaluate_gate("final", summary(0.70, 0.80, 0.90))["status"] == "FAIL"
    assert mod.evaluate_gate("final", summary(0.95, 0.80, 0.90, n=2))["status"] == "INCONCLUSIVE"


def test_saliency_cpu_offload_remains_backward_compatible_default():
    from kiaomni.adapters.saliency import SaliencyAdapter

    param = inspect.signature(SaliencyAdapter.__init__).parameters["offload_to_cpu"]
    assert param.default is True
