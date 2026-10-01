from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
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
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_launcher():
    spec = importlib.util.spec_from_file_location("phase03_modal_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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
    assert data["retention"]["ratios"] == [0.25, 0.125, 0.0625]
    assert data["cost_guard"]["one_pass_total_gpu_ceiling_minutes"] == 150
    assert data["stages"]["smoke"]["real_cases"] == 2
    assert data["stages"]["smoke"]["cases"] == 6
    assert "past_key_values" in " ".join(data["claim_boundary"]["forbidden"])
    assert data["final_gate"]["16x"].startswith("Stress diagnostic")


def test_runner_reuses_saliency_and_does_not_use_apply_wrapper():
    src = _source(RUNNER)
    assert "SaliencyAdapter(probe, offload_to_cpu=False)" in src
    assert "policy_scores = score_fn(saliency)" in src
    assert "apply_kiaomni" not in src
    assert "CPU/disk offload is forbidden" in src
    assert "Non-finite saliency" in src
    assert "top128_jaccard" in src
    assert "RECENCY = RECENCY_DEFAULT" in src
    assert 'p.add_argument("--dataset-index", required=True)' in src
    assert 'p.add_argument("--repo-revision", required=True)' in src
    assert "KIAOMNI_QWEN3_30B_SCALE_GATE_ERROR_V1" in src


def test_modal_remote_paths_are_posix_strings_on_windows_hosts():
    src = _source(LAUNCHER)
    assert 'ASSET_ROOT = "/assets"' in src
    assert 'RESULTS_ROOT = "/results/phase_03_qwen3_30b_scale_gate"' in src
    assert 'REMOTE_REPO = "/root/kiaomni"' in src
    assert 'Path("/assets")' not in src
    assert 'Path("/root/kiaomni")' not in src
    assert 'remote_path=f"{REMOTE_REPO}/kiaomni"' in src
    assert 'remote_path=REMOTE_RUNNER' in src
    assert ".workdir(" not in src


def test_modal_launcher_imports_under_pinned_sdk():
    mod = _load_launcher()
    assert mod.APP_NAME == "kiaomni-qwen3-30b-scale-gate"
    assert mod.GPU_STAGE_TIMEOUTS == {
        "preflight": 20 * 60,
        "smoke": 40 * 60,
        "final": 90 * 60,
    }


def test_detach_is_fully_remote_and_no_invalid_scale_down():
    src = _source(LAUNCHER)
    assert "scaledown_window=0" not in src
    assert "def orchestrate(" in src
    assert "orchestrate.spawn(" in src
    assert "single_use_containers=True" in src
    assert "prepare_assets.remote()" in src
    assert "run_stage.with_options(" in src
    assert '"--untracked-files=no"' in src
    assert "Refusing to mix revisions across a frozen gate" in src
    assert "Frozen LongBench-v2 token index identity mismatch" in src


def test_launcher_pins_assets_and_has_hard_timeouts():
    src = _source(LAUNCHER)
    assert 'MODEL_REVISION = "0d7cf23"' in src
    assert 'DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"' in src
    assert '"preflight": 20 * 60' in src
    assert '"smoke": 40 * 60' in src
    assert '"final": 90 * 60' in src
    assert "max_containers=1" in src
    assert "Previous stage did not PASS" in src
    assert "with_options(" in src
    assert '"--dataset-index", DATASET_INDEX' in src
    assert '"--repo-revision", repo_revision' in src
    assert "results.reload()" in src
    assert "assets.reload()" in src


def test_final_gate_pass_fail_and_inconclusive():
    mod = _load_runner()
    rows = [
        {"case_id": f"s{i}", "result": {"method": "full_context", "success": True}}
        for i in range(4)
    ] + [
        {"case_id": f"r{i}", "result": {"method": "full_context", "success": True}}
        for i in range(3)
    ]
    base = [
        {
            "source": "synthetic",
            "method": "kiaomni_r0.25",
            "full_context_solved_n": 4,
            "full_context_conditioned_accuracy": 1.0,
        },
        {
            "source": "synthetic",
            "method": "kiaomni_r0.125",
            "full_context_solved_n": 4,
            "full_context_conditioned_accuracy": 0.75,
        },
        {
            "source": "longbench_v2",
            "method": "kiaomni_r0.125",
            "full_context_solved_n": 3,
            "full_context_conditioned_accuracy": 2 / 3,
        },
        {
            "source": "synthetic",
            "method": "random_r0.125",
            "full_context_solved_n": 4,
            "full_context_conditioned_accuracy": 0.50,
        },
        {
            "source": "synthetic",
            "method": "recency_r0.125",
            "full_context_solved_n": 4,
            "full_context_conditioned_accuracy": 0.50,
        },
    ]
    assert mod.build_gate("final", rows, base, None, 7)["status"] == "PASS"

    failed = [dict(x) for x in base]
    failed[1] = dict(failed[1], full_context_conditioned_accuracy=0.50)
    assert mod.build_gate("final", rows, failed, None, 7)["status"] == "FAIL"

    inconclusive = [dict(x) for x in base]
    inconclusive[2] = dict(
        inconclusive[2],
        full_context_solved_n=2,
        full_context_conditioned_accuracy=1.0,
    )
    assert mod.build_gate("final", rows, inconclusive, None, 7)["status"] == "INCONCLUSIVE"


def test_saliency_cpu_offload_remains_backward_compatible_default():
    from kiaomni.adapters.saliency import SaliencyAdapter

    param = inspect.signature(SaliencyAdapter.__init__).parameters["offload_to_cpu"]
    assert param.default is True
