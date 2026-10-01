from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-30b-scale-gate"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23"
DATASET_ID = "THUDM/LongBench-v2"
DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"

ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"
RESULTS_VOLUME_NAME = "kiaomni-qwen3-results"
ASSET_ROOT = "/assets"
MODEL_DIR = "/assets/models/qwen3-30b-a3b-instruct-2507_0d7cf23"
DATASET_DIR = "/assets/datasets/longbench-v2_b0db4901"
DATASET_INDEX = "/assets/indices/longbench_v2_qwen3_0d7cf23.json"
ASSET_MANIFEST = "/assets/PHASE03_ASSET_MANIFEST.json"
RESULTS_ROOT = "/results/phase_03_qwen3_30b_scale_gate"
REMOTE_REPO = "/root/kiaomni"
REMOTE_RUNNER = f"{REMOTE_REPO}/experiments/qwen3_30b_scale_gate.py"

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = LOCAL_REPO / "kiaomni"
LOCAL_RUNNER = LOCAL_REPO / "experiments" / "qwen3_30b_scale_gate.py"

GPU_STAGE_TIMEOUTS = {
    "preflight": 20 * 60,
    "smoke": 40 * 60,
    "final": 90 * 60,
}

app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "accelerate>=1.10,<2",
        "datasets>=4.0,<5",
        "huggingface_hub>=0.35,<2",
        "hf_xet>=1.1,<2",
        "numpy>=1.26,<3",
        "safetensors>=0.5",
    )
    .add_local_dir(
        LOCAL_PACKAGE,
        remote_path=f"{REMOTE_REPO}/kiaomni",
        copy=False,
        ignore=["__pycache__/**", "*.pyc"],
    )
    .add_local_file(
        LOCAL_RUNNER,
        remote_path=REMOTE_RUNNER,
        copy=False,
    )
)

control_image = modal.Image.debian_slim(python_version="3.11")


def _local_git_state() -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=LOCAL_REPO,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=LOCAL_REPO,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Tracked working-tree changes detected. Commit/stash them before running Phase 03:\n"
            + dirty
        )
    return head


def _render_for_length(tokenizer, context: str, row: dict) -> str:
    question = (
        f"{row['question']}\n\nChoices:\n"
        f"A: {row['choice_A']}\n"
        f"B: {row['choice_B']}\n"
        f"C: {row['choice_C']}\n"
        f"D: {row['choice_D']}\n\n"
        "Answer with one letter only: A, B, C, or D."
    )
    user = f"Document:\n{context}\n\nQuestion:\n{question}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user


@app.function(
    image=runtime_image,
    volumes={ASSET_ROOT: assets},
    cpu=4,
    memory=8192,
    timeout=3 * 60 * 60,
    max_containers=1,
    single_use_containers=True,
)
def prepare_assets() -> dict[str, object]:
    """Download and validate all large assets without reserving a GPU."""
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    assets.reload()

    model_dir = Path(MODEL_DIR)
    dataset_dir = Path(DATASET_DIR)
    dataset_index = Path(DATASET_INDEX)
    asset_manifest = Path(ASSET_MANIFEST)

    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_index.parent.mkdir(parents=True, exist_ok=True)

    model_marker = model_dir / ".complete"
    if not model_marker.exists():
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIR,
            max_workers=8,
        )
        model_marker.write_text(MODEL_REVISION, encoding="utf-8")
        assets.commit()

    dataset_marker = dataset_dir / ".complete"
    if not dataset_marker.exists():
        ds = load_dataset(
            DATASET_ID,
            split="train",
            revision=DATASET_REVISION,
        )
        ds.save_to_disk(DATASET_DIR)
        dataset_marker.write_text(DATASET_REVISION, encoding="utf-8")
        assets.commit()

    ds = load_from_disk(DATASET_DIR)

    required_columns = {
        "_id", "domain", "sub_domain", "difficulty", "length", "question",
        "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context",
    }
    missing = required_columns - set(ds.column_names)
    if missing:
        raise RuntimeError(f"LongBench-v2 schema mismatch; missing {sorted(missing)}")

    bad_answers = sorted({
        str(x).strip().upper()
        for x in ds["answer"]
        if str(x).strip().upper() not in {"A", "B", "C", "D"}
    })
    if bad_answers:
        raise RuntimeError(f"LongBench-v2 contains invalid answer labels: {bad_answers[:10]}")

    cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    expected_cfg = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "head_dim": 128,
    }
    observed_cfg = {k: getattr(cfg, k, None) for k in expected_cfg}
    if observed_cfg != expected_cfg:
        raise RuntimeError(
            f"Frozen Qwen config mismatch. expected={expected_cfg} observed={observed_cfg}"
        )

    index_json = model_dir / "model.safetensors.index.json"
    if not index_json.exists():
        raise RuntimeError("Frozen model is missing model.safetensors.index.json")
    weight_index = json.loads(index_json.read_text(encoding="utf-8"))
    weight_files = sorted(set(weight_index.get("weight_map", {}).values()))
    missing_weights = [name for name in weight_files if not (model_dir / name).exists()]
    if missing_weights:
        raise RuntimeError(f"Frozen model snapshot is incomplete: {missing_weights[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Frozen tokenizer has no chat template")

    if not dataset_index.exists():
        rows = []
        smoke_eligible = 0
        final_eligible = 0
        for raw in ds:
            row = dict(raw)
            if str(row.get("length", "")).lower() != "short":
                continue
            rendered = _render_for_length(tokenizer, str(row["context"]), row)
            n = len(tokenizer(rendered, add_special_tokens=False).input_ids)
            if 8192 <= n <= 16384:
                rows.append({
                    "_id": str(row["_id"]),
                    "domain": str(row.get("domain", "unknown")),
                    "sub_domain": str(row.get("sub_domain", "unknown")),
                    "rendered_tokens": int(n),
                })
                final_eligible += 1
                if n <= 12288:
                    smoke_eligible += 1

        payload = {
            "model_repo": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_repo": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_length_category": "short",
            "eligible_rows": rows,
            "smoke_eligible_8k_12k": smoke_eligible,
            "final_eligible_8k_16k": final_eligible,
        }
        dataset_index.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        assets.commit()

    index_payload = json.loads(dataset_index.read_text(encoding="utf-8"))
    expected_index_identity = {
        "model_repo": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_repo": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_length_category": "short",
    }
    observed_index_identity = {
        k: index_payload.get(k) for k in expected_index_identity
    }
    if observed_index_identity != expected_index_identity:
        raise RuntimeError(
            "Frozen LongBench-v2 token index identity mismatch: "
            f"expected={expected_index_identity} observed={observed_index_identity}"
        )
    if int(index_payload.get("smoke_eligible_8k_12k", 0)) < 2:
        raise RuntimeError("Fewer than 2 LongBench-v2 cases fit the frozen smoke window")
    if int(index_payload.get("final_eligible_8k_16k", 0)) < 6:
        raise RuntimeError("Fewer than 6 LongBench-v2 cases fit the frozen final window")

    api = HfApi()
    model_info = api.model_info(MODEL_ID, revision=MODEL_REVISION)
    dataset_info = api.dataset_info(DATASET_ID, revision=DATASET_REVISION)

    manifest = {
        "model_repo": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": model_info.sha,
        "model_dir": MODEL_DIR,
        "model_config": observed_cfg,
        "weight_file_count": len(weight_files),
        "dataset_repo": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_revision_resolved": dataset_info.sha,
        "dataset_dir": DATASET_DIR,
        "dataset_index": DATASET_INDEX,
        "dataset_rows": len(ds),
        "dataset_columns": sorted(ds.column_names),
        "smoke_eligible_8k_12k": int(index_payload["smoke_eligible_8k_12k"]),
        "final_eligible_8k_16k": int(index_payload["final_eligible_8k_16k"]),
    }
    asset_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assets.commit()
    return manifest


def _dependency_path(stage: str) -> str | None:
    if stage == "smoke":
        return f"{RESULTS_ROOT}/preflight.json"
    if stage == "final":
        return f"{RESULTS_ROOT}/smoke.json"
    return None


@app.function(
    image=runtime_image,
    volumes={ASSET_ROOT: assets, "/results": results},
    cpu=4,
    memory=32768,
    timeout=GPU_STAGE_TIMEOUTS["final"],
    max_containers=1,
    single_use_containers=True,
)
def run_stage(stage: str, repo_revision: str) -> dict[str, object]:
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"unknown stage: {stage}")

    assets.reload()
    results.reload()

    asset_manifest = Path(ASSET_MANIFEST)
    results_root = Path(RESULTS_ROOT)
    if not asset_manifest.exists():
        raise RuntimeError("Assets are missing. Run preflight with --prepare first.")

    dep_raw = _dependency_path(stage)
    if dep_raw is not None:
        dep = Path(dep_raw)
        if not dep.exists():
            raise RuntimeError(f"Required previous-stage artifact is missing: {dep}")
        dep_payload = json.loads(dep.read_text(encoding="utf-8"))
        dep_status = dep_payload.get("gate", {}).get("status")
        dep_revision = dep_payload.get("repo_revision")
        if dep_status != "PASS":
            raise RuntimeError(
                f"Previous stage did not PASS ({dep.name}: {dep_status}); refusing {stage}."
            )
        if dep_revision != repo_revision:
            raise RuntimeError(
                f"Previous stage used repo revision {dep_revision}, but current run uses "
                f"{repo_revision}. Refusing to mix revisions across a frozen gate."
            )

    results_root.mkdir(parents=True, exist_ok=True)
    out_path = results_root / f"{stage}.json"
    log_path = results_root / f"{stage}.log"

    # Never let a failed rerun accidentally expose a stale artifact.
    out_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    results.commit()

    cmd = [
        sys.executable,
        REMOTE_RUNNER,
        "--stage", stage,
        "--model", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--dataset", DATASET_ID,
        "--dataset-revision", DATASET_REVISION,
        "--model-dir", MODEL_DIR,
        "--dataset-dir", DATASET_DIR,
        "--dataset-index", DATASET_INDEX,
        "--asset-manifest", ASSET_MANIFEST,
        "--repo-revision", repo_revision,
        "--output", str(out_path),
        "--min-free-gb", "8",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_REPO
    env["TOKENIZERS_PARALLELISM"] = "false"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=REMOTE_REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = proc.wait()

    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema": "KIAOMNI_QWEN3_30B_SCALE_GATE_ERROR_V1",
            "stage": stage,
            "repo_revision": repo_revision,
            "gate": {
                "status": "ERROR",
                "reason": "runner exited without writing an artifact",
                "exit_code": return_code,
            },
            "log_file": str(log_path),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    results.commit()

    summary = {
        "stage": stage,
        "exit_code": return_code,
        "output": str(out_path),
        "log": str(log_path),
        "gate": payload.get("gate", {}),
        "aggregate": payload.get("aggregate", []),
        "post_load_safety": payload.get("post_load_safety", {}),
        "probe": payload.get("probe", {}),
        "wall_seconds": payload.get("wall_seconds"),
    }
    if return_code != 0:
        raise RuntimeError(
            f"Phase-03 {stage} exited {return_code}. "
            f"Artifact/log preserved. Gate={summary['gate']}"
        )
    return summary


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=4 * 60 * 60,
    max_containers=1,
    single_use_containers=True,
)
def orchestrate(stage: str, gpu: str, prepare: bool, repo_revision: str) -> dict[str, object]:
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"unknown stage: {stage}")
    if prepare and stage != "preflight":
        raise ValueError("--prepare is only valid with the preflight stage")

    prepared = None
    if prepare:
        prepared = prepare_assets.remote()

    stage_fn = run_stage.with_options(
        gpu=gpu,
        timeout=GPU_STAGE_TIMEOUTS[stage],
    )
    summary = stage_fn.remote(stage, repo_revision)
    return {
        "repo_revision": repo_revision,
        "prepared": prepared,
        "stage_summary": summary,
    }


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    gpu: str = "A100-80GB",
    prepare: bool = False,
):
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(GPU_STAGE_TIMEOUTS)}")
    if prepare and stage != "preflight":
        raise ValueError("--prepare is only valid with --stage preflight")

    repo_revision = _local_git_state()
    print(
        f"KiaOmni Phase-03 stage={stage} gpu={gpu} "
        f"repo={repo_revision} hard_gpu_timeout={GPU_STAGE_TIMEOUTS[stage] / 60:.0f} min"
    )

    call = orchestrate.spawn(stage, gpu, prepare, repo_revision)
    print(f"Remote orchestration started: {call.object_id}")
    print("The workflow is now remote. With 'modal run --detach', closing PowerShell will not stop it.")
    print(f"Logs: modal app logs {APP_NAME}")
    print(
        f"Artifact: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_qwen3_30b_scale_gate/{stage}.json ./{stage}.json"
    )
    print(
        f"Run log: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_qwen3_30b_scale_gate/{stage}.log ./{stage}.log"
    )
