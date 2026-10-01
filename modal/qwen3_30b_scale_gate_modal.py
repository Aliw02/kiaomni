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
ASSET_ROOT = Path("/assets")
MODEL_DIR = ASSET_ROOT / "models" / "qwen3-30b-a3b-instruct-2507_0d7cf23"
DATASET_DIR = ASSET_ROOT / "datasets" / "longbench-v2_b0db4901"
ASSET_MANIFEST = ASSET_ROOT / "PHASE03_ASSET_MANIFEST.json"
RESULTS_ROOT = Path("/results") / "phase_03_qwen3_30b_scale_gate"
REMOTE_REPO = Path("/root/kiaomni")
LOCAL_REPO = Path(__file__).resolve().parents[1]

# Absolute GPU ceilings for a one-pass run.
# preflight + smoke + final = 150 configured GPU minutes total.
GPU_STAGE_TIMEOUTS = {
    "preflight": 20 * 60,
    "smoke": 40 * 60,
    "final": 90 * 60,
}

app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
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
        LOCAL_REPO,
        remote_path=str(REMOTE_REPO),
        copy=False,
        ignore=[".git/**", ".venv/**", "__pycache__/**", "*.pyc"],
    )
    .workdir(str(REMOTE_REPO))
)


@app.function(
    image=image,
    volumes={str(ASSET_ROOT): assets},
    cpu=4,
    memory=8192,
    timeout=3 * 60 * 60,
    max_containers=1,
)
def prepare_assets() -> dict[str, str]:
    """Download model + frozen real benchmark without reserving a GPU."""
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

    model_marker = MODEL_DIR / ".complete"
    if not model_marker.exists():
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
        )
        model_marker.write_text(MODEL_REVISION, encoding="utf-8")
        assets.commit()

    dataset_marker = DATASET_DIR / ".complete"
    if not dataset_marker.exists():
        ds = load_dataset(
            DATASET_ID,
            split="train",
            revision=DATASET_REVISION,
        )
        ds.save_to_disk(str(DATASET_DIR))
        dataset_marker.write_text(DATASET_REVISION, encoding="utf-8")
        assets.commit()
    else:
        _ = load_from_disk(str(DATASET_DIR))

    api = HfApi()
    model_info = api.model_info(MODEL_ID, revision=MODEL_REVISION)
    dataset_info = api.dataset_info(DATASET_ID, revision=DATASET_REVISION)
    manifest = {
        "model_repo": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": model_info.sha,
        "model_dir": str(MODEL_DIR),
        "dataset_repo": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_revision_resolved": dataset_info.sha,
        "dataset_dir": str(DATASET_DIR),
    }
    ASSET_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    assets.commit()
    return manifest


def _dependency_path(stage: str) -> Path | None:
    if stage == "smoke":
        return RESULTS_ROOT / "preflight.json"
    if stage == "final":
        return RESULTS_ROOT / "smoke.json"
    return None


@app.function(
    image=image,
    volumes={str(ASSET_ROOT): assets, "/results": results},
    cpu=4,
    memory=32768,
    timeout=GPU_STAGE_TIMEOUTS["final"],
    max_containers=1,
    scaledown_window=0,
)
def run_stage(stage: str) -> dict[str, object]:
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"unknown stage: {stage}")

    if not ASSET_MANIFEST.exists():
        raise RuntimeError("Assets are missing. Run --prepare with preflight first.")

    dep = _dependency_path(stage)
    if dep is not None:
        if not dep.exists():
            raise RuntimeError(f"Required previous-stage artifact is missing: {dep}")
        dep_payload = json.loads(dep.read_text(encoding="utf-8"))
        dep_status = dep_payload.get("gate", {}).get("status")
        if dep_status != "PASS":
            raise RuntimeError(
                f"Previous stage did not PASS ({dep.name}: {dep_status}); refusing {stage}."
            )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_ROOT / f"{stage}.json"
    cmd = [
        sys.executable,
        str(REMOTE_REPO / "experiments" / "qwen3_30b_scale_gate.py"),
        "--stage", stage,
        "--model", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--dataset", DATASET_ID,
        "--dataset-revision", DATASET_REVISION,
        "--model-dir", str(MODEL_DIR),
        "--dataset-dir", str(DATASET_DIR),
        "--asset-manifest", str(ASSET_MANIFEST),
        "--output", str(out_path),
        "--min-free-gb", "8",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_REPO)
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.run(
        cmd,
        cwd=str(REMOTE_REPO),
        env=env,
        check=False,
    )
    if out_path.exists():
        results.commit()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    summary = {
        "stage": stage,
        "exit_code": proc.returncode,
        "output": str(out_path),
        "gate": payload.get("gate", {}),
        "aggregate": payload.get("aggregate", []),
        "post_load_safety": payload.get("post_load_safety", {}),
        "probe": payload.get("probe", {}),
        "wall_seconds": payload.get("wall_seconds"),
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"Phase-03 {stage} exited {proc.returncode}. "
            f"Artifact was preserved at {out_path}. Gate={summary['gate']}"
        )
    return summary


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    gpu: str = "A100-80GB",
    prepare: bool = False,
):
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(GPU_STAGE_TIMEOUTS)}")

    print(
        f"KiaOmni Phase-03 stage={stage} gpu={gpu} "
        f"hard_timeout={GPU_STAGE_TIMEOUTS[stage] / 60:.0f} min"
    )

    if prepare:
        print("Preparing frozen model + LongBench-v2 on CPU volume...")
        print(json.dumps(prepare_assets.remote(), indent=2))

    fn = run_stage.with_options(
        gpu=gpu,
        timeout=GPU_STAGE_TIMEOUTS[stage],
    )
    summary = fn.remote(stage)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Download artifact:")
    print(
        f"modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_qwen3_30b_scale_gate/{stage}.json ./{stage}.json"
    )
