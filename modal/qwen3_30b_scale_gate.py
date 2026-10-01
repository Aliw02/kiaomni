from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-30b-scale-gate"
VOLUME_NAME = "kiaomni-qwen3-30b-cache"
VOLUME_PATH = Path("/cache")
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "b9b7053e66b5de60c03b1913dbc21e900ef7ded7"
MODEL_DIR = VOLUME_PATH / "models" / "qwen3-30b-a3b-instruct-2507"
DATASET_ID = "THUDM/LongBench-v2"
DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"
DATASET_DIR = VOLUME_PATH / "datasets" / "longbench-v2"
OUTPUT_DIR = VOLUME_PATH / "outputs" / "phase03_qwen3_30b_scale_gate"

# Hard ceilings: even if a stage hangs, these stop the GPU burn.
STAGE_TIMEOUTS = {
    "preflight": 30 * 60,
    "smoke": 45 * 60,
    "final": 90 * 60,
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.57.6",
        "accelerate>=1.10,<2",
        "datasets>=3.6,<5",
        "huggingface_hub>=0.36,<2",
        "hf_xet>=1.1",
        "numpy>=1.26,<3",
        "safetensors>=0.5",
    )
    .add_local_dir(
        ".",
        remote_path="/root/kiaomni",
        copy=False,
        ignore=[".git/**", ".venv/**", "__pycache__/**", "*.pyc", "results/**"],
    )
    .workdir("/root/kiaomni")
)


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
    max_containers=1,
)
def prefetch_assets() -> dict[str, str]:
    """Download large public assets without reserving a GPU."""
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import snapshot_download

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
        volume.commit()

    dataset_marker = DATASET_DIR / ".complete"
    if not dataset_marker.exists():
        ds = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
        ds.save_to_disk(str(DATASET_DIR))
        dataset_marker.write_text(DATASET_REVISION, encoding="utf-8")
        volume.commit()
    else:
        # Fail early if the cached dataset is corrupted.
        _ = load_from_disk(str(DATASET_DIR))

    return {
        "model_dir": str(MODEL_DIR),
        "model_revision": MODEL_REVISION,
        "dataset_dir": str(DATASET_DIR),
        "dataset_revision": DATASET_REVISION,
    }


@app.function(
    image=image,
    volumes={str(VOLUME_PATH): volume},
    cpu=4,
    memory=32768,
    timeout=STAGE_TIMEOUTS["final"],
    max_containers=1,
    scaledown_window=0,
)
def run_stage(stage: str) -> dict:
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"Unknown stage: {stage}")

    model_marker = MODEL_DIR / ".complete"
    dataset_marker = DATASET_DIR / ".complete"
    if not model_marker.exists() or not dataset_marker.exists():
        raise RuntimeError("Assets are missing. Run prefetch_assets first.")
    if model_marker.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        raise RuntimeError("Cached model revision does not match frozen protocol")
    if dataset_marker.read_text(encoding="utf-8").strip() != DATASET_REVISION:
        raise RuntimeError("Cached dataset revision does not match frozen protocol")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/kiaomni"
    cmd = [
        sys.executable,
        "/root/kiaomni/experiments/modal_qwen3_30b_scale_gate.py",
        "--stage", stage,
        "--model", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--dataset", DATASET_ID,
        "--dataset-revision", DATASET_REVISION,
        "--model-dir", str(MODEL_DIR),
        "--dataset-dir", str(DATASET_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--target-tokens", "8192",
        "--max-context", "12288",
        "--min-free-gb", "8",
    ]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=False, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Phase-03 {stage} failed with exit code {proc.returncode}")

    artifact = OUTPUT_DIR / f"{stage}.json"
    if not artifact.exists():
        raise RuntimeError(f"Runner returned success but artifact is missing: {artifact}")
    volume.commit()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return {
        "stage": stage,
        "artifact": str(artifact),
        "summary": payload.get("summary", {}),
        "gate": payload.get("gate", {}),
        "cuda_after_load": payload.get("metadata", {}).get("cuda_after_load", {}),
    }


@app.local_entrypoint()
def main(stage: str = "preflight", gpu: str = "A100", prefetch: bool = True):
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(STAGE_TIMEOUTS)}")

    print(f"KiaOmni Phase-03 stage={stage} gpu={gpu}")
    print(f"Hard GPU timeout: {STAGE_TIMEOUTS[stage] / 60:.0f} minutes")

    if prefetch:
        print("Prefetching model + LongBench-v2 on CPU volume...")
        assets = prefetch_assets.remote()
        print(json.dumps(assets, indent=2))

    fn = run_stage.with_options(gpu=gpu, timeout=STAGE_TIMEOUTS[stage])
    result = fn.remote(stage)
    print(json.dumps(result, indent=2))
    print("Download artifact with:")
    print(
        f"modal volume get {VOLUME_NAME} "
        f"outputs/phase03_qwen3_30b_scale_gate/{stage}.json ."
    )
