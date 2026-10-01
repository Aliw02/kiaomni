from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-30b-scale-gate"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "b9b7053e66b5de60c03b1913dbc21e900ef7ded7"
DATASET_ID = "THUDM/LongBench-v2"
DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/kiaomni"
CACHE_ROOT = Path("/cache")
HF_HOME = CACHE_ROOT / "huggingface"
MODEL_DIR = CACHE_ROOT / "models" / "qwen3-30b-a3b-instruct-2507"
RESULTS_ROOT = Path("/results") / "phase_03_qwen3_30b_scale_gate"

GPU_STAGE_TIMEOUTS = {
    "preflight": 30 * 60,
    "smoke": 45 * 60,
    "final": 120 * 60,
}

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name(
    "kiaomni-qwen3-model-cache", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "kiaomni-qwen3-results", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "accelerate>=1.10,<2",
        "numpy>=1.26,<3",
        "datasets>=4.0,<5",
        "huggingface_hub>=0.35,<1",
        "hf_xet>=1.1,<2",
        "safetensors>=0.5",
    )
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=False)
)


@app.function(
    image=image,
    volumes={str(CACHE_ROOT): model_cache},
    timeout=2 * 60 * 60,
    cpu=4.0,
    memory=8192,
    max_containers=1,
)
def prepare_assets() -> dict[str, str]:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_marker = MODEL_DIR / ".kiaomni_revision"
    if (
        not model_marker.exists()
        or model_marker.read_text(encoding="utf-8").strip() != MODEL_REVISION
    ):
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
        )
        model_marker.write_text(MODEL_REVISION, encoding="utf-8")
        model_cache.commit()

    ds = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
        cache_dir=str(HF_HOME / "datasets"),
    )
    _ = len(ds)
    dataset_marker = HF_HOME / "datasets" / ".longbench_v2_revision"
    dataset_marker.parent.mkdir(parents=True, exist_ok=True)
    dataset_marker.write_text(DATASET_REVISION, encoding="utf-8")
    model_cache.commit()

    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dir": str(MODEL_DIR),
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
    }


@app.function(
    image=image,
    volumes={
        str(CACHE_ROOT): model_cache,
        "/results": results_volume,
    },
    cpu=4.0,
    memory=32768,
    max_containers=1,
    scaledown_window=5,
)
def run_stage(stage: str) -> dict[str, object]:
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"Unknown stage: {stage}")

    model_marker = MODEL_DIR / ".kiaomni_revision"
    dataset_marker = HF_HOME / "datasets" / ".longbench_v2_revision"
    if not model_marker.exists():
        raise RuntimeError("Pinned model cache is missing; run --prepare first")
    if model_marker.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        raise RuntimeError("Pinned model revision marker does not match protocol")
    if not dataset_marker.exists():
        raise RuntimeError("Pinned LongBench-v2 cache is missing; run --prepare first")
    if dataset_marker.read_text(encoding="utf-8").strip() != DATASET_REVISION:
        raise RuntimeError("Pinned dataset revision marker does not match protocol")

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONPATH"] = REMOTE_REPO

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_ROOT / f"{stage}.json"
    cmd = [
        "python",
        f"{REMOTE_REPO}/experiments/qwen3_30b_scale_gate.py",
        "--stage", stage,
        "--model", str(MODEL_DIR),
        "--model-revision", MODEL_REVISION,
        "--dataset", DATASET_ID,
        "--dataset-revision", DATASET_REVISION,
        "--cache-dir", str(HF_HOME),
        "--local-files-only",
        "--max-wall-seconds", str(GPU_STAGE_TIMEOUTS[stage] - 120),
        "--output", str(out_path),
    ]

    proc = subprocess.run(cmd, cwd=REMOTE_REPO, check=False)
    results_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Phase-03 {stage} failed with exit code {proc.returncode}. "
            f"If a checkpoint exists it was committed to {out_path}."
        )
    if not out_path.exists():
        raise RuntimeError(f"Runner succeeded but artifact is missing: {out_path}")

    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    return {
        "stage": stage,
        "output": str(out_path),
        "complete": artifact.get("complete"),
        "gate": artifact.get("gate"),
        "summary": artifact.get("summary"),
        "post_load_headroom": artifact.get("post_load_headroom"),
        "saliency_parity": artifact.get("saliency_parity"),
    }


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    gpu: str = "A100-80GB",
    prepare: bool = False,
):
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(GPU_STAGE_TIMEOUTS)}")

    if prepare:
        print("Preparing pinned model + LongBench-v2 on CPU...")
        print(json.dumps(prepare_assets.remote(), indent=2))

    print(
        f"Running stage={stage} gpu={gpu} "
        f"hard_timeout={GPU_STAGE_TIMEOUTS[stage] // 60}m"
    )
    fn = run_stage.with_options(
        gpu=gpu,
        timeout=GPU_STAGE_TIMEOUTS[stage],
    )
    summary = fn.remote(stage)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        "Download artifact with:\n"
        f"modal volume get kiaomni-qwen3-results "
        f"phase_03_qwen3_30b_scale_gate/{stage}.json ."
    )
