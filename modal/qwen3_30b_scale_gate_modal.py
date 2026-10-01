from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-30b-scale-gate"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GPU_TYPE = os.environ.get("KIAOMNI_GPU", "A100")
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/kiaomni"
HF_HOME = "/cache/huggingface"
RESULTS_ROOT = "/results/phase_03_qwen3_30b_scale_gate"

# Hard upper bounds. At Modal's documented A100 rate around $2.50/h, the three
# GPU stages sum to <= 4 hours if each hits its timeout. Asset download is CPU.
GPU_STAGE_TIMEOUTS = {
    "preflight": 20 * 60,
    "smoke": 40 * 60,
    "final": 180 * 60,
}

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("kiaomni-qwen3-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("kiaomni-qwen3-results", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "accelerate>=1.10,<2",
        "numpy>=1.26,<3",
        "datasets>=4.0,<5",
        "huggingface_hub>=0.35,<1",
    )
    .add_local_dir(REPO_ROOT, REMOTE_REPO)
)


@app.function(
    image=base_image,
    volumes={HF_HOME: model_cache},
    timeout=2 * 60 * 60,
    max_containers=1,
)
def prepare_assets() -> dict[str, str]:
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{HF_HOME}/datasets"

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_HOME,
    )
    # Warm the selected real LongBench QA subsets on CPU so GPU minutes are not
    # spent downloading/parsing benchmark data.
    longbench_sets = ("qasper", "hotpotqa", "2wikimqa", "musique", "multifieldqa_en")
    for name in longbench_sets:
        ds = load_dataset(
            "THUDM/LongBench",
            name,
            split="test",
            cache_dir=f"{HF_HOME}/datasets",
        )
        _ = len(ds)
    model_cache.commit()
    return {"model_path": model_path, "dataset": "THUDM/LongBench QA subsets"}


@app.function(
    image=base_image,
    gpu=GPU_TYPE,
    volumes={HF_HOME: model_cache, "/results": results_volume},
    timeout=max(GPU_STAGE_TIMEOUTS.values()) + 10 * 60,
    max_containers=1,
    scaledown_window=30,
)
def run_stage(stage: str) -> dict[str, object]:
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"unknown stage: {stage}")

    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{HF_HOME}/datasets"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONPATH"] = REMOTE_REPO

    out_path = f"{RESULTS_ROOT}/qwen3_30b_{stage}.json"
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        f"{REMOTE_REPO}/experiments/qwen3_30b_scale_gate.py",
        "--stage",
        stage,
        "--model",
        MODEL_ID,
        "--cache-dir",
        HF_HOME,
        "--local-files-only",
        "--max-wall-seconds",
        str(GPU_STAGE_TIMEOUTS[stage] - 120),
        "--output",
        out_path,
    ]
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)
    results_volume.commit()

    artifact = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return {
        "stage": stage,
        "output": out_path,
        "wall_seconds": artifact.get("wall_seconds"),
        "aggregate": artifact.get("aggregate"),
        "post_load_headroom": artifact.get("post_load_headroom"),
        "probe": artifact.get("probe"),
    }


@app.local_entrypoint()
def main(stage: str = "preflight", prepare: bool = False):
    if stage not in GPU_STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(GPU_STAGE_TIMEOUTS)}")
    if prepare:
        print("Preparing model and LongBench QA subsets on CPU...")
        print(prepare_assets.remote())
    print(f"Running {stage} on {GPU_TYPE}...")
    summary = run_stage.remote(stage)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        "Artifact is stored in Modal volume 'kiaomni-qwen3-results' at "
        f"{summary['output']}"
    )
