"""Run the frozen True-KV nine-way ablation on one Modal T4 at a time.

Usage (after modal token authentication):
  modal run experiments/true_kv_memory_ablation_v2/modal_runner.py --mode smoke
  modal run experiments/true_kv_memory_ablation_v2/modal_runner.py --mode pilot
  modal run experiments/true_kv_memory_ablation_v2/modal_runner.py --mode final

The T4 is required for the primary VRAM/OOM comparison with Kaggle T4.
No GPU is started by importing this module or by creating the Git branch.
"""
from __future__ import annotations

import subprocess

import modal

APP_NAME = "kiaomni-true-kv-memory-ablation-v2"
REPO = "https://github.com/Aliw02/kiaomni.git"
BRANCH = "exp/kiaomni-true-kv-memory-ablation-v2-research"
ROOT = "/opt/kiaomni"
RESULTS = "/outputs/true_kv_memory_ablation_v2"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(
    "kiaomni-true-kv-ablation-results", create_if_missing=True
)
hf_volume = modal.Volume.from_name(
    "kiaomni-true-kv-ablation-hf-cache", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "bitsandbytes==0.48.2",
        "accelerate>=1.2,<2",
        "scipy>=1.12,<2",
        "numpy>=1.26,<3",
        "pandas>=2.2,<3",
        "safetensors",
    )
    .run_commands(
        f"git clone --depth 1 --single-branch --branch {BRANCH} {REPO} {ROOT}",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16000,
    max_containers=1,
    volumes={
        "/outputs": results_volume,
        "/root/.cache/huggingface": hf_volume,
    },
)
def run_on_t4(mode: str, context: int, run_tag: str,
              max_new_tokens: int = 0) -> dict:
    from pathlib import Path
    import os

    if not run_tag or "/" in run_tag or ".." in run_tag:
        raise ValueError("Use a nonempty stable run-tag without path separators")
    folder = f"{RESULTS}/{run_tag}/{mode}/context_{context}"
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "experiments/true_kv_memory_ablation_v2/run_ablation.py",
        "--mode", mode, "--contexts", str(context),
        "--output-dir", folder,
    ]
    if (path / "runs.jsonl").exists():
        cmd.append("--resume")
    if max_new_tokens:
        cmd += ["--max-new-tokens", str(max_new_tokens)]

    env = os.environ.copy()
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["KIAOMNI_TRUEKV_RESEARCH_ONLY"] = "1"
    try:
        status = subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode
        return {
            "mode": mode, "context": context, "run_tag": run_tag,
            "exit_code": status, "output_dir": folder,
            "has_manifest": (path / "manifest.json").exists(),
        }
    finally:
        # Flush all completed JSONL rows, including on a failed/partial job.
        results_volume.commit()
        hf_volume.commit()


@app.local_entrypoint()
def main(mode: str = "smoke", contexts: str = "",
         run_tag: str = "frozen_v2a", max_new_tokens: int = 0):
    profiles = {
        "smoke": [2048], "pilot": [4096], "final": [4096, 8192],
    }
    if mode not in profiles:
        raise ValueError("mode must be smoke, pilot or final")
    selected = [int(x.strip()) for x in contexts.split(",") if x.strip()] if contexts else profiles[mode]
    for ctx in selected:
        print(f"[Modal T4] mode={mode} context={ctx} run_tag={run_tag}", flush=True)
        result = run_on_t4.remote(mode, ctx, run_tag, max_new_tokens)
        print(result, flush=True)
        if result["exit_code"]:
            raise RuntimeError(
                "GPU ablation failed or was partial; inspect persisted results "
                f"at {result['output_dir']} before starting the next context"
            )
