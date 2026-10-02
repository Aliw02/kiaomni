from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-ruler-niah-frontier-v1"
MODEL_REVISION = "0d7cf23"
RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3")
RULER_LENGTHS = (8192, 16384)
RULER_SAMPLES = 4

ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"
RESULTS_VOLUME_NAME = "kiaomni-qwen3-frontier-results"
ASSET_ROOT = "/assets"
MODEL_DIR = "/assets/models/qwen3-30b-a3b-instruct-2507_0d7cf23"
RULER_DATA_ROOT = f"/assets/ruler_niah_{RULER_REVISION[:8]}"
RULER_MANIFEST = f"{RULER_DATA_ROOT}/MANIFEST.json"

RESULTS_ROOT = "/results/phase_03_ruler_niah_frontier_v1"
REMOTE_REPO = "/root/kiaomni"
REMOTE_RUNNER = f"{REMOTE_REPO}/experiments/qwen3_30b_ruler_niah_frontier.py"
REMOTE_CORE = f"{REMOTE_REPO}/experiments/qwen3_30b_percentage_replay.py"

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = LOCAL_REPO / "kiaomni"
LOCAL_RUNNER = LOCAL_REPO / "experiments" / "qwen3_30b_ruler_niah_frontier.py"
LOCAL_CORE = LOCAL_REPO / "experiments" / "qwen3_30b_percentage_replay.py"

STAGE_TIMEOUTS = {
    "preflight": 20 * 60,
    "final": 120 * 60,
}

app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

runtime_image = (
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
        "scipy>=1.11,<2",
        "safetensors>=0.5",
        "nltk>=3.9,<4",
        "wonderwords>=2.2,<3",
        "pyyaml>=6,<7",
        "tqdm>=4.66,<5",
    )
    .add_local_dir(
        LOCAL_PACKAGE,
        remote_path=f"{REMOTE_REPO}/kiaomni",
        copy=False,
        ignore=["__pycache__/**", "*.pyc"],
    )
    .add_local_file(LOCAL_RUNNER, remote_path=REMOTE_RUNNER, copy=False)
    .add_local_file(LOCAL_CORE, remote_path=REMOTE_CORE, copy=False)
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
        raise RuntimeError("Tracked working-tree changes detected. Commit/stash before running:\n" + dirty)
    return head


@app.function(
    image=runtime_image,
    volumes={ASSET_ROOT: assets},
    cpu=8,
    memory=16384,
    timeout=45 * 60,
    max_containers=1,
    single_use_containers=True,
)
def prepare_ruler_data() -> dict[str, object]:
    assets.reload()
    if not Path(MODEL_DIR).exists():
        raise RuntimeError(f"Frozen Qwen model asset is missing: {MODEL_DIR}")

    root = Path(RULER_DATA_ROOT)
    manifest_path = Path(RULER_MANIFEST)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("ruler_revision") == RULER_REVISION
            and tuple(manifest.get("tasks", [])) == RULER_TASKS
            and tuple(manifest.get("lengths", [])) == RULER_LENGTHS
            and int(manifest.get("samples_per_task", -1)) == RULER_SAMPLES
        ):
            return {"status": "EXISTS", "manifest": RULER_MANIFEST}

    work = Path("/tmp/ruler")
    shutil.rmtree(work, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "https://github.com/NVIDIA/RULER.git", str(work)],
        check=True,
    )
    subprocess.run(["git", "checkout", RULER_REVISION], cwd=work, check=True)

    data_script_dir = work / "scripts" / "data"
    prepare = data_script_dir / "prepare.py"
    if not prepare.exists():
        raise RuntimeError("Pinned RULER prepare.py not found")

    for length in RULER_LENGTHS:
        save_dir = root / str(length)
        save_dir.mkdir(parents=True, exist_ok=True)
        for task in RULER_TASKS:
            cmd = [
                sys.executable,
                str(prepare),
                "--save_dir", str(save_dir),
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", MODEL_DIR,
                "--tokenizer_type", "hf",
                "--max_seq_length", str(length),
                "--model_template_type", "base",
                "--num_samples", str(RULER_SAMPLES),
                "--random_seed", "42",
            ]
            subprocess.run(cmd, cwd=data_script_dir, check=True)
            expected = save_dir / task / "validation.jsonl"
            if not expected.exists():
                raise RuntimeError(f"RULER generation did not create {expected}")

    manifest = {
        "schema": "KIAOMNI_RULER_NIAH_ASSET_V1",
        "ruler_repo": "NVIDIA/RULER",
        "ruler_revision": RULER_REVISION,
        "tasks": list(RULER_TASKS),
        "lengths": list(RULER_LENGTHS),
        "samples_per_task": RULER_SAMPLES,
        "tokenizer_model_revision": MODEL_REVISION,
        "generation": "Official NVIDIA/RULER scripts/data/prepare.py at pinned revision; tokenizer_type=hf; model_template_type=base; random_seed=42.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assets.commit()
    return {"status": "CREATED", "manifest": RULER_MANIFEST}


def _dependency(stage: str) -> str | None:
    if stage == "final":
        return f"{RESULTS_ROOT}/preflight.json"
    return None


@app.function(
    image=runtime_image,
    volumes={ASSET_ROOT: assets, "/results": results},
    cpu=4,
    memory=32768,
    timeout=STAGE_TIMEOUTS["final"],
    max_containers=1,
    single_use_containers=True,
)
def run_stage(stage: str, repo_revision: str) -> dict[str, object]:
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"Unknown stage: {stage}")
    assets.reload()
    results.reload()

    for required in (MODEL_DIR, RULER_DATA_ROOT, RULER_MANIFEST):
        if not Path(required).exists():
            raise RuntimeError(f"Required RULER asset missing: {required}")

    dep_raw = _dependency(stage)
    if dep_raw:
        dep = Path(dep_raw)
        if not dep.exists():
            raise RuntimeError(f"Required previous-stage artifact missing: {dep}")
        previous = json.loads(dep.read_text(encoding="utf-8"))
        if previous.get("execution_gate", {}).get("status") != "PASS":
            raise RuntimeError("RULER preflight is not PASS")
        if previous.get("repo_revision") != repo_revision:
            raise RuntimeError("RULER preflight was run from a different repo revision")

    root = Path(RESULTS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{stage}.json"
    log = root / f"{stage}.log"
    out.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    results.commit()

    cmd = [
        sys.executable,
        REMOTE_RUNNER,
        "--stage", stage,
        "--model-dir", MODEL_DIR,
        "--ruler-data-root", RULER_DATA_ROOT,
        "--ruler-manifest", RULER_MANIFEST,
        "--repo-revision", repo_revision,
        "--output", str(out),
        "--min-free-gb", "8",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_REPO
    env["TOKENIZERS_PARALLELISM"] = "false"

    with log.open("w", encoding="utf-8", buffering=1) as fp:
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
            fp.write(line)
        code = proc.wait()

    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {
        "execution_gate": {"status": "ERROR", "reason": "runner wrote no artifact", "exit_code": code}
    }
    results.commit()
    if code != 0:
        raise RuntimeError(f"RULER {stage} exited {code}: {payload.get('execution_gate')}")
    return {
        "stage": stage,
        "output": str(out),
        "log": str(log),
        "execution_gate": payload.get("execution_gate"),
        "wall_seconds": payload.get("wall_seconds"),
    }


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=3 * 60 * 60,
    max_containers=1,
    single_use_containers=True,
)
def orchestrate(stage: str, gpu: str, prepare_data: bool, repo_revision: str) -> dict[str, object]:
    prepared = prepare_ruler_data.remote() if prepare_data else None
    fn = run_stage.with_options(gpu=gpu, timeout=STAGE_TIMEOUTS[stage])
    summary = fn.remote(stage, repo_revision)
    return {"prepared": prepared, "summary": summary}


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    gpu: str = "A100-80GB",
    prepare_data: bool = False,
):
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(STAGE_TIMEOUTS)}")
    if prepare_data and stage != "preflight":
        raise ValueError("--prepare-data only applies to preflight")
    repo_revision = _local_git_state()
    print(
        f"KiaOmni RULER NIAH stage={stage} gpu={gpu} repo={repo_revision} "
        f"timeout={STAGE_TIMEOUTS[stage] / 60:.0f} min"
    )
    call = orchestrate.spawn(stage, gpu, prepare_data, repo_revision)
    print(f"Remote orchestration started: {call.object_id}")
    print(f"Logs: modal app logs {APP_NAME}")
    print(
        f"Artifact: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_ruler_niah_frontier_v1/{stage}.json ./{stage}.json"
    )
    print(
        f"Run log: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_ruler_niah_frontier_v1/{stage}.log ./{stage}.log"
    )
