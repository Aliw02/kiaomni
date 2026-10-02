from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-ruler-niah-v1"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23"

RULER_DATASET_ID = "VenusChenyy/RULER_50"
RULER_OFFICIAL_GENERATION_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
RULER_TASKS = (
    "niah_single_2",
    "niah_multikey_1",
    "niah_multivalue",
    "niah_multiquery",
)
RULER_LENGTHS = (8192, 16384)
RULER_MIRROR_ROWS_PER_GROUP = 50
SAMPLES_PER_GROUP = 5

ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"
RESULTS_VOLUME_NAME = "kiaomni-qwen3-frontier-results"
ASSET_ROOT = "/assets"
MODEL_DIR = "/assets/models/qwen3-30b-a3b-instruct-2507_0d7cf23"
RULER_ROOT = "/assets/ruler_50_qwen3"
RULER_INDEX = "/assets/indices/ruler_qwen3_niah_8k16k_v1.json"

RESULTS_ROOT = "/results/phase_03_ruler_niah_v1"
REMOTE_REPO = "/root/kiaomni"
REMOTE_CORE = f"{REMOTE_REPO}/experiments/qwen3_30b_percentage_replay.py"
REMOTE_RUNNER = f"{REMOTE_REPO}/experiments/qwen3_30b_ruler_niah.py"

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = LOCAL_REPO / "kiaomni"
LOCAL_CORE = LOCAL_REPO / "experiments" / "qwen3_30b_percentage_replay.py"
LOCAL_RUNNER = LOCAL_REPO / "experiments" / "qwen3_30b_ruler_niah.py"

STAGE_TIMEOUTS = {
    "preflight": 20 * 60,
    "final": 110 * 60,
}

app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "accelerate>=1.10,<2",
        "huggingface_hub>=0.35,<2",
        "hf_xet>=1.1,<2",
        "numpy>=1.26,<3",
        "scipy>=1.11,<2",
        "safetensors>=0.5",
    )
    .add_local_dir(
        LOCAL_PACKAGE,
        remote_path=f"{REMOTE_REPO}/kiaomni",
        copy=False,
        ignore=["__pycache__/**", "*.pyc"],
    )
    .add_local_file(
        LOCAL_CORE,
        remote_path=REMOTE_CORE,
        copy=False,
    )
    .add_local_file(
        LOCAL_RUNNER,
        remote_path=REMOTE_RUNNER,
        copy=False,
    )
)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface_hub>=0.35,<2",
        "hf_xet>=1.1,<2",
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
            "Tracked working-tree changes detected. Commit/stash before running:\n" + dirty
        )
    return head


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_depth_stratified(rows: list[dict], count: int = 5) -> list[dict]:
    centers = [10.0, 30.0, 50.0, 70.0, 90.0]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    if count != len(centers):
        raise ValueError("This frozen selector expects five depth bins")

    candidates = []
    for line_idx, row in enumerate(rows):
        if "token_position_answer" not in row:
            continue
        denom = int(row.get("length_w_model_temp", row.get("length", 0)))
        if denom <= 0:
            continue
        depth = 100.0 * int(row["token_position_answer"]) / denom
        candidates.append((line_idx, depth, row))

    if len(candidates) < count:
        raise RuntimeError(f"Not enough depth-aware RULER rows: {len(candidates)}")

    selected = []
    used = set()
    for center, label in zip(centers, labels):
        lo, hi = center - 10.0, center + 10.0
        in_bin = [
            x for x in candidates
            if x[0] not in used and lo <= x[1] < hi
        ]
        pool = in_bin if in_bin else [x for x in candidates if x[0] not in used]
        chosen = min(pool, key=lambda x: (abs(x[1] - center), x[0]))
        used.add(chosen[0])
        selected.append({
            "source_line": int(chosen[0]),
            "depth_bin": label,
            "official_depth_pct": float(chosen[1]),
            "declared_length": int(chosen[2].get("length", 0)),
            "declared_length_w_model_temp": int(
                chosen[2].get("length_w_model_temp", chosen[2].get("length", 0))
            ),
            "outputs_count": len(chosen[2].get("outputs", [])),
        })
    return selected


@app.function(
    image=data_image,
    volumes={ASSET_ROOT: assets},
    cpu=2,
    memory=4096,
    timeout=45 * 60,
    max_containers=1,
    single_use_containers=True,
)
def prepare_ruler_assets() -> dict[str, object]:
    from huggingface_hub import HfApi, hf_hub_download

    assets.reload()
    root = Path(RULER_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    Path(RULER_INDEX).parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(RULER_DATASET_ID)
    resolved_revision = str(info.sha)
    repo_files = set(
        api.list_repo_files(
            RULER_DATASET_ID,
            repo_type="dataset",
            revision=resolved_revision,
        )
    )

    manifest_path = Path(
        hf_hub_download(
            repo_id=RULER_DATASET_ID,
            repo_type="dataset",
            filename="source_official_generation_manifest.json",
            revision=resolved_revision,
            local_dir=RULER_ROOT,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("official_commit") != RULER_OFFICIAL_GENERATION_COMMIT:
        raise RuntimeError(
            "RULER mirror official generation commit mismatch: "
            f"{manifest.get('official_commit')}"
        )
    if int(manifest.get("expected_rows_per_group", 0)) != 500:
        raise RuntimeError("Unexpected RULER rows-per-group in generation manifest")

    manifest_groups = {
        (str(x["task"]), int(x["context_length"])): x
        for x in manifest.get("files", [])
        if "task" in x and "context_length" in x
    }

    frozen_groups = []
    for task in RULER_TASKS:
        for length in RULER_LENGTHS:
            key = (task, length)
            if key not in manifest_groups:
                raise RuntimeError(f"Missing RULER manifest group: {task}/{length}")
            source_entry = manifest_groups[key]
            source_rows = int(source_entry.get("rows", 0))
            if source_rows != 500:
                raise RuntimeError(
                    f"Unexpected official-source row count for {task}/{length}: {source_rows}"
                )

            # The public RULER_50 mirror is a deterministic 50-row extraction
            # from each official 500-row source group. Its repository layout is
            # <task>/<context_length>.jsonl, not the original source output_path.
            filename = f"{task}/{length}.jsonl"
            if filename not in repo_files:
                raise RuntimeError(
                    f"Required RULER_50 mirror file is missing: {filename}"
                )

            local_path = Path(
                hf_hub_download(
                    repo_id=RULER_DATASET_ID,
                    repo_type="dataset",
                    filename=filename,
                    revision=resolved_revision,
                    local_dir=RULER_ROOT,
                )
            )
            observed_sha = _sha256(local_path)

            rows = _read_jsonl(local_path)
            if len(rows) != RULER_MIRROR_ROWS_PER_GROUP:
                raise RuntimeError(
                    f"RULER_50 mirror row-count mismatch {task}/{length}: "
                    f"{len(rows)} != {RULER_MIRROR_ROWS_PER_GROUP}"
                )

            selected = _select_depth_stratified(rows, SAMPLES_PER_GROUP)
            frozen_groups.append({
                "task": task,
                "context_length": length,
                "repo_filename": filename,
                "relative_path": str(local_path.relative_to(root)).replace("\\", "/"),
                "mirror_sha256": observed_sha,
                "mirror_rows": len(rows),
                "official_source_output_path": str(source_entry.get("output_path", "")),
                "official_source_sha256": str(source_entry.get("sha256", "")),
                "official_source_rows": source_rows,
                "selected": selected,
            })

    payload = {
        "schema": "KIAOMNI_RULER_NIAH_INDEX_V1",
        "date": "2026-10-02",
        "dataset_repo": RULER_DATASET_ID,
        "dataset_revision": resolved_revision,
        "official_repo": "NVIDIA/RULER",
        "official_generation_commit": RULER_OFFICIAL_GENERATION_COMMIT,
        "generation_method": manifest.get("generation_method"),
        "generation_tokenizer_path": manifest.get("tokenizer_path"),
        "generation_tokenizer_type": manifest.get("tokenizer_type"),
        "random_seed": manifest.get("random_seed"),
        "tasks": list(RULER_TASKS),
        "lengths": list(RULER_LENGTHS),
        "mirror_rows_per_group": RULER_MIRROR_ROWS_PER_GROUP,
        "samples_per_group": SAMPLES_PER_GROUP,
        "integrity_rule": (
            "Dataset revision pins mirror bytes; each downloaded mirror file is hashed "
            "and frozen in this index. Official-source provenance/500-row SHA is recorded "
            "separately and is never compared to the 50-row mirror file."
        ),
        "selection_rule": (
            "Five deterministic target-answer depth strata per task/length: "
            "closest unused sample to 10/30/50/70/90 percent, preferring its 20-point bin."
        ),
        "answer_prefix_policy": (
            "Reattach stored answer_prefix to input before applying the Qwen chat template."
        ),
        "groups": frozen_groups,
    }
    Path(RULER_INDEX).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assets.commit()
    return {
        "dataset_revision": resolved_revision,
        "groups": len(frozen_groups),
        "cases": sum(len(x["selected"]) for x in frozen_groups),
        "index": RULER_INDEX,
    }


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

    for required in (MODEL_DIR, RULER_INDEX):
        if not Path(required).exists():
            raise RuntimeError(f"Required RULER asset missing: {required}")

    dep_raw = _dependency(stage)
    if dep_raw:
        dep = Path(dep_raw)
        if not dep.exists():
            raise RuntimeError(f"Required previous-stage artifact missing: {dep}")
        d = json.loads(dep.read_text(encoding="utf-8"))
        status = d.get("execution_gate", {}).get("status")
        previous_revision = d.get("repo_revision")
        if status != "PASS":
            raise RuntimeError(f"RULER preflight is not PASS: {status}")
        if previous_revision != repo_revision:
            raise RuntimeError(
                f"RULER preflight revision {previous_revision} != current {repo_revision}"
            )

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
        "--ruler-root", RULER_ROOT,
        "--ruler-index", RULER_INDEX,
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

    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema": "KIAOMNI_QWEN3_30B_RULER_NIAH_ERROR_V1",
            "stage": stage,
            "repo_revision": repo_revision,
            "execution_gate": {
                "status": "ERROR",
                "reason": "RULER runner exited without writing an artifact",
                "exit_code": code,
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    results.commit()
    if code != 0:
        raise RuntimeError(
            f"RULER {stage} exited {code}; artifact/log preserved: "
            f"{payload.get('execution_gate')}"
        )
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
def orchestrate(
    stage: str,
    gpu: str,
    prepare_ruler: bool,
    repo_revision: str,
) -> dict[str, object]:
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"Unknown stage: {stage}")
    if prepare_ruler and stage != "preflight":
        raise ValueError("--prepare-ruler is only valid for preflight")

    prepared = None
    if prepare_ruler:
        prepared = prepare_ruler_assets.remote()

    fn = run_stage.with_options(gpu=gpu, timeout=STAGE_TIMEOUTS[stage])
    summary = fn.remote(stage, repo_revision)
    return {"prepared": prepared, "summary": summary}


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    gpu: str = "A100-80GB",
    prepare_ruler: bool = False,
):
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(STAGE_TIMEOUTS)}")
    if prepare_ruler and stage != "preflight":
        raise ValueError("--prepare-ruler only applies to preflight")

    repo_revision = _local_git_state()
    print(
        f"KiaOmni RULER NIAH stage={stage} gpu={gpu} repo={repo_revision} "
        f"timeout={STAGE_TIMEOUTS[stage] / 60:.0f} min"
    )
    call = orchestrate.spawn(stage, gpu, prepare_ruler, repo_revision)
    print(f"Remote orchestration started: {call.object_id}")
    print(f"Logs: modal app logs {APP_NAME}")
    print(
        f"Artifact: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_ruler_niah_v1/{stage}.json ./{stage}.json"
    )
    print(
        f"Run log: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_ruler_niah_v1/{stage}.log ./{stage}.log"
    )
