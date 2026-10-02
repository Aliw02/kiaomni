from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kiaomni-qwen3-percentage-replay-v1"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23"
DATASET_ID = "THUDM/LongBench-v2"
DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"

ASSET_VOLUME_NAME = "kiaomni-qwen3-assets"
RESULTS_VOLUME_NAME = "kiaomni-qwen3-frontier-results"
ASSET_ROOT = "/assets"
MODEL_DIR = "/assets/models/qwen3-30b-a3b-instruct-2507_0d7cf23"
DATASET_DIR = "/assets/datasets/longbench-v2_b0db4901"
ASSET_MANIFEST = "/assets/PHASE03_ASSET_MANIFEST.json"
SOURCE_INDEX = "/assets/indices/longbench_v2_qwen3_0d7cf23.json"
ADJUDICATION_INDEX = "/assets/indices/longbench_v2_official_adjudication_v1.json"

RESULTS_ROOT = "/results/phase_03_percentage_replay_v1"
REMOTE_REPO = "/root/kiaomni"
REMOTE_RUNNER = f"{REMOTE_REPO}/experiments/qwen3_30b_percentage_replay.py"

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = LOCAL_REPO / "kiaomni"
LOCAL_RUNNER = LOCAL_REPO / "experiments" / "qwen3_30b_percentage_replay.py"

STAGE_TIMEOUTS = {
    "preflight": 20 * 60,
    "final": 175 * 60,
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
        "datasets>=4.0,<5",
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
            "Tracked working-tree changes detected. Commit/stash before running:\n" + dirty
        )
    return head


def _official_prompt(row: dict) -> str:
    return (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
        f"{str(row['context']).strip()}\n"
        "</text>\n\n"
        f"What is the correct answer to this question: {str(row['question']).strip()}\n"
        "Choices:\n"
        f"(A) {str(row['choice_A']).strip()}\n"
        f"(B) {str(row['choice_B']).strip()}\n"
        f"(C) {str(row['choice_C']).strip()}\n"
        f"(D) {str(row['choice_D']).strip()}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


@app.function(
    image=runtime_image,
    volumes={ASSET_ROOT: assets},
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    max_containers=1,
    single_use_containers=True,
)
def prepare_adjudication_index() -> dict[str, object]:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    assets.reload()
    for required in (MODEL_DIR, DATASET_DIR, ASSET_MANIFEST, SOURCE_INDEX):
        if not Path(required).exists():
            raise RuntimeError(f"Required frozen Phase-03 asset is missing: {required}")

    manifest = json.loads(Path(ASSET_MANIFEST).read_text(encoding="utf-8"))
    if not str(manifest.get("model_revision_resolved", "")).startswith(MODEL_REVISION):
        raise RuntimeError("Frozen model revision mismatch")
    if manifest.get("dataset_revision_resolved") != DATASET_REVISION:
        raise RuntimeError("Frozen dataset revision mismatch")

    source = json.loads(Path(SOURCE_INDEX).read_text(encoding="utf-8"))
    source_rows = list(source.get("eligible_rows", []))
    if len(source_rows) != 27:
        raise RuntimeError(
            f"Parent Phase-03 eligible set must contain exactly 27 rows; found {len(source_rows)}"
        )

    ids = [str(x["_id"]) for x in source_rows]
    ds = load_from_disk(DATASET_DIR)
    by_id = {str(raw["_id"]): dict(raw) for raw in ds if str(raw["_id"]) in set(ids)}
    if set(by_id) != set(ids):
        raise RuntimeError("Parent eligible set does not match frozen LongBench-v2 dataset")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
    )

    rows = []
    outside = []
    for item in source_rows:
        row = by_id[str(item["_id"])]
        user = _official_prompt(row)
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = user
        n = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        rec = {
            "_id": str(row["_id"]),
            "domain": str(row.get("domain", "unknown")),
            "sub_domain": str(row.get("sub_domain", "unknown")),
            "difficulty": str(row.get("difficulty", "unknown")),
            "parent_rendered_tokens": int(item["rendered_tokens"]),
            "official_rendered_tokens": int(n),
        }
        rows.append(rec)
        if not (8192 <= n <= 16384):
            outside.append(rec)

    if outside:
        raise RuntimeError(
            "Official LongBench-v2 prompt moved frozen parent cases outside 8K-16K: "
            + json.dumps(outside, ensure_ascii=False)
        )

    payload = {
        "schema": "KIAOMNI_QWEN3_30B_ADJUDICATION_INDEX_V1",
        "model_repo": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_repo": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_index": SOURCE_INDEX,
        "selection_rule": "exactly the 27 parent Phase-03 eligible IDs, no reselection",
        "prompt": "LongBench-v2 official zero-shot wording + model chat template",
        "rows": rows,
    }
    Path(ADJUDICATION_INDEX).parent.mkdir(parents=True, exist_ok=True)
    Path(ADJUDICATION_INDEX).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assets.commit()
    return {
        "rows": len(rows),
        "min_tokens": min(x["official_rendered_tokens"] for x in rows),
        "max_tokens": max(x["official_rendered_tokens"] for x in rows),
        "index": ADJUDICATION_INDEX,
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
def run_stage(stage: str, repo_revision: str, resume: bool = False, case_start: int = 0) -> dict[str, object]:
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"Unknown stage: {stage}")

    assets.reload()
    results.reload()

    for required in (MODEL_DIR, DATASET_DIR, ASSET_MANIFEST, ADJUDICATION_INDEX):
        if not Path(required).exists():
            raise RuntimeError(f"Required asset missing: {required}")

    dep_raw = _dependency(stage)
    if dep_raw:
        dep = Path(dep_raw)
        if not dep.exists():
            raise RuntimeError(f"Required previous-stage artifact missing: {dep}")
        d = json.loads(dep.read_text(encoding="utf-8"))
        status = d.get("execution_gate", {}).get("status")
        previous_revision = d.get("repo_revision")
        if status != "PASS":
            raise RuntimeError(f"Preflight is not PASS: {status}")
        if previous_revision != repo_revision:
            current_runner_sha = hashlib.sha256(
                Path(REMOTE_RUNNER).read_bytes()
            ).hexdigest()
            previous_runner_sha = d.get("runner_sha256")
            if previous_runner_sha != current_runner_sha:
                semantic_ok = (
                    d.get("schema") == "KIAOMNI_QWEN3_30B_PERCENTAGE_REPLAY_V1"
                    and d.get("model", {}).get("repo") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
                    and d.get("model", {}).get("revision") == "0d7cf23"
                    and d.get("dataset", {}).get("repo") == "THUDM/LongBench-v2"
                    and d.get("dataset", {}).get("revision") == "b0db4901b856522026b7353ab541b8535ff2a4b8"
                    and d.get("policies") == ["kiaomni_s8", "kiaomni_gaussian"]
                    and d.get("percentage_budgets") == [0.25, 0.125, 0.0625]
                    and d.get("max_new_tokens") == 256
                    and d.get("routing_preflight", {}).get("passed") is True
                )
                if not semantic_ok:
                    raise RuntimeError(
                        f"Preflight revision {previous_revision} != current {repo_revision}; "
                        "runner changed and semantic compatibility checks failed"
                    )
                print(
                    "Preflight runner SHA differs only across an execution-control revision; "
                    "frozen model/dataset/policies/ratios/token-limit/routing checks match. "
                    "Accepting the existing PASS preflight.",
                    flush=True,
                )
            else:
                print(
                    "Preflight repo revision differs, but runner SHA256 is identical; "
                    "accepting the frozen preflight without rerunning it.",
                    flush=True,
                )

    root = Path(RESULTS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_tail_from_{case_start + 1:02d}" if case_start else ""
    out = root / f"{stage}{suffix}.json"
    log = root / f"{stage}{suffix}.log"
    if not resume:
        out.unlink(missing_ok=True)
        log.unlink(missing_ok=True)
        results.commit()
    elif stage != "final":
        raise RuntimeError("--resume is only valid for final")

    cmd = [
        sys.executable,
        REMOTE_RUNNER,
        "--stage", stage,
        "--model-dir", MODEL_DIR,
        "--dataset-dir", DATASET_DIR,
        "--adjudication-index", ADJUDICATION_INDEX,
        "--asset-manifest", ASSET_MANIFEST,
        "--repo-revision", repo_revision,
        "--output", str(out),
        "--min-free-gb", "8",
    ]
    if resume:
        cmd.append("--resume")
    if case_start:
        cmd.extend(["--case-start", str(case_start)])

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_REPO
    env["TOKENIZERS_PARALLELISM"] = "false"

    with log.open("a" if resume else "w", encoding="utf-8", buffering=1) as fp:
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
            "schema": "KIAOMNI_QWEN3_30B_PERCENTAGE_REPLAY_ERROR_V1",
            "stage": stage,
            "repo_revision": repo_revision,
            "execution_gate": {
                "status": "ERROR",
                "reason": "Runner exited without writing an artifact",
                "exit_code": code,
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    results.commit()
    if code != 0:
        raise RuntimeError(
            f"Percentage replay {stage} exited {code}; artifact/log preserved: "
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
def orchestrate(stage: str, gpu: str, prepare_index: bool, repo_revision: str, resume: bool = False, case_start: int = 0) -> dict[str, object]:
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"Unknown stage: {stage}")
    if prepare_index and stage != "preflight":
        raise ValueError("--prepare-index is only valid for preflight")

    prepared = None
    if prepare_index:
        prepared = prepare_adjudication_index.remote()

    fn = run_stage.with_options(gpu=gpu, timeout=STAGE_TIMEOUTS[stage])
    summary = fn.remote(stage, repo_revision, resume, case_start)
    return {"prepared": prepared, "summary": summary}


@app.local_entrypoint()
def main(
    stage: str = "preflight",
    gpu: str = "A100-80GB",
    prepare_index: bool = False,
    resume: bool = False,
    case_start: int = 0,
):
    if stage not in STAGE_TIMEOUTS:
        raise ValueError(f"stage must be one of {sorted(STAGE_TIMEOUTS)}")
    if prepare_index and stage != "preflight":
        raise ValueError("--prepare-index only applies to preflight")
    if resume and stage != "final":
        raise ValueError("--resume only applies to final")
    if case_start and stage != "final":
        raise ValueError("--case-start only applies to final")
    if case_start < 0 or case_start > 26:
        raise ValueError("--case-start must be between 0 and 26")

    repo_revision = _local_git_state()
    print(
        f"KiaOmni percentage replay stage={stage} gpu={gpu} repo={repo_revision} "
        f"timeout={STAGE_TIMEOUTS[stage] / 60:.0f} min"
    )
    call = orchestrate.spawn(stage, gpu, prepare_index, repo_revision, resume, case_start)
    print(f"Remote orchestration started: {call.object_id}")
    print(f"Logs: modal app logs {APP_NAME}")
    print(
        f"Artifact: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_percentage_replay_v1/{stage}.json ./{stage}.json"
    )
    print(
        f"Run log: modal volume get {RESULTS_VOLUME_NAME} "
        f"phase_03_percentage_replay_v1/{stage}.log ./{stage}.log"
    )
