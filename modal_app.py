"""Modal entry point for the KiaOmni Chat web app.

The image installs kiaomni from PyPI (matching the demo notebook's
``pip install kiaomni[gaussian]`` style), builds the FastAPI ASGI app, and
serves it on a single A100-40GB. No manual GC anywhere — the v3 plan
mandates that VRAM accumulates naturally.
"""
from __future__ import annotations

import os

import modal

# ── App definition ────────────────────────────────────────────────────────
app = modal.App("kiaomni-chat")

# ── Image ────────────────────────────────────────────────────────────────
# kiaomni is built from GitHub. The [gaussian] extra pulls scipy.
# We pin torch to 2.7.0+cpu+cu126 for CUDA 12.6 compatibility on A100;
# kiaomni requires torch>=2.6. CUDA 13 wheels force torch 2.13 which our
# CUDA 12.1 base image can't load.
#
# The kiaomni_chat/ web package is installed via `pip install -e ./kiaomni_chat`
# so `from kiaomni_chat.app import app` resolves reliably. The local dir is
# copied into the image with add_local_dir() first.
#
# Build marker bumped to v3 (was v1, then v2) to force a fresh image layer
# — earlier cached layers did not include the kiaomni_chat package and the
# container kept crash-looping with ModuleNotFoundError on every request.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",   # base has no Python — add_python installs 3.11 fresh
    )
    .apt_install("git")
    .pip_install(
        # torch 2.6.0 ships cu121 wheels, satisfying kiaomni (>=2.6) and
        # our CUDA 12.1 base. NO bitsandbytes — engine loads in pure fp16.
        "torch==2.6.0",
        "transformers>=4.50,<5.0",
        "accelerate<2",
        "scipy>=1.12",                  # kiaomni_gaussian
        "fastapi==0.115.0",
        "uvicorn==0.32.0",
        "pydantic==2.9.2",
        "hf-transfer==0.1.6",
        "kvpress",
    )
    .add_local_dir("./kiaomni", "/root/kiaomni", copy=True)
    .add_local_dir("./kiaomni_chat", "/root/kiaomni_chat", copy=True)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "KIAOMNI_BUILD": "v17", "PYTHONPATH": "/root:/root/kiaomni:/root/kiaomni_chat"})




)

# ── Modal ASGI deployment ───────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=8192,
    timeout=7200,
    scaledown_window=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve() -> object:
    # Qwen2.5-7B-Instruct is Apache 2.0 / ungated, no HF token needed.
    os.environ.setdefault("KIAOMNI_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    os.environ.setdefault("KIAOMNI_ATTN", "sdpa")
    os.environ.setdefault("KIAOMNI_QUANTIZE_4BIT", "1")

    from kiaomni_chat.app import app as fastapi_app
    return fastapi_app


# ── Acceptance test: 4 demo tasks × kiaomni_gaussian @ B=512 ─────────────
@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=8192,
    timeout=3600,
    scaledown_window=600,
    max_containers=1,
)
def test_demo_acceptance(n_samples: int = 3) -> dict:
    """Run the 4 demo tasks with kiaomni_gaussian @ B=512. Compare to
    the demo notebook's published numbers."""
    os.environ.setdefault("KIAOMNI_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    os.environ.setdefault("KIAOMNI_ATTN", "sdpa")
    os.environ.setdefault("KIAOMNI_QUANTIZE_4BIT", "1")

    import time
    from kiaomni_chat.engine import get_engine
    from kiaomni_chat.tasks import list_tasks, make_sample

    eng = get_engine()
    eng.ensure_loaded()

    BUDGET = 512
    targets = {"single": 0.90, "multi": 0.90, "reason": 0.30, "summary": 0.90}
    out: dict = {"budget": BUDGET, "results": {}, "vram_peak_gb": 0.0}

    for task in list_tasks():
        scores: list[float] = []
        t0 = time.perf_counter()
        for sid in range(n_samples):
            sample = make_sample(task, sid, eng._tokenizer)
            messages = [{"role": "user", "content": f"{sample.context}\n\n{sample.question}"}]
            gen = eng.generate_full(
                messages, policy="kiaomni_gaussian", budget=BUDGET,  # type: ignore[arg-type]
                max_new_tokens=sample.max_new,
            )
            score, detail = sample.grade(gen.text)
            scores.append(score)
        elapsed = time.perf_counter() - t0
        mean = sum(scores) / len(scores)
        vram_peak = eng._vram()["max_allocated_mb"] / 1024.0
        out["results"][task] = {
            "n": n_samples,
            "mean": mean,
            "target": targets[task],
            "pass": mean >= targets[task],
            "elapsed_s": elapsed,
        }
        out["vram_peak_gb"] = max(out["vram_peak_gb"], vram_peak)

    out["all_pass"] = all(r["pass"] for r in out["results"].values())
    return out


# ── Stress test: 100 sequential chat requests, observe VRAM drift ────────
@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=8192,
    timeout=7200,
    scaledown_window=600,
    max_containers=1,
)
def test_vram_accumulation(n_requests: int = 100, budget: int = 512) -> dict:
    """Sequential chat requests, snapshot VRAM after each, return the
    full series. No manual GC — this is the v3 production stress test."""
    os.environ.setdefault("KIAOMNI_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    os.environ.setdefault("KIAOMNI_ATTN", "sdpa")
    os.environ.setdefault("KIAOMNI_QUANTIZE_4BIT", "1")

    import torch
    from kiaomni_chat.engine import get_engine
    from kiaomni_chat.telemetry import get_telemetry

    eng = get_engine()
    eng.ensure_loaded()
    tel = get_telemetry()
    series: list[dict] = []

    for i in range(n_requests):
        messages = [{"role": "user", "content": f"Turn {i+1}. Tell me a short fact."}]
        gen = eng.generate_full(
            messages,  # type: ignore[arg-type]
            policy="kiaomni_gaussian", budget=budget, max_new_tokens=64,
        )
        v = eng._vram()
        series.append({
            "i": i,
            "allocated_mb": v["allocated_mb"],
            "reserved_mb": v["reserved_mb"],
            "max_allocated_mb": v["max_allocated_mb"],
            "fragmentation_pct": v["fragmentation_pct"],
        })

    reserved = [s["reserved_mb"] for s in series]
    return {
        "n_requests": n_requests,
        "series": series,
        "summary": {
            "reserved_first": reserved[0],
            "reserved_last": reserved[-1],
            "reserved_max": max(reserved),
            "reserved_plateau_idx": _plateau_index(reserved),
            "vram_peak_gb": max(s["max_allocated_mb"] for s in series) / 1024.0,
        },
    }


# ── Q1–Q9 regression test ────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=8192,
    timeout=3600,
    scaledown_window=600,
    max_containers=1,
)
def test_q1_q9_acceptance(policy: str = "kiaomni_gaussian", budget: int = 512) -> dict:
    """Run the 9 Starship questions with the given policy at B=budget.

    Scores all 6 tier-1 questions (Q1–Q6) against
    ``expected_answers.json`` and asserts >= 80% mean (= at least 5/6
    or equivalent partial credit). Also runs fullcontext as a sanity
    check that the grading pipeline is working.
    """
    import json
    import numpy as np
    from pathlib import Path
    from kiaomni_chat.engine import get_engine
    from kiaomni_chat.grade_export import grade as grade_export

    REPO = Path(__file__).resolve().parent
    article_path = REPO / "kiaomni_chat" / "static" / "sample_data" / "article.txt"
    expected_path = REPO / "kiaomni_chat" / "sample_data" / "expected_answers.json"

    os.environ.setdefault("KIAOMNI_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    os.environ.setdefault("KIAOMNI_ATTN", "sdpa")
    os.environ.setdefault("KIAOMNI_QUANTIZE_4BIT", "1")

    eng = get_engine()
    eng.ensure_loaded()

    article = article_path.read_text("utf-8")
    expected = json.loads(expected_path.read_text("utf-8"))

    messages = [
        {
            "role": "user",
            "content": (
                f"Read this article and then answer Q1 through Q9 about it.\n\n"
                f"{article}\n\n"
                f"Q1: On what date did the third integrated flight test (IFT-3) of Starship launch?\n"
                f"Q2: What altitude did the Starship upper stage reach during IFT-3?\n"
                f"Q3: What peak temperature did the heatshield experience during IFT-4's re-entry?\n"
                f"Q4: On what date was the first successful booster catch achieved?\n"
                f"Q5: List the dates and outcomes of the five integrated flight tests (IFT-3 through IFT-7).\n"
                f"Q6: List the specifications of the Starship vehicle: total height, total liftoff mass, "
                f"total liftoff thrust, number of first-stage engines, number of upper-stage engines.\n"
                f"Q7: Why is the chopstick-catch approach more ambitious than leg-landing, "
                f"and what engineering capabilities had to be proven before attempting the catch?\n"
                f"Q8: Why did IFT-6 catch the booster successfully but lose the ship during the same mission, "
                f"and what does the pattern of ship failures tell us about the engineering maturity of the two stages?\n"
                f"Q9: Summarize the SpaceX Starship 2024–2025 test campaign in exactly 5 bullet points, "
                f"covering: the vehicle, the early test flights, the catch breakthrough, "
                f"the January 2025 milestone, and the broader industry impact.\n"
            ),
        },
    ]

    results: dict[str, dict] = {}

    for p in (policy, "fullcontext"):
        import time
        t0 = time.perf_counter()
        gen = eng.generate_full(
            messages,  # type: ignore[arg-type]
            policy=p, budget=budget, max_new_tokens=2048,
        )
        elapsed = time.perf_counter() - t0

        # Grade via grade_export
        export = {
            "mode": "side_by_side",
            "deploy_url": "modal",
            "created_at": "2026-07-17T00:00:00Z",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "budget": budget,
            "max_new_tokens": 2048,
            "policies": [p],
            "turns": [
                {
                    "ts": "2026-07-17T00:00:01Z",
                    "user": messages[0]["content"],
                    "responses": {
                        p: {
                            "text": gen.text,
                            "stats": gen.stats.model_dump(),
                        },
                    },
                }
            ],
        }
        graded = grade_export(export, expected)
        tier1_mean = graded["summary_by_policy_tier1"].get(p, {}).get("mean", 0.0)
        breakdown = {
            r["question_id"]: r["score"]
            for r in graded["rows"]
            if r["policy"] == p and r.get("tier") == 1
        }
        results[p] = {
            "tier1_mean": tier1_mean,
            "tier1_breakdown": breakdown,
            "elapsed_s": round(elapsed, 2),
            "tokens_in": gen.stats.tokens_in,
            "tokens_kept": gen.stats.tokens_kept,
        }

    # Fullcontext must still work (sanity)
    fc = results.get("fullcontext", {})
    assert fc.get("tier1_mean", 0) >= 0.95, (
        f"fullcontext sanity check failed: {fc.get('tier1_mean')}"
    )

    # The kiaomni policy must score >= 80% on tier-1 questions
    kia = results.get(policy, {})
    assert kia.get("tier1_mean", 0) >= 0.80, (
        f"{policy} @ B={budget} tier-1 mean={kia.get('tier1_mean'):.3f} "
        f"below 0.80 threshold. Breakdown: {kia.get('tier1_breakdown')}"
    )

    return {
        "budget": budget,
        "results": results,
        "all_pass": True,
        "vram_peak_gb": eng._vram()["max_allocated_mb"] / 1024.0,
    }


def _plateau_index(values: list[float], window: int = 10, tol_pct: float = 2.0) -> int:
    """Return the first index after which ``values`` stays within ``tol_pct``
    of the running mean for ``window`` consecutive samples, or -1 if never."""
    if len(values) < window + 1:
        return -1
    for start in range(window, len(values)):
        chunk = values[start:start + window]
        if not chunk:
            break
        m = sum(chunk) / len(chunk)
        if m == 0:
            continue
        if all(abs(v - m) / m * 100.0 <= tol_pct for v in chunk):
            return start
    return -1
