"""Demo task runner — runs the 4 NIAH-style tasks from notebook/demo/."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from ..engine import EngineNotReady, EngineOOM, get_engine
from ..schemas import DemoRunRequest
from ..tasks import list_tasks, make_sample
from ..telemetry import get_telemetry

router = APIRouter()


@router.post("/api/demo/run")
def demo_run(req: DemoRunRequest) -> dict:
    eng = get_engine()
    if not eng.is_ready():
        raise HTTPException(status_code=503, detail="model not ready")

    tok = eng._tokenizer  # type: ignore[attr-defined]
    if tok is None:
        raise HTTPException(status_code=503, detail="tokenizer not loaded")

    tasks_to_run: list[str] = list_tasks() if req.task == "all" else [req.task]
    if tasks_to_run[0] not in list_tasks():
        raise HTTPException(status_code=400, detail=f"unknown task {req.task!r}")

    out: dict = {"task": req.task, "policy": req.policy, "budget": req.budget,
                 "results": []}

    for task in tasks_to_run:
        per_task: list[dict] = []
        for sid in range(req.n_samples):
            sample = make_sample(task, sid, tok)
            messages = [
                {"role": "user", "content": f"{sample.context}\n\n{sample.question}"}
            ]
            oom = False
            try:
                t0 = time.perf_counter()
                gen = eng.generate_full(
                    messages,  # type: ignore[arg-type]
                    policy=req.policy,
                    budget=req.budget,
                    max_new_tokens=sample.max_new,
                )
                wall_ms = (time.perf_counter() - t0) * 1000.0
                score, detail = sample.grade(gen.text)
                per_task.append({
                    "sid": sid,
                    "info": sample.info,
                    "answer": gen.text,
                    "score": score,
                    "detail": detail,
                    "gold": sample.gold,
                    "stats": gen.stats.model_dump(),
                    "wall_ms": wall_ms,
                })
                get_telemetry().record_request(
                    endpoint="demo", policy=req.policy, budget=req.budget,
                    tokens_in=gen.stats.tokens_in, tokens_kept=gen.stats.tokens_kept,
                    prefill_ms=gen.stats.prefill_ms, decode_ms=gen.stats.decode_ms,
                    tok_per_sec=gen.stats.tok_per_sec,
                    vram_allocated_mb=gen.stats.vram_allocated_mb, oom=False,
                )
            except EngineOOM as exc:
                oom = True
                per_task.append({
                    "sid": sid, "info": sample.info, "error": "oom", "message": str(exc),
                })
        # Aggregate
        scored = [r for r in per_task if "score" in r]
        if scored:
            mean_score = sum(r["score"] for r in scored) / len(scored)
            n_pass = sum(1 for r in scored if r["score"] >= 0.99)
        else:
            mean_score = 0.0
            n_pass = 0
        out["results"].append({
            "task": task,
            "n_samples": len(per_task),
            "n_scored": len(scored),
            "mean_score": mean_score,
            "n_pass": n_pass,
            "samples": per_task,
        })

    return out
