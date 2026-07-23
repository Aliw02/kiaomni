"""Side-by-side policy comparison endpoint.

Two flavours:

* ``/api/compare``            — single-shot. Run one prompt through N
                                policies. Stateless.
* ``/api/compare/turn``       — multi-turn. The client sends the full
                                conversation history; the server runs the
                                N policies on the same history and
                                returns N replies. This is what powers
                                the Side-by-Side chat panel.

Both run sequentially on a single model (the kiaomni patch is a singleton
on ``model.generate``). Latency is N× the per-call latency.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from ..engine import EngineNotReady, EngineOOM, get_engine
from ..schemas import ChatMessage, CompareRequest, CompareTurnRequest
from ..telemetry import get_telemetry

router = APIRouter()


@router.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    eng = get_engine()
    if not eng.is_ready():
        raise HTTPException(status_code=503, detail="model not ready")
    if not req.policies:
        raise HTTPException(status_code=400, detail="policies must be non-empty")

    results: list[dict] = []
    overall_oom = False
    for policy in req.policies:
        oom = False
        try:
            t0 = time.perf_counter()
            gen = eng.generate_full(
                req.messages,
                policy=policy,
                budget=req.budget,
                max_new_tokens=req.max_new_tokens,
                use_system_prompt=req.use_system_prompt,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            results.append({
                "policy": policy,
                "text": gen.text,
                "stats": gen.stats.model_dump(),
                "wall_ms": wall_ms,
            })
        except EngineOOM as exc:
            overall_oom = True
            oom = True
            results.append({"policy": policy, "error": "oom", "message": str(exc)})
        except EngineNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        finally:
            last = results[-1]
            if "stats" in last:
                sb = last["stats"]
                get_telemetry().record_request(
                    endpoint="compare", policy=policy, budget=req.budget,
                    tokens_in=sb["tokens_in"], tokens_kept=sb["tokens_kept"],
                    prefill_ms=sb["prefill_ms"], decode_ms=sb["decode_ms"],
                    tok_per_sec=sb["tok_per_sec"],
                    vram_allocated_mb=sb["vram_allocated_mb"], oom=oom,
                )
    return {"results": results, "oom": overall_oom, "budget": req.budget}


@router.post("/api/compare/turn")
def compare_turn(req: CompareTurnRequest) -> dict:
    """Multi-turn side-by-side. The client's full conversation history is
    sent on every turn; the server appends a new user turn, rebuilds the
    message list, and runs all 3 policies against the same history. Each
    policy keeps its own response thread.
    """
    eng = get_engine()
    if not eng.is_ready():
        raise HTTPException(status_code=503, detail="model not ready")
    if not req.policies:
        raise HTTPException(status_code=400, detail="policies must be non-empty")
    if not req.history:
        raise HTTPException(status_code=400, detail="history must not be empty")

    overall_oom = False
    turn_results: list[dict] = []
    for policy in req.policies:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        oom = False
        t0 = time.perf_counter()
        try:
            gen = eng.generate_full(
                req.history,
                policy=policy,
                budget=req.budget,
                max_new_tokens=req.max_new_tokens,
                use_system_prompt=req.use_system_prompt,
            )

            wall_ms = (time.perf_counter() - t0) * 1000.0
            turn_results.append({
                "policy": policy,
                "text": gen.text,
                "stats": gen.stats.model_dump(),
                "wall_ms": wall_ms,
            })
        except EngineOOM as exc:
            overall_oom = True
            oom = True
            turn_results.append({"policy": policy, "error": "oom", "message": str(exc)})
        except EngineNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            turn_results.append({"policy": policy, "error": "stream", "message": str(exc)})
        finally:
            if turn_results:
                last = turn_results[-1]
                if "stats" in last:
                    sb = last["stats"]
                    get_telemetry().record_request(
                        endpoint="compare_turn", policy=policy, budget=req.budget,
                        tokens_in=sb["tokens_in"], tokens_kept=sb["tokens_kept"],
                        prefill_ms=sb["prefill_ms"], decode_ms=sb["decode_ms"],
                        tok_per_sec=sb["tok_per_sec"],
                        vram_allocated_mb=sb["vram_allocated_mb"], oom=oom,
                    )
    return {
        "results": turn_results,
        "oom": overall_oom,
        "budget": req.budget,
        "policies": req.policies,
    }
