"""Single-policy streaming chat endpoint."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..engine import EngineNotReady, EngineOOM, get_engine
from ..schemas import ChatRequest
from ..sessions import get_session_store
from ..telemetry import get_telemetry

router = APIRouter()


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


@router.post("/api/chat")
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    eng = get_engine()
    if not eng.is_ready():
        async def _err():  # type: ignore[no-untyped-def]
            yield _sse({"type": "error", "error": "not_ready",
                        "message": "model is still loading — try again in a moment"})
        return StreamingResponse(_err(), media_type="text/event-stream")

    # Session glue: if a session_id is provided, append the latest user turn
    # and replace the messages list with the session's full history.
    messages = list(req.messages)
    session_id = req.session_id
    store = get_session_store()
    if session_id is not None:
        existing = store.get(session_id)
        if existing is not None and existing.messages:
            # Use the session's stored history verbatim, including the just-arrived user turn.
            messages = list(existing.messages)
            if messages and messages[-1].role == "user":
                pass  # already in session
            elif req.messages and req.messages[-1].role == "user":
                # Append latest user turn to the session
                snap = store.append(session_id, "user", req.messages[-1].content)
                if snap is not None:
                    messages = list(snap.messages)

    def _gen():  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        oom = False
        try:
            for ev in eng.stream_generate(
                messages,
                policy=req.policy,
                budget=req.budget,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            ):
                if ev.get("type") == "error" and ev.get("error") == "oom":
                    oom = True
                yield _sse(ev)
                if ev.get("type") == "stats":
                    sb = ev["stats"]
                    get_telemetry().record_request(
                        endpoint="chat",
                        policy=req.policy,
                        budget=req.budget,
                        tokens_in=sb["tokens_in"],
                        tokens_kept=sb["tokens_kept"],
                        prefill_ms=sb["prefill_ms"],
                        decode_ms=sb["decode_ms"],
                        tok_per_sec=sb["tok_per_sec"],
                        vram_allocated_mb=sb["vram_allocated_mb"],
                        oom=oom,
                    )
        except EngineNotReady as exc:
            yield _sse({"type": "error", "error": "not_ready", "message": str(exc)})
        except EngineOOM as exc:
            yield _sse({"type": "error", "error": "oom", "message": str(exc)})
        finally:
            # Append assistant reply to the session if applicable
            if session_id is not None and not oom:
                # We don't have the full text here (it streamed out);
                # the client is expected to POST /api/session/append with the
                # assistant's reply it just received. This keeps the engine
                # and the client state machine simple.
                pass

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )
