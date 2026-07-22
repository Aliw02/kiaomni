"""Document Q&A endpoint — paste a document, ask 1+ questions."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from ..engine import EngineNotReady, EngineOOM, get_engine
from ..schemas import DocQARequest
from ..telemetry import get_telemetry

router = APIRouter()


@router.post("/api/docqa")
def docqa(req: DocQARequest) -> dict:
    eng = get_engine()
    if not eng.is_ready():
        raise HTTPException(status_code=503, detail="model not ready")

    out: list[dict] = []
    for q in req.questions:
        messages = [
            {"role": "system", "content":
             "You answer questions about a document the user provides. "
             "Answer only based on the document. If the answer is not in the "
             "document, say 'Not found in document'."},
            {"role": "user", "content":
             f"Document:\n```\n{req.document}\n```\n\nQuestion: {q}\n\nAnswer:"},
        ]
        oom = False
        try:
            t0 = time.perf_counter()
            gen = eng.generate_full(
                messages,  # type: ignore[arg-type]
                policy=req.policy,
                budget=req.budget,
                max_new_tokens=req.max_new_tokens,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            out.append({
                "question": q,
                "answer": gen.text,
                "stats": gen.stats.model_dump(),
                "wall_ms": wall_ms,
            })
            get_telemetry().record_request(
                endpoint="docqa", policy=req.policy, budget=req.budget,
                tokens_in=gen.stats.tokens_in, tokens_kept=gen.stats.tokens_kept,
                prefill_ms=gen.stats.prefill_ms, decode_ms=gen.stats.decode_ms,
                tok_per_sec=gen.stats.tok_per_sec,
                vram_allocated_mb=gen.stats.vram_allocated_mb, oom=False,
            )
        except EngineOOM as exc:
            oom = True
            out.append({"question": q, "error": "oom", "message": str(exc)})
    return {"document_chars": len(req.document), "answers": out, "oom": oom}
