"""Multi-turn session CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import (
    ChatMessage,
    SessionAppendRequest,
    SessionCreateRequest,
    SessionResponse,
)
from ..sessions import get_session_store

router = APIRouter()


@router.post("/api/session/create", response_model=SessionResponse)
def create(req: SessionCreateRequest) -> SessionResponse:
    return get_session_store().create(req.system_prompt)


# /stats must be registered BEFORE the /{session_id} path-param route,
# otherwise FastAPI's route matching will capture "stats" as a session_id.
@router.get("/api/session/stats")
def stats() -> dict:
    return get_session_store().stats()


@router.get("/api/session/{session_id}", response_model=SessionResponse | None)
def get(session_id: str) -> SessionResponse | None:
    snap = get_session_store().get(session_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="session not found")
    return snap


@router.post("/api/session/append", response_model=SessionResponse | None)
def append(req: SessionAppendRequest) -> SessionResponse | None:
    snap = get_session_store().append(req.session_id, req.role, req.content)
    if snap is None:
        raise HTTPException(status_code=404, detail="session not found")
    return snap


@router.delete("/api/session/{session_id}")
def clear(session_id: str) -> dict:
    ok = get_session_store().clear(session_id)
    return {"cleared": ok}
