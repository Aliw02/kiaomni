"""VRAM telemetry endpoint — serves the ring buffer to the frontend."""
from __future__ import annotations

from fastapi import APIRouter

from ..telemetry import get_telemetry

router = APIRouter()


@router.get("/api/telemetry")
def telemetry() -> dict:
    return get_telemetry().view()
