"""Health + readiness endpoints."""
from __future__ import annotations

import time

from fastapi import APIRouter

import kiaomni

from ..engine import get_engine
from ..schemas import HealthResponse
from ..telemetry import get_telemetry

router = APIRouter()


# Known context windows for the models the chat supports. Falls back to
# 32 K when the model id isn't in this table.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "Qwen/Qwen2.5-7B-Instruct": 131_072,   # 128 K (Qwen2.5 official)
    "Qwen/Qwen2.5-14B-Instruct": 131_072,
    "Qwen/Qwen2.5-3B-Instruct": 131_072,
    "Qwen/Qwen2.5-1.5B-Instruct": 131_072,
    "Qwen/Qwen2.5-0.5B-Instruct": 131_072,
    "Qwen/Qwen3-8B": 32_768,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 131_072,
    "meta-llama/Llama-3.1-8B-Instruct": 131_072,
    "mistralai/Mistral-7B-Instruct-v0.3": 32_768,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2_048,
}
_DEFAULT_BUDGET_BY_CONTEXT: list[tuple[int, int]] = [
    # (context_window_min, default_budget)
    (100_000, 512),
    (16_000, 512),
    (4_000, 256),
    (0, 128),
]


def _resolve_window(model_id: str) -> int:
    for prefix, ctx in _MODEL_CONTEXT_WINDOWS.items():
        if model_id.startswith(prefix):
            return ctx
    return 32_768  # safe default


def _default_budget(ctx: int) -> int:
    for threshold, budget in _DEFAULT_BUDGET_BY_CONTEXT:
        if ctx >= threshold:
            return budget
    return 128


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    eng = get_engine()
    h = eng._health_dict()  # internal, but safe
    model_id = str(h.get("model", ""))
    ctx = _resolve_window(model_id)
    return HealthResponse(
        ready=bool(h.get("ready", False)),
        model=model_id,
        gpu=str(h.get("device", "cpu")),
        vram_allocated_mb=float(h.get("vram", {}).get("allocated_mb", 0.0)),
        vram_reserved_mb=float(h.get("vram", {}).get("reserved_mb", 0.0)),
        kiaomni_version=kiaomni.__version__,
        uptime_s=float(h.get("uptime_s", 0.0)),
        context_window=ctx,
        default_budget=_default_budget(ctx),
    )
