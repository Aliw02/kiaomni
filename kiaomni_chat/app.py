"""FastAPI ASGI app — assembled here so the Modal entry point can mount it."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import get_engine
from .routes import admin, chat, compare, demo, docqa, health, session, telemetry
from .static_dir import STATIC_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    # Lazy warm-up: do not block startup, but kick off a background load
    # so the first user request is faster.
    import threading
    eng = get_engine()

    def _warm() -> None:
        model_id = os.environ.get("KIAOMNI_MODEL_ID")
        token = os.environ.get("HF_TOKEN")
        attn = os.environ.get("KIAOMNI_ATTN", "sdpa")
        quantize = os.environ.get("KIAOMNI_QUANTIZE_4BIT", "1") == "1"
        try:
            eng.ensure_loaded(model_id=model_id, attn_impl=attn,
                              quantize_4bit=quantize, token=token)
        except Exception as exc:  # noqa: BLE001
            import sys
            print(f"[warmup] model load failed: {exc!r}", file=sys.stderr)

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="KiaOmni Web Chat",
    version="0.1.0",
    description="Production stress-test for kiaomni KV-cache eviction. "
                "No garbage collection. A100-40GB. All scenarios are run on the "
                "real Qwen2.5-7B-Instruct (4-bit NF4) model with the published "
                "kiaomni library.",
    lifespan=lifespan,
)

# Routes
app.include_router(health.router, tags=["health"])
app.include_router(telemetry.router, tags=["telemetry"])
app.include_router(chat.router, tags=["chat"])
app.include_router(compare.router, tags=["chat"])
app.include_router(demo.router, tags=["demo"])
app.include_router(docqa.router, tags=["chat"])
app.include_router(session.router, tags=["session"])
app.include_router(admin.router, tags=["admin"])


# Static frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))
