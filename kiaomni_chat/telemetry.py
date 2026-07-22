"""VRAM telemetry ring buffer.

The engine snapshots ``memory_allocated`` and ``memory_reserved`` after every
request and pushes them into this ring. The frontend reads it via
``/api/telemetry`` to plot VRAM-over-time.

Critically, we never call ``torch.cuda.empty_cache()`` anywhere in this module
or the engine — the buffer observes what PyTorch's caching allocator does on
its own.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any

try:
    import torch
except ImportError:  # CPU-only sanity-test path
    torch = None  # type: ignore[assignment]

_MAX_SNAPSHOTS: int = 2000
_MAX_REQUESTS: int = 500


@dataclass
class VRAMSnapshot:
    t: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    fragmentation_pct: float


@dataclass
class RequestRecord:
    t: float
    endpoint: str
    policy: str
    budget: int
    tokens_in: int
    tokens_kept: int
    prefill_ms: float
    decode_ms: float
    tok_per_sec: float
    vram_allocated_mb: float
    oom: bool


class Telemetry:
    def __init__(self) -> None:
        self._snapshots: deque[VRAMSnapshot] = deque(maxlen=_MAX_SNAPSHOTS)
        self._requests: deque[RequestRecord] = deque(maxlen=_MAX_REQUESTS)
        self._started_at: float = time.time()
        self._lock = RLock()
        self._oom_count: int = 0

    def snapshot(self) -> VRAMSnapshot | None:
        if torch is None or not torch.cuda.is_available():
            return None
        with self._lock:
            alloc = torch.cuda.memory_allocated() / 2**20
            resrv = torch.cuda.memory_reserved() / 2**20
            maxalloc = torch.cuda.max_memory_allocated() / 2**20
            frag = ((resrv - alloc) / resrv * 100.0) if resrv > 0 else 0.0
            s = VRAMSnapshot(
                t=time.time(), allocated_mb=alloc, reserved_mb=resrv,
                max_allocated_mb=maxalloc, fragmentation_pct=frag,
            )
            self._snapshots.append(s)
            return s

    def record_request(
        self, *, endpoint: str, policy: str, budget: int,
        tokens_in: int, tokens_kept: int,
        prefill_ms: float, decode_ms: float, tok_per_sec: float,
        vram_allocated_mb: float, oom: bool = False,
    ) -> None:
        with self._lock:
            self._requests.append(RequestRecord(
                t=time.time(), endpoint=endpoint, policy=policy, budget=budget,
                tokens_in=tokens_in, tokens_kept=tokens_kept,
                prefill_ms=prefill_ms, decode_ms=decode_ms, tok_per_sec=tok_per_sec,
                vram_allocated_mb=vram_allocated_mb, oom=oom,
            ))
            if oom:
                self._oom_count += 1

    def recent_oom_count(self, window_s: float = 300.0) -> int:
        cutoff = time.time() - window_s
        with self._lock:
            return sum(1 for r in self._requests if r.oom and r.t > cutoff)

    def view(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": time.time() - self._started_at,
                "snapshots": [
                    {"t": s.t, "allocated_mb": s.allocated_mb, "reserved_mb": s.reserved_mb,
                     "max_allocated_mb": s.max_allocated_mb, "fragmentation_pct": s.fragmentation_pct}
                    for s in self._snapshots
                ],
                "requests": [
                    {"t": r.t, "endpoint": r.endpoint, "policy": r.policy, "budget": r.budget,
                     "tokens_in": r.tokens_in, "tokens_kept": r.tokens_kept,
                     "prefill_ms": r.prefill_ms, "decode_ms": r.decode_ms,
                     "tok_per_sec": r.tok_per_sec, "vram_allocated_mb": r.vram_allocated_mb,
                     "oom": r.oom}
                    for r in self._requests
                ],
                "oom_count": self._oom_count,
                "stats": {
                    "total_requests": len(self._requests),
                    "active_snapshots": len(self._snapshots),
                },
            }


_singleton: Telemetry | None = None


def get_telemetry() -> Telemetry:
    global _singleton
    if _singleton is None:
        _singleton = Telemetry()
    return _singleton
