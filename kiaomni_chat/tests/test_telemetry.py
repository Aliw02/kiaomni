"""Test the VRAM telemetry ring buffer."""
from __future__ import annotations

from kiaomni_chat.telemetry import Telemetry, get_telemetry


def test_singleton() -> None:
    a = get_telemetry()
    b = get_telemetry()
    assert a is b


def test_record_request_increments_oom_count() -> None:
    t = Telemetry()
    t.record_request(endpoint="x", policy="y", budget=128, tokens_in=1000,
                     tokens_kept=200, prefill_ms=10, decode_ms=20,
                     tok_per_sec=10, vram_allocated_mb=100)
    t.record_request(endpoint="x", policy="y", budget=128, tokens_in=1000,
                     tokens_kept=200, prefill_ms=10, decode_ms=20,
                     tok_per_sec=10, vram_allocated_mb=100, oom=True)
    view = t.view()
    assert view["oom_count"] == 1
    assert view["stats"]["total_requests"] == 2


def test_view_shape() -> None:
    t = Telemetry()
    view = t.view()
    assert "snapshots" in view
    assert "requests" in view
    assert "uptime_s" in view
    assert "oom_count" in view


def test_recent_oom_window() -> None:
    t = Telemetry()
    t.record_request(endpoint="x", policy="y", budget=128, tokens_in=1,
                     tokens_kept=1, prefill_ms=1, decode_ms=1, tok_per_sec=1,
                     vram_allocated_mb=1, oom=True)
    assert t.recent_oom_count(window_s=10) == 1
    assert t.recent_oom_count(window_s=0) == 0
