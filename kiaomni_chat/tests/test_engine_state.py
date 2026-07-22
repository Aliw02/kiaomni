"""Tests for engine state hygiene — no model load required."""
from __future__ import annotations

from kiaomni_chat.engine import KiaOmniEngine, get_engine


def test_engine_not_ready_by_default() -> None:
    eng = get_engine()
    assert not eng.is_ready()
    assert eng._active_policy == ""
    assert eng._active_budget == 0


def test_health_dict_before_ready() -> None:
    eng = get_engine()
    h = eng._health_dict()
    assert h["ready"] is False
    assert "model" in h
    assert "device" in h
    assert "vram" in h
    assert "uptime_s" in h


def test_engine_singleton() -> None:
    a = get_engine()
    b = get_engine()
    assert a is b


def test_vram_on_cpu() -> None:
    v = KiaOmniEngine._vram()
    assert "allocated_mb" in v
    assert "reserved_mb" in v
    assert "fragmentation_pct" in v
