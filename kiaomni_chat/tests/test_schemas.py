"""Test Pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kiaomni_chat.schemas import (
    ChatMessage, ChatRequest, CompareRequest, DemoRunRequest, DocQARequest,
    HealthResponse, SessionAppendRequest, SessionCreateRequest, StatsBlock,
)


def test_chat_request_defaults() -> None:
    r = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert r.policy == "kiaomni_gaussian"
    assert r.budget == 512
    assert r.max_new_tokens == 2048   # raised from 256 so the model can answer multi-part questions
    assert r.temperature == 0.0


def test_chat_request_policy_literal() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="x")],
            policy="not_a_real_policy",
        )


def test_chat_request_budget_bounds() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="x")],
            budget=16,  # below min
        )
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="x")],
            budget=9999,  # above max
        )


def test_compare_request() -> None:
    r = CompareRequest(messages=[ChatMessage(role="user", content="hi")])
    assert "fullcontext" in r.policies
    assert "kiaomni_s8" in r.policies
    assert "kiaomni_gaussian" in r.policies


def test_demo_request_task_literal() -> None:
    with pytest.raises(ValidationError):
        DemoRunRequest(task="invalid")


def test_docqa_request_min_questions() -> None:
    with pytest.raises(ValidationError):
        DocQARequest(document="x", questions=[])


def test_health_response() -> None:
    h = HealthResponse(ready=True, model="x", gpu="cuda:0",
                       vram_allocated_mb=100.0, vram_reserved_mb=200.0,
                       kiaomni_version="0.3.0", uptime_s=10.0,
                       context_window=131072, default_budget=512)
    assert h.ready is True
    assert h.context_window == 131072


def test_stats_block_defaults() -> None:
    s = StatsBlock(tokens_in=100, tokens_kept=20, compression_ratio=0.2,
                   prefill_ms=10.0, decode_ms=20.0, tok_per_sec=5.0,
                   vram_allocated_mb=100.0, vram_reserved_mb=200.0,
                   vram_max_allocated_mb=300.0, fragmentation_pct=50.0)
    assert s.compression_ratio == 0.2
    assert s.keep_indices == []


def test_stats_block_keep_indices() -> None:
    s = StatsBlock(tokens_in=100, tokens_kept=20, compression_ratio=0.2,
                   prefill_ms=10.0, decode_ms=20.0, tok_per_sec=5.0,
                   vram_allocated_mb=100.0, vram_reserved_mb=200.0,
                   vram_max_allocated_mb=300.0, fragmentation_pct=50.0,
                   keep_indices=[0, 1, 2, 3, 4])
    assert s.keep_indices == [0, 1, 2, 3, 4]
