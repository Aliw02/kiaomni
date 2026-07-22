"""Pydantic schemas for the KiaOmni web chat API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PolicyName = Literal["fullcontext", "kiaomni_s8", "kiaomni_gaussian", "snapkv"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    policy: PolicyName = "kiaomni_gaussian"
    budget: int = Field(default=512, ge=64, le=4096)
    # Cap is 8192 so the model can fully answer multi-part questions; the
    # model still stops at EOS naturally, so this is just a safety ceiling.
    max_new_tokens: int = Field(default=2048, ge=8, le=8192)
    session_id: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class CompareRequest(BaseModel):
    messages: list[ChatMessage]
    budget: int = Field(default=512, ge=64, le=4096)
    max_new_tokens: int = Field(default=2048, ge=8, le=8192)
    policies: list[PolicyName] = Field(
        default_factory=lambda: ["fullcontext", "kiaomni_s8", "kiaomni_gaussian", "snapkv"]
    )


class CompareTurnRequest(BaseModel):
    """Multi-turn compare: send the full conversation history + a new user
    turn; server runs all 4 policies on the same history and returns 4
    replies in one response."""
    history: list[ChatMessage]
    budget: int = Field(default=512, ge=64, le=4096)
    max_new_tokens: int = Field(default=2048, ge=8, le=8192)
    policies: list[PolicyName] = Field(
        default_factory=lambda: ["fullcontext", "kiaomni_s8", "kiaomni_gaussian", "snapkv"]
    )


class DemoRunRequest(BaseModel):
    task: Literal["single", "multi", "reason", "summary", "all"] = "single"
    policy: PolicyName = "kiaomni_gaussian"
    budget: int = Field(default=512, ge=64, le=4096)
    n_samples: int = Field(default=3, ge=1, le=20)


class DocQARequest(BaseModel):
    document: str
    questions: list[str] = Field(min_length=1, max_length=10)
    policy: PolicyName = "kiaomni_gaussian"
    budget: int = Field(default=512, ge=64, le=4096)
    max_new_tokens: int = Field(default=2048, ge=8, le=8192)


class SessionCreateRequest(BaseModel):
    system_prompt: str | None = None


class SessionAppendRequest(BaseModel):
    session_id: str
    role: Literal["user", "assistant"]
    content: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    tokens: int
    created_at: float
    last_access: float


class HealthResponse(BaseModel):
    ready: bool
    model: str
    gpu: str
    vram_allocated_mb: float
    vram_reserved_mb: float
    kiaomni_version: str
    uptime_s: float
    context_window: int = 0       # model's max context, in tokens
    default_budget: int = 0      # a sane default for the budget slider


class StatsBlock(BaseModel):
    tokens_in: int
    tokens_kept: int
    compression_ratio: float
    prefill_ms: float
    decode_ms: float
    tok_per_sec: float
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_max_allocated_mb: float
    fragmentation_pct: float
    keep_indices: list[int] = Field(default_factory=list)
