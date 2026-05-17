"""Unit tests for ArchitectureProbe using hand-built nn.Module fakes.

These tests run in CI without downloading any real model — they
construct minimal modules whose structure mimics each supported
architecture family.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kiaomni.adapters import ArchitectureProbe, KiaomniConfigError


class _FakeConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _SeparateAttn(nn.Module):
    def __init__(self, hidden: int = 64, nh: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        # Marker for RoPE detection.
        self.rotary_emb = nn.Identity()


class _SeparateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SeparateAttn()
        self.mlp = nn.Linear(64, 64)


class _SeparateModel(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_SeparateBlock() for _ in range(n_layers)])

        self.model = _Inner()
        self.config = _FakeConfig(
            num_attention_heads=8,
            hidden_size=64,
            rope_theta=10000.0,
            _attn_implementation="eager",
        )


class _FusedConcatAttn(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.c_attn = nn.Linear(hidden, 3 * hidden, bias=False)


class _GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FusedConcatAttn()
        self.mlp = nn.Linear(64, 64)


class _GPT2LikeModel(nn.Module):
    def __init__(self):
        super().__init__()

        class _Trans(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.ModuleList([_GPT2Block() for _ in range(3)])
                self.wpe = nn.Embedding(1024, 64)

        self.transformer = _Trans()
        self.config = _FakeConfig(
            n_head=8,
            n_embd=64,
            _attn_implementation="eager",
        )


# ── tests ────────────────────────────────────────────────────────────────────


def test_probe_separate_llama_like():
    m = _SeparateModel()
    r = ArchitectureProbe.probe(m, force=True)
    assert r.layer_container_path == "model.layers"
    assert r.attn_module_name == "self_attn"
    assert r.qkv_pattern == "separate"
    assert r.q_module_name == "q_proj"
    assert r.num_attention_heads == 8
    assert r.hidden_size == 64
    assert r.head_dim == 8
    assert r.pos_encoding == "rope"
    assert r.confidence == "high"
    assert r.batch_dim_pos == 0


def test_probe_gpt2_like_fused_concat_and_learned_pe():
    m = _GPT2LikeModel()
    r = ArchitectureProbe.probe(m, force=True)
    assert r.layer_container_path == "transformer.h"
    assert r.attn_module_name == "attn"
    assert r.qkv_pattern == "fused_concat"
    assert r.fused_module_name == "c_attn"
    assert r.num_attention_heads == 8
    assert r.hidden_size == 64
    assert r.pos_encoding == "learned"
    assert r.confidence == "high"


def test_probe_rejects_sdpa_attn_implementation():
    m = _SeparateModel()
    m.config._attn_implementation = "sdpa"
    with pytest.raises(KiaomniConfigError, match="eager"):
        ArchitectureProbe.probe(m, force=True)


def test_probe_rejects_flash_attention_2():
    m = _SeparateModel()
    m.config._attn_implementation = "flash_attention_2"
    with pytest.raises(KiaomniConfigError, match="eager"):
        ArchitectureProbe.probe(m, force=True)


def test_probe_result_cached_on_model():
    m = _SeparateModel()
    r1 = ArchitectureProbe.probe(m)
    r2 = ArchitectureProbe.probe(m)
    assert r1 is r2  # same instance, came from cache
