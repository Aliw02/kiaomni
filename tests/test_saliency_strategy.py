"""Tests for SaliencyAdapter strategy selection.

Verifies the upfront safety routing: fused-interleaved + GQA must
auto-fall back to output_attentions instead of raising at hook time.
"""
from __future__ import annotations

from kiaomni.adapters.probe import ProbeResult
from kiaomni.adapters.saliency import SaliencyAdapter


def _make_probe(**overrides) -> ProbeResult:
    defaults = dict(
        layer_container_path="model.layers",
        attn_module_name="self_attn",
        qkv_pattern="separate",
        q_module_name="q_proj",
        k_module_name="k_proj",
        v_module_name="v_proj",
        fused_module_name=None,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=8,
        hidden_size=64,
        batch_dim_pos=0,
        pos_encoding="rope",
        confidence="high",
        attn_implementation="eager",
        detection_notes=(),
    )
    defaults.update(overrides)
    return ProbeResult(**defaults)


def test_separate_picks_hook_separate():
    a = SaliencyAdapter(_make_probe(qkv_pattern="separate"))
    assert a._strategy == "hook-separate"


def test_fused_concat_picks_hook_fused_concat():
    a = SaliencyAdapter(_make_probe(
        qkv_pattern="fused_concat",
        q_module_name=None, k_module_name=None, v_module_name=None,
        fused_module_name="c_attn",
    ))
    assert a._strategy == "hook-fused-concat"


def test_fused_interleaved_picks_hook_when_nh_eq_nkv():
    a = SaliencyAdapter(_make_probe(
        qkv_pattern="fused_interleaved",
        q_module_name=None, k_module_name=None, v_module_name=None,
        fused_module_name="query_key_value",
        num_attention_heads=8, num_key_value_heads=8,
    ))
    assert a._strategy == "hook-fused-interleaved"


def test_fused_interleaved_plus_gqa_auto_routes_to_fallback():
    """The fix for review issue #2 — must not raise, must reroute."""
    a = SaliencyAdapter(_make_probe(
        qkv_pattern="fused_interleaved",
        q_module_name=None, k_module_name=None, v_module_name=None,
        fused_module_name="query_key_value",
        num_attention_heads=8, num_key_value_heads=2,  # GQA: 8 Q heads, 2 KV heads
    ))
    assert a._strategy == "fallback-attentions"


def test_low_confidence_routes_to_fallback():
    a = SaliencyAdapter(_make_probe(qkv_pattern="unknown", confidence="low"))
    assert a._strategy == "fallback-attentions"


def test_unknown_pattern_routes_to_fallback():
    a = SaliencyAdapter(_make_probe(qkv_pattern="unknown"))
    assert a._strategy == "fallback-attentions"
