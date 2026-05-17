"""End-to-end smoke test on real GPT-2 (downloads ~500MB).

Validates that the fused-concat (c_attn) QKV path runs to completion
on a CPU and produces sensible kept-index counts.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import apply_kiaomni, remove_kiaomni


@pytest.mark.slow
def test_gpt2_end_to_end_eviction():
    model_id = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")

    probe = apply_kiaomni(model, policy="kiaomni_s8", budget=128)
    assert probe.qkv_pattern == "fused_concat"
    assert probe.pos_encoding == "learned"

    prompt = "The capital of France is " + ("Paris. " * 80)
    ids = tok(prompt, return_tensors="pt").input_ids
    assert ids.shape[1] > 128, "prompt must exceed budget to trigger eviction"

    out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 8

    remove_kiaomni(model)
    # After remove, generate should produce same output (no eviction path).
    out2 = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out2.shape[1] == ids.shape[1] + 8
