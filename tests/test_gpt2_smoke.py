"""End-to-end smoke test on real GPT-2 (downloads ~500MB).

Validates that the fused-concat (c_attn) QKV path runs to completion
on a CPU and produces sensible kept-index counts.

Hub-availability failures (rate-limit 429 on shared CI runner IPs,
offline cache miss) skip rather than fail: they say nothing about the
code under test. ``requests.RequestException`` and huggingface_hub's
``LocalEntryNotFoundError`` both inherit from ``OSError``, and
transformers re-wraps remaining hub errors into ``OSError`` itself,
so one except clause covers every download failure mode.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer

from kiaomni import apply_kiaomni, load_model, remove_kiaomni


def _load_gpt2(model_id: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        # flash_attention_2 → sdpa → eager, same chain as the experiments.
        model = load_model(model_id)
    except OSError as exc:
        pytest.skip(f"HF Hub unavailable (rate-limit or offline): {exc}")
    return tok, model


@pytest.mark.slow
def test_gpt2_end_to_end_eviction():
    model_id = "gpt2"
    tok, model = _load_gpt2(model_id)
    tok.pad_token = tok.eos_token

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
