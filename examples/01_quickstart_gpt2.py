"""GPT-2 — tests the fused-concat (c_attn) QKV path."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import apply_kiaomni

MODEL_ID = "gpt2"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="eager")

    probe = apply_kiaomni(model, policy="kiaomni_s8", budget=128, verbose=True)
    assert probe.qkv_pattern == "fused_concat", probe
    assert probe.pos_encoding == "learned", probe

    prompt = "Once upon a time " * 60
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=32, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
