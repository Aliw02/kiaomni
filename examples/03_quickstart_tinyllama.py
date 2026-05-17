"""TinyLlama quickstart — open / no HuggingFace auth required.

Note: TinyLlama is an independent community model, NOT Meta's gated
Llama-3 / Llama-3.1. Anyone can download it without accepting a license.
For Meta Llama, you must first request access at
https://huggingface.co/meta-llama and authenticate with `huggingface-cli login`.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import apply_kiaomni

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # public, no auth


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    probe = apply_kiaomni(model, policy="kiaomni_s8", budget=256, verbose=True)
    print("Probed:", probe)

    prompt = "Summarise the following long document:\n" + ("foo bar " * 300) + "\n\nSummary:"
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=64, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
