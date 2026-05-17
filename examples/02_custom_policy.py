"""Register a custom scoring policy."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import apply_kiaomni, register_policy


def my_sqrt_policy(saliency: np.ndarray) -> np.ndarray:
    """Square-root of saliency — flattens the distribution."""
    return np.sqrt(np.maximum(saliency, 0.0))


def main() -> None:
    register_policy("sqrt", my_sqrt_policy)

    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    apply_kiaomni(model, policy="sqrt", budget=256, verbose=True)

    prompt = "List five facts about " + ("topic " * 200)
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=48, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
