"""Hugging Face Gate-0 runner for a real causal LM.

Run this only in an environment with torch + transformers and model access.
The first target should be a small full-attention model such as Qwen/Qwen2-0.5B-Instruct.

Important: arbitrary sparse position eviction can interact with RoPE/cache_position semantics.
This runner is intentionally a Gate-0 diagnostic, not a production cache implementation.
"""

from __future__ import annotations

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kv_gaussian_policy import kv_gaussian_smooth, kv_norm_proxy_score, select_kv_positions
from kv_tensor_cache import cache_nbytes, compact_legacy_kv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--prompt", default="The secret code is 7391. Remember it. What is the secret code?")
    parser.add_argument("--retention", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def greedy_decode(model, tokenizer, input_ids, cache, original_prompt_length, max_new_tokens):
    generated = []
    current = input_ids
    # RoPE/cache_position must follow the original absolute sequence, not compacted cache length.
    next_absolute_position = original_prompt_length
    attention_mask = torch.ones(
        (1, cache.get_seq_length() + 1), dtype=torch.long, device=input_ids.device
    )

    for _ in range(max_new_tokens):
        cache_position = torch.tensor([next_absolute_position], device=input_ids.device)
        out = model(
            input_ids=current,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        current = token
        next_absolute_position += 1
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=-1,
        )
        if tokenizer.eos_token_id is not None and int(token.item()) == tokenizer.eos_token_id:
            break

    return torch.cat(generated, dim=-1) if generated else input_ids[:, :0]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_length = encoded.input_ids.shape[-1]

    # Full prompt reaches prefill unchanged.
    with torch.no_grad():
        full_cache = DynamicCache(config=model.config)
        out = model(**encoded, past_key_values=full_cache, use_cache=True)

    legacy = full_cache.to_legacy_cache()
    before = cache_nbytes(legacy)
    raw = kv_norm_proxy_score(*legacy[0])
    scores = kv_gaussian_smooth(raw, args.sigma)
    budget = max(1, int(round(prompt_length * args.retention)))
    keep = select_kv_positions(scores, budget, n_sink=min(4, budget), recency=min(8, budget))
    compacted_legacy = compact_legacy_kv(legacy, keep)
    after = cache_nbytes(compacted_legacy)
    compacted_cache = DynamicCache.from_legacy_cache(compacted_legacy)

    first = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [first]
    if args.max_new_tokens > 1:
        tail = greedy_decode(
            model,
            tokenizer,
            first,
            compacted_cache,
            original_prompt_length=prompt_length,
            max_new_tokens=args.max_new_tokens - 1,
        )
        if tail.numel():
            generated.append(tail)

    ids = torch.cat(generated, dim=-1)
    print(json.dumps({
        "model": args.model,
        "prompt_tokens": prompt_length,
        "kept_kv_tokens": int(keep.numel()),
        "kv_bytes_before": before,
        "kv_bytes_after": after,
        "physical_reduction_pct": 100.0 * (1.0 - after / before),
        "output": tokenizer.decode(ids[0], skip_special_tokens=True),
        "warning": "Gate-0 diagnostic only; sparse eviction position semantics require confirmation.",
    }, indent=2))


if __name__ == "__main__":
    main()
