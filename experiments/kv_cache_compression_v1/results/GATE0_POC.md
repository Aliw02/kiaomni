# Gate 0 POC Result

## Status

PASS for isolated plumbing only.

This POC demonstrates post-prefill physical K/V compaction in a minimal PyTorch causal-attention implementation with RoPE-style absolute positions.

## What was tested

- Full prefill is performed before compression.
- K and V tensors are physically reduced with index selection on the sequence axis.
- Absolute positions are retained separately from compacted cache length.
- Decode continues after compaction.
- Identity compaction (keeping all positions) is bitwise exact.
- A KV-specific Gaussian policy copy is isolated from the frozen prompt-selection implementation.
- Six synthetic NIAH plumbing cases cover multiple context lengths and budgets.

## Result

- Unit tests: 3/3 passed.
- Decode after physical compaction: 6/6 successful.
- Synthetic needle survival: 6/6.
- Physical KV reduction observed in every case.
- 64 -> 24 KV positions: 2.67x byte reduction.
- 96 -> 32 KV positions: 3.00x byte reduction.
- 128 -> 40 KV positions: 3.20x byte reduction.
- Absolute next-token position remains the original full-prefill position rather than the compacted cache length.

## Important limitation

This is not yet a model-quality result and must not be cited as evidence that KiaOmni preserves LLM quality under true KV compression.

The model is a custom tiny PyTorch attention module and the dataset is synthetic. The policy-input saliency at the synthetic needle is intentionally strengthened to isolate cache plumbing from saliency quality.

## Next gate

Implement the same separation of cache length vs absolute/cache position using a real Hugging Face decoder cache API, then run a real pretrained model on a small NIAH/passkey set before moving to Qwen3 MoE.
