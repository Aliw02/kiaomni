# V1 Protocol

## Question

Can the KiaOmni retention policy be moved from pre-model prompt selection to **post-prefill physical KV-cache retention** while preserving useful model quality and reducing the actual stored KV cache?

## Gate 0 — Cache correctness

Before benchmarking quality:

1. Run full prefill with `use_cache=True`.
2. Inspect the model's native cache representation.
3. Measure K/V token count and allocated bytes.
4. Apply a deterministic retention mask to the cache token axis.
5. Continue autoregressive decode from the compacted cache.
6. Verify no shape, cache-position, RoPE, causal-mask, GQA/MQA, or generation-state corruption.

A simple deterministic policy is allowed only for Gate 0 plumbing validation. It is not a KiaOmni result.

## Gate 1 — KiaOmni policy transfer

Adapt the existing frozen selection/ranking policy to emit KV-position indices without truncating the input prompt.

The prompt must remain identical to `full_kv` through prefill.

## Gate 2 — Three-way controlled evaluation

Compare:

- full KV;
- existing prompt-selection KiaOmni;
- true KV KiaOmni.

Use matched examples and generation configuration.

Primary systems claim requires measured physical cache reduction. Primary quality claim requires paired evaluation against full KV.

## Gate 3 — MoE confirmation

After dense/small-model correctness is established, repeat on the current Qwen3 MoE line using the same percentage budgets already retained in Phase 03 where feasible.

Do not infer success on MoE from dense-model success.

## Promotion criterion

V1 is promoted only when the true-KV path:

- demonstrably reduces stored KV state;
- decodes correctly;
- has reproducible paired quality measurements;
- reports compression overhead;
- does not depend on pre-prefill prompt truncation.

Otherwise record the failure mode and keep the existing Phase 03 result classified as prompt/context compression.
