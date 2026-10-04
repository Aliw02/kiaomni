# KiaOmni True KV-Cache Compression V1

## Purpose

This directory starts a clean experimental line for **true post-prefill KV-cache compression/eviction**.

The frozen Phase 03 MoE evidence remains immutable on:
`freeze/kiaomni-qwen3-moe-phase03-2026-10-04`
at commit:
`62fd80b8631d7aa9c69b46be1b2c5967f339b445`

## Scientific boundary

The existing KiaOmni Phase 03 experiments reduce/select **input context tokens before model execution**. They must not be described in this line as direct KV-cache compression.

This experiment changes the intervention point:

```text
full prompt
  -> full prefill
  -> materialized K/V cache
  -> KiaOmni KV retention policy
  -> physically reduced K/V tensors/cache
  -> decode from reduced cache
```

A run qualifies as true KV-cache compression only if:

1. The full prompt reaches prefill unchanged.
2. K/V states are materialized before KiaOmni selection.
3. Compression changes the stored K/V cache itself, not only attention masks or input IDs.
4. Decode consumes the reduced cache.
5. Actual cache bytes/tokens before and after compression are recorded.
6. Quality and systems metrics are compared against an uncompressed full-KV control.

## V1 controls

Every comparable run must include:

- `full_kv`: full prompt, full KV cache.
- `prompt_selection`: frozen/current KiaOmni pre-model selection behavior.
- `true_kv_kiaomni`: full prefill followed by physical KV eviction/compression.

No result may be promoted if model, dataset, generation settings, seed policy, or evaluation protocol differs across these controls without being explicitly reported.

## Initial implementation scope

Start in Hugging Face/PyTorch before any vLLM integration.

V1 should establish:

- cache API compatibility;
- exact K/V tensor/cache shapes;
- safe token-axis compaction;
- position/cache-position correctness;
- decode correctness after compaction;
- measured cache-memory reduction;
- quality retention;
- compression overhead.

Only after this gate passes should runtime-specific integration (for example vLLM) begin.

## Required metrics

At minimum record:

- input tokens;
- pre-compression KV tokens and bytes;
- post-compression KV tokens and bytes;
- physical KV reduction ratio;
- peak VRAM;
- TTFT/prefill time;
- decode tokens/s;
- compression-policy time;
- total latency;
- task accuracy/recall;
- gold-answer perplexity where valid;
- failures and incompatibilities.

For MoE models, retain the existing routing diagnostics where technically meaningful.

## Non-negotiable rules

- Do not modify frozen Phase 03 evidence.
- Do not relabel prompt compression as KV-cache compression.
- Do not silently change budgets or baselines.
- Preserve raw per-case outputs.
- Freeze protocol/config before looking at final benchmark outcomes.
- Failed runs remain evidence and are not deleted.
- Separate exploratory POC results from confirmatory results.
