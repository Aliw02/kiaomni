# Streaming KV Eviction POC — Gate S0

## Status

**PASS for mechanical correctness and peak-KV behavior.**

This POC does not make an LLM quality claim. It establishes the implementation invariants required before a real Qwen2.5-7B Kaggle comparison.

## Compared paths

- Full one-shot prefill
- Post-prefill Global KV compaction
- Post-prefill Layerwise KV compaction
- Streaming Global KV eviction
- Streaming Layerwise KV eviction

The existing post-prefill code is not modified. Streaming uses isolated files under `streaming_poc/`.

## Critical correctness invariant: chunking itself changes nothing

A 96-token prompt was processed both as:

1. one full prefill; and
2. six 16-token chunks with **no eviction**.

Results:

- Hidden-output max absolute difference: **0.0**
- K-cache max absolute difference: **0.0**
- V-cache max absolute difference: **0.0**
- Absolute-position metadata: **exact**

Therefore any later difference with streaming enabled is caused by eviction, not by chunked execution.

## Tests

**5/5 passed**

The tests cover:

- exact no-eviction chunk/full equivalence;
- reduced peak KV during streaming prefill;
- equal per-layer memory budget;
- decode continuation with original absolute positions;
- smaller chunks producing lower peak KV for the same budget.

## Peak-KV comparison

At equal final budgets, post-prefill compaction must first materialize the full cache. Streaming evicts after each chunk, so its transient cache approaches `budget + chunk_size` rather than the full context length.

| Context | Budget | Chunk | Post-prefill peak | Streaming peak | Streaming peak reduction | Final reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 24 | 16 | 65,536 B | 40,960 B | 1.60x | 2.67x |
| 96 | 32 | 16 | 98,304 B | 49,152 B | 2.00x | 3.00x |
| 128 | 40 | 16 | 131,072 B | 57,344 B | 2.29x | 3.20x |

All Global and Layerwise streaming runs continued decode successfully.

## Chunk-size tradeoff

For context 128 / budget 40:

| Chunk | Peak KV | Peak reduction vs full |
|---:|---:|---:|
| 8 | 49,152 B | 2.67x |
| 16 | 57,344 B | 2.29x |
| 32 | 73,728 B | 1.78x |

Smaller chunks lower transient KV memory, but a real model may pay more Python/kernel-launch and repeated-prefill overhead. Chunk size therefore must be measured rather than assumed.

## Absolute-axis Gaussian

After eviction, two adjacent cache entries may have been far apart in the original prompt. Streaming therefore does **not** run Gaussian smoothing over compact-cache indices as if they were original neighbors.

The POC projects current saliency back onto the original absolute token axis, smooths on that axis, then gathers scores for the surviving candidates.

## Important limitation: irreversible early eviction

The POC includes an adversarial condition where an old token becomes visibly important only in the final chunk.

Its survival rate was **0%** after it had already been evicted.

This is not an implementation bug. It is the defining risk of online eviction:

> a KV entry removed before future queries reveal its importance cannot be recovered.

A real benchmark must therefore compare streaming quality directly against post-prefill KV and FullKV. It must not assume that matching final KV bytes implies matching quality.

## Real-model promotion requirements

The Kaggle implementation must preserve these invariants:

1. Chunked Qwen prefill with eviction disabled must match one-shot prefill.
2. `position_ids` and `cache_position` remain absolute.
3. The custom chunk causal mask must allow all retained historical KV entries but only the causal prefix inside the new chunk.
4. Global and Layerwise receive identical per-layer KV budgets.
5. Sink/recency protection uses absolute positions.
6. Saliency uses only information available up to the current chunk; no future leakage.
7. Peak KV bytes are measured during prefill, not only after final compression.
8. Evicted KV entries are never silently reconstructed.
9. Strict first-answer scoring is used.
10. The same deterministic cases and budgets are used for FullKV, prompt selection, post-prefill KV, and streaming KV.

## Recommended Kaggle comparison

Do **not** repeat the already-running FullKV/post-prefill Final unnecessarily.

Use its raw deterministic cases as the reference and run a streaming supplement on the exact same:

- case IDs / seeds
- context lengths
- budgets
- Qwen2.5-7B model
- generation settings

Then combine:

- `full_kv`
- `prompt_selection`
- `kv_global`
- `kv_layerwise`
- `stream_global`
- `stream_layerwise`

into one strict paired analysis.
