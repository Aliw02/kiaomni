# Streaming Policy V2 POC

## Decision

Promote **Persistent Streaming** only to the next Qwen2.5 smoke.

Do not promote temporal-landmark or block-coverage variants yet.

## Setup

- synthetic absolute context length: 512
- streaming budget: 64 entries (12.5%)
- chunk size: 32
- 50 deterministic trials per policy/regime
- identical peak cache: 96 entries
- identical final cache: 64 entries

Two regimes were separated deliberately.

### Early-observable, late-needed

The important record emits a strong saliency signal when it first appears, but the final query arrives much later.

| Policy | Mean important-position survival |
|---|---:|
| Current streaming | 8.5% |
| Persistent | **100%** |
| Persistent + landmarks | 100% |
| Persistent + blocks | 100% |

Persistent history solves the failure without using more KV memory.

### Future-only importance

The old token has no importance signal until the final query.

| Policy | Mean important-position survival |
|---|---:|
| Current streaming | 11.5% |
| Persistent | 11.0% |
| Persistent + landmarks | 12.5% |
| Persistent + blocks | 12.5% |

No method solves this regime. Under a 12.5% retention budget, an arbitrary old token with no earlier observable signal cannot be guaranteed to survive.

## Interpretation

The current Qwen streaming smoke likely fails if useful key records briefly receive attention when introduced and that evidence is later forgotten by the per-chunk selector. Persistent scoring is the minimal change that directly tests this hypothesis.

Promotion rule:

- Run Persistent Global and Persistent Layerwise on the exact failed multi-key 4K/8K smoke cases.
- Keep chunked-full-KV as the control.
- If Persistent does not materially improve answer correctness, stop this streaming line before a full benchmark.
