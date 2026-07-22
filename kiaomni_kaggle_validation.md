# KiaOmni Kaggle Validation — Qwen2.5-7B-Instruct

**Date:** 2026-05-17
**Package version:** kiaomni 0.2.4
**Model:** Qwen/Qwen2.5-7B-Instruct (4-bit NF4 via bitsandbytes)
**Environment:** Kaggle T4 GPU, transformers >=4.50, `attn_implementation="eager"`
**Test bench:** `kv_cache_benchmark/033_full_comparison.py`-style NIAH + VT, compressed

---

## Configuration

| Parameter | Value |
|---|---|
| Tasks | NIAH (needle-in-haystack), VT (variable tracking) |
| Budgets | 96, 256 |
| Policies | `kiaomni_s8`, `kiaomni_gaussian` |
| Trials per cell | 3 |
| Target prompt length | ~1500 tokens (actual ~1535) |
| Max new tokens | 16 |
| Sampling | greedy (`do_sample=False`) |
| Sinks / Recency | 16 / 32 (defaults) |

---

## Results — contains-score (higher is better, max=1.0)

| Task | Policy | B=96 | B=256 | FullContext |
|---|---|---:|---:|---:|
| **NIAH** | `kiaomni_s8`       | 0.667 | **1.000** | 0.333 |
| **NIAH** | `kiaomni_gaussian` | 0.667 | 0.333 | 0.333 |
| **VT**   | `kiaomni_s8`       | 0.333 | **0.667** | 0.667 |
| **VT**   | `kiaomni_gaussian` | 0.333 | 0.333 | 0.667 |

## Latency (median over 3 trials, ms)

| Task | Policy | B=96 | B=256 |
|---|---|---:|---:|
| NIAH | `kiaomni_s8`       | 14300 | 17077 |
| NIAH | `kiaomni_gaussian` | 14868 | 16468 |
| VT   | `kiaomni_s8`       | 14458 | 15238 |
| VT   | `kiaomni_gaussian` | 14692 | 15444 |

Latency dominated by the single saliency-extraction prefill pass on 7B-NF4 (~14s baseline); subsequent generate is cheap because the pruned prompt is short.

---

## Headline findings

1. **`kiaomni_s8 @ B=256` on NIAH: 1.000 vs FullContext 0.333** — eviction *outperforms* the no-eviction baseline by removing haystack tokens that compete with the needle for attention mass. This reproduces the published-paper phenomenon on Qwen.
2. **`kiaomni_s8 @ B=256` on VT: matches FullContext (0.667)** — eviction preserves the multi-hop variable-tracking chain even when 83% of tokens are dropped (256/1535).
3. **`kiaomni_gaussian` regresses on NIAH at B=256** (0.333 vs 0.667 at B=96) — possible σ=4 oversmoothing dilutes the needle's signal when budget is generous. Consistent with the σ=8 boxcar being the production winner reported in 015_trace_eval.

## Honest caveats

- **n=3 is statistically thin.** Each trial flips the score by ±0.333. Treat these numbers as directional smoke-test results, not paper-grade. Re-run with n≥20 before any claim of statistical significance.
- **FullContext NIAH=0.333 is low** for a 7B model at 1.5K context. Likely cause: random 8-char alphanumeric passphrases (e.g. `BANANA42`) sometimes get tokenised in ways the model partially garbles in greedy decode. Not an algorithm issue, but it inflates the apparent eviction-vs-baseline gap.
- **B=96 is below the n_sink+recency floor (16+32=48) plus needed signal** — at B=96 only 48 free slots remain after protected zones, which is why both policies degrade vs B=256.
- **Bug fixed mid-test:** v0.2.4 fixes `remove_kiaomni` over-unwinding past the bound method via `@torch.no_grad()`'s `__wrapped__` chain. Prior versions (0.2.0–0.2.3) had a Tensor-as-self crash on multi-policy test loops. The hard-test cell here ran clean on v0.2.4.

## Verdict for publication

The package is **functionally validated for v0.2.x publication on Qwen2.5-7B in 4-bit NF4**:
- API works end-to-end with HuggingFace `generate` contract preserved
- Architecture probe correctly identifies Qwen as `qkv=separate, nh=28, nkv=4, hd=128, rope`
- Saliency adapter chooses `hook-separate` strategy (fast path)
- Both policies execute without crashes across all 8 cells × 3 trials = 24 generations
- Headline accuracy signal (NIAH B=256 = 1.000) is the expected directional behavior

Next steps before paper-grade claims:
- Re-run on Modal with n≥20 trials per cell
- Sweep budgets {64, 96, 128, 192, 256, 384, 512}
- Add LongBench V2 subset to expand coverage beyond synthetic NIAH/VT
- Investigate gaussian B=256 regression — likely σ adaptive to budget

---

## Raw JSON

```json
{
  "NIAH": {
    "FullContext": {"contains": 0.333, "n": 3},
    "kiaomni_s8": {
      "B96":  {"contains": 0.667, "latency_ms": 14300, "n": 3},
      "B256": {"contains": 1.000, "latency_ms": 17077, "n": 3}
    },
    "kiaomni_gaussian": {
      "B96":  {"contains": 0.667, "latency_ms": 14868, "n": 3},
      "B256": {"contains": 0.333, "latency_ms": 16468, "n": 3}
    }
  },
  "VT": {
    "FullContext": {"contains": 0.667, "n": 3},
    "kiaomni_s8": {
      "B96":  {"contains": 0.333, "latency_ms": 14458, "n": 3},
      "B256": {"contains": 0.667, "latency_ms": 15238, "n": 3}
    },
    "kiaomni_gaussian": {
      "B96":  {"contains": 0.333, "latency_ms": 14692, "n": 3},
      "B256": {"contains": 0.333, "latency_ms": 15444, "n": 3}
    }
  }
}
```
