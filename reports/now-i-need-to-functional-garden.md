# Plan: 033 — Full 6-Policy Comparison Benchmark

## Context

All prior experiments (029–032) used only 4 policies: KiaOmni_σ8, SnapKV, H2O, StreamingLLM.
PyramidKV and AdaKV are fully implemented in `004_production_eval_algorithms.py` but have never
been benchmarked head-to-head against KiaOmni variants on live-model tasks. This experiment
provides the complete comparison table needed for the paper's Related Work section, with speed,
VRAM, PPL, and task-accuracy metrics — all in one reproducible script.

**Platform:** Kaggle T4×2 (32 GB VRAM total)
**Context lengths:** 4 096, 8 192, 16 384 tokens (16K needs Flash Attention 2)
**Tasks:** RULER (niah_single + niah_multikey + vt) + LongBench (qasper + hotpotqa + multifieldqa_en)
**Output file:** `033_full_comparison.py`
**Results dir:** `033_full_comparison_results/`

---

## Policies (6)

| # | Name | Saliency input | Selection strategy |
|---|------|---------------|-------------------|
| 1 | SnapKV | mean-over-heads, last layer | Block-mean, block_size=16 |
| 2 | H2O | mean-over-heads, last layer | Raw top-k pointwise |
| 3 | KiaOmni_σ8 | mean-over-heads, last layer | Boxcar σ=8 + log1p |
| 4 | KiaOmni_Adaptive | mean-over-heads, last layer | Entropy-driven σ, σ_max=64 |
| 5 | PyramidKV | layer-weighted aggregate (all layers) | Block-mean SnapKV on multi-layer signal |
| 6 | AdaKV | per-head last layer [n_heads, L] | Entropy budget/head → union of per-head SnapKV |

**StreamingLLM excluded** — already confirmed functionally broken on all tasks (score 1/10).

---

## Metrics (per trial)

| Metric | How measured |
|--------|-------------|
| Token F1 | `compute_metrics(pred, gt)["f1"]` |
| Exact Match | `compute_metrics(pred, gt)["em"]` |
| ROUGE-L | rouge_score library |
| Contains | substring match after normalization |
| **PPL** | `exp(model(ids_evicted, labels=ids_evicted).loss)` on evicted context |
| **Tokens/sec** | `new_tokens / (gen_end - gen_start)` via `time.perf_counter()` |
| **VRAM peak (saliency)** | `torch.cuda.max_memory_allocated()` reset before saliency pass |
| **VRAM peak (generation)** | `torch.cuda.max_memory_allocated()` reset before generate() |
| **Saliency time (ms)** | Time for single forward pass with hooks |

---

## Saliency Extraction — Single Forward Pass

All 6 policies are served by ONE forward pass per trial via three registered hooks:

```python
def extract_all_saliency(ids, model) -> dict:
    # Hook 1: last layer q_proj + k_proj → per-head scores [n_heads, L]
    # Hook 2: ALL layers q_proj + k_proj → layer-weighted aggregate [L]
    #   weight(l) = (L - l) / sum(L - l)   ← lower layers weighted more (PyramidKV insight)
    # Returns:
    #   sal_mean      : np.ndarray[L]      ← mean over heads, last layer
    #   sal_per_head  : np.ndarray[H, L]   ← per head, last layer (AdaKV)
    #   sal_multilayer: np.ndarray[L]      ← layer-weighted aggregate (PyramidKV)
```

This avoids 2–3 forward passes and keeps VRAM overhead identical for all policies.

---

## Policy Implementations

### KiaOmni_Adaptive (σ_max = 64, Qwen-calibrated from 015v3)
```python
def get_adaptive_sigma(sal, budget, seq_len):
    p = sal / (sal.sum() + 1e-12)
    H_norm = -np.sum(p * np.log(p + 1e-12)) / np.log(seq_len)
    peakiness = max(0.0, 1.0 - H_norm)
    return int(max(1, round(64 * peakiness * np.sqrt(budget / seq_len))))
```

### PyramidKV (pre-eviction approximation)
```python
def pyramidkv_keep(sal_multilayer, budget, seq_len):
    # Uses sal_multilayer (lower-layer weighted) instead of last-layer only
    # Selection: SnapKV block-mean on this richer signal
    return snapkv_keep(sal_multilayer, budget, seq_len)
```

### AdaKV (per-head union)
```python
def adakv_keep(sal_per_head, budget, seq_len):
    # Compute per-head entropy → budget(h) proportional to H(h)
    # For each head: SnapKV selection with budget(h), obs_window=32
    # Return union of all kept sets, pruned to budget by max-head-score
    # (matches AdaKVPolicy in 004_production_eval_algorithms.py:961-1070)
```

---

## Config (top of file, all tunable)

```python
MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
CTX_LENS    = [4096, 8192, 16384]
BUDGETS     = [128, 256, 512]
N_TRIALS    = 15          # RULER trials per cell
LB_SAMPLES  = 15          # LongBench samples per task
RULER_TASKS = ["niah_single", "niah_multikey", "vt"]
LB_TASKS    = ["qasper", "hotpotqa", "multifieldqa_en"]
N_SINK      = 16
RECENCY     = 32
BLOCK_SIZE  = 16
SIGMA_FIXED = 8
SIGMA_MAX   = 64          # Adaptive KiaOmni calibrated for Qwen
MAX_NEW     = 96
```

---

## File Structure

```
notebook/kv_cache_benchmark/
├── 033_full_comparison.py               ← NEW script
└── 033_full_comparison_results/
    ├── results.json                     ← aggregated per-task/ctx/policy/budget metrics
    ├── predictions.csv                  ← raw predictions + all metrics + judge columns
    ├── speed_vram.csv                   ← per-trial: policy, ctx, budget, tps, vram_sal_mb, vram_gen_mb, sal_ms
    ├── ppl.csv                          ← per-trial: policy, ctx, budget, ppl
    └── checkpoints/                     ← per-trial JSON for resume
```

---

## Critical Files Referenced

| File | Role |
|------|------|
| `032_ruler_eval.py` | Source: qk_saliency, kiaomni_keep, snapkv_keep, h2o_keep, task builders, checkpointing pattern |
| `031_longbench_eval.py` | Source: load_model (FA2 fallback), LongBench zip loader, compute_metrics |
| `004_production_eval_algorithms.py:961-1070` | AdaKV per-head entropy logic to port |
| `004_production_eval_algorithms.py:886-953` | PyramidKV layer-weight logic to adapt |

---

## Implementation Steps

1. **Copy base structure** from `032_ruler_eval.py` (checkpointing, CSV, main loop)
2. **Port model loader** from `031_longbench_eval.py:118-142` (FA2 → SDPA → eager fallback)
3. **Write `extract_all_saliency()`** — single forward pass with 3 hook types
4. **Port `adakv_keep()`** from `004:961-1070` — replace evict() interface with keep-index interface
5. **Write `pyramidkv_keep()`** — pass `sal_multilayer` to existing `snapkv_keep()`
6. **Write `kiaomni_adaptive_keep()`** — use σ_max=64 formula above
7. **Add `measure_ppl()`** — model(ids_evicted, labels=ids_evicted).loss → exp()
8. **Add speed/VRAM timing** — perf_counter + reset_peak_memory_stats per trial
9. **Port RULER task builders** from `032:284-336` (niah_single, niah_multikey, vt)
10. **Port LongBench loader** from `031_longbench_eval.py` (zip-based HF download)
11. **Write aggregation + CSV output** — results.json + predictions.csv + speed_vram.csv + ppl.csv

---

## Verification

1. Run with `CTX_LENS=[4096]`, `N_TRIALS=2`, `LB_SAMPLES=2` — confirm all 6 policies produce outputs and no OOM
2. Check `speed_vram.csv` has non-zero tps and vram values for every policy
3. Check `ppl.csv` — FullContext PPL should be lowest; StreamingLLM-equivalent (recency-only) should be highest
4. Confirm `results.json` macro_avg keys match all 6 policy names
5. Full run on T4×2: expected ~3–4 hours total
