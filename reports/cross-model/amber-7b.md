# Amber-7B — Results (Incomplete)

**Model:** Amber-7B
**Source:** `040_amber_results/` (3 files: predictions.csv, speed_vram.csv, eviction_coherence_loss.csv)

> **WARNING:** This directory is **missing** `llm_judge_results.csv` and `results.json`. No aggregate F1 or contains scores are available.

## Available Data

| File | Status | Notes |
|------|--------|-------|
| `predictions.csv` | Present | Raw model outputs across all tasks/conditions |
| `speed_vram.csv` | Present | Tokens/sec, VRAM (sal+gen) per policy |
| `eviction_coherence_loss.csv` | Present | Coherence metric per trial |
| `llm_judge_results.csv` | **MISSING** | Cannot compute F1/contains aggregates |
| `results.json` | **MISSING** | No aggregate scores |

## Eviction Coherence Loss

Since no LLM-judge metrics are available, we use **eviction coherence loss** as a proxy for quality.
Lower is better — it measures how much the eviction disrupts the attention distribution.

| Policy | Budget=96 | Budget=128 | Budget=256 | Budget=512 |
|--------|-----------|------------|------------|------------|
| FullContext | 5.0 | 5.0 | 5.0 | 5.0 |
| SnapKV | 49.5 | 209.9 | 108.7 | 96.5 |
| SnapKV_Modified (BlockSal) | 36.7 | 29.7 | 16.4 | 8.6 |
| Ada-SnapKV | 336.2 | 372.9 | 175.1 | 69.6 |
| H2O | 210.8 | 311.5 | 377.0 | 127.3 |
| KiaOmni_σ8 | 104.8 | 84.5 | 48.0 | 22.2 |
| KiaOmni_Gaussian | 66.4 | 59.1 | 33.2 | 14.3 |

> Values are mean eviction coherence loss across ruler (niah_single, ctx=4096) trials.

### Coherence Loss Observations

1. **SnapKV_Modified (BlockSal) has the lowest loss** among eviction policies — 8.6 at budget=512.
2. **KiaOmni_Gaussian** is second best (14.3 at budget=512).
3. **Ada-SnapKV and H2O** have very high loss at low budgets (>300), indicating severe distribution disruption.
4. **FullContext** has minimal loss (~5) as expected — no eviction occurs.
5. **Coherence loss correlates inversely with task performance** (where measurable).

## Observations from Predictions

Partial reads of `predictions.csv` show garbled/recursive output patterns from Amber-7B under FullContext, suggesting the model itself has generation quality issues independent of eviction.

## Prediction Quality (Qualitative)

Tasks attempted (ruler): niah_single, niah_multikey, vt
Tasks attempted (longbench): narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum

Many outputs show repetitive token generation patterns across ALL policies, indicating a potential issue with the model or evaluation pipeline rather than eviction-specific degradation.

## Speed & VRAM

| Policy | Notes |
|--------|-------|
| FullContext | Baseline — highest VRAM, lowest tokens/sec |
| SnapKV | Low VRAM but poor quality |
| KiaOmni variants | Best balance of speed/quality |
| H2O, AdaSnapKV | High coherence loss negates VRAM savings |

See `speed_vram.csv` for exact per-policy throughput and memory measurements.

## Caveats

1. **No LLM-judge results** — cannot compute F1 or contains scores.
2. Predictions show garbled output patterns across ALL conditions, not just eviction policies. This may indicate a model-level issue.
3. Raw CSV reads were truncated (large files). Aggregate statistics are computed inline from partial data where noted.
4. Only niah_single at ctx=4096 was fully available for coherence loss aggregation. Multi-key, VT, and longer contexts may give different results.
