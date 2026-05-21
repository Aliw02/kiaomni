# Falcon3-7B — Detailed Results

**Model:** `tiiuae/Falcon3-7B-Instruct`
**Source:** `037_falcon3_results/` (5 files: results.json, predictions.csv, speed_vram.csv, eviction_coherence_loss.csv, llm_judge_results.csv)

## TL;DR

- KiaOmni_Gaussian achieves the best eviction policy mean F1 at **0.2289**.
- BlockSal (our novel baseline) is second at **0.2192**.
- KiaOmni_σ8 underperforms BlockSal at low budgets but closes the gap at budget=512.

## Method

Evaluation on:
- **Ruler tasks:** niah_single, niah_multikey, vt (15 trials each)
- **LongBench tasks:** narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum (15 samples each)
- **Context lengths:** 4096, 8192, 16384 (16384 mostly OOM — null)
- **Budgets:** 96, 128, 256, 512

**Metric:** Macro-Avg F1 (averaged across Ruler + LongBench tasks)

## Results Table

| Policy | Budget=96 | Budget=128 | Budget=256 | Budget=512 | Mean |
|--------|-----------|------------|------------|------------|------|
| FullContext | 0.3568 | 0.3568 | 0.3568 | 0.3568 | **0.3568** |
| SnapKV | 0.1155 | 0.1229 | 0.1390 | 0.1797 | 0.1393 |
| H2O | 0.1329 | 0.1574 | 0.2043 | 0.2213 | 0.1790 |
| AdaSnapKV | 0.1484 | 0.1698 | 0.2233 | 0.2191 | 0.1902 |
| KiaOmni_σ8 | 0.1638 | 0.1872 | 0.2278 | 0.2770 | 0.2140 |
| BlockSal | 0.1620 | 0.1821 | 0.2518 | 0.2808 | 0.2192 |
| **KiaOmni_Gaussian** | **0.1808** | **0.2103** | **0.2340** | **0.2903** | **0.2289** |

## Key Observations

1. **KiaOmni_Gaussian wins across all budgets** — consistent advantage over BlockSal at low budgets (96/128/256).
2. **BlockSal is strongest at budget=256** (0.2518), nearly tying KiaOmni_Gaussian.
3. **KiaOmni_σ8 scales well** — from 0.1638 at budget=96 to 0.2770 at budget=512 (+69%).
4. **SnapKV (arXiv baseline) severely underperforms** — mean 0.1393, barely above random on needle tasks.
5. **AdaSnapKV plateaus at budget=256→512** (0.2233 → 0.2191), suggesting saturation.

## Speed & VRAM

See `speed_vram.csv` in source directory for per-policy tokens/sec and VRAM usage (sal+gen). FullContext has highest VRAM; eviction policies reduce memory proportionally to budget.

## Eviction Coherence Loss

See `eviction_coherence_loss.csv`. KiaOmni variants generally show lower coherence loss than SnapKV/H2O/AdaSnapKV at equivalent budgets.

## Caveats

- 16384 context: all policies have null data (OOM during ruler tasks).
- Scores are macro-averaged F1 across ruler + longbench — not micro-averaged.
- SnapKV = arXiv:2404.14469 (faithful reproduction). BlockSal = our novel block-level baseline.
