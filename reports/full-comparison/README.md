# L8 Master Comparison

## TL;DR
**FullContext** wins the mean column across all 4 models (x̄ = 1.442).

## Methodology
Aggregated from:
- **L1 — Qwen2.5-7B**: LongBench 11-task suite, `overall` score
- **L2 — Mistral-7B**: RULER macro F1 at B=256
- **L4 — Falcon3-7B**: Mean across budgets 96/128/256/512
- **L4 — BioMistral-7B**: Mean across budgets 96/128/256/512
The **Mean** column is the arithmetic average of the 4 model columns.

## Master Comparison Table
| Policy | Qwen Overall | Mistral F1 | Falcon3 Mean | BioMistral Mean | Mean |
|--------|-------------|-----------|-------------|----------------|------|
| FullContext | 4.695 | 0.352 | 0.357 | 0.362 | 1.442 |
| KiaOmni_Gaussian | 4.176 | 0.334 | 0.229 | 0.304 | 1.261 |
| KiaOmni_σ8 | 4.133 | 0.333 | 0.214 | 0.306 | 1.246 |
| BlockSal | 4.074 | 0.341 | 0.219 | 0.304 | 1.235 |
| AdaSnapKV | 3.092 | 0.257 | 0.190 | 0.315 | 0.964 |
| H2O | 2.940 | 0.229 | 0.179 | 0.294 | 0.911 |
| SnapKV | 2.563 | 0.131 | 0.139 | 0.213 | 0.762 |

## Master Heatmap
![Master Heatmap](plots/master_heatmap.png)
*Figure: Cross-model comparison heatmap. Rows sorted by descending mean. Color intensity reflects relative score.*

## Caveats
- **SnapKV** = faithful arXiv:2404.14469 implementation. **BlockSal** = our novel block-level baseline (paper §4).
- **Amber** is excluded from this comparison; results reported separately in the Amber section.
- Each model uses a different metric suite: LongBench overall (Qwen), RULER macro F1 (Mistral), passkey accuracy mean across budgets (Falcon3, BioMistral). Direct cross-model comparability is limited — the **Mean** column should be treated as an aggregated indicator, not a rigorous apples-to-apples benchmark.

## Reproduce
```bash
python reports/full-comparison/_build.py
```

## Full Data
See [`data/master_table.csv`](data/master_table.csv) for the canonical CSV.

### All Values (by model)
| Policy | Qwen Overall | Mistral F1 | Falcon3 Mean | BioMistral Mean |
|--------|-------------|-----------|-------------|----------------|
| FullContext | 4.695 | 0.352 | 0.357 | 0.362 |
| KiaOmni_Gaussian | 4.176 | 0.334 | 0.229 | 0.304 |
| KiaOmni_σ8 | 4.133 | 0.333 | 0.214 | 0.306 |
| BlockSal | 4.074 | 0.341 | 0.219 | 0.304 |
| AdaSnapKV | 3.092 | 0.257 | 0.190 | 0.315 |
| H2O | 2.940 | 0.229 | 0.179 | 0.294 |
| SnapKV | 2.563 | 0.131 | 0.139 | 0.213 |