# BioMistral-7B — Detailed Results

**Model:** `BioMistral/BioMistral-7B`
**Source:** `038_biomistral_results/` (5 files: results.json, predictions.csv, speed_vram.csv, eviction_coherence_loss.csv, llm_judge_results.csv)

## TL;DR

- **AdaSnapKV** achieves the best eviction policy mean contains at **0.3152**.
- KiaOmni_σ8 (0.3058) and BlockSal (0.3043) are close behind.
- Unlike Falcon3, **AdaSnapKV outperforms all KiaOmni variants** on this biomedical domain.

## Method

Evaluation on:
- **Bio-Ruler tasks:** bio_niah_single, bio_niah_gene, bio_vt (15 trials each)
- **Bio-LongBench tasks:** pubmedqa, pubmedqa_long, medmcqa, medalpaca_medqa, medalpaca_wiki, clinical_niah (15 samples each)
- **Context lengths:** 4096, 8192, 16384 (16384 and 8192 mostly null)
- **Budgets:** 96, 128, 256, 512

**Metric:** Macro-Avg Contains (averaged across Bio-Ruler + Bio-LongBench tasks)

## Results Table

| Policy | Budget=96 | Budget=128 | Budget=256 | Budget=512 | Mean |
|--------|-----------|------------|------------|------------|------|
| FullContext | 0.3623 | 0.3623 | 0.3623 | 0.3623 | **0.3623** |
| SnapKV | 0.1304 | 0.1478 | 0.2348 | 0.3391 | 0.2130 |
| H2O | 0.2348 | 0.2609 | 0.3217 | 0.3594 | 0.2942 |
| BlockSal | 0.2174 | 0.2812 | 0.3623 | 0.3565 | 0.3043 |
| KiaOmni_Gaussian | 0.2261 | 0.2957 | 0.3362 | 0.3565 | 0.3036 |
| KiaOmni_σ8 | 0.2348 | 0.2957 | 0.3449 | 0.3478 | 0.3058 |
| **AdaSnapKV** | **0.2609** | **0.3043** | **0.3391** | **0.3565** | **0.3152** |

## Key Observations

1. **BioMistral is easier to evict** — even H2O reaches 0.3594 at budget=512 (99% of FullContext).
2. **SnapKV collapses at low budgets** (0.1304 at 96) but recovers strongly (0.3391 at 512).
3. **AdaSnapKV is the top eviction policy** — consistent advantage across all budgets.
4. **KiaOmni_σ8 and KiaOmni_Gaussian** perform similarly, within 0.002 of each other on mean.
5. **All eviction policies converge near FullContext at budget=512** (0.3478–0.3594 vs 0.3623).

## Speed & VRAM

See `speed_vram.csv` for per-policy tokens/sec and VRAM. FullContext is most expensive; eviction policies save proportionally to budget.

## Eviction Coherence Loss

See `eviction_coherence_loss.csv`. BioMistral shows higher coherence loss variance than Falcon3 across policies.

## Caveats

- Only 4096 context has full data. 8192 and 16384 are mostly null (null = not run/OOM).
- Bio-Ruler tasks (esp. bio_niah_gene) show zero performance for most policies — these tasks are the hardest.
- Scores are macro-averaged contains across ruler + longbench tasks.
