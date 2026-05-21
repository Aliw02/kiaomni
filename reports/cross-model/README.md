# Cross-Architecture Evaluation: KiaOmni on Falcon3-7B, BioMistral-7B, and Amber-7B

> **Lane L4** of the KiaOmni publication plan. Whitelist-only policies (§4b).

## TL;DR

- **KiaOmni_Gaussian** leads eviction policies on **Falcon3-7B** (mean F1 = 0.2289, +5.1% vs BlockSal).
- **AdaSnapKV** leads on **BioMistral-7B** (mean contains = 0.3152, +3.6% vs BlockSal).
- **Amber-7B** has no LLM-judge results; only eviction coherence loss + speed/VRAM available.
- All eviction policies fall well below FullContext upper bound (0.3568 F1 / 0.3623 contains).

## Models Evaluated

| Model | Architecture | Params | Domain | Source |
|-------|-------------|--------|--------|--------|
| Falcon3-7B | Decoder-only (GeLU, RoPE) | 7B | General | `037_falcon3_results/` |
| BioMistral-7B | Decoder-only (Mistral-based) | 7B | Biomedical | `038_biomistral_results/` |
| Amber-7B | Decoder-only | 7B | General | `040_amber_results/` |

## Metrics

| Model | Primary Metric | Tasks |
|-------|---------------|-------|
| Falcon3-7B | Macro-Avg F1 | Ruler (niah_single, niah_multikey, vt) + LongBench (8 tasks) |
| BioMistral-7B | Macro-Avg Contains | Bio-Ruler (bio_niah_single, bio_niah_gene, bio_vt) + Bio-LongBench (6 tasks) |
| Amber-7B | Eviction Coherence Loss (proxy) | Ruler + LongBench; **no LLM-judge available** |

## Experimental Setup

- **Context lengths:** 4096, 8192, 16384
- **Budgets:** 96, 128, 256, 512 (tokens)
- **Ruler trials:** 15 per condition
- **LongBench samples:** 15 per task
- **Hardware:** NVIDIA GPU (CUDA)

## Policy Naming Convention

| Source Label | Published Label | Description |
|-------------|----------------|-------------|
| `FullContext` | FullContext | No eviction (upper bound) |
| `RealSnapKV` | SnapKV | Faithful implementation of arXiv:2404.14469 |
| `SnapKV_Modified` | BlockSal | Our novel block-level baseline (blocks instead of tokens) |
| `Ada-SnapKV` | AdaSnapKV | Adaptive SnapKV variant |
| `H2O` | H2O | Heavy Hitter Oracle baseline |
| `KiaOmni_σ8` | KiaOmni_σ8 | KiaOmni with sigma=8 scoring |
| `KiaOmni_Gaussian` | KiaOmni_Gaussian | KiaOmni with Gaussian-weighted scoring |

> **Critical:** SnapKV = faithful arXiv:2404.14469. BlockSal = our novel block-level baseline.

## Headline Results

### Falcon3-7B — Macro-Avg F1 (mean across budgets 96–512)

| Policy | Budget=96 | Budget=128 | Budget=256 | Budget=512 | Mean |
|--------|-----------|------------|------------|------------|------|
| FullContext | 0.3568 | 0.3568 | 0.3568 | 0.3568 | **0.3568** |
| SnapKV | 0.1155 | 0.1229 | 0.1390 | 0.1797 | 0.1393 |
| H2O | 0.1329 | 0.1574 | 0.2043 | 0.2213 | 0.1790 |
| AdaSnapKV | 0.1484 | 0.1698 | 0.2233 | 0.2191 | 0.1902 |
| BlockSal | 0.1620 | 0.1821 | 0.2518 | 0.2808 | 0.2192 |
| KiaOmni_σ8 | 0.1638 | 0.1872 | 0.2278 | 0.2770 | 0.2140 |
| **KiaOmni_Gaussian** | **0.1808** | **0.2103** | **0.2340** | **0.2903** | **0.2289** |

### BioMistral-7B — Macro-Avg Contains (mean across budgets 96–512)

| Policy | Budget=96 | Budget=128 | Budget=256 | Budget=512 | Mean |
|--------|-----------|------------|------------|------------|------|
| FullContext | 0.3623 | 0.3623 | 0.3623 | 0.3623 | **0.3623** |
| SnapKV | 0.1304 | 0.1478 | 0.2348 | 0.3391 | 0.2130 |
| H2O | 0.2348 | 0.2609 | 0.3217 | 0.3594 | 0.2942 |
| BlockSal | 0.2174 | 0.2812 | 0.3623 | 0.3565 | 0.3043 |
| KiaOmni_σ8 | 0.2348 | 0.2957 | 0.3449 | 0.3478 | 0.3058 |
| KiaOmni_Gaussian | 0.2261 | 0.2957 | 0.3362 | 0.3565 | 0.3036 |
| **AdaSnapKV** | **0.2609** | **0.3043** | **0.3391** | **0.3565** | **0.3152** |

### Amber-7B

**No LLM-judge results available.** The directory `040_amber_results/` contains only:
- `predictions.csv` (raw generations)
- `speed_vram.csv` (throughput + memory)
- `eviction_coherence_loss.csv` (coherence metric)

See [`amber-7b.md`](amber-7b.md) for eviction coherence loss and speed/VRAM analysis.

## Figures

| Figure | File |
|--------|------|
| Cross-architecture bar chart (mean scores) | [`plots/cross_arch_comparison.png`](plots/cross_arch_comparison.png) |
| Falcon3-7B F1 vs Budget | [`plots/falcon3_scores.png`](plots/falcon3_scores.png) |
| BioMistral-7B Contains vs Budget | [`plots/biomistral_scores.png`](plots/biomistral_scores.png) |
| Amber-7B Coherence Loss vs Budget | [`plots/amber_coherence.png`](plots/amber_coherence.png) |

## Caveats

1. **Amber-7B missing results.json & llm_judge_results.csv** — no aggregate F1/contains scores available. Only raw predictions and eviction coherence loss are present.
2. **All file reads from source directories were truncated** (large CSVs). Aggregate scores from `results.json` are authoritative where available.
3. **SnapKV = faithful arXiv:2404.14469.** BlockSal = our novel block-level baseline.
4. **Context length 16384** has null data for most policies across all models (OOM/not run). All reported scores are from 4096/8192 context.

## Full Data

| File | Contents |
|------|----------|
| [`data/falcon3_final_scores.csv`](data/falcon3_final_scores.csv) | Falcon3 macro_avg_f1 per policy per budget |
| [`data/biomistral_final_scores.csv`](data/biomistral_final_scores.csv) | BioMistral macro_avg_contains per policy per budget |
| [`data/amber_final_scores.csv`](data/amber_final_scores.csv) | Amber placeholder (no aggregate scores) |
| [`provenance.json`](provenance.json) | Source→number mapping for every reported value |

## Reproduce

```bash
# View raw aggregate scores
cat notebook/kv_cache_benchmark/037_falcon3_results/results.json
cat notebook/kv_cache_benchmark/038_biomistral_results/results.json
ls notebook/kv_cache_benchmark/040_amber_results/
```

## Per-Model Reports

- [Falcon3-7B](falcon3-7b.md)
- [BioMistral-7B](biomistral-7b.md)
- [Amber-7B](amber-7b.md)
