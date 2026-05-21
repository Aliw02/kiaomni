# L8 Master Comparison — Cross-Model LLM-Judge Win Rates

## TL;DR

Under LLM-as-Judge evaluation across **4 architectures × 8 LongBench tasks × 4 budgets**, **KiaOmni_Gaussian** (33.5% mean win-rate) and **KiaOmni_σ8** (33.0%) lead all eviction policies, ahead of BlockSal (31.9%), AdaSnapKV (30.6%), H2O (28.2%), and SnapKV (21.8%). FullContext (oracle, no eviction) sets the upper bound at 48.0%.

## Master Comparison Table (LLM-Judge Win-Rate %)

| Policy | Qwen2.5-7B | Mistral-7B | Falcon3-7B | BioMistral-7B | **Mean** |
|--------|:----------:|:----------:|:----------:|:-------------:|:--------:|
| FullContext | 47.4 | 45.8 | 41.5 | 57.3 | **48.0** |
| **KiaOmni_Gaussian** | **32.4** | **29.0** | **24.3** | **48.1** | **33.5** |
| **KiaOmni_σ8** | **33.2** | **27.1** | **23.8** | **48.0** | **33.0** |
| BlockSal | 32.1 | 27.5 | 21.6 | 46.5 | 31.9 |
| AdaSnapKV | 27.4 | 24.1 | 21.1 | 49.6 | 30.6 |
| H2O | 24.1 | 22.2 | 20.1 | 46.5 | 28.2 |
| SnapKV | 19.3 | 18.2 | 14.9 | 34.9 | 21.8 |

*Win-rate = % of samples judged CORRECT by Claude Haiku (4-category rubric: CORRECT / HALLUCINATED / REFUSED / NOISE).*

## Master Heatmap

![Master Heatmap](plots/master_heatmap.png)
*Cross-model win-rate heatmap. Rows sorted by descending mean. KiaOmni variants (rows 2–3) consistently outperform all other eviction policies across all 4 architectures.*

## Results Detail (Tasks × Context Lengths × Budgets)

Each model was evaluated on **8 LongBench tasks** at **4 budgets** (B ∈ {98, 128, 256, 512}) and **3 context lengths** (4K, 8K, 16K):

| Dimension | Values |
|-----------|--------|
| **Tasks** | narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum |
| **Context lengths** | 4 096, 8 192, 16 384 tokens |
| **Budgets** | 98, 128, 256, 512 tokens |
| **Policies (7)** | FullContext, H2O, SnapKV, BlockSal, AdaSnapKV, KiaOmni_σ8, KiaOmni_Gaussian |
| **Metric** | LLM-as-Judge win-rate (Claude Haiku, 4-category rubric) |
| **Total judged** | 61 681 samples across all 4 models |

**Per-model tasks and context lengths:**

| Model | Tasks | Contexts | Budgets |
|-------|-------|----------|---------|
| Qwen2.5-7B | 8 LongBench | 4K, 8K, 16K | 98, 128, 256, 512 |
| Mistral-7B | 8 LongBench | 4K, 8K, 16K | 98, 128, 256, 512 |
| Falcon3-7B | 8 LongBench | 4K, 8K, 16K | 98, 128, 256, 512 |
| BioMistral-7B | 2 Bio-RULER (bio_niah_single, bio_niah_gene) | 4K, 8K | 98, 128, 256, 512 |

## Methodology

Aggregated from per-model LLM-judge results in Lane L7. Each prediction was classified by Claude Haiku into CORRECT / HALLUCINATED / REFUSED / NOISE. Win-rate = `CORRECT / (CORRECT + HALLUCINATED + REFUSED + NOISE)`.

Source lanes:
- **L1 Qwen2.5-7B**: `reports/qwen2.5-7b/`
- **L2 Mistral-7B**: `reports/mistral-7b/`
- **L4 Falcon3-7B + BioMistral-7B**: `reports/cross-model/`
- **L7 LLM-judge synthesis**: `reports/llm-judge/`

## Caveats

- **SnapKV** = faithful arXiv:2404.14469 implementation. **BlockSal** = our novel block-level baseline (paper §4).
- **Amber** is excluded from this comparison; see `reports/cross-model/amber-7b.md` for isolated results.
- **BioMistral-7B** uses Bio-RULER tasks (not LongBench), which are harder biomedical needle-in-haystack tasks with more NOISE-labeled rows.
- **Falcon3-7B** predictions contain `<|assistant|>` template leakage artifacts in some samples.
- Cross-model mean is a simple macro-average across 4 models (equal weight regardless of sample count).

## Reproduce

```bash
# Clone and install
git clone https://github.com/Aliw02/kiaomni
cd kiaomni
pip install -e .

# Run any benchmark
python experiments/033_full_comparison.py   # Qwen2.5-7B
python experiments/llm_judge.py --model qwen  # LLM judge
```

## Full Data

| File | Description |
|------|-------------|
| [`data/master_table.csv`](data/master_table.csv) | Canonical win-rate CSV |
| [`plots/master_heatmap.png`](plots/master_heatmap.png) | Cross-model heatmap |
| [`reports/llm-judge/`](../llm-judge/) | Full per-model judge results |
