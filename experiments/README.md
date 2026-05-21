# KV-Cache Benchmark Experiments

Canonical benchmark scripts for evaluating KV-cache eviction policies (KiaOmni, SnapKV, H2O, etc.) across multiple LLMs, context lengths, and compression budgets.

## Script Index

| #  | Script | Model | Tasks | Budgets | Contexts |
|----|--------|-------|-------|---------|----------|
| 033 | [`033_full_comparison.py`](033_full_comparison.py) | Qwen2.5-7B-Instruct | RULER + LongBench | 98, 128, 256, 512 | 4K, 8K, 16K |
| 034 | [`034_mistral_benchmark.py`](034_mistral_benchmark.py) | Mistral-7B-v0.3 | RULER + LongBench | 96, 128, 256, 512 | 4K, 8K, 16K |
| 035a | [`035_niah_heatmap.py`](035_niah_heatmap.py) | configurable | NIAH depth grid | 128, 256 | 4K–32K |
| 035b | [`035_ppl_wikitext2.py`](035_ppl_wikitext2.py) | configurable | WikiText-2 PPL | 128, 256 | 4K, 8K |
| 035c | [`035_passkey_retrieval.py`](035_passkey_retrieval.py) | configurable | Passkey retrieval | 128, 256 | 4K–32K |
| 037 | [`037_falcon3_benchmark.py`](037_falcon3_benchmark.py) | Falcon3-7B-Instruct | RULER + LongBench | 96, 128, 256, 512 | 4K, 8K, 16K |
| 038 | [`038_biomistral_benchmark.py`](038_biomistral_benchmark.py) | BioMistral-7B | Bio-RULER + Bio-LongBench | 96, 128, 256, 512 | 4K, 8K |
| 039 | [`039_swap_experiment.py`](039_swap_experiment.py) | Qwen2.5-7B-Instruct | RULER swap test | 98, 128 | 4K |
| 040 | [`040_amber_benchmark.py`](040_amber_benchmark.py) | LLM360/Amber | RULER + LongBench | 98, 128, 256, 512 | 512, 1K, 2K |
| — | [`llm_judge.py`](llm_judge.py) | — | LLM-as-Judge scorer | — | — |

## Quick Start

```bash
# Install core dependencies
pip install -U -q transformers bitsandbytes accelerate rouge-score scipy

# Run a benchmark (results go to experiments/results/<name>_results/)
python experiments/033_full_comparison.py

# Run LLM-as-Judge on predictions
export LIGHTNING_API_KEY="your-key-here"
python experiments/llm_judge.py
python experiments/llm_judge.py --model qwen
```

## Output Structure

```
experiments/results/
├── 033_full_comparison_results/
│   ├── results.json
│   ├── predictions.csv
│   ├── speed_vram.csv
│   ├── eviction_coherence_loss.csv
│   └── checkpoints/
├── 034_mistral_results/
├── 035_heatmap_results/
├── 035_ppl_results/
├── 035_passkey_results/
├── 037_falcon3_results/
├── 038_biomistral_results/
├── 039_swap_results/
└── 040_amber_results/
```

## Policies Evaluated

Each benchmark script inlines its own policy implementations (not importing from `kiaomni.policies`) to remain fully self-contained. Common policies:

- **FullContext** — no eviction (baseline upper bound)
- **SnapKV_Modified** — observability-window page scoring
- **RealSnapKV** — per-head max-pooling with budget trim
- **H2O** — Heavy Hitter + recent tokens
- **KiaOmni_σ8** — boxcar-smoothed log1p saliency (σ=8)
- **KiaOmni_Adaptive** — entropy-driven adaptive σ
- **KiaOmni_RatioAdaptive** — compression-ratio adaptive σ
- **KiaOmni_Quest** — Quest-style max-filter envelope (window=2σ+1)
- **KiaOmni_Gaussian** — Gaussian-smoothed saliency
- **KiaOmni_AnchorExp** — anchor-expand clustering
- **KiaOmni_Scissorhands** — per-layer saliency mixture

## Metrics

- **Token-F1** — macro-averaged token overlap
- **EM** — exact match (normalised)
- **ROUGE-L** — longest common subsequence F1
- **Contains** — ground truth substring containment
- **tokens/sec** — generation throughput
- **VRAM peak** — peak CUDA memory (saliency + generation)
- **Saliency latency** — time for single-forward saliency extraction
- **Eviction coherence loss** — PPL on the evicted subset

## Notes

- All paths are relative (`./results/...`). Override with `OUT_DIR` env var.
- Context lengths use conservative defaults to fit common GPU VRAM (T4 16 GB, L4 24 GB).
- Scripts are resumable: per-trial checkpoints in `checkpoints/` directory.
- 4-bit NF4 quantisation via `bitsandbytes` for all large models.
- LongBench data auto-downloaded from HuggingFace (`THUDM/LongBench`).
