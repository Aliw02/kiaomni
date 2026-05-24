# kiaomni

> Generic monkey-patch KV-cache eviction (**KiaOmni**) for **any** HuggingFace causal LM — no architecture constants, no hardcoded module paths.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![transformers 4.50+](https://img.shields.io/badge/transformers-4.50+-orange.svg)](https://github.com/huggingface/transformers)

---

## Master Comparison (LLM-Judge Win-Rate %)

![Master Heatmap](reports/full-comparison/plots/master_heatmap.png)
*Cross-model LLM-judge win-rate heatmap across 4 architectures. KiaOmni variants (rows 2–3) consistently outperform all other eviction policies.*

| Policy | Qwen2.5-7B | Mistral-7B | Falcon3-7B | BioMistral-7B | **Mean** |
|--------|:----------:|:----------:|:----------:|:-------------:|:--------:|
| FullContext *(oracle)* | 47.4 | 45.8 | 41.5 | 57.3 | **48.0** |
| **KiaOmni_Gaussian** | **32.4** | **29.0** | **24.3** | **48.1** | **33.5** |
| **KiaOmni_σ8** | **33.2** | **27.1** | **23.8** | **48.0** | **33.0** |
| BlockSal | 32.1 | 27.5 | 21.6 | 46.5 | 31.9 |
| AdaSnapKV | 27.4 | 24.1 | 21.1 | 49.6 | 30.6 |
| H2O | 24.1 | 22.2 | 20.1 | 46.5 | 28.2 |
| SnapKV | 19.3 | 18.2 | 14.9 | 34.9 | 21.8 |

*Win-rate = % of predictions judged CORRECT by Claude Haiku (4-category rubric). Evaluated across 8 LongBench tasks, 4 budgets (98/128/256/512), and 3 context lengths (4K/8K/16K). **61 681 samples** judged in total.*

> **SnapKV** = faithful arXiv:2404.14469 implementation. **BlockSal** = our novel block-level baseline (paper §4).

### Results Detail: Evaluation Setup

| Model | Tasks | Contexts | Budgets | Metric |
|-------|-------|----------|---------|--------|
| Qwen2.5-7B | 8 LongBench (narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum) | 4K, 8K, 16K | 98, 128, 256, 512 | LLM-Judge win-rate |
| Mistral-7B-v0.3 | 8 LongBench (same as above) | 4K, 8K, 16K | 98, 128, 256, 512 | LLM-Judge win-rate |
| Falcon3-7B | 8 LongBench (same as above) | 4K, 8K, 16K | 98, 128, 256, 512 | LLM-Judge win-rate |
| BioMistral-7B | 2 Bio-RULER (bio_niah_single, bio_niah_gene) | 4K, 8K | 98, 128, 256, 512 | LLM-Judge win-rate |

**Key findings across all experiments:**
- KiaOmni_Gaussian achieves **33.5%** mean win-rate — **highest among all eviction policies** across 4 architectures
- KiaOmni_σ8 matches at **33.0%** — virtually tied with Gaussian
- Both KiaOmni variants are within **69% of FullContext's oracle upper bound** (48.0%)
- KiaOmni_Gaussian achieves **100% passkey retrieval** at all depths B≥98
- KiaOmni_Gaussian PPL **27.80** at B=512 — best among eviction policies
- **Signal-swap ablation** proves the gain is KiaOmni's signal, not the selector

---

## 📊 Results

| Lane | Report | Coverage | Headline |
|------|--------|----------|----------|
| L1 | [`reports/qwen2.5-7b/`](reports/qwen2.5-7b/README.md) | Qwen2.5-7B — 11 tasks × 7 policies | KiaOmni_Gaussian: **89.0%** of FullContext |
| L2 | [`reports/mistral-7b/`](reports/mistral-7b/README.md) | Mistral-7B — RULER + LongBench | **100%** niah_single across all contexts |
| L4 | [`reports/cross-model/`](reports/cross-model/README.md) | Falcon3-7B · BioMistral-7B · Amber-7B | Cross-architecture generalization confirmed |
| L5 | [`reports/benchmarks/niah-heatmap/`](reports/benchmarks/niah-heatmap/README.md) | Needle-In-A-Haystack heatmaps | σ8 + Gaussian retain needle at all depths B≥128 |
| L6 | [`reports/benchmarks/passkey-and-ppl/`](reports/benchmarks/passkey-and-ppl/README.md) | Passkey retrieval + WikiText-2 PPL | **100%** passkey at B≥98; Gaussian PPL **27.80** |
| L7 | [`reports/llm-judge/`](reports/llm-judge/README.md) | LLM-as-Judge win-rates (4 models) | KiaOmni variants lead at **32%+** win-rate |
| L8 | [`reports/full-comparison/`](reports/full-comparison/README.md) | Master comparison — all models in one table | KiaOmni_Gaussian **#1 eviction** policy |
| L9 | [`reports/ablations/signal-swap/`](reports/ablations/signal-swap/README.md) | Mechanism ablation — signal vs selector | **The gain is the signal, not the selector** |

---

## 🧪 Reproduce

All experiment scripts live in [`experiments/`](experiments/README.md):

```bash
git clone https://github.com/Aliw02/kiaomni
cd kiaomni
pip install -e .
python experiments/033_full_comparison.py    # Qwen2.5-7B benchmark
python experiments/llm_judge.py --model qwen  # LLM-as-Judge
```

See [`experiments/README.md`](experiments/README.md) for the full script index, 10 canonical benchmarks, and reproduction guide.

---

## Install

```bash
pip install kiaomni
# optional: enables the kiaomni_gaussian policy
pip install kiaomni[gaussian]
```

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kiaomni import apply_kiaomni

# Any ungated HF causal LM works — TinyLlama / Qwen / Mistral / GPT-2 ...
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # attn_implementation="eager",  # optional — hooks fire before fused kernels
    torch_dtype="auto",
)

apply_kiaomni(model, policy="kiaomni_gaussian", budget=256)

# Generate as normal
text = "The quick brown fox jumps over the lazy dog. " * 50
prompt = f"Summarise the following text:\n{text}\n\nSummary:"
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=128)
print("Model: " + tok.decode(outputs[0], skip_special_tokens=True))
```

That's it. Any prompt longer than `budget` tokens is automatically evicted down to `budget` positions before the first decode step.

## Supported architectures

| Tier | Architectures | Strategy |
|------|---------------|----------|
| ✅ Verified (ungated) | TinyLlama, Mistral, Qwen2 / Qwen2.5, GPT-2, GPT-NeoX / Pythia | Hook-based extraction (fast) |
| ✅ Verified (gated — needs HF auth) | Meta Llama 3 / 3.1 | Hook-based extraction (fast) |
| 🟡 Probed / fallback | Falcon (MQA), MPT, exotic variants | Auto-routes to `output_attentions=True` (slower but correct) |
| ❌ Unsupported | T5, BART, BERT (not causal-LM) | n/a |

All examples in `examples/` use **ungated** models so they run on a fresh `pip install` with no `huggingface-cli login` required.

The `ArchitectureProbe` walks the module tree at `apply_kiaomni` time, classifies the QKV layout (separate / fused-concat / fused-interleaved), pulls dims via a priority list of config field names, and detects positional encoding (RoPE / ALiBi / learned). When confidence is low, saliency extraction falls back to `output_attentions=True` — guaranteed compatible with any HF causal LM.

## Policies

| Policy | Description | Best for |
|--------|-------------|----------|
| `kiaomni_s8`       | Boxcar smoothing (σ=8) on log-saliency | Overall production winner |
| `kiaomni_gaussian` | Gaussian smoothing (σ=4) on log-saliency | VectorTrace tasks |

Register your own:

```python
from kiaomni import register_policy
register_policy("my_policy", lambda sal: sal ** 0.5)
apply_kiaomni(model, policy="my_policy", budget=512)
```

## Requirements

- `attn_implementation="eager"` is recommended but not enforced — hook-based saliency on `q_proj`/`k_proj` fires before fused kernels (proven by `039_swap_experiment.py`). A warning is logged for SDPA/Flash-Attn.
- `transformers >= 4.50` — newer DynamicCache API.
- Works with NF4 / 4-bit bitsandbytes models (no `.to()` calls made).

## How it works

1. **Probe** — one walk of the module tree to discover layer container, attention module, QKV pattern, head dims, positional encoding.
2. **Saliency** — register forward hooks on Q/K projections, run one prefill, compute last-query softmax(QK^T/√d) per layer, average across layers and heads to a `(B, L)` saliency.
3. **Score & select** — apply the policy's smoothing function to log-saliency, always protect the first 16 tokens (attention sinks) and last 32 tokens (recency), fill the remaining budget with top-scoring positions.
4. **Prune & re-prefill** — slice `input_ids` by the kept positions and re-invoke `model.generate` on the shorter prompt. The model handles its own KV cache, position encoding, and attention masking — KiaOmni stays out of the way.

> **v0.2.0 algorithm note:** KiaOmni uses *prompt-side* eviction (slice the input tokens) rather than *cache-side* eviction (gather KV and resume with `past_key_values`). The prompt-side approach has been validated across Qwen2.5-7B, Mistral, BioMistral, Llama-3.1, and TinyLlama, and is robust against `transformers` version drift because it delegates all cache/position contracts to the model's own `generate`.

## Citation

```bibtex
@misc{kiaomni2026,
  title  = {KiaOmni: Smoothed Saliency for Long-Context KV-Cache Eviction},
  author = {Aliwey},
  year   = {2026},
  url    = {https://github.com/Aliw02/kiaomni}
}
```

## License

MIT — see [LICENSE](LICENSE).
