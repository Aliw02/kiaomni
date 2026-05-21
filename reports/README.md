# Experimental Results — KiaOmni KV-Cache Eviction

Curated, public-facing results for every experiment in the KiaOmni paper. Each lane below is self-contained — read the one you care about.

---

## 📊 Results by Lane

| Lane | Report | What it covers | Headline |
|------|--------|---------------|----------|
| L1 | [`reports/qwen2.5-7b/`](qwen2.5-7b/README.md) | Qwen2.5-7B — 11 tasks × 7 policies | KiaOmni_Gaussian: **89.0%** of FullContext |
| L2 | [`reports/mistral-7b/`](mistral-7b/README.md) | Mistral-7B — RULER + LongBench | **100%** niah_single across all contexts |
| L4 | [`reports/cross-model/`](cross-model/README.md) | Falcon3-7B · BioMistral-7B · Amber-7B | Confirms cross-architecture generalization |
| L5 | [`reports/benchmarks/niah-heatmap/`](benchmarks/niah-heatmap/README.md) | Needle-In-A-Haystack heatmaps | σ8 + Gaussian retain needle at all depths B≥128 |
| L6 | [`reports/benchmarks/passkey-and-ppl/`](benchmarks/passkey-and-ppl/README.md) | Passkey retrieval + WikiText-2 PPL | **100%** passkey at B≥98; Gaussian PPL **27.80** |
| L7 | [`reports/llm-judge/`](llm-judge/README.md) | LLM-as-Judge win-rates (4 models) | KiaOmni variants lead at **32%+** win-rate |
| L8 | [`reports/full-comparison/`](full-comparison/README.md) | **Master comparison** — all models in one table | KiaOmni_Gaussian **#1 eviction** policy |
| L9 | [`reports/ablations/signal-swap/`](ablations/signal-swap/README.md) | Mechanism ablation — signal vs selector | **The gain is the signal, not the selector** |

---

## 🧪 Reproduce

All experiment scripts live in [`experiments/`](../experiments/README.md):

```bash
git clone https://github.com/Aliw02/kiaomni
cd kiaomni
pip install -e .
python experiments/033_full_comparison.py    # Qwen2.5-7B benchmark
python experiments/llm_judge.py --model qwen  # LLM-as-Judge
```

See [`experiments/README.md`](../experiments/README.md) for the full script index and reproduction guide.

---

> **SnapKV** = faithful arXiv:2404.14469 implementation. **BlockSal** = our novel block-level baseline (paper §4).
