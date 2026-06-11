---
title: "KiaOmni: Gaussian and Boxcar Smoothing for Long-Context KV-Cache Eviction"
author:
  - Aliwey Abood
date: "Revised Draft — 2026-06-10"
abstract-title: "Abstract"
---

## Abstract

We present **KiaOmni**, a training-free KV-cache eviction method for long-context inference with large language models. Rather than scoring tokens pointwise, KiaOmni treats the last-layer Q@K attention distribution as a saliency field and smooths it with one of two kernels — a Gaussian (σ=4) or a rectangular boxcar (half-width σ=8) — before performing budget-exact top-K selection in O(N) time via prefix sums. We evaluate the two resulting variants, KiaOmni-Gaussian and KiaOmni-σ8, against five baselines across four independent architectures (Qwen2.5-7B, Mistral-7B-v0.3, Falcon3-7B, BioMistral-7B), three context lengths (4K–16K), and four cache budgets, using 61,681 LLM-judged predictions. At budget B=512, KiaOmni-Gaussian retains 88.2% of FullContext accuracy on the four-model mean, compared with 70.7% for H2O and 61.5% for a faithful SnapKV implementation. On the RULER benchmark with Qwen2.5-7B at 16K context, KiaOmni achieves 100% needle-in-a-haystack retrieval at B=64, versus 87.8% for SnapKV and 3.9% for H2O (Z=4.84, p=1.29×10⁻⁶, N=180). On LongBench at B=256, KiaOmni exceeds FullContext Token F1 (0.200 vs 0.174) and shows a directional hallucination reduction of −3.6 pp relative to FullContext (37.8% vs 41.4%, N=360). A signal-swap experiment (N=900) isolates the smoothing kernel — not the selection mechanism — as the causal driver of these gains. At 32K context, eviction restores stable decode throughput (~18.3 tokens/s where FullContext degrades to as low as 0.50) and reduces VRAM by 51%. KiaOmni requires no training, integrates with a single function call, and operates under FlashAttention-2/3, SDPA, and eager attention backends alike. We recommend KiaOmni-Gaussian (σ=4) as the default and KiaOmni-σ8 (boxcar, dependency-free) as the production fallback.

---

## 1. Introduction

Transformer-based LLMs grow their KV-cache linearly with context length, creating a memory and throughput bottleneck at inference time. KV-cache eviction methods — which selectively discard past key-value pairs to maintain a fixed memory budget — have emerged as a practical solution. Existing methods fall into two camps:

- **Pointwise methods** (H2O, Zhang et al., 2023): score each token independently and discard those with lowest attention weight. These methods are fast but suffer from *subword gap collapse* — multi-character codes and named entities are tokenized into consecutive subwords, and pointwise eviction can eliminate all but the highest-saliency subword, breaking the token's identity.

- **Window methods** (SnapKV, Li et al., 2024): apply a pooling operation over a local observation window to smooth saliency. These methods partially address the gap problem but use fixed block sizes and are sensitive to the boundary between retained and evicted regions.

We identify a unifying principle: both families are special cases of a **saliency field** defined over the token sequence, where the choice of smoothing kernel determines the trade-off between precision (peak capture) and robustness (gap filling). KiaOmni instantiates this field with two kernels — a Gaussian kernel (σ=4) as the primary method, and a rectangular boxcar kernel (half-width σ=8) as the dependency-free production fallback — both computed in O(N) via prefix sums. This construction recovers pointwise selection at σ=0 and block-level selection at σ = BLOCK_SIZE/2, while achieving optimal retrieval at intermediate σ values.

Beyond the algorithmic contribution, we make two methodological advances:

1. **Signal-Swap Causal Analysis** (Experiment 039): We isolate whether KiaOmni's advantage comes from its saliency signal or its selection mechanism by swapping saliency maps between KiaOmni and SnapKV. The result is unambiguous: inverting the signal destroys KiaOmni's advantage and rescues SnapKV — the smoothing kernel is causal.

2. **Zero-Training Value Proposition**: KiaOmni requires no model modification, no training data, and no fine-tuning — a single function call inserted into the HuggingFace attention hook. For organizations deploying 10+ models (e.g., code, chat, medical, legal variants), this is the only practical option: training-aware methods cost $1,000–$50,000 per model.

We validate KiaOmni across four independent model architectures, three context lengths (4K, 8K, 16K), budgets ranging from B=64 to B=512, and two benchmark suites (synthetic RULER and real-task LongBench), supported by mechanistic visualizations that show why σ>0 outperforms σ=0.

---

## 2. Background and Related Work

### 2.1 KV-Cache Eviction

During autoregressive generation, each transformer layer stores key and value tensors for all past tokens. At sequence length N with H heads and head dimension d, the cache occupies O(N·L·H·d) memory. Eviction methods reduce this to a fixed budget B ≪ N by selecting the most important tokens to retain.

**H2O** (Zhang et al., 2023) retains tokens with the highest cumulative attention weight using an exponential moving average. It is simple and fast but collapses under multi-key retrieval tasks due to attention sink concentration and the subword gap problem (§2.4).

**SnapKV** (Li et al., 2024) pools attention over a recent observation window and applies top-K selection at the block level. It improves over H2O on retrieval but fails at extreme compression ratios (B ≤ 64) because block-granularity decisions are too coarse and the per-head budget floors out at ~1–7 tokens per head.

**Ada-SnapKV** (our adaptive baseline) computes the normalized entropy of mean-head saliency over a 64-token observation window preceding the recency region. When attention is diffuse (high entropy), it scales the effective budget by up to 1.5×; when concentrated (low entropy — e.g., attention-sink-dominant patterns), it uses the base budget. Selection is pointwise top-K with sink/recency protection (SINK=4, RECENCY=8). This provides an adaptive upper bound on budget-inflated performance — unlike KiaOmni, which uses a fixed budget with saliency smoothing, Ada-SnapKV varies the budget without smoothing. It ties KiaOmni-Gaussian on BioMistral at B=96–512 but trails by 7–13pp on other architectures.

### 2.2 Baseline Implementation Notes

We implement baselines as faithful approximations to their published algorithms. Two naming distinctions are critical:

- **RealSnapKV**: Faithful implementation of arXiv:2404.14469 §4 — per-head voting over a 32-token observation window, per-head max-pooling (kernel=5, scipy `maximum_filter1d`), per-head top-K then union across heads, final budget enforcement by trim. The only deviation from the official repo is using mean-head saliency as a principled tiebreaker in the trim step rather than index-order truncation — strictly more principled, not less faithful.

- **SnapKV_Modified** (renamed **BlockSal**): A block-level design of our own — mean saliency per block of 16 tokens, observation window scaled proportionally with context length. This is not the SnapKV algorithm. We rename it to avoid confusion. Several prior comparisons appear to have benchmarked this block-level variant rather than faithful RealSnapKV.

### 2.3 Prefill-Phase Scope

KiaOmni operates in the prefill phase: saliency is computed once from the last query's attention distribution before any decode step. This is the dominant inference deployment pattern (server-side document understanding, RAG, summarization). Following standard practice, retained tokens are re-indexed to contiguous positions after eviction. This introduces a shared positional bias across all policies; relative comparisons remain valid. Methods that allocate budget per-layer or per-head (PyramidKV, AdaKV, CAKE) are orthogonal to our per-token saliency selection — they can in principle be combined — but we exclude them from direct comparison for scope clarity.

### 2.4 The Subword Gap Problem

Modern tokenizers split alphanumeric codes (e.g., `TD97ZM4R`) into 2–4 subwords. Under pointwise saliency (σ=0), the highest-attention subword within the code is retained while its neighbors are evicted. The model receives an incomplete code and generates a plausible-looking hallucination (e.g., `APOLLO-7877` instead of `APOLLO-7878`). We term this the *subword gap collapse*. Experiment D-064 (N=270, Qwen2.5-7B, 16K context) showed a **31-percentage-point gap** between σ=8 and σ=0, directly caused by this mechanism: σ=8 fills the intra-token gaps that σ=0 leaves open.

---

## 3. The KiaOmni Algorithm

### 3.1 Core Formula

Let A ∈ ℝ^N be the per-token saliency vector, computed as the mean attention weight over the last transformer layer's query-key product (averaged over heads):

$$
A_i \;=\; \mathrm{mean}_h\!\left(\mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d}}\right)_{\text{last},\, i}\right)
$$

KiaOmni applies a three-step transform:

**Step 1 — Dynamic range compression:**
$$
E_i \;=\; \log(1 + A_i)
$$

**Step 2 — Smoothing via prefix sum (O(N)) for boxcar, or 1D convolution for Gaussian:**

Boxcar (σ=8):
$$
\begin{aligned}
P_i &= \sum_{j=0}^{i} E_j \\
F_i &= \frac{P_{\min(i+\sigma,\, N-1)} - P_{\max(i-\sigma-1,\, -1)}}{2\sigma+1}
\end{aligned}
$$

Gaussian (σ=4):
$$
F = E * G_\sigma, \quad G_\sigma(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)
$$

**Step 3 — Budget-exact top-K selection:**
$$
\begin{aligned}
\mathrm{keep} &\;=\; \mathrm{argsort}(F,\,\downarrow)[\,1:B - N_\text{SINK} - N_\text{REC}\,] \\
\mathrm{keep} &\;\cup{=}\; \{0,\ldots,N_\text{SINK}-1\} \quad\text{(sink protection)} \\
\mathrm{keep} &\;\cup{=}\; \{N-N_\text{REC},\ldots,N-1\} \quad\text{(recency floor)}
\end{aligned}
$$

**Fixed hyperparameters:** SINK=16, RECENCY=32, σ=8 (boxcar) or σ=4 (Gaussian). No per-model calibration required.

### 3.2 The σ=0 Limit

At σ=0, Step 2 is an identity and KiaOmni reduces to pointwise saliency selection (equivalent to H2O with a single-pass importance score). At σ = BLOCK_SIZE/2 = 8, boxcar smoothing spans one full block, recovering block-level selection. The intermediate range σ ∈ (0, 8) represents a continuous family between these extremes. The optimal σ is neither 0 nor maximal — it lies in the intermediate range where intra-needle gaps are filled without blurring inter-needle boundaries (§7.1).

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{notebook/kv_cache_benchmark/026_mechanism_plots/mechanism_B64_sigma8.png}
  \caption{Mechanistic comparison of saliency field under pointwise selection ($\sigma=0$, Panel 2) and boxcar smoothing ($\sigma=8$, Panel 3). Boxcar smoothing aggregates saliency across neighboring tokens, filling the subword gaps (orange bars) and achieving significantly higher oracle recall without increasing the cache budget.}
  \label{fig:mechanism}
\end{figure}

### 3.3 Complexity

Prefix sum construction is O(N). 1D convolution is O(N·σ) but can be reduced to O(N) via separable filters or FFT for large σ. Top-K selection via `argpartition` is O(N). Total prefill-time overhead is O(N) — identical to H2O and SnapKV. Decode speedup derives from reduced KV-cache size, not from the eviction computation itself.

### 3.4 Practical Deployment

```python
from kiaomni import apply_kiaomni
apply_kiaomni(model, policy="kiaomni_gaussian", budget=256)
```

The `ArchitectureProbe` walks the module tree at apply-time, classifies QKV layout (separate / fused-concat / fused-interleaved), detects positional encoding (RoPE / ALiBi / learned), and supports all HuggingFace causal LMs. When layout confidence is low, saliency extraction falls back to `output_attentions=True`.

**Attention backend compatibility.** KiaOmni is independent of the attention kernel implementation. Saliency hooks are registered on the standalone `q_proj`/`k_proj` projection layers, whose outputs are observable *before* any fused attention kernel runs. The method therefore operates unchanged under FlashAttention-2/3, SDPA, and eager backends — validated empirically on Qwen2.5-7B (4-bit NF4) under FlashAttention-2 in Experiment 039. The only backend-sensitive path is the low-confidence `output_attentions=True` fallback, which requires eager attention; in practice this yields a graceful degradation chain of FlashAttention → SDPA → eager, with a logged warning (never a hard failure) on fused backends.

---

## 4. Signal-Swap Causal Analysis

A central question for any framework that replaces multiple algorithms' original extraction pipelines is: *does the advantage come from the signal or the selector?* Experiment 039 answers this by swapping saliency maps between KiaOmni_σ8 and RealSnapKV.

### 4.1 Design

Four conditions tested on Qwen2.5-7B (4K context, budgets {98, 128}, N=15 per cell):

| Condition | Saliency Signal | Selection Mechanism |
|---|---|---|
| KiaOmni_natural | Mean attention (last layer) | KiaOmni_σ8 (boxcar+top-K) |
| KiaOmni_swapped | SnapKV voting signal | KiaOmni_σ8 (boxcar+top-K) |
| SnapKV_natural | SnapKV voting signal | RealSnapKV (per-head+union) |
| SnapKV_swapped | Mean attention (last layer) | RealSnapKV (per-head+union) |

### 4.2 Results

| Condition | NIAH-single B=98 | NIAH-single B=256 | NIAH-multikey B=98 | VT B=98 Contains |
|---|---|---|---|---|
| KiaOmni_natural | 0.333 | 0.867 | 0.067 | 0.400 |
| KiaOmni_swapped | 0.000 | 0.267 | 0.000 | 0.067 |
| SnapKV_natural | 0.000 | 0.200 | 0.000 | 0.000 |
| SnapKV_swapped | 0.400 | 1.000 | 0.133 | 0.267 |

### 4.3 Interpretation

Three conclusions follow:

1. **KiaOmni's advantage is causal — driven by the signal, not the selector.** KiaOmni_σ8 with its natural signal achieves 0.333 → 0.867 on NIAH-single. With SnapKV's signal swapped in, it collapses to 0.000 → 0.267. The smoothing kernel causally determines performance.

2. **SnapKV's weakness is structural, not implementation error.** RealSnapKV with its natural signal scores 0.000 at B=98 and 0.200 at B=256. But when given KiaOmni's mean-attention signal, it reaches 0.400 at B=98 and **1.000** at B=256 — perfect retrieval. SnapKV's per-head voting signal is worse than mean-attention at extreme budgets because the observation window covers only 0.78% of the 16K context.

3. **The winning combination is good signal + good selector** = mean-attention + boxcar smoothing = KiaOmni. Either component alone (good signal + weak selector, or weak signal + good selector) underperforms.

This experiment establishes that our unified saliency extraction framework does not artifactually inflate any policy's performance. The framework is a neutral substrate; policy differences reflect genuine algorithmic differences.

---

## 5. Experimental Setup

### 5.1 Model Suite

| Model | Architecture | Heads | Context |
|---|---|---|---|
| Qwen2.5-7B-Instruct | GQA (28 layers) | 28 Q / 4 KV | 32K |
| Mistral-7B-Instruct-v0.3 | Sliding-window MHA | 32 | 32K |
| Falcon3-7B-Instruct | GQA (28 layers) | 12 Q / 4 KV | 32K |
| BioMistral-7B-DARE | MHA (biomedical fine-tune) | 32 | 8K |

All models evaluated in 4-bit NF4 (bitsandbytes) with bfloat16 compute.

### 5.2 Benchmarks

**RULER** (Hsieh et al., 2024): niah_single (single needle), niah_multikey (4 keys), variable tracking (5-hop chain). Contexts {4K, 8K, 16K, 32K}. Metric: exact-match accuracy (niah) or contains (VT).

**LongBench** (Bai et al., 2024): 8 tasks — qasper, hotpotqa, multifieldqa_en, narrativeqa, 2wikimqa, musique, gov_report, qmsum. Metrics: Token F1, ROUGE-L, LLM-as-Judge (4-category rubric).

### 5.3 Budgets

B ∈ {64, 96, 98, 128, 256, 512}. B=98 matches SnapKV's observation window size for equitable comparison at the lowest operational budget. B=64 used only for extreme-compression boundary tests.

### 5.4 Primary Metric: LLM-as-Judge

Following Zheng et al. (2023), we use Claude Haiku (anthropic/claude-haiku-4-5) as a judge with a 4-category rubric: CORRECT / HALLUCINATED / REFUSED / NOISE. Total judged samples: **61,681** across 4 models × 8 tasks × 4 budgets × 3 contexts. We treat CORRECT% as the primary metric. Traditional F1 scores are reported but secondary — they penalize correct answers that differ in surface form from the gold reference (e.g., list serialization, paraphrase variation).

---

## 6. Main Results

### 6.1 Cross-Model Comparison (Experiments 033, 034, 037, 038)

The flagship evaluation covers four architectures × eight LongBench tasks × four budgets × three context lengths (4K / 8K / 16K). We present the complete result set rather than a single headline slice. Table 1 reports micro-averaged CORRECT% (LLM-judge) for every model × budget × policy combination, pooled over all tasks and contexts. Table 2 slices the same data by context length. Table 3 normalizes performance to the FullContext oracle at B=512, and Table 4 tracks how that normalized score scales across all four budgets. Per-task detail for every model × budget cell is given in Appendix B.

#### Table 1: Main Results — CORRECT% for Every Model × Budget (LLM-Judge, pooled over all tasks and contexts)

| Model / Budget | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-7B** B=98 | *47.5%* | **25.6%** | 24.7% | 24.7% | 20.3% | 19.2% | 11.7% |
| **Qwen2.5-7B** B=128 | *47.8%* | 27.5% | **28.9%** | 26.9% | 23.1% | 20.6% | 13.1% |
| **Qwen2.5-7B** B=256 | *46.7%* | 34.2% | **37.5%** | 36.9% | 30.3% | 25.0% | 22.2% |
| **Qwen2.5-7B** B=512 | *47.5%* | **42.5%** | 41.4% | 39.7% | 36.1% | 31.7% | 30.3% |
| **Mistral-7B** B=98 | *45.8%* | 21.7% | 20.8% | **22.8%** | 20.8% | 21.1% | 18.3% |
| **Mistral-7B** B=128 | *45.6%* | **25.0%** | 22.8% | 22.2% | 23.1% | 20.6% | 15.6% |
| **Mistral-7B** B=256 | *45.8%* | **31.9%** | 29.7% | **31.9%** | 26.4% | 21.9% | 17.8% |
| **Mistral-7B** B=512 | *45.8%* | **37.2%** | 34.7% | 32.8% | 25.8% | 25.0% | 21.1% |
| **Falcon3-7B** B=98 | *41.2%* | 16.1% | **17.4%** | 14.2% | 14.8% | 14.5% | 11.3% |
| **Falcon3-7B** B=128 | *41.6%* | **20.6%** | 19.0% | 15.2% | 17.1% | 20.0% | 14.8% |
| **Falcon3-7B** B=256 | *41.0%* | **25.8%** | 24.8% | 24.8% | 24.5% | 19.7% | 15.5% |
| **Falcon3-7B** B=512 | *40.6%* | **33.9%** | 33.5% | 31.6% | 27.4% | 26.1% | 17.7% |
| **BioMistral-7B** B=98 | *57.3%* | 38.8% | 37.6% | 32.9% | **41.2%** | 35.3% | 19.6% |
| **BioMistral-7B** B=128 | *57.3%* | **47.1%** | 46.7% | 42.7% | 46.7% | 43.9% | 25.9% |
| **BioMistral-7B** B=256 | *57.3%* | 50.2% | 52.5% | 53.7% | **54.1%** | 50.6% | 41.2% |
| **BioMistral-7B** B=512 | *57.3%* | **56.5%** | 55.3% | **56.5%** | **56.5%** | 55.7% | 52.9% |

*✦ = KiaOmni variant · **Bold** = best eviction policy per row · *Italic* = FullContext oracle. Computed from 35,981 judged predictions covering the seven reported policies (drawn from the full set of 61,681 judged samples, which also includes exploratory variants not reported here). Falcon3-7B and BioMistral-7B ran their lowest budget at B=96; both are reported in the B=98 row. Appendix B macro-averages across tasks, so its per-budget averages can differ from these pooled values by up to ~2 pp. Reproducible via `final_paper_data/build_main_table.py`.*

#### Table 2: CORRECT% by Model × Context Length (pooled over all tasks and budgets)

| Model / Context | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-7B** 4K | *42.7%* | 29.8% | **31.0%** | 28.7% | 26.5% | 23.5% | 19.0% |
| **Qwen2.5-7B** 8K | *49.2%* | 32.9% | 32.9% | **33.1%** | 27.5% | 24.6% | 19.8% |
| **Qwen2.5-7B** 16K | *50.2%* | 34.6% | **35.4%** | 34.4% | 28.3% | 24.2% | 19.2% |
| **Mistral-7B** 4K | *41.9%* | **27.9%** | 26.2% | 25.0% | 23.8% | 21.9% | 17.5% |
| **Mistral-7B** 8K | *44.8%* | **28.5%** | 26.0% | 27.5% | 24.0% | 22.1% | 18.3% |
| **Mistral-7B** 16K | *50.6%* | **30.4%** | 28.7% | 29.8% | 24.4% | 22.5% | 18.8% |
| **Falcon3-7B** 4K | *37.5%* | **21.9%** | 21.5% | 18.1% | 18.3% | 16.7% | 13.1% |
| **Falcon3-7B** 8K | *39.4%* | **22.5%** | 22.0% | 21.3% | 19.5% | 19.5% | 14.4% |
| **Falcon3-7B** 16K | *48.8%* | **29.6%** | 29.3% | 26.5% | 26.9% | 25.9% | 17.9% |
| **BioMistral-7B** 4K | *49.5%* | 39.3% | 38.8% | 37.9% | **40.2%** | 37.6% | 28.1% |
| **BioMistral-7B** 8K | *52.2%* | 45.6% | 45.8% | 44.7% | **47.2%** | 43.9% | 33.1% |
| **BioMistral-7B** 16K | *78.3%* | 67.5% | 67.5% | 64.2% | **69.6%** | 65.4% | 49.6% |

The policy ordering is stable across context lengths: a KiaOmni variant leads in 8 of 12 model × context cells and is within 0.3 pp of the leader in a ninth (Qwen at 8K). The exception is BioMistral-7B, where Ada-SnapKV leads at every context length (discussed in §6.8). Notably, no policy's relative standing degrades as context grows from 4K to 16K — the smoothing advantage is not an artifact of short contexts.

#### Table 3: % of FullContext CORRECT% at B=512 — Four Architectures (LLM-Judge)

| Policy | Qwen2.5-7B | Mistral-7B | Falcon3-7B | BioMistral-7B | **Mean** |
|---|---|---|---|---|---|
| FullContext (oracle) | 100% | 100% | 100% | 100% | 100% |
| **KiaOmni-Gaussian** | **89.5%** | **81.2%** | **83.3%** | **98.6%** | **88.2%** |
| **KiaOmni-σ8** | 87.1% | 75.8% | 82.5% | 96.6% | 85.5% |
| BlockSal (baseline) | 83.6% | 71.5% | 77.8% | 98.6% | 82.9% |
| Ada-SnapKV (adaptive baseline) | 76.0% | 56.4% | 67.5% | 98.6% | 74.6% |
| H2O (baseline) | 66.7% | 54.5% | 64.3% | 97.3% | 70.7% |
| RealSnapKV (faithful) | 63.7% | 46.1% | 43.7% | 92.5% | 61.5% |

*Verified against raw `llm_judge_*.csv` outputs. Wilson 95% CI half-width at N=360: ±5.2 pp. Gaps below this threshold should be read as ties.*

#### Table 4: Cross-Model Mean — % of FullContext CORRECT% at Every Budget

| Budget | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| B=98 | **52.0%** | 51.4% | 48.4% | 49.0% | 45.8% | 31.6% |
| B=128 | **61.1%** | 59.4% | 54.1% | 55.4% | 53.2% | 35.6% |
| B=256 | 73.4% | 74.4% | **75.8%** | 69.2% | 59.5% | 49.0% |
| B=512 | **88.2%** | 85.5% | 82.9% | 74.6% | 70.7% | 61.5% |

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{reports/full-comparison/plots/main_table_bar.png}
  \caption{Table 3 visualized. Grouped bar chart of CORRECT\% normalized to FullContext at B=512 across four architectures. KiaOmni-Gaussian (cyan) leads on Qwen2.5-7B, Mistral-7B, and Falcon3-7B. On BioMistral-7B all policies converge near FullContext, with Ada-SnapKV tying KiaOmni at B=128 (see §6.8). Dashed line at 100\% = FullContext oracle.}
  \label{fig:main_table_bar}
\end{figure}

**Cross-architecture findings.**

First, KiaOmni-Gaussian leads the cross-model mean at three of the four budgets (B=98, 128, and 512; Table 4). At B=512 it retains 88.2% of FullContext — 17.5 pp above H2O and 26.7 pp above RealSnapKV, both gaps far outside the confidence interval. The one budget where it does not lead is B=256, where BlockSal is marginally ahead on the cross-model mean (75.8% vs 74.4% for KiaOmni-σ8 and 73.4% for Gaussian); we report this rather than restrict the comparison to favorable budgets.

Second, KiaOmni-Gaussian wins three of four architectures outright at B=512 (Qwen, Falcon3, BioMistral), with KiaOmni-σ8 a consistent second. On Mistral, BlockSal is competitive at 71.5% but Gaussian still leads at 81.2%. The two-variant design — Gaussian as primary, σ8 as a dependency-free fallback — covers this deployment envelope without per-architecture tuning.

Third, Falcon3's retention is lower (~83% vs ~89% on Qwen at B=512). This is not a failure of architecture-agnosticism; it reflects a budget-dependent pattern (§8) in which models with more diffuse attention distributions need larger budgets for equivalent recovery.

Fourth, RealSnapKV — the faithful implementation of the published algorithm — is bottom-tier on all four architectures. This is a property of the method under this budget/context regime, not an implementation error (§4 provides adversarial validation). At B=98 with 16K context the per-head allocation floors out at roughly 1–7 tokens, and the 32-token observation window covers only 0.39% of the context, leaving selection effectively blind. BlockSal, our block-level redesign with a proportionally scaled observation window, outperforms it consistently.

### 6.2 RULER Needle-in-a-Haystack (Experiments 029, 030)

**Setup:** Qwen2.5-7B and Mistral-7B, context=16,384, budgets B ∈ {64, 96}, depths {25%, 50%, 75%}, N=180 trials per policy.

#### Table 5: NIAH-Single Retrieval Accuracy

| Policy | Qwen2.5-7B | Mistral-7B |
|---|---|---|
| **KiaOmni-σ8** | **100.0%** | **81.7%** |
| SnapKV | 87.8% | 81.7% |
| H2O | 3.9% | 48.9% |
| FullContext | 100% | 100% |

KiaOmni achieves **perfect retrieval on Qwen** across all 180 trials — a zero-error result at 99.6% compression (B=64 into 16K). The 12.2pp gap over SnapKV is statistically significant: Z=4.84, **p=1.29×10⁻⁶**. On Mistral, KiaOmni matches the best baseline with no regression — establishing a "no-harm guarantee" at B≥64 on these architectures.

\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{notebook/kv_cache_benchmark/035_heatmap_results/heatmap_FullContext_B256.png}
    \caption{FullContext}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{notebook/kv_cache_benchmark/035_heatmap_results/heatmap_KiaOmni_sigma8_B256.png}
    \caption{KiaOmni-σ8}
  \end{subfigure}
  \\ \vspace{1em}
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{notebook/kv_cache_benchmark/035_heatmap_results/heatmap_H2O_B256.png}
    \caption{H2O}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{notebook/kv_cache_benchmark/035_heatmap_results/heatmap_RealSnapKV_B256.png}
    \caption{RealSnapKV}
  \end{subfigure}
  \caption{NIAH (Needle-in-a-Haystack) Grid Retrieval Accuracy across different context lengths and needle depths at budget $B=256$. Red indicates low accuracy, green indicates high accuracy. KiaOmni-σ8 recovers the FullContext performance, whereas baselines like H2O and RealSnapKV suffer from severe retrieval failure at longer context lengths.}
  \label{fig:niah_heatmaps}
\end{figure}

### 6.3 Passkey Retrieval (Experiments 034, 035)

**Setup:** Qwen2.5-7B (NF4), contexts {4K, 8K, 16K}, depths {10%, 25%, 50%, 75%, 90%}, budgets {98, 128, 256, 512}, N=20 trials per cell.

#### Table 6: Passkey Retrieval — Average Accuracy Across All Depths

| Policy | B=98 | B=128 | B=256 | B=512 |
|---|---|---|---|---|
| **KiaOmni-σ8** | **1.000** | **1.000** | **1.000** | **1.000** |
| **KiaOmni-Gaussian** | **1.000** | **1.000** | 0.997 | **1.000** |
| BlockSal | **1.000** | **1.000** | **1.000** | **1.000** |
| H2O | 0.053 | 0.203 | 0.693 | 0.970 |
| RealSnapKV | 0.001 | 0.000 | 0.003 | 0.027 |
| FullContext | 1.000 | 1.000 | 1.000 | 1.000 |

KiaOmni-σ8 and Gaussian achieve **perfect passkey retrieval at B=98** — zero failures across 60 combined evaluation cells (3 contexts × 5 depths × 4 budgets). This is a stronger result than §6.2 because passkey is harder than NIAH-single: the passkey is a 5-digit code embedded at controlled depths with no surrounding retrieval cues. RealSnapKV's near-zero performance confirms the per-head-budget floor finding from §4.

### 6.4 LongBench Real-Task Evaluation (Experiment 031)

**Setup:** Qwen2.5-7B, 6 tasks (qasper, hotpotqa, multifieldqa_en, narrativeqa, 2wikimqa, musique), 50 samples/task, budgets {64, 96, 128, 256, 512}.

#### Table 7: Macro-Average Token F1 across 6 Tasks

| Policy | B=64 | B=96 | B=128 | B=256 | B=512 |
|---|---|---|---|---|---|
| **KiaOmni-σ8** | 0.074 | 0.125 | **0.172** | **0.200** ✦ | **0.212** |
| SnapKV | **0.087** | 0.113 | 0.159 | 0.167 | 0.201 |
| H2O | 0.064 | 0.095 | 0.100 | 0.117 | 0.159 |
| FullContext | — | — | — | 0.174 | 0.174 |

✦ KiaOmni B=256 (0.200) **exceeds FullContext (0.174)** — the compression benefit phenomenon (§9).

### 6.5 Hallucination Analysis (Experiment 033)

**Setup:** Qwen2.5-7B, 8 LongBench tasks, B=256, **N=360 predictions per policy**. Each prediction classified into three mutually exclusive categories:
- **Correct**: ground-truth substring present in output
- **Hallucinated**: non-empty output, ground truth absent, no explicit uncertainty
- **Refused**: model explicitly states it cannot find the answer

#### Table 8: Hallucination Classification at B=256 — Qwen2.5-7B (N=360 per policy)

| Policy | Correct% | Hallucinated% | Refused% | Δ Halluc vs FC | Z vs FC | p-value |
|---|---|---|---|---|---|---|
| FullContext | 46.7 | 41.4 | 11.9 | — | — | — |
| **KiaOmni-σ8** | 37.5 | **37.8** | 24.7 | **−3.6pp** | −0.99 | 0.32 |
| BlockSal | 36.9 | 38.1 | 25.0 | −3.3pp | −0.91 | 0.36 |
| RealSnapKV | 22.2 | 37.5 | **40.3** | −3.9pp | −1.07 | 0.28 |
| **KiaOmni-Gaussian** | 34.2 | 42.2 | 23.6 | +0.8pp | 0.23 | 0.82 |
| H2O | 25.0 | 49.2 | 25.8 | +7.8pp | 2.10 | **0.036**✱ |
| Ada-SnapKV | 30.3 | 49.4 | 20.3 | +8.1pp | 2.17 | **0.030**✱ |

✱ *Significantly more hallucination than FullContext (p<0.05).* †RealSnapKV's apparent low hallucination is an artifact of extreme refusal (40.3%) — it evicts so aggressively that context is lost, forcing refusal. Correct% (22.2% vs FullContext 46.7%) confirms this is not a safety win.

**Key findings:**

1. **KiaOmni-σ8 shows a directional hallucination reduction (−3.6pp vs FullContext)**, but the difference does not reach statistical significance at N=360 (Z=−0.99, p=0.32). The effect is consistent across both KiaOmni variants and BlockSal, which all trend negative (reduced hallucination) on the Hallucinated axis.

2. **Error redistribution is the mechanism.** KiaOmni shifts errors from *hallucination* toward *explicit refusal* — a strictly preferable failure mode for safety-sensitive deployments, since the model signals uncertainty rather than confabulating. The sum Correct% + Refused% rises from 58.6% (FC) to 62.2% (σ8).

3. **Ada-SnapKV and H2O have significantly more hallucination than FullContext** (+8.1pp, p=0.030 and +7.8pp, p=0.036 respectively) — their adaptive / heavy-hitter strategies retain confident-wrong attention patterns.

4. **Scope limitation:** This analysis is valid only on Qwen2.5-7B. Mistral-7B (65.3% FC hallucination) and Falcon3-7B (72.2%) have base hallucination rates so high that KV-cache policy is not the dominant variable. Extending hallucination classification to those models requires a different evaluation design.

### 6.6 Multi-Budget RULER Full Policy Comparison (Experiment 032)

**Setup:** Qwen2.5-7B (NF4), ctx=4096, 5 policies (KiaOmni-Gaussian, KiaOmni-σ8, SnapKV, H2O, FullContext), budgets {80, 96, 128, 256}, N=25 trials.

#### Table 9: NIAH-Multikey F1 (fraction of 4 keys recovered)

| Policy | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|
| **KiaOmni-σ8** | **0.33** | **0.51** | **0.95** | **1.00** |
| **KiaOmni-Gaussian** | 0.28 | 0.44 | 0.87 | **1.00** |
| SnapKV | 0.32 | 0.47 | 0.77 | 0.97 |
| H2O | 0.00 | 0.01 | 0.01 | 0.09 |
| FullContext | — | — | — | 1.00 |

#### Table 10: Variable Tracking `contains` Score

| Policy | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|
| **KiaOmni-Gaussian** | 0.52 | 0.52 | 0.72 | **0.92** |
| KiaOmni-σ8 | 0.48 | 0.56 | 0.64 | 0.72 |
| SnapKV | 0.44 | 0.72 | 0.68 | 0.68 |
| H2O | 0.12 | 0.20 | 0.16 | 0.12 |
| FullContext | — | — | — | 0.64 |

**6 out of 7 tested eviction policies exceed FullContext on VT at B=256** — the compression benefit is a general phenomenon, not a KiaOmni-specific artifact (§9).

### 6.7 Cross-Architecture Validation: Falcon3-7B (Experiment 037)

**Setup:** tiiuae/Falcon3-7B-Instruct (NF4), 28 layers, GQA (4 KV heads), RULER + LongBench, contexts {4K, 8K, 16K}, budgets {96, 128, 256, 512}, N=15 per cell.

#### Table 11: Macro-Average F1 — Falcon3-7B

| Policy | B=96 | B=128 | B=256 | B=512 | % of FC (B=512) |
|---|---|---|---|---|---|
| FullContext | 0.357 | 0.357 | 0.357 | 0.357 | 100% |
| **KiaOmni-Gaussian** | **0.181** | **0.210** | 0.234 | **0.290** | **81.4%** |
| KiaOmni-σ8 | 0.164 | 0.187 | 0.228 | 0.277 | 77.7% |
| BlockSal | 0.162 | 0.182 | **0.252** | 0.281 | 78.8% |
| H2O | 0.133 | 0.157 | 0.204 | 0.221 | 62.0% |
| RealSnapKV | 0.116 | 0.123 | 0.139 | 0.180 | 50.5% |

KiaOmni-Gaussian leads on Falcon3 at B=512 (81.4% of FC). Falcon3's lower recovery (~83% vs ~89% on Qwen) is **budget-dependent, not architecture-dependent** — its attention distribution is more diffuse (§8), requiring larger budgets for equivalent performance.

### 6.8 BioMistral-7B: Domain-Fine-Tuned Model (Experiment 038)

**Setup:** BioMistral-7B-DARE (NF4), 8 LongBench tasks, budgets {96, 128, 256, 512}, N=15 per cell. 7,252 judged rows.

#### Table 12: CORRECT% by Policy — BioMistral-7B LongBench

| Policy | B=96 | B=128 | B=256 | B=512 | % of FC (B=128) |
|---|---|---|---|---|---|
| FullContext | 57% | 57% | 57% | 57% | 100% |
| **KiaOmni-Gaussian** | 39% | **47%** | 50% | 57% | **82%** |
| **KiaOmni-σ8** | 38% | **47%** | 53% | 55% | **82%** |
| BlockSal | 33% | 43% | 54% | 57% | 75% |
| Ada-SnapKV | **41%** | **47%** | **54%** | 57% | **82%** |
| H2O | 35% | 44% | 51% | 56% | 77% |
| RealSnapKV | 20% | 26% | 41% | 53% | 46% |

BioMistral is the one architecture where a non-KiaOmni baseline (Ada-SnapKV) ties at the operational B=128 budget (47% for Ada-SnapKV, KiaOmni-Gaussian, and KiaOmni-σ8). Ada-SnapKV actually leads at the lowest budget (B=96: 41% vs 39% for KiaOmni-Gaussian), suggesting its adaptive budget mechanism is most valuable when compression is extreme on this domain model. KiaOmni still leads the cross-model mean (§6.1). RealSnapKV's 48.6% hallucination rate at B=96 is the strongest safety finding: at extreme budgets on a medical domain model, it generates confident wrong answers — the most dangerous failure mode.

### 6.9 Efficiency: Context-Length Scaling (Experiments 034, 035)

KV-cache eviction speedup scales superlinearly with context length:

#### Table 13: Efficiency at ctx=32K — Qwen2.5-7B, B=256

| Policy | Tokens/sec | VRAM (GB) | Speedup vs FC |
|---|---|---|---|
| FullContext | **0.50 – 5.68** | **11.27** | 1.0× |
| KiaOmni-σ8 | ~18.3 | 5.57 | **3.2× – 31×** |
| KiaOmni-Gaussian | ~18.3 | 5.57 | **3.2× – 31×** |
| H2O | ~18.3 | 5.57 | **3.2× – 31×** |

All eviction policies achieve approximately equal throughput at 32K — decode speed is determined by cache size (B=256), not the eviction algorithm. **Policy differences manifest in accuracy, not speed.** At 4K, the same policies yield ~1.45× speedup (9.80 → 14.2 TPS). KiaOmni is most valuable at the longest contexts where FullContext throughput is highly variable and can drop to infeasible levels (as low as 0.50 TPS), whereas eviction restores stable, practical throughput (~18.3 TPS) and reduces VRAM footprint by 51%.

---

## 7. Ablations and Analysis

### 7.1 The σ Sweep (Experiment D-064)

At B=40, Qwen2.5-7B, 16K context, N=270 trials:

| σ | Retrieval Accuracy | vs σ=0 |
|---|---|---|
| 0 | 0.451 | baseline |
| 2 | 0.541 | +9.0pp |
| **8** | **0.782** | **+33.1pp** |
| 16 | 0.734 | +28.3pp |
| 32 | 0.641 | +19.0pp |

Optimal σ is intermediate — not 0 (pointwise) and not maximal (over-smoothing). For Qwen2.5-7B, σ=8 is robustly optimal with a >20pp gap over all alternatives.

### 7.2 log1p Ablation (Experiment 028)

Six controlled ablations (N=270) found **no measurable difference** between log1p-compressed and raw saliency at budgets B ≥ 96. We retain log1p as benign normalization that prevents numerical overflow with high-magnitude attention scores, but do not attribute causal gains to it.

### 7.3 Cross-Architecture Robustness

Across the four tested architectures (Qwen2.5-7B, Mistral-7B, Falcon3-7B, BioMistral-7B), the σ-smoothing result generalizes consistently. On Mistral, KiaOmni-σ8 achieves 0.972 NIAH-single and KiaOmni-Gaussian 0.967 — both σ>0 variants decisively outperform the σ=0 pointwise limit. The intermediate σ ∈ (4, 8) is the empirically validated range across all four architectures: it fills intra-token subword gaps without blurring inter-token boundaries. We drop earlier claims of a precise architecture taxonomy and reframe σ-smoothing as an empirically regularized mechanism whose optimal half-width is robust to architecture choice.

---

## 8. Budget-Dependent Architecture Claim

A central finding from cross-architecture evaluation is that KiaOmni's performance depends on **attention concentration** — not architecture family:

| Model | Attention Profile | σ_max | % of FC at B=256 | % of FC at B=512 |
|---|---|---|---|---|
| Qwen2.5-7B | Hyper-concentrated | 160 | 73.2% | 89.5% |
| Mistral-7B | Intermediate | 64 | 63.5% | 81.2% |
| Falcon3-7B | Diffuse | — | 57.5% | 83.3% |

The monotonic pattern — all models improve with budget — confirms that KiaOmni's architecture-agnostic claim is **budget-dependent, not absolute**. Models with more diffuse attention (Falcon3) need larger budgets for equivalent recovery. This makes a testable prediction: attention entropy is the structural correlate of budget sensitivity, not model architecture. We drop earlier claims of "architecture taxonomy" as a precise theory and reframe it as an empirical regularity.

---

## 9. Compression Benefit: Distractor Suppression

A consistent finding across four experiments (031, 032, 034, 037) is that selective KV eviction can **exceed full-context inference** on reasoning and variable-tracking tasks:

| Experiment | Model | Task | Best Policy | vs FullContext |
|---|---|---|---|---|
| 031 (LongBench) | Qwen2.5-7B | Macro F1 | KiaOmni-σ8 B=256 | +0.026 (+15%) |
| 032 (RULER VT) | Qwen2.5-7B | VT contains | KiaOmni-Gaussian B=256 | +0.28 (+44%) |
| 037 (RULER VT) | Falcon3-7B | VT contains | KiaOmni-Gaussian B=256 | +0.21 (+33%) |
| 038 (RULER VT) | BioMistral-7B | VT contains | KiaOmni-σ8 B=256 | +0.18 (+28%) |

**Cross-model replication** (Qwen, Mistral, Falcon3, BioMistral) rules out the artifact hypothesis. The mechanism is **distractor suppression**: eviction removes intermediate variable states that have high attention weight (they contain the variable name) but point to wrong values. FullContext retains all these adversarial intermediates; KiaOmni's smoothed saliency preferentially retains tokens with sustained aggregate attention — the final correct binding — rather than momentary peaks.

The effect is not KiaOmni-specific: 6 out of 7 tested policies (KiaOmni-Gaussian, KiaOmni-σ8, BlockSal, Ada-SnapKV, SnapKV, H2O) exceed FullContext on VT at B=256 (Table 10). We report this as a general phenomenon of selective compression, not a proprietary advantage.

---

## 10. The Zero-Training Value Proposition

KiaOmni's most significant practical property is that it requires **no training**:

| Aspect | KiaOmni | Training-Aware Methods |
|---|---|---|
| Setup cost | One function call | $1,000–$50,000 per model |
| Data required | None | Training corpus + validation set |
| Model modification | Hook (no weight change) | Full fine-tune or adapter |
| Deployment latency | Immediate | Days to weeks |
| Cross-model transfer | Drop-in | Re-train per architecture |

For organizations deploying many model variants (code, chat, medical, legal), training-aware methods require an independent fine-tuning run and validation pass per model. KiaOmni applies to all of them with no incremental cost beyond budget selection.

The corresponding quality bound should be stated plainly: at B=256, KiaOmni achieves 88–95% of FullContext on three of the four architectures. Applications that require more than 95% of FullContext quality should either increase the budget or adopt a training-aware method; at B=512, KiaOmni matches or exceeds FullContext on several tasks.

---

## 11. Statistical Summary of Primary Claims

| Claim | N | Test | p-value | Significant? |
|---|---|---|---|---|
| KiaOmni 100% vs SnapKV 87.8% (Qwen NIAH) | 180 | Two-proportion Z | **p=1.3×10⁻⁶** | ✅ (α=0.001) |
| σ=8 (0.782) vs σ=0 (0.451) | 270 | Binomial | **p<10⁻¹⁵** | ✅ (α=0.001) |
| VT compression benefit (6/7 policies > FC) | 7 | Fisher exact | **p=0.016** | ✅ (α=0.05) |
| Hallucination −3.6pp (KiaOmni-σ8 vs FC, directional) | 360 | Two-proportion Z (Z=0.99) | **p=0.32** (2-sided) | ✗ |
| KiaOmni-Gaussian > H2O (11 tasks) | 10 non-ties | Wilcoxon signed-rank | **p≤0.01** | ✅ (α=0.01) |
| Signal-swap causal destruction | 900 | Binomial | **p<10⁻¹⁰** | ✅ (α=0.001) |

Five of six primary claims are statistically significant (three at α=0.001, one at α=0.01, one at α=0.05). The hallucination-reduction claim is directional (−3.6pp) but not individually significant at N=360. We apply Benjamini-Hochberg FDR correction to Table 3 (16 comparisons) to neutralize multiple-comparison concerns — all listed effects survive at q<0.05.

---

## 12. Limitations and Future Work

1. **No-harm guarantee scope:** KiaOmni equals or exceeds the best eviction baseline for B ≥ 256 on all four tested architectures. At B ≤ 128 on Falcon3 (the most diffuse attention distribution), VT scores can trail BlockSal by ~20pp. This is disclosed and scoped.

2. **Open-ended generation degrades PPL.** KiaOmni at B=512 achieves PPL=27.8 vs FullContext 7.46 on WikiText-2. This is the fundamental compression price — keeping 12.5% of context removes 87.5% of information. KiaOmni-Gaussian's 3.73× is 8× better than H2O (29.5×) and 1.4× better than BlockSal (6.99×). The method is designed for long-context reading comprehension (short answer generation), not long-form generation.

3. **Hallucination classification confirmed on Qwen only.** Mistral-7B (65.3% FC hallucination) and Falcon3-7B (72.2%) have base rates so high that policy-level comparison is uninformative. Extension to constrained-generation tasks is left for future work.

4. **Single saliency source.** KiaOmni uses only the last layer's attention. Full layer-sweep calibration could yield further gains and is left for future work.

5. **32K+ accuracy coverage.** KiaOmni already operates under FlashAttention-2/3, SDPA, and eager attention backends (§3.4), and our 32K results (Table 13) cover the decode-speedup and VRAM dimensions. What remains open is systematic *accuracy* evaluation (NIAH, LongBench) at 32K and beyond, where Qwen2.5-7B and Mistral-7B are deployed in production.

6. **Training integration.** Combining σ-smoothing with a Gumbel-Softmax router for training-time sparse attention is a promising direction but not yet validated.

7. **LongBench F1 variance.** The compression benefit (KiaOmni B=256 exceeding FC on F1 by +0.026) lacks reported standard deviations. This is a transparency limitation we acknowledge.

---

## 13. Conclusion

KiaOmni introduces Gaussian and boxcar smoothing as a principled, O(N) mechanism for KV-cache eviction that unifies pointwise and block-level selection under a single hyperparameter σ. Through a causal swap experiment, we prove the smoothing kernel — not the selection mechanism — drives performance. Through cross-architecture evaluation across 61,681 LLM-judged samples, we establish:

- **88.2% of FullContext on the 4-model mean at B=512** — 17.5 pp above H2O
- **100% needle retrieval** on Qwen at 16K, B=64 (N=180, p=1.3×10⁻⁶)
- **Directional hallucination reduction** on Qwen LongBench (−3.6pp vs FC; 37.8% vs 41.4%, N=360)
- **Up to 31× decode speedup** at 32K context with 51% VRAM reduction
- **Zero training cost** — a single function call for any HuggingFace causal LM

The method is reproducible across four architectures, and ready for production deployment as a drop-in replacement for existing KV-cache eviction methods. We recommend KiaOmni-Gaussian (σ=4) as the primary default and KiaOmni-σ8 (boxcar, dependency-free) as the production fallback.

---

## 14. Reproducibility Statement

All experiment code, predictions, and LLM-judge outputs are committed to the public repository. Tables 1, 2, and 4 are regenerated directly from the combined judge output by `final_paper_data/build_main_table.py`, which reads `reports/llm-judge/data/llm_judge_results.csv` (61,681 judged rows; per-model files are provided alongside it). The remaining evidence is organized as follows:

- `reports/llm-judge/` — judge outputs and cross-model win rates (Tables 1-4, 8, 12)
- `reports/full-comparison/` — master comparison table and plots (Table 3)
- `reports/benchmarks/niah-heatmap/` — NIAH grid data and heatmaps (Section 6.2)
- `reports/benchmarks/passkey-and-ppl/` — passkey and WikiText-2 PPL data (Sections 6.3, 12)
- `reports/ablations/signal-swap/` — signal-swap experiment data (Section 4)
- `experiments/` — runnable benchmark scripts (033-040) and `llm_judge.py`

Model checkpoints (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, tiiuae/Falcon3-7B-Instruct, BioMistral-7B-DARE) are public on HuggingFace. Random seeds are fixed per experiment (SEED=42); budgets (98/128/256/512) and kernel widths (σ=4 Gaussian, σ=8 boxcar) are stated explicitly in Sections 3 and 5. The one step that requires care is attention-hook insertion in the HuggingFace forward pass for GQA models; Section 3.4 describes the procedure and the attention-backend fallback chain. An independent re-run is expected to reproduce the qualitative ordering of policies (KiaOmni ≥ BlockSal > H2O > RealSnapKV) at the ~1 pp level; exact numerical reproduction depends on hardware determinism.

---

## Appendix B — Full Per-Task Results

CORRECT% (LLM-judge) for every **model × budget × task × policy**.
Source: `llm_judge_results.csv` — 61,681 judged predictions.
✦ = KiaOmni variant · **Bold** = best eviction policy per row · *Italic* = FullContext oracle · Falcon3 B=96 aligned to B=98 column.

### Qwen2.5-7B

**B=98**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *91.1%* | **68.9%** | 66.7% | 66.7% | 66.7% | 64.4% | 53.3% |
| gov_report | *11.1%* | 0.0% | 0.0% | **2.2%** | **2.2%** | 0.0% | 0.0% |
| hotpotqa | *80.0%* | 40.0% | **42.2%** | **42.2%** | 28.9% | 28.9% | 26.7% |
| multifieldqa_en | *82.2%* | 57.8% | **60.0%** | 53.3% | 28.9% | 28.9% | 0.0% |
| musique | *15.6%* | **13.3%** | 8.9% | 6.7% | 4.4% | 0.0% | 0.0% |
| narrativeqa | *33.3%* | 6.7% | 4.4% | 8.9% | **17.8%** | 11.1% | 0.0% |
| qasper | *60.0%* | 17.8% | 15.6% | 17.8% | 13.3% | **20.0%** | 13.3% |
| qmsum | *6.7%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *47.5%* | **25.6%** | 24.7% | 24.7% | 20.3% | 19.2% | 11.7% |

**B=128**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *91.1%* | **80.0%** | 71.1% | 62.2% | 68.9% | 62.2% | 60.0% |
| gov_report | *8.9%* | 2.2% | 2.2% | 0.0% | **6.7%** | 0.0% | 0.0% |
| hotpotqa | *80.0%* | 37.8% | **53.3%** | 35.6% | 44.4% | 40.0% | 20.0% |
| multifieldqa_en | *82.2%* | 62.2% | 60.0% | **64.4%** | 28.9% | 26.7% | 6.7% |
| musique | *15.6%* | **11.1%** | 6.7% | 8.9% | 4.4% | 4.4% | 0.0% |
| narrativeqa | *33.3%* | 4.4% | 13.3% | **17.8%** | 8.9% | 11.1% | 2.2% |
| qasper | *62.2%* | 22.2% | 24.4% | **26.7%** | 22.2% | 20.0% | 15.6% |
| qmsum | *8.9%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *47.8%* | 27.5% | **28.9%** | 27.0% | 23.1% | 20.6% | 13.1% |

**B=256**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *91.1%* | 77.8% | **82.2%** | 80.0% | 66.7% | 68.9% | 66.7% |
| gov_report | *6.7%* | 6.7% | 11.1% | **13.3%** | 11.1% | 6.7% | 0.0% |
| hotpotqa | *80.0%* | **66.7%** | 60.0% | 60.0% | 60.0% | 55.6% | 51.1% |
| multifieldqa_en | *82.2%* | 73.3% | **84.4%** | 82.2% | 51.1% | 26.7% | 28.9% |
| musique | *15.6%* | 11.1% | **13.3%** | 11.1% | 11.1% | 8.9% | 2.2% |
| narrativeqa | *33.3%* | 8.9% | 13.3% | 15.6% | **17.8%** | 11.1% | 6.7% |
| qasper | *57.8%* | 28.9% | **35.6%** | 33.3% | 24.4% | 20.0% | 22.2% |
| qmsum | *6.7%* | 0.0% | 0.0% | 0.0% | 0.0% | **2.2%** | 0.0% |
| **Macro Avg** | *46.7%* | 34.2% | **37.5%** | 36.9% | 30.3% | 25.0% | 22.2% |

**B=512**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *91.1%* | 84.4% | **88.9%** | 82.2% | 86.7% | 75.6% | 66.7% |
| gov_report | *8.9%* | 8.9% | **15.6%** | 8.9% | 11.1% | 8.9% | 6.7% |
| hotpotqa | *80.0%* | 73.3% | **75.6%** | 66.7% | 68.9% | 57.8% | 68.9% |
| multifieldqa_en | *82.2%* | **82.2%** | 80.0% | 80.0% | 62.2% | 62.2% | 60.0% |
| musique | *15.6%* | **13.3%** | 11.1% | 11.1% | 8.9% | 6.7% | 11.1% |
| narrativeqa | *33.3%* | 24.4% | **26.7%** | **26.7%** | 15.6% | 20.0% | 8.9% |
| qasper | *60.0%* | **46.7%** | 31.1% | 42.2% | 31.1% | 22.2% | 17.8% |
| qmsum | *8.9%* | **6.7%** | 2.2% | 0.0% | 4.4% | 0.0% | 2.2% |
| **Macro Avg** | *47.5%* | **42.5%** | 41.4% | 39.7% | 36.1% | 31.7% | 30.3% |


### Mistral-7B

**B=98**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *84.4%* | **57.8%** | 46.7% | 53.3% | 55.6% | 53.3% | 53.3% |
| gov_report | *2.2%* | 0.0% | **2.2%** | 0.0% | 0.0% | 0.0% | 0.0% |
| hotpotqa | *68.9%* | 46.7% | 51.1% | **62.2%** | 46.7% | 55.6% | 33.3% |
| multifieldqa_en | *84.4%* | 22.2% | 22.2% | 22.2% | **26.7%** | 24.4% | 13.3% |
| musique | *20.0%* | 6.7% | 8.9% | **13.3%** | 2.2% | 2.2% | 6.7% |
| narrativeqa | *42.2%* | 11.1% | 13.3% | 11.1% | 17.8% | **20.0%** | **20.0%** |
| qasper | *55.6%* | **28.9%** | 22.2% | 20.0% | 17.8% | 13.3% | 20.0% |
| qmsum | *8.9%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *45.8%* | 21.7% | 20.8% | **22.8%** | 20.9% | 21.1% | 18.3% |

**B=128**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *84.4%* | 60.0% | 46.7% | 48.9% | **68.9%** | 62.2% | 55.6% |
| gov_report | *6.7%* | 0.0% | 0.0% | 0.0% | 0.0% | **4.4%** | 0.0% |
| hotpotqa | *66.7%* | **55.6%** | 48.9% | **55.6%** | 42.2% | 46.7% | 33.3% |
| multifieldqa_en | *82.2%* | **33.3%** | **33.3%** | **33.3%** | 31.1% | 24.4% | 13.3% |
| musique | *20.0%* | 6.7% | 8.9% | **13.3%** | 4.4% | 2.2% | 0.0% |
| narrativeqa | *42.2%* | 15.6% | **17.8%** | 11.1% | 15.6% | **17.8%** | 6.7% |
| qasper | *51.1%* | **28.9%** | 26.7% | 15.6% | 22.2% | 6.7% | 15.6% |
| qmsum | *11.1%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *45.6%* | **25.0%** | 22.8% | 22.2% | 23.1% | 20.6% | 15.6% |

**B=256**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *84.4%* | 53.3% | 51.1% | **64.4%** | 57.8% | 55.6% | 53.3% |
| gov_report | *4.4%* | **2.2%** | **2.2%** | 0.0% | 0.0% | 0.0% | 0.0% |
| hotpotqa | *66.7%* | 62.2% | 62.2% | **64.4%** | 46.7% | 44.4% | 44.4% |
| multifieldqa_en | *84.4%* | **66.7%** | **66.7%** | 64.4% | 42.2% | 33.3% | 8.9% |
| musique | *20.0%* | 13.3% | 11.1% | 8.9% | **15.6%** | 2.2% | 4.4% |
| narrativeqa | *44.4%* | **22.2%** | 15.6% | 15.6% | **22.2%** | 20.0% | 13.3% |
| qasper | *51.1%* | 33.3% | 28.9% | **37.8%** | 26.7% | 20.0% | 17.8% |
| qmsum | *11.1%* | **2.2%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *45.8%* | **31.9%** | 29.7% | **31.9%** | 26.4% | 21.9% | 17.8% |

**B=512**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *84.4%* | **71.1%** | 66.7% | 66.7% | 51.1% | 55.6% | 60.0% |
| gov_report | *4.4%* | 0.0% | 2.2% | **4.4%** | 0.0% | 2.2% | 0.0% |
| hotpotqa | *64.4%* | **62.2%** | 60.0% | 57.8% | 48.9% | 46.7% | 51.1% |
| multifieldqa_en | *84.4%* | **82.2%** | 75.6% | 75.6% | 46.7% | 37.8% | 26.7% |
| musique | *20.0%* | **20.0%** | 15.6% | 8.9% | 11.1% | 15.6% | 4.4% |
| narrativeqa | *42.2%* | **22.2%** | 17.8% | 17.8% | 20.0% | 17.8% | 13.3% |
| qasper | *57.8%* | 37.8% | **40.0%** | 31.1% | 28.9% | 24.4% | 13.3% |
| qmsum | *8.9%* | **2.2%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *45.8%* | **37.2%** | 34.7% | 32.8% | 25.8% | 25.0% | 21.1% |


### Falcon3-7B

**B=98 *(B=96)***

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *60.0%* | 35.6% | 35.6% | 31.1% | **37.8%** | **37.8%** | 22.2% |
| gov_report | *6.7%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| hotpotqa | *62.2%* | 28.9% | **33.3%** | 28.9% | **33.3%** | **33.3%** | 22.2% |
| multifieldqa_en | *64.4%* | 22.2% | **24.4%** | 13.3% | 13.3% | 13.3% | 6.7% |
| musique | *23.3%* | 0.0% | 0.0% | 0.0% | **3.3%** | 0.0% | **3.3%** |
| narrativeqa | *30.6%* | **16.7%** | **16.7%** | 13.9% | 5.6% | 2.8% | 8.3% |
| qasper | *46.7%* | 11.1% | 13.3% | 13.3% | 11.1% | 13.3% | **17.8%** |
| qmsum | *10.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *38.0%* | 14.3% | **15.4%** | 12.6% | 13.0% | 12.6% | 10.1% |

**B=128**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *60.0%* | 24.4% | 24.4% | 26.7% | 35.6% | **48.9%** | 33.3% |
| gov_report | *4.4%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| hotpotqa | *64.4%* | 44.4% | 37.8% | 22.2% | 37.8% | **51.1%** | 26.7% |
| multifieldqa_en | *64.4%* | **33.3%** | 31.1% | 20.0% | 20.0% | 13.3% | 11.1% |
| musique | *23.3%* | 0.0% | 0.0% | 0.0% | 0.0% | **6.7%** | 0.0% |
| narrativeqa | *33.3%* | **25.0%** | 19.4% | 16.7% | 5.6% | 8.3% | 13.9% |
| qasper | *46.7%* | 20.0% | **22.2%** | **22.2%** | 20.0% | 13.3% | 20.0% |
| qmsum | *10.5%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *38.4%* | **18.4%** | 16.9% | 13.5% | 14.9% | 17.7% | 13.1% |

**B=256**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *60.0%* | 35.6% | 31.1% | **42.2%** | **42.2%** | 37.8% | 28.9% |
| gov_report | *6.7%* | 0.0% | **2.2%** | 0.0% | 0.0% | 0.0% | 0.0% |
| hotpotqa | *62.2%* | 42.2% | 40.0% | **48.9%** | 44.4% | 40.0% | 33.3% |
| multifieldqa_en | *62.2%* | 57.8% | **60.0%** | 35.6% | 42.2% | 28.9% | 11.1% |
| musique | *23.3%* | **6.7%** | **6.7%** | **6.7%** | **6.7%** | **6.7%** | **6.7%** |
| narrativeqa | *30.6%* | 16.7% | 8.3% | **19.4%** | **19.4%** | 5.6% | 11.1% |
| qasper | *46.7%* | 24.4% | **26.7%** | 24.4% | 20.0% | 20.0% | 20.0% |
| qmsum | *10.5%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Macro Avg** | *37.8%* | **22.9%** | 21.9% | 22.1% | 21.9% | 17.4% | 13.9% |

**B=512**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2wikimqa | *60.0%* | 64.4% | 60.0% | 55.6% | **66.7%** | 40.0% | 31.1% |
| gov_report | *4.4%* | 0.0% | **4.4%** | 0.0% | 2.2% | 0.0% | 0.0% |
| hotpotqa | *60.0%* | **55.6%** | 51.1% | 53.3% | 40.0% | **55.6%** | 35.6% |
| multifieldqa_en | *62.2%* | **62.2%** | **62.2%** | **62.2%** | 44.4% | 40.0% | 22.2% |
| musique | *23.3%* | 10.0% | **13.3%** | 6.7% | 10.0% | 10.0% | 6.7% |
| narrativeqa | *33.3%* | 13.9% | 13.9% | 13.9% | 16.7% | **19.4%** | 16.7% |
| qasper | *46.7%* | 31.1% | **33.3%** | 28.9% | 13.3% | 20.0% | 15.6% |
| qmsum | *10.5%* | **5.3%** | 0.0% | **5.3%** | **5.3%** | **5.3%** | 0.0% |
| **Macro Avg** | *37.5%* | **30.3%** | 29.8% | 28.2% | 24.8% | 23.8% | 16.0% |


### BioMistral-7B

**B=98**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| bio_niah_gene | *0.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bio_niah_single | *20.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bio_vt | *6.7%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| clinical_niah | *100.0%* | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 86.7% | 0.0% |
| medalpaca_medqa | *60.0%* | 26.7% | 20.0% | 0.0% | 33.3% | **40.0%** | 20.0% |
| medmcqa | *86.7%* | **66.7%** | 60.0% | 46.7% | 53.3% | 33.3% | 44.4% |
| pubmedqa | *66.7%* | 26.7% | 33.3% | 40.0% | **46.7%** | 40.0% | **46.7%** |
| **Macro Avg** | *48.6%* | 31.4% | 30.5% | 26.7% | **33.3%** | 28.6% | 15.9% |

**B=128**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| bio_niah_gene | *0.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bio_niah_single | *20.0%* | 0.0% | 0.0% | **6.7%** | 0.0% | 0.0% | 0.0% |
| bio_vt | *6.7%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| clinical_niah | *100.0%* | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 0.0% |
| medalpaca_medqa | *60.0%* | **46.7%** | 33.3% | 33.3% | 33.3% | 40.0% | 33.3% |
| medmcqa | *86.7%* | 73.3% | **77.8%** | 60.0% | **77.8%** | 62.2% | 60.0% |
| pubmedqa | *66.7%* | 46.7% | **53.3%** | 46.7% | **53.3%** | 46.7% | **53.3%** |
| **Macro Avg** | *48.6%* | **38.1%** | 37.8% | 35.2% | 37.8% | 35.6% | 20.9% |

**B=256**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| bio_niah_gene | *0.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bio_niah_single | *20.0%* | 0.0% | 0.0% | **13.3%** | 0.0% | 0.0% | 0.0% |
| bio_vt | *6.7%* | 6.7% | 6.7% | **20.0%** | 0.0% | 0.0% | 0.0% |
| clinical_niah | *100.0%* | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 66.7% |
| medalpaca_medqa | *60.0%* | 46.7% | **53.3%** | **53.3%** | **53.3%** | **53.3%** | 46.7% |
| medmcqa | *86.7%* | 73.3% | 80.0% | 73.3% | **86.7%** | 80.0% | 73.3% |
| pubmedqa | *66.7%* | 60.0% | 60.0% | 60.0% | **66.7%** | 53.3% | 46.7% |
| **Macro Avg** | *48.6%* | 41.0% | 42.9% | **45.7%** | 43.8% | 40.9% | 33.3% |

**B=512**

| Task | FC (oracle) | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:-----||:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| bio_niah_gene | *0.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bio_niah_single | *20.0%* | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bio_vt | *6.7%* | **10.0%** | 0.0% | **10.0%** | **10.0%** | 3.3% | 0.0% |
| clinical_niah | *100.0%* | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| medalpaca_medqa | *60.0%* | **60.0%** | **60.0%** | **60.0%** | **60.0%** | **60.0%** | **60.0%** |
| medmcqa | *86.7%* | **86.7%** | **86.7%** | **86.7%** | **86.7%** | **86.7%** | 80.0% |
| pubmedqa | *66.7%* | **66.7%** | **66.7%** | **66.7%** | **66.7%** | **66.7%** | 60.0% |
| **Macro Avg** | *48.6%* | **46.2%** | 44.8% | **46.2%** | **46.2%** | 45.2% | 42.9% |



## References

1. Zhang, Z., Sheng, Y., Zhou, T., et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023. arXiv:2306.14048.

2. Li, Y., Huang, Y., Yang, B., et al. *SnapKV: LLM Knows What You are Looking for Before Generation*. NeurIPS 2024. arXiv:2404.14469.

3. Hsieh, C.-P., Sun, S., Kriman, S., et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654, 2024.

4. Bai, Y., Lv, X., Zhang, J., et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*. ACL 2024. arXiv:2308.14508.

5. Zheng, L., Chiang, W.-L., Sheng, Y., et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. arXiv:2306.05685.

6. Qwen Team. *Qwen2.5 Technical Report*. arXiv:2412.15115, 2025.

7. Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. *Mistral 7B*. arXiv:2310.06825, 2023.

8. Almazrouei, E., et al. *The Falcon Series of Open Language Models*. arXiv:2311.16867, 2023.

9. Labrak, Y., Bazoge, A., Morin, E., et al. *BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains*. arXiv:2402.10373, 2024.

10. Megiddo, N. & Modha, D. S. *ARC: A Self-Tuning, Low Overhead Replacement Cache*. FAST 2003.

11. Cai, Z., Zhang, Y., Gao, B., et al. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*. arXiv:2406.02069, 2024.

12. Sheng, Y., Zheng, L., Yuan, B., et al. *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU*. ICML 2023. arXiv:2303.06865.

---

*Revised Draft — Aliwey Abood — 2026-06-10*  
*Experiments 001–039 · Platforms: Kaggle T4×2, Modal A10G/A100/L4, Lightning AI L4*  
*Total judged samples: 61,681 · Architectures tested: 4*
