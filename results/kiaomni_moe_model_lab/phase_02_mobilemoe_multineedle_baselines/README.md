# Phase 02 — MobileMoE Multi-Needle + Baseline Validation

## Goal

Test KiaOmni on `facebook/MobileMoE-M-SFT` at Kaggle-friendly ~4K final
prompt length against independently sourced external baselines and the internal
BlockSal variant.

This phase does **not** test adaptive MoE route intervention yet. Routing alpha
ablations remain a later phase after the context-compression comparison is
validated.

## Methods

| Method | Classification | Source / semantics |
|---|---|---|
| FullContext | upper bound | no compression |
| KiaOmni-s8 | our method | current KiaOmni prompt-side saliency compression |
| BlockSal | our internal method | mean-saliency block ranking with exact-budget boundary adapter |
| SnapKV | external baseline | NVIDIA kvpress `SnapKVPress` |
| StreamingLLM | external baseline | NVIDIA kvpress `StreamingLLMPress` |

AdaSnapKV is intentionally excluded from this phase. H2O is deferred until its
implementation is independently parity-validated.

### BlockSal disclosure

BlockSal is **not** an external SnapKV baseline. It is an internal design that
groups evictable prompt positions into blocks and applies our saliency-driven
whole-block selection logic. The canonical Phase-02 implementation uses
`BLOCK_SIZE=16`, matching the full-comparison/paper Section 2.2 lineage. It
preserves the historical whole-block behavior rather than silently forcing
exact token budgets.

Because whole blocks are evicted, actual retained tokens may be up to
`block_size - 1` below the requested budget. Every result records
`actual_kept_tokens` and `budget_delta`.

## External baseline provenance

NVIDIA kvpress is pinned to:

`7331c23da9e6f1510d89ea651d0dea77a57b3252`

Package version at that ref: `0.5.5`.

SnapKV settings are frozen to the reference defaults used by the existing
KiaOmni demo:

- `window_size=64`
- `kernel_size=5`
- exact per-prompt retained-token ratio

StreamingLLM uses NVIDIA's `StreamingLLMPress(n_sink=4)` wrapped by
`KeyRerotationPress`, because kvpress explicitly requires key rerotation for
full paper-parity RoPE handling after pruning.

## Validation gate

External methods do not enter benchmark tables unless they first pass **every
budget in the requested grid** (512, 256, 128, 98 by default):

1. The press installs and executes on MobileMoE without exception.
2. Every observed transformer layer has exactly the requested KV length after
   prefill compression.
3. One-token greedy generation advances at that budget.
4. The kvpress context manager restores its forward hooks after the check.

The full comparison is fail-closed: if SnapKV or StreamingLLM fails validation
at any budget, the script writes a validation-failure artifact and refuses to
produce comparison scores. `--skip-external` exists only for internal
engineering checks and must not be used for the external-baseline comparison.

KiaOmni is also validated at every requested budget. BlockSal has a separate
grid validation that preserves its historical whole-block semantics and records
the actual kept-token count and budget delta.

## Benchmark

Final prompt target: `3900` tokens after the model chat template.

Hard limit: `4096` tokens.

Budgets:

`512, 256, 128, 98`

Tasks:

- `single`: one planted fact at varying depth.
- `multi`: three independent facts distributed early / middle / late.
- `hard_multi`: three target facts mixed with structurally matched distractor
  records from another project.
- `reason`: four-hop variable tracking plus a distractor chain.

Scoring records exact success, fact recall, distractor hits, latency,
tokens/second, peak allocated VRAM, and compression telemetry.

The artifact also reports **FullContext-conditioned** accuracy/recall: compressed
methods are evaluated separately on the subset of cases that MobileMoE itself
solves with no compression. Absolute scores are still retained for every case.
This prevents model capability failures from being misattributed to compression.

## Kaggle setup

Use one T4 for the model. The Phase-02 experiment requires
`transformers==4.57.6`.

```bash
%cd /kaggle/working/kiaomni
%pip install -q -r experiments/phase02_moe_baselines_requirements.txt
%pip install -q -e /kaggle/working/kiaomni --no-deps
```

Then update KiaOmni:

```bash
%cd /kaggle/working/kiaomni
!git fetch origin
!git switch exp/kiaomni-moe-model-lab
!git pull --ff-only
%pip install -q -e /kaggle/working/kiaomni --no-deps
!git rev-parse HEAD
```

### 1. Validation only

Run this first. Do not run the benchmark until the validation table is
understood.

```bash
!python /kaggle/working/kiaomni/experiments/kaggle_moe_phase02_multineedle_baselines.py \
  --validation-only \
  --output /kaggle/working/phase02_validation.json
```

### 2. One-sample smoke

Only after validation succeeds:

```bash
!python /kaggle/working/kiaomni/experiments/kaggle_moe_phase02_multineedle_baselines.py \
  --samples-per-task 1 \
  --output /kaggle/working/phase02_smoke_n1.json
```

### 3. Comparison run

After the smoke run is clean, use ten deterministic samples per task:

```bash
!python /kaggle/working/kiaomni/experiments/kaggle_moe_phase02_multineedle_baselines.py \
  --samples-per-task 10 \
  --output /kaggle/working/phase02_mobilemoe_multineedle_n10.json
```

The N=1 smoke is engineering validation only. The N=10 run is the first
comparison artifact intended for analysis.

## Implementation

Runner:

`experiments/kaggle_moe_phase02_multineedle_baselines.py`

CI semantics tests:

`tests/test_moe_phase02_benchmark.py`


## MobileMoE SnapKV compatibility adapter

SnapKV validation initially failed on MobileMoE before producing any score.
The Phase-02 runner now uses a narrow compatibility adapter in
`kiaomni/baselines/mobilemoe_snapkv.py`.

The adapter is allowed to change only architecture representation plumbing:

- reconstruct queries with MobileMoE's native `q_proj` + `q_norm`;
- normalize the model's RoPE representation into the form required by kvpress.

The following SnapKV semantics remain pinned to NVIDIA kvpress at
`7331c23da9e6f1510d89ea651d0dea77a57b3252`:

- observation window;
- attention-based importance equation;
- pooling kernel;
- GQA grouping;
- score ranking;
- KV Top-K pruning.

Validation is fail-closed. SnapKV is accepted only if:

1. native MobileMoE QK-Norm query reconstruction matches numerically;
2. zero-compression adapter output is bit-identical to the unadapted model;
3. all requested budgets produce exact per-layer KV lengths;
4. one-token generation advances;
5. hooks are restored after each press context.

If the model exposes a RoPE form that cannot be converted without changing
attention semantics, the adapter raises instead of approximating.


## SnapKV deferred status

SnapKV is **deferred from Phase 02 scoring** for MobileMoE.

The compatibility adapter can execute SnapKV, compress all 26 layers to the
requested budgets, and advance generation. Zero-compression identity is exact.
However, the adapter's reconstructed post-RoPE query does not match the native
MobileMoE query entering SDPA closely enough for a faithful external-baseline
claim.

Therefore Phase 02 proceeds with:

- FullContext
- KiaOmni-s8
- BlockSal
- StreamingLLM

Use `--exclude-methods snapkv` for the canonical Phase-02 comparison run.
SnapKV may be revisited later as a separate compatibility study.


## N25 final benchmark contract

The final MobileMoE comparison uses **25 deterministic samples per task**:

- 25 single-needle cases
- 25 multi-needle cases
- 25 hard multi-needle cases
- 25 reasoning cases

That is 100 cases total for each method/budget condition.

Active methods:

- FullContext
- KiaOmni-s8
- BlockSal
- RecencyOnly
- RandomRetention

SnapKV is deferred because MobileMoE query-representation parity is not
faithful enough for a clean external-baseline claim. StreamingLLM is also
deferred from N25 because the smoke run produced repetitive/degenerate
generation despite technically correct cache-length validation.

### Metrics saved per case

Quality:
- exact success
- fact recall
- FullContext-conditioned success

Efficiency:
- elapsed time
- end-to-end output tokens/second
- peak allocated VRAM
- peak reserved VRAM
- retained-token/compression telemetry

Likelihood:
- generated-answer mean log probability
- generated-answer NLL
- generated-answer PPL

Routing:
- decode-only projected raw Top-1 jitter
- raw Top-1 continuity
- raw Top-K Jaccard
- counterfactual stabilized jitter/continuity/Jaccard
- intervention rate / alpha / uncertainty / hidden similarity
- per-layer raw and counterfactual Top-1 expert sequences
- per-layer raw and counterfactual Top-K expert sequences
- mean/longest expert dwell and expert diversity

The raw route trace is non-mutating projected routing from the observed decode
hidden states and router weights. The counterfactual stable path is diagnostic
only and is **not** claimed as executed expert dispatch.

Statistical summaries include Wilson 95% confidence intervals and paired exact
McNemar comparisons for KiaOmni versus FullContext, BlockSal, RecencyOnly, and
RandomRetention.

Known repetitive SDPA/kvpress compatibility warnings are suppressed from stdout;
unexpected warnings and errors remain visible.

### Canonical N25 command

```bash
python experiments/kaggle_moe_phase02_multineedle_baselines.py \
  --exclude-methods snapkv,streamingllm \
  --samples-per-task 25 \
  --budgets 512,256,128,98 \
  --target-tokens 3900 \
  --max-context 4096 \
  --route-telemetry \
  --route-alpha-max 0.10 \
  --output /kaggle/working/phase02_mobilemoe_multineedle_n25.json
```
