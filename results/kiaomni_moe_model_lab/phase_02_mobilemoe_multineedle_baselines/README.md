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
| BlockSal | our internal method | historical whole-block mean-saliency selector |
| SnapKV | external baseline | NVIDIA kvpress `SnapKVPress` |
| StreamingLLM | external baseline | NVIDIA kvpress `StreamingLLMPress` |

AdaSnapKV is intentionally excluded from this phase. H2O is deferred until its
implementation is independently parity-validated.

### BlockSal disclosure

BlockSal is **not** an external SnapKV baseline. It is an internal design that
groups evictable prompt positions into blocks and applies our saliency-driven
whole-block selection logic. The implementation preserves the historical
whole-block behavior rather than silently forcing exact token budgets.

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

External methods do not enter benchmark tables unless they first pass:

1. The press installs and executes on MobileMoE without exception.
2. A prefill at requested budget 256 produces a measurable KV cache.
3. The measured cache length is exactly 256.
4. One-token greedy generation advances successfully.

A failed method is recorded as `VALIDATION_FAIL` and is skipped instead of
producing a misleading score.

KiaOmni must also report exactly 256 retained prompt tokens in its validation
case. BlockSal has a separate frozen historical-semantics gate.

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

## Kaggle setup

Use one T4 for the model. The Phase-02 experiment requires
`transformers==4.57.6`.

```python
%pip install -q \
  "git+https://github.com/NVIDIA/kvpress.git@7331c23da9e6f1510d89ea651d0dea77a57b3252"

%pip install -q \
  "transformers==4.57.6" \
  "accelerate==1.13.0" \
  scipy
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
