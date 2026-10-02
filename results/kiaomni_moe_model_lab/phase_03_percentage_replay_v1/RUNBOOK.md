# KiaOmni Qwen3 Practical Frontier V1 — Runbook

## Immediate goal

Replay the exact frozen 27 LongBench-v2 cases with the historical percentage budgets:

- 25%
- 12.5%
- 6.25%

The fixed-budget results at B512/B256/B128/B98 remain frozen in commit `95cda0f` and are merged only at reporting time.

## Branch

```text
exp/kiaomni-qwen3-practical-budget-frontier-v1
```

## Pull

```powershell
git fetch origin
git checkout exp/kiaomni-qwen3-practical-budget-frontier-v1
git pull origin exp/kiaomni-qwen3-practical-budget-frontier-v1
```

## Local validation

```powershell
python -m pytest tests/test_phase03_percentage_replay.py -q
```

## Modal preflight

The frozen adjudication index should already exist in the `kiaomni-qwen3-assets` volume from Phase 03.

```powershell
modal run modal/qwen3_30b_percentage_replay_modal.py --stage preflight
```

If Modal reports that the adjudication index is missing, rebuild the index from the frozen parent 27 IDs only:

```powershell
modal run modal/qwen3_30b_percentage_replay_modal.py --stage preflight --prepare-index
```

## Modal final

Only run after the new percentage preflight reports PASS.

```powershell
modal run modal/qwen3_30b_percentage_replay_modal.py --stage final
```

## Download new percentage artifacts

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/final.json .\percentage_final.json
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/final.log .\percentage_final.log
```

Optional preflight artifacts:

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/preflight.json .\percentage_preflight.json
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/preflight.log .\percentage_preflight.log
```

## Download frozen fixed-budget artifact

```powershell
modal volume get kiaomni-qwen3-adjudication-results phase_03_adjudication_routing_v1/final.json .\fixed_final.json
```

## Build paper/company report

```powershell
python -m pip install matplotlib
python analysis/phase03_frontier_report.py --fixed .\fixed_final.json --percentage .\percentage_final.json --outdir .\frontier_report
```

Outputs:

```text
frontier_report/frontier_summary.csv
frontier_report/frontier_pairwise.csv
frontier_report/frontier_plot_data.json
frontier_report/quality_vs_compression.png
frontier_report/routing_vs_compression.png
frontier_report/memory_vs_compression.png
frontier_report/preservation_vs_compression.png
```

## Frozen semantics

Percentage budget per case:

```text
budget = round(input_tokens * requested_ratio)
budget = clamp(budget, N_SINK + RECENCY, input_tokens - 1)
```

with:

```text
N_SINK = 16
RECENCY = 32
```

This matches the historical Phase-03 ratio-budget semantics.

## Recorded metrics

Quality:
- accuracy
- 95% bootstrap accuracy CI
- parse rate
- gold-answer NLL/PPL

Paired FullContext comparison:
- FullContext-correct preservation rate
- regression rate
- rescue rate
- net rescues minus regressions
- exact McNemar p-value

Compression:
- requested retention %
- actual retention %
- kept tokens
- effective compression ratio

Actual MoE routing:
- top-1 expert agreement
- top-8 set Jaccard
- dispatch-weight cosine
- entropy delta
- expert-load JSD
- median and worst layer top-1 agreement
- median and worst layer top-8 Jaccard
- full layerwise routing values

Systems:
- generation peak VRAM
- saliency peak VRAM
- pipeline peak VRAM
- output tokens/s
- generation time
- saliency time
- inference-path elapsed time
- measurement-pipeline elapsed time

## Scientific boundary

This run is a paired percentage-budget replay on the exact same 27 LongBench-v2 cases. It does not replace the fixed-budget evidence and it does not claim runtime KV-cache eviction.


---

# Fast official RULER NIAH frontier

This is the second-stage controlled retention test. It is pinned to:

```text
NVIDIA/RULER
revision c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
```

Fast same-day suite:

```text
tasks:
  niah_single_1
  niah_multikey_2
  niah_multikey_3

lengths:
  8192
  16384

samples/task/length:
  4

ratios:
  25%
  12.5%
  6.25%
```

The Modal preparation step uses upstream `scripts/data/prepare.py` with the frozen Qwen HF tokenizer and seed 42.

## RULER preflight + official data generation

Run this after launching the LongBench percentage final, or in parallel if your Modal concurrency/budget allows it:

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage preflight --prepare-data
```

If the RULER assets were already prepared successfully, later preflights can omit `--prepare-data`.

## RULER final

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage final
```

## Download RULER results

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_frontier_v1/final.json .\ruler_niah_final.json
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_frontier_v1/final.log .\ruler_niah_final.log
```

RULER-specific metrics include:

```text
official-style string_match_all score
all-reference accuracy
gold/reference token recall after pruning
complete-reference survival
all-references-survived rate
needle depth
gold PPL
actual MoE routing
VRAM
throughput
latency components
```

The RULER fast suite is a same-day controlled validation, not the final paper-scale RULER sample count.
