# KiaOmni Qwen3 RULER NIAH V1 — Runbook

## Goal

Run a controlled long-context retention audit using official-code-generated RULER NIAH data.

Frozen suite:

- tasks:
  - niah_single_2
  - niah_multikey_1
  - niah_multivalue
  - niah_multiquery
- lengths:
  - 8192
  - 16384
- five deterministic answer-depth strata per task/length
- total cases: 40
- percentage budgets:
  - 25%
  - 12.5%
  - 6.25%

The RULER source data is mirrored from `VenusChenyy/RULER_50`, but every source file is checked against the SHA256 recorded in its official-generation manifest. That manifest pins NVIDIA/RULER generation commit:

```text
38da79d79519ef87aa46ae804f838e1eab7f86d7
```

## Pull branch

```powershell
git fetch origin
git checkout exp/kiaomni-qwen3-practical-budget-frontier-v1
git pull origin exp/kiaomni-qwen3-practical-budget-frontier-v1
```

## Local validation

```powershell
python -m pytest tests/test_phase03_percentage_replay.py tests/test_phase03_ruler_niah.py -q
```

## Prepare RULER assets + preflight

This downloads only the eight required 8K/16K task files, verifies their manifest SHA256 values, freezes the 40-case depth-stratified index, then runs one GPU case.

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage preflight --prepare-ruler
```

If the RULER index already exists and was prepared successfully before, use:

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage preflight
```

## Final RULER run

Run only after preflight PASS:

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage final
```

## Download artifacts

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/final.json .\ruler_final.json
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/final.log .\ruler_final.log
```

Optional preflight:

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/preflight.json .\ruler_preflight.json
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/preflight.log .\ruler_preflight.log
```

## Build plots

```powershell
python -m pip install matplotlib
python analysis/phase03_ruler_report.py --ruler .\ruler_final.json --outdir .\ruler_report
```

Default heatmap uses:

```text
kiaomni_s8_r0.125
```

To render another method:

```powershell
python analysis/phase03_ruler_report.py --ruler .\ruler_final.json --outdir .\ruler_report_r25 --heatmap-method kiaomni_s8_r0.25
```

## Outputs

```text
ruler_report/ruler_cases.csv
ruler_report/ruler_global_summary.csv
ruler_report/ruler_depth_summary.csv
ruler_report/ruler_plot_data.json
ruler_report/ruler_quality_vs_compression.png
ruler_report/ruler_needle_survival_vs_compression.png
ruler_report/heatmap_score_kiaomni_s8_r0.125_8192.png (plus per-method/per-length heatmaps)
ruler_report/ruler_survival_vs_quality_scatter.png
```

## Main metrics

Quality:
- RULER official-compatible all-reference string-match fraction
- all-required-answers-correct rate
- 95% bootstrap confidence interval
- gold-answer NLL/PPL

Selection mechanism:
- required-answer token recall
- complete required-answer survival
- per-answer token span
- answer depth
- depth-bin robustness

Paired FullContext comparison:
- FullContext-success preservation
- regressions
- rescues
- exact McNemar p-value
- score delta

Actual MoE routing:
- top-1 agreement
- top-8 Jaccard
- dispatch-weight cosine
- expert-load JSD
- median/worst-layer routing
- full layerwise routing values

Systems:
- measured time to first generated token
- decode-after-first-token time
- generation time
- saliency time
- output tokens/s
- generation peak VRAM
- saliency peak VRAM
- end-to-end measured pipeline peak VRAM

## Claim boundary

This 40-case suite is a controlled, pre-frozen RULER audit for mechanism and retention analysis. It must not be presented as a full official 500-sample-per-task RULER leaderboard submission.

It remains prompt-side pruning. Runtime KV-cache eviction is not yet implemented.
