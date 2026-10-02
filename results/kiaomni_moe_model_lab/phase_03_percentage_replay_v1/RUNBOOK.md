# KiaOmni Qwen3 Practical Frontier V1 — Runbook

## Branch

```text
exp/kiaomni-qwen3-practical-budget-frontier-v1
```

Base scientific recovery point:

```text
95cda0f
```

This branch does not rewrite the frozen Phase-03 fixed-budget results.

---

# A. Pull and validate locally

```powershell
git fetch origin
git checkout exp/kiaomni-qwen3-practical-budget-frontier-v1
git pull origin exp/kiaomni-qwen3-practical-budget-frontier-v1
```

Run the two new protocol tests:

```powershell
python -m pytest tests/test_phase03_percentage_replay.py tests/test_phase03_ruler_niah.py -q
```

---

# B. Priority 1 — exact LongBench-27 percentage replay

This re-runs the exact frozen 27 LongBench-v2 cases with:

```text
25%
12.5%
6.25%
```

for both frozen policies:

```text
kiaomni_s8
kiaomni_gaussian
```

The old fixed-budget results remain the comparison baseline:

```text
B512
B256
B128
B98
```

## B1. Percentage preflight

The adjudication index should already exist in `kiaomni-qwen3-assets`:

```powershell
modal run modal/qwen3_30b_percentage_replay_modal.py --stage preflight
```

Only if Modal says the frozen adjudication index is missing:

```powershell
modal run modal/qwen3_30b_percentage_replay_modal.py --stage preflight --prepare-index
```

## B2. Percentage final

Run only after the percentage preflight artifact reports PASS:

```powershell
modal run modal/qwen3_30b_percentage_replay_modal.py --stage final
```

Follow logs:

```powershell
modal app logs kiaomni-qwen3-percentage-replay-v1
```

## B3. Download percentage results

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/final.json .\percentage_final.json
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/final.log .\percentage_final.log
```

Optional preflight artifacts:

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/preflight.json .\percentage_preflight.json
modal volume get kiaomni-qwen3-frontier-results phase_03_percentage_replay_v1/preflight.log .\percentage_preflight.log
```

---

# C. Build fixed-vs-percentage paper/company plots

Download the frozen fixed-budget artifact:

```powershell
modal volume get kiaomni-qwen3-adjudication-results phase_03_adjudication_routing_v1/final.json .\fixed_final.json
```

Build the report:

```powershell
python -m pip install matplotlib
python analysis/phase03_frontier_report.py --fixed .\fixed_final.json --percentage .\percentage_final.json --outdir .\frontier_report
```

Core outputs:

```text
frontier_report/frontier_summary.csv
frontier_report/frontier_pairwise.csv
frontier_report/fullcontext_reproducibility.csv
frontier_report/fullcontext_reproducibility.json
frontier_report/frontier_plot_data.json
frontier_report/quality_vs_compression.png
frontier_report/routing_vs_compression.png
frontier_report/memory_vs_compression.png
frontier_report/ppl_vs_compression.png
frontier_report/ttft_vs_compression.png
frontier_report/preservation_vs_compression.png
```

The fixed rows are reconstructed from their actual kept/input token counts, so fixed and percentage conditions are plotted on the same effective-compression axis.

---

# D. Priority 2 — controlled RULER NIAH suite

Frozen suite:

```text
Official-code-generated RULER data mirror:
VenusChenyy/RULER_50

Official generation provenance:
NVIDIA/RULER
commit 38da79d79519ef87aa46ae804f838e1eab7f86d7

Tasks:
niah_single_2
niah_multikey_1
niah_multivalue
niah_multiquery

Lengths:
8192
16384

Depth strata per task/length:
0–20%
20–40%
40–60%
60–80%
80–100%

Cases:
4 tasks × 2 lengths × 5 depths = 40
```

The preparation stage downloads only the eight required JSONL files, resolves and freezes the Hugging Face dataset revision, verifies every source file against the SHA256 in the generation manifest, then freezes the 40 selected source lines before GPU evaluation.

The stored RULER `answer_prefix` is explicitly reattached before applying the Qwen chat template.

## D1. First RULER preflight + asset preparation

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage preflight --prepare-ruler
```

Future preflights can omit `--prepare-ruler` after the frozen asset/index exists:

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage preflight
```

Follow logs:

```powershell
modal app logs kiaomni-qwen3-ruler-niah-v1
```

## D2. RULER final

Run after RULER preflight PASS:

```powershell
modal run modal/qwen3_30b_ruler_niah_modal.py --stage final
```

## D3. Download RULER results

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/final.json .\ruler_niah_final.json
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/final.log .\ruler_niah_final.log
```

Optional preflight artifacts:

```powershell
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/preflight.json .\ruler_niah_preflight.json
modal volume get kiaomni-qwen3-frontier-results phase_03_ruler_niah_v1/preflight.log .\ruler_niah_preflight.log
```

---

# E. Build RULER paper/company plots

```powershell
python analysis/phase03_ruler_report.py --ruler .\ruler_niah_final.json --outdir .\ruler_report
```

Core outputs include:

```text
ruler_report/ruler_global_summary.csv
ruler_report/ruler_cases.csv
ruler_report/ruler_pairwise.csv
ruler_report/ruler_depth_summary.csv
ruler_report/ruler_plot_data.json
ruler_report/ruler_quality_vs_compression.png
ruler_report/ruler_needle_survival_vs_compression.png
ruler_report/ruler_routing_vs_compression.png
ruler_report/ruler_ttft_vs_compression.png
ruler_report/ruler_survival_vs_quality_scatter.png
ruler_report/heatmap_score_<method>_<length>.png
```

---

# F. Metrics frozen for the new runs

## Quality

```text
LongBench:
accuracy
95% bootstrap CI
gold-answer NLL/PPL
parse rate

RULER:
official-compatible string_match_all score
all-required-outputs correctness
95% bootstrap CI
gold-answer NLL/PPL
```

## FullContext paired behavior

```text
preservation rate
regression rate
rescue rate
net rescues - regressions
exact McNemar p-value
```

## Compression

```text
requested retention %
actual retention %
kept tokens
effective compression ratio
```

## RULER evidence survival

```text
required-answer token recall
complete required-answer survival
all-required-answers-complete rate
per-answer token spans
answer/needle depth
depth-bin performance
```

## Actual MoE routing

```text
top-1 expert agreement
top-8 set Jaccard
dispatch-weight cosine
entropy delta
expert-load JSD
median layer routing
worst layer routing
full layerwise values
actual expert-dispatch verification
```

## Systems

```text
generation peak VRAM
saliency peak VRAM
pipeline peak VRAM
output tokens/s
generation time
measured time to first generated token
decode time after first token
saliency time
inference-path time
measurement-pipeline time
```

---

# G. Scientific boundary

These experiments still test prompt-side token selection followed by a fresh generation pass.

They do not yet prove:

```text
runtime past_key_values eviction
production KV-cache mutation
production end-to-end memory reduction
production end-to-end latency reduction
```

The fixed-budget Phase-03 evidence remains frozen and is used only as the historical comparison track.


## Baseline reproducibility audit

Because the fixed-budget artifact and percentage artifact were produced in separate runs, the report explicitly compares the 27 FullContext rows case-by-case.

It records:

```text
parsed-answer agreement
correctness agreement
old vs replay FullContext correct count
gold-PPL absolute drift
```

Treat fixed-vs-percentage conclusions as clean only after inspecting this audit.
