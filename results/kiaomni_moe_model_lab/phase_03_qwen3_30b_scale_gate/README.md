# Phase 03 — Qwen3-30B-A3B Modal Scale Gate

## What this phase proves

This phase asks one narrow question:

**Does the current prompt-side KiaOmni selection policy scale from MobileMoE to Qwen3-30B-A3B-Instruct-2507 without collapsing quality?**

It does **not** claim production KV-cache eviction. The current public KiaOmni path still scores the full prompt, selects retained token positions, and generates from the shorter prompt. Real `past_key_values` eviction is Phase 04.

## Frozen model and benchmark

Model:

- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- revision `0d7cf23`
- BF16
- SDPA
- no quantization
- no CPU/disk offload

Real benchmark:

- `THUDM/LongBench-v2`
- revision `b0db4901b856522026b7353ab541b8535ff2a4b8`
- naturally fitting 8K–16K examples only
- no truncation
- exact multiple-choice scoring

Controlled synthetic tasks remain:

- single retrieval
- multi-record retrieval
- distractor-heavy relational binding
- chained reasoning

## Important correction from the earlier scaffold

Phase 02 used KiaOmni's package defaults:

- `n_sink=16`
- `recency=32`

The first Phase-03 scaffold accidentally used `recency=256`. That would have changed the algorithm and weakened comparability, so V2 restores `recency=32`.

Saliency is also computed once per case and reused across 4x / 8x / 16x retention conditions. This avoids paying for three identical full-prompt saliency forwards.

For Qwen3 BF16, the Phase-03 runner performs FP32 saliency math on GPU instead of offloading Q/K to CPU. The preflight compares this path against the historical CPU-offload path on a short frozen case and fails closed unless:

- Pearson correlation >= 0.999
- Top-128 Jaccard >= 0.98

The default KiaOmni library behavior is not changed.

## Cost contract

Large model and dataset downloads run in a CPU-only Modal function and are stored in a persistent Volume before any GPU is allocated.

| Stage | Work | Hard GPU ceiling |
|---|---|---:|
| preflight | load + probe + saliency parity + 1 synthetic case | 20 min |
| smoke | 4 synthetic task families | 40 min |
| final | 8 synthetic + 6 LongBench-v2 cases | 90 min |

A one-pass run therefore has a hard configured ceiling of **150 A100 GPU minutes**. Re-running failed stages consumes additional credit and is outside this contract.

## Retention conditions

The final run uses per-case retention ratios rather than fixed token budgets:

- 25% retained ≈ 4x compression
- 12.5% retained ≈ 8x compression — primary operating point
- 6.25% retained ≈ 16x compression — stress diagnostic

At 8x the final run also evaluates deterministic RecencyOnly and RandomRetention controls.

## Final PASS / FAIL / INCONCLUSIVE gate

PASS requires all of the following:

- synthetic FullContext-conditioned accuracy at 4x >= 0.80
- synthetic FullContext-conditioned accuracy at 8x >= 0.70
- LongBench-v2 FullContext-conditioned accuracy at 8x >= 0.60
- KiaOmni at 8x is not below RandomRetention on synthetic cases
- KiaOmni at 8x is not below RecencyOnly on synthetic cases

If FullContext itself solves fewer than 4 synthetic or 3 real frozen cases, the result is **INCONCLUSIVE** rather than moving the thresholds after seeing the data.

16x is diagnostic only and cannot fail the gate by itself.

## Run order

Use the stages in order. The Modal launcher refuses to start the next stage unless the previous artifact says `PASS`.

```powershell
python -m pip install -U "modal>=1.5,<2"
modal setup

modal run modal/qwen3_30b_scale_gate_modal.py --stage preflight --gpu A100-80GB --prepare
modal run modal/qwen3_30b_scale_gate_modal.py --stage smoke --gpu A100-80GB
modal run modal/qwen3_30b_scale_gate_modal.py --stage final --gpu A100-80GB
```

Artifacts are stored in Modal Volume `kiaomni-qwen3-results`:

```text
phase_03_qwen3_30b_scale_gate/
  preflight.json
  smoke.json
  final.json
```

Download, for example:

```powershell
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/preflight.json .\preflight.json
```

## Hard-stop policy

Do not work around an experiment failure with 4-bit quantization, CPU offload, changed retention ratios, changed cases, or moved quality thresholds. Preserve the artifact and debug the root cause first.
