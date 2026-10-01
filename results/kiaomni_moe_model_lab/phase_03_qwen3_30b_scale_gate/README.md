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
- six frozen cases in the final stage

Controlled synthetic tasks:

- single retrieval
- multi-record retrieval
- distractor-heavy relational binding
- chained reasoning

## Important corrections from the earlier scaffold

Phase 02 used KiaOmni package defaults:

- `n_sink=16`
- `recency=32`

The first Phase-03 scaffold accidentally used `recency=256`. V2 restores `recency=32` so this gate does not silently change the algorithm.

Saliency is computed **once per case** and reused across all KiaOmni retention ratios. This avoids paying for multiple identical full-prompt saliency forwards.

For Qwen3 BF16, Phase 03 performs the FP32 saliency math on GPU instead of offloading captured Q/K to CPU. The historical CPU-offload behavior remains the library default. Preflight compares CPU and GPU saliency on a frozen short case and fails closed unless:

- Pearson correlation >= 0.999
- Top-128 Jaccard >= 0.98

## Cost contract

The large model and LongBench-v2 dataset are downloaded by a **CPU-only** Modal function into a persistent Volume before any GPU is allocated.

| Stage | Work | Hard GPU ceiling |
|---|---|---:|
| preflight | load + probe + CPU/GPU saliency parity + 1 synthetic case | 20 min |
| smoke | 4 synthetic + 2 LongBench-v2 cases (8K–12K memory/real-path canary) | 40 min |
| final | 8 synthetic + 6 naturally fitting LongBench-v2 cases | 90 min |

One pass through all three stages therefore has a hard configured ceiling of **150 A100 GPU minutes**. At Modal's 2026-10-01 listed A100-80GB base rate of about $2.50/hour, the theoretical GPU ceiling is about **$6.25**, before comparatively small CPU/memory/storage charges. Re-running failed stages is outside the one-pass budget contract.

The launcher is sequential and fail-closed:

- smoke refuses to start unless preflight artifact says `PASS`
- smoke returns `FAIL` if KiaOmni@8x solves zero FullContext-solvable smoke cases
- smoke returns `INCONCLUSIVE` if FullContext solves fewer than two smoke cases
- final refuses to start unless smoke artifact says `PASS`

## Retention conditions

Final:

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

If FullContext itself solves fewer than 4 synthetic or 3 real frozen cases, the result is **INCONCLUSIVE** rather than changing thresholds after seeing the data.

16x is diagnostic only and cannot fail the gate by itself.

## Pull the branch

From the KiaOmni repository:

```powershell
git fetch origin
git switch exp/kiaomni-qwen3-30b-modal-scale-gate
git pull --ff-only
git rev-parse HEAD
```

If the branch does not exist locally yet, `git switch` will create the tracking branch when Git can resolve it. If your Git version does not do that automatically:

```powershell
git switch -c exp/kiaomni-qwen3-30b-modal-scale-gate --track origin/exp/kiaomni-qwen3-30b-modal-scale-gate
```

## Modal setup

Use the frozen current stable Modal CLI:

```powershell
python -m pip install -U "modal==1.6.0"
modal setup
modal billing rates
modal billing summary
modal billing report --for "this month" --show-resources
```

## Run order

Run **only one stage at a time** and inspect/download its artifact before continuing.

### 1. Preflight + CPU-only asset preparation

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage preflight --gpu A100-80GB --prepare
```

Download:

```powershell
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/preflight.json .\preflight.json
```

Do not run smoke unless `preflight.json -> gate.status == "PASS"`.

### 2. Smoke

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage smoke --gpu A100-80GB
```

Download:

```powershell
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/smoke.json .\smoke.json
```

Do not run final unless `smoke.json -> gate.status == "PASS"`.

### 3. Final

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage final --gpu A100-80GB
```

Download:

```powershell
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/final.json .\final.json
```

Artifacts remain in Modal Volume `kiaomni-qwen3-results`:

```text
phase_03_qwen3_30b_scale_gate/
  preflight.json
  smoke.json
  final.json
```

## If A100 runs out of memory

Only if preflight fails specifically because of OOM or the 8 GiB post-load headroom gate, rerun that stage on H200:

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage preflight --gpu H200
```

Do **not** switch GPU for a code, architecture-probe, saliency-parity, or dataset failure. Fix that failure first. If H200 is required for preflight, use H200 for the later stages too so the execution environment stays consistent.

## After the final artifact is backed up

Check the bill again:

```powershell
modal billing report --for "this month" --show-resources
```

The large model lives in the persistent asset volume and continues to incur storage charges while retained. After `final.json` is downloaded and backed up, remove the asset volume:

```powershell
modal volume delete kiaomni-qwen3-assets
```

Keep `kiaomni-qwen3-results` until the JSON artifacts are safely copied locally.

## Hard-stop policy

Do not work around a failure with 4-bit quantization, CPU offload, changed retention ratios, changed cases, or moved quality thresholds. Preserve the artifact and debug the root cause first.

This Phase-03 result remains a **prompt-side scaling result**. Phase 04 is the separate real cache-side `past_key_values` gate.
