# Phase 03 — Qwen3-30B-A3B Modal Scale Gate

## Question

Does the existing **prompt-side** KiaOmni selection policy scale from MobileMoE to a substantially larger MoE without collapsing quality?

This phase deliberately does **not** claim real `past_key_values` eviction. Cache-side KiaOmni and external KV-cache baselines are Phase 04.

## Frozen setup

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, pinned revision in `PROTOCOL_FREEZE.json`
- Precision: BF16, no quantization, no CPU/disk offload
- Context target: ~8192 rendered tokens
- KiaOmni budgets: 2048 (~4x), 1024 (~8x), 512 (~16x stress)
- `n_sink=16`, `recency=32`
- Real validation: pinned `THUDM/LongBench-v2`, naturally fitting cases only, no truncation
- Synthetic validation: single, multi, hard_multi, reason

The runner computes saliency once per case and reuses it across budgets. Phase-03 uses FP32 saliency math on GPU. Preflight compares it against the historical CPU-FP32 path before allowing the scale run.

## Cost guard

Large assets are prepared on CPU into a Modal Volume before a GPU is requested.

| Stage | Cases | Budgets | Hard GPU timeout |
|---|---|---|---:|
| preflight | 1 synthetic | 1024 | 30 min |
| smoke | 4 synthetic + 2 real | 2048, 1024 | 45 min |
| final | 8 synthetic + 8 real | 2048, 1024, 512 | 120 min |

A one-pass run is capped at **195 configured GPU minutes**. `max_containers=1` prevents accidental parallel GPU burn. Re-running a stage costs additional credit.

## Pull and install

```powershell
git fetch origin
git switch exp/kiaomni-qwen3-30b-modal-scale-gate
git pull --ff-only

py -m pip install -U "modal>=1.6,<2"
modal setup
modal billing rates
modal billing summary --for "this month"
```

## Run order

Prepare the pinned model/dataset on CPU and run preflight:

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage preflight --gpu A100-80GB --prepare
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/preflight.json .
```

Read `preflight.json` before continuing. If clean:

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage smoke --gpu A100-80GB
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/smoke.json .
```

Then, only after the smoke artifact is clean:

```powershell
modal run modal/qwen3_30b_scale_gate_modal.py --stage final --gpu A100-80GB
modal volume get kiaomni-qwen3-results phase_03_qwen3_30b_scale_gate/final.json .
```

If A100 fails the **memory-headroom gate**, do not quantize or offload. Re-run only the failed stage with `--gpu H200`.

After the final artifact is safely downloaded, the large model cache can be deleted to stop persistent storage charges:

```powershell
modal volume delete kiaomni-qwen3-model-cache
```

Keep `kiaomni-qwen3-results` until the JSON artifacts are backed up.
