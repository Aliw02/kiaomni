# Phase 03 — Qwen3-30B-A3B Modal Scale Gate

## Purpose

This phase answers one narrow question: **does the existing KiaOmni selection policy scale from MobileMoE to a substantially larger MoE without collapsing quality?**

It intentionally does **not** claim production KV-cache eviction. The current public KiaOmni implementation remains prompt-side: it scores the full prompt, selects retained token positions, then regenerates from the shorter prompt. Real `past_key_values` eviction and apples-to-apples KV-cache baselines are deferred to Phase 04.

## Frozen model

- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- BF16 only
- SDPA
- no quantization
- no CPU offload
- default GPU: A100 80GB

## Cost design

Model weights and LongBench are downloaded by a **CPU-only** Modal function into a persistent Volume before a GPU is requested.

The GPU stages are separate and fail-closed:

| Stage | Context target | KiaOmni budgets | Configured GPU timeout |
|---|---:|---:|---:|
| preflight | 2K | 1K | 20 min |
| smoke | 4K | 1K, 512 | 40 min |
| final | 8K | 2K, 1K, 512 | 180 min |

One pass through all three stages is capped at 240 configured GPU minutes. Re-running a stage spends additional credit and is not part of the one-pass budget contract.

## Quality data

1. Controlled synthetic tasks preserve the old failure modes: single retrieval, multi-record retrieval, distractor-heavy relational binding, and short chained reasoning.
2. Real validation uses naturally fitting `THUDM/LongBench` QA subset examples. The runner never truncates LongBench context to force a case into the window; it selects only examples whose rendered prompt naturally fits the stage token range.

## Run order

Do not run `final` before reading the `preflight` and `smoke` artifacts.

```bash
python -m pip install -U "modal>=1.5,<2"
modal setup

modal run modal/qwen3_30b_scale_gate_modal.py --stage preflight --prepare
modal run modal/qwen3_30b_scale_gate_modal.py --stage smoke
modal run modal/qwen3_30b_scale_gate_modal.py --stage final
```

Artifacts live in the Modal Volume `kiaomni-qwen3-results` under:

```text
phase_03_qwen3_30b_scale_gate/
  qwen3_30b_preflight.json
  qwen3_30b_smoke.json
  qwen3_30b_final.json
```

Download one with:

```bash
modal volume get kiaomni-qwen3-results \
  phase_03_qwen3_30b_scale_gate/qwen3_30b_preflight.json \
  ./qwen3_30b_preflight.json
```

## Hard stop conditions

The gate refuses to continue if:

- the GPU is not BF16-capable;
- fewer than 7 GiB remain free after model load;
- the architecture probe fails;
- an offline cache is missing after preparation;
- the stage wall-time budget is reached.

Do not work around these failures with 4-bit quantization or CPU offload; those change the experiment.
