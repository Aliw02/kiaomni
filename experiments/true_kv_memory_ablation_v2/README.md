# True-KV Memory Ablation V2 — reproducibility and execution

**Status:** source-built and syntax-checked; GPU result pending. Streaming eviction is excluded.

## Scope and frozen conditions

The new branch forks `exp/kiaomni-repro-sdk-v1`; do not edit the Phase 03 frozen
results or the legacy V1 experiment. All variants use `Qwen/Qwen2.5-7B-Instruct`,
4-bit NF4 model weights, FP16 compute, SDPA, and a single NVIDIA GPU.

| ID | Prefill | Saliency | KV selection |
|---|---|---|---|
| F0 | One-shot | None | Full KV |
| F1 | No-eviction chunks 128 | None | Full KV |
| F2 | No-eviction chunks 256 | None | Full KV |
| F3 | No-eviction chunks 512 | None | Full KV |
| K0 | One-shot | Legacy in-forward Q/K hooks | Post-prefill layerwise Gaussian |
| K1 | One-shot | Deferred Q/cache-K computation | Post-prefill layerwise Gaussian |
| K2 | No-eviction chunks 128 | Deferred Q/cache-K | Post-prefill layerwise Gaussian |
| K3 | No-eviction chunks 256 | Deferred Q/cache-K | Post-prefill layerwise Gaussian |
| K4 | No-eviction chunks 512 | Deferred Q/cache-K | Post-prefill layerwise Gaussian |

**No online eviction:** all input tokens are always forwarded and their KV remains intact
until all prefill chunks are finished. Any KV deletion happens once, after prefill.

## Benchmark and budgets

Source: the same `kiaomni.repro.controlled_benchmark.make_case` and
`DEFAULT_TASKS` as the earlier Kaggle true-KV experiment.

- Tasks: single_needle, multi_key, multi_value, multi_query, variable_tracking,
  summary_facts.
- Deterministic seed: `20261004 + context * 10 + repeat * 1000 + task_index * 37`.
- Fixed budgets: B98, B128, B256, B512.
- Relative budgets: r0.0625, r0.125, r0.25.
- Same numeric budget/positions are executed once per variant, but every alias is
  preserved in `budget_aliases`. At 4K there are 5 unique budgets; at 8K, 6.
- Smoke: 2K, 2 tasks, one repeat, 32 generated-token cap.
- Pilot: 4K, 6 tasks, one repeat, all seven labels, 128 generated-token cap.
- Final: 4K and 8K, 6 tasks × 2 repeats, all labels, 256 generated-token cap.
- Optional 16K is exploratory and separately labeled. The full 4K/8K Final matrix
  has 756 distinct GPU executions: (5+6) × 5 × 12 KV and 4 × 24 Full-KV.
- Strict first-answer-line accuracy/recall; no credit for answers corrected later
  in an explanation. Output tokens/s and greedy self-perplexity are exploratory.

## Critical preflight

Before running real comparisons:

1. One-shot full-prefill vs no-eviction chunk sizes 128/256/512: same full KV
   lengths and same next-token argmax on a deterministic 1024-token case.
2. Original `K0` vs deferred `K1` saliency: exact selected layerwise
   positions for B128, B256, and B512 on the same case.
3. If either check fails, stop before the wider run and inspect
   `preflight.json`; do not make an ablation claim.
4. Each experiment logs its own errors and `error_stage`; an OOM in F0
   **does not skip** K1/K2/K3/K4.
5. Per-phase CUDA allocated, reserved, peak allocated, driver free and duration
   are recorded in each raw row under `phase_memory`. Comparing raw `kv_after_bytes`
   with `peak_allocated_gib` is required; they are not interchangeable.
6. K0/K1/K2/K3/K4 post-compaction mask hashes are exported as
   `mask_equivalence.csv`; a mismatch weakens a clean memory-only interpretation.

## Kaggle T4 smoke

Create a Kaggle Python notebook with a **single T4 GPU** and Internet.
Run the following commands as notebook cells:

```python
%pip install -q "transformers==4.57.6" "bitsandbytes==0.48.2" "accelerate>=1.2,<2" "scipy>=1.12,<2" "pandas>=2.2,<3"
```

```python
%cd /kaggle/working
!git clone --depth 1 --single-branch --branch exp/kiaomni-true-kv-memory-ablation-v2 https://github.com/Aliw02/kiaomni.git KiaOmni
%cd /kaggle/working/KiaOmni
!git rev-parse HEAD
!nvidia-smi
```

```python
!python experiments/true_kv_memory_ablation_v2/run_ablation.py \
    --mode smoke \
    --output-dir /kaggle/working/true_kv_ablation_v2/smoke
```

Inspect `preflight.json`, `runs.jsonl`, `summary.csv`,
`mask_equivalence.csv`, and `manifest.json` before pilot.

After smoke passes, execute the full-budget 4K pilot:

```python
!python experiments/true_kv_memory_ablation_v2/run_ablation.py \
    --mode pilot \
    --output-dir /kaggle/working/true_kv_ablation_v2/pilot
```

Retry safely with the **same** code revision/config and `--resume`.
Do not change the budget/seed/method list while resuming the same directory.

## Modal T4 Final

Modal must already be authenticated on the computer invoking the CLI.
The runner provisions the GPU only when explicitly executed:

```bash
modal run experiments/true_kv_memory_ablation_v2/modal_runner.py --mode smoke
modal run experiments/true_kv_memory_ablation_v2/modal_runner.py --mode pilot
modal run experiments/true_kv_memory_ablation_v2/modal_runner.py --mode final
```

- One T4 GPU per invocation; contexts are sequential on a single GPU.
- Default Final contexts: 4K, then 8K. GPU allocation per method is measured
  without parallel processes on the same device.
- Two Modal volumes: model cache and raw experimental results; run output is
  committed even after a nonzero subprocess exit.
- Use `--run-tag` to isolate experiment revisions; the stored protocol SHA-256
  prevents resuming with changed model/config/source.
- GPU billing applies; no experiment is started by creating the branch.

## Acceptance / interpretation

PASS requires completed preflight, independently executed variants, frozen
paired cases and budgets, physical cache reduction, and real per-phase memory
results. Do not claim a lower GPU peak from KV token ratios alone.

The same selected positions for K0/K1 is an implementation validation, **not**
proof of downstream quality. If original and deferred positions diverge,
classify K1–K4 as changes in both policy and runtime, rather than a clean
memory-only ablation. If F0 OOMs, report valid OOM evidence; do not invent an
accuracy baseline for that particular pair.

A reduction in post-compaction live CUDA allocation, but not global GPU peak,
is a legitimate distinct finding. Streaming and quantized-KV methods remain
outside this frozen experiment.
