# Experimental Streaming Repair Gate

**Status: UNVALIDATED ON GPU — do not send to Ahmed Zaki or promote as default.**

The existing `full_kv`, `prompt_selection`, and post-prefill `true_kv` paths were left unchanged. This branch adds an optional Qwen2/Qwen2.5 streaming evaluation.

## New candidate methods

- `stream_persistent_global`
- `stream_persistent_layerwise`
- `trigger_x1`: evict as soon as live KV exceeds budget
- `trigger_x2`: defer eviction until live KV exceeds twice the budget, then compact to budget; the final chunk always compacts

This is a **quality/peak-memory trade-off**, not a proven accuracy fix. Higher triggers can increase peak memory. Future-only fact importance cannot always be predicted by any online eviction policy.

## Reproduce in Kaggle (NVIDIA GPU)

Install the branch after the standard requirements and provide your private evaluation license via the `KIAOMNI_LICENSE_KEY` environment variable in your own session.

```bash
kiaomni-repro streaming-gate --profile smoke --context 4096 --chunk-size 128 --budgets B256 --methods stream_persistent_global stream_persistent_layerwise --trigger-multipliers 1 2 --output-dir /kaggle/working/kiaomni_streaming_gate
```

If no candidate passes, the exit status is 2 and the gate report says `SKIP`. Do not launch the expensive pilot or include streaming as supported in the Windows demo.

If at least one passes, run:

```bash
kiaomni-repro streaming-gate --profile pilot --context 4096 --chunk-size 128 --budgets B98 B128 B256 B512 r0.0625 r0.125 r0.25 --methods stream_persistent_global stream_persistent_layerwise --trigger-multipliers 1 2 --output-dir /kaggle/working/kiaomni_streaming_gate_pilot
```

## Acceptance gate

- Chunked prefill without eviction must retain the same cache lengths and next-token argmax as one-shot prefill.
- Same task, case, model, seed, quantization and generation settings for reference and candidate.
- For every case where either Full KV or post-prefill True KV is correct, Streaming must not regress.
- Streaming must actually reduce stored KV bytes versus Full KV.
- Any error, missing paired control, quality regression or fake compression causes `SKIP` for that candidate/budget.
- A `PASS_CANDIDATE` only permits further investigation. It is **not** proof of real conversation, long-context, multi-model or OOM capability.
- If every Streaming candidate fails, retain Full KV / Prompt Selection / True KV, disclose Streaming as experimental and failed in the handoff.

Generated files: `protocol.json`, `model_info.json`, `equivalence.json`, `streaming_runs.jsonl`, `streaming_gate_report.json`.

This has passed source syntax and offline decision-rule unit checks. **An actual GPU run has not been performed in this environment.**
