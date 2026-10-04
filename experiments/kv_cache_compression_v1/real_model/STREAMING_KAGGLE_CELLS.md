# Kaggle Streaming Supplement — after the current Final finishes

This supplement reuses the already-computed Final raw cases and runs only:

- stream_global
- stream_layerwise

It then merges those rows with the existing:

- full_kv
- prompt_selection
- kv_global
- kv_layerwise

and applies strict first-answer scoring to all six methods.

## Cell S1 — Pull the streaming implementation

```python
%cd /kaggle/working/KiaOmni
!git pull
!git rev-parse HEAD
```

Do this only after the currently running Final process has finished.

## Cell S2 — One-case streaming smoke

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_streaming_supplement.py \
  --reference-runs /kaggle/working/kiaomni_kv_results/final/final_runs.jsonl \
  --output-dir /kaggle/working/kiaomni_kv_results/streaming_smoke \
  --chunk-size 128 \
  --limit-cases 1 \
  --limit-budgets 1
```

This also runs a real no-eviction equivalence check between one-shot and chunked Qwen prefill.

## Cell S3 — Inspect the streaming smoke

```python
import json
import pandas as pd
from pathlib import Path

out = Path("/kaggle/working/kiaomni_kv_results/streaming_smoke")

print("EQUIVALENCE")
print(json.dumps(json.loads((out / "streaming_equivalence.json").read_text()), indent=2))

print("\nMANIFEST")
print(json.dumps(json.loads((out / "streaming_manifest.json").read_text()), indent=2))

display(
    pd.read_csv(out / "summary.csv")
    .sort_values(["context_tokens", "budget_label", "method"])
)
```

Do not run the full streaming supplement if the smoke has cache-position, mask-shape, or decode errors.

## Cell S4 — Full streaming supplement

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_streaming_supplement.py \
  --reference-runs /kaggle/working/kiaomni_kv_results/final/final_runs.jsonl \
  --output-dir /kaggle/working/kiaomni_kv_results/streaming_final \
  --chunk-size 128
```

Resume form:

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_streaming_supplement.py \
  --reference-runs /kaggle/working/kiaomni_kv_results/final/final_runs.jsonl \
  --output-dir /kaggle/working/kiaomni_kv_results/streaming_final \
  --chunk-size 128 \
  --resume \
  --skip-equivalence
```

## Cell S5 — Six-way comparison

```python
import pandas as pd
from pathlib import Path

out = Path("/kaggle/working/kiaomni_kv_results/streaming_final")

summary = pd.read_csv(out / "summary.csv")
paired = pd.read_csv(out / "paired_summary.csv")

cols = [
    "context_tokens",
    "method",
    "budget_label",
    "n",
    "accuracy",
    "answer_recall",
    "kv_reduction_ratio",
    "peak_allocated_gb",
    "ttft_seconds",
    "total_seconds",
    "tokens_per_second",
    "greedy_self_ppl",
]

display(
    summary[cols]
    .sort_values(["context_tokens", "budget_label", "method"])
)

display(
    paired
    .sort_values(["context_tokens", "budget_label", "method"])
)
```

For streaming-specific peak-KV metrics, inspect the raw rows:

```python
import json
import pandas as pd

rows = [
    json.loads(line)
    for line in open(
        "/kaggle/working/kiaomni_kv_results/streaming_final/streaming_runs.jsonl",
        encoding="utf-8",
    )
    if line.strip()
]

stream = pd.DataFrame(rows)

display(
    stream[[
        "context_tokens",
        "task",
        "method",
        "budget_label",
        "all_correct",
        "answer_recall",
        "kv_reduction_ratio",
        "prefill_peak_kv_reduction_ratio",
        "prefill_peak_kv_bytes",
        "peak_allocated_gb",
        "ttft_seconds",
        "tokens_per_second",
        "eviction_events",
    ]]
    .sort_values(["context_tokens", "budget_label", "method", "task"])
)
```

## Cell S6 — Archive all streaming comparison outputs

```python
%cd /kaggle/working
!zip -qr kiaomni_streaming_final.zip kiaomni_kv_results/streaming_final
!ls -lh kiaomni_streaming_final.zip
```
