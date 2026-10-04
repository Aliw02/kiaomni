# Streaming Policy V2 — Persistent Smoke

Run only after the previous streaming smoke and Final outputs already exist.

## Cell P1 — Pull latest branch

```python
%cd /kaggle/working/KiaOmni
!git pull
!git rev-parse HEAD
```

## Cell P2 — Persistent streaming smoke on the exact 4K and 8K multi-key cases

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_streaming_supplement.py \
  --reference-runs /kaggle/working/kiaomni_kv_results/final/final_runs.jsonl \
  --output-dir /kaggle/working/kiaomni_kv_results/persistent_streaming_smoke \
  --chunk-size 128 \
  --context-lengths 4096 8192 \
  --limit-cases 1 \
  --budget-labels B256 \
  --variants persistent_global persistent_layerwise
```

The runner also executes `chunked_full_kv` as the no-eviction chunking control.

## Cell P3 — Show only the decisive rows

```python
import pandas as pd
import json
from pathlib import Path

out = Path("/kaggle/working/kiaomni_kv_results/persistent_streaming_smoke")

rows = [
    json.loads(line)
    for line in open(out / "streaming_runs.jsonl", encoding="utf-8")
    if line.strip()
]
df = pd.DataFrame(rows)

cols = [
    "context_tokens",
    "task",
    "method",
    "budget_label",
    "status",
    "all_correct",
    "answer_recall",
    "kv_reduction_ratio",
    "prefill_peak_kv_reduction_ratio",
    "peak_allocated_gb",
    "ttft_seconds",
    "total_seconds",
    "tokens_per_second",
    "eviction_events",
]

display(df[cols].sort_values(["context_tokens", "method"]))
```

## Promotion rule

Continue to a broader Persistent Streaming benchmark only if:

- at least one persistent variant recovers the 4K multi-key case that V1 streaming missed; and
- 8K remains executable without OOM.

If both persistent variants remain wrong on the same multi-key case, stop streaming-policy expansion for this round and retain post-prefill B256 as the validated KiaOmni true-KV result.
