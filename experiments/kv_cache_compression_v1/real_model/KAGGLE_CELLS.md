# Kaggle Cells — Qwen2.5-7B True KV-Cache Compression V1

Use a Kaggle notebook with **GPU enabled** and **Internet enabled**. The V1 harness uses one CUDA device.

## Cell 1 — Install pinned runtime

```python
%pip install -q --upgrade "transformers==4.57.6" "bitsandbytes>=0.46.0" "accelerate>=1.2.0" "scipy>=1.12" "pandas>=2.2"
```

## Cell 2 — Check GPU

```python
import torch, subprocess, sys
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
subprocess.run(["nvidia-smi"])
```

## Cell 3 — Clone the isolated branch

```python
%cd /kaggle/working
!rm -rf KiaOmni
!git clone --depth 1 --single-branch --branch exp/kiaomni-true-kv-cache-compression-v1 https://github.com/Aliw02/KiaOmni.git
%cd /kaggle/working/KiaOmni
!git rev-parse HEAD
```

## Cell 4 — Verify exact software versions and files

```python
import transformers, bitsandbytes, torch
from pathlib import Path

print("transformers:", transformers.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("torch:", torch.__version__)

root = Path("/kaggle/working/KiaOmni/experiments/kv_cache_compression_v1/real_model")
for name in [
    "run_kaggle_qwen25.py",
    "qwen25_kv_runtime.py",
    "kv_policy_qwen25.py",
    "controlled_benchmark.py",
]:
    p = root / name
    print(name, "OK" if p.exists() else "MISSING")
```

## Cell 5 — Real-model smoke

This is a real Qwen2.5-7B run, but intentionally small. It checks model loading, real DynamicCache compaction, absolute position handling, Global and Layerwise decode, and answer scoring.

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_qwen25.py \
  --mode smoke \
  --output-dir /kaggle/working/kiaomni_kv_results/smoke
```

## Cell 6 — Inspect smoke results

```python
import pandas as pd
from pathlib import Path

out = Path("/kaggle/working/kiaomni_kv_results/smoke")
display(pd.read_csv(out / "summary.csv"))
if (out / "paired_summary.csv").exists():
    display(pd.read_csv(out / "paired_summary.csv"))

print((out / "smoke_manifest.json").read_text())
```

## Cell 7 — Pilot sweep on 4K

Runs all four methods:

- full_kv
- prompt_selection
- kv_global
- kv_layerwise

The pilot uses multiple budgets on the complete controlled task set.

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_qwen25.py \
  --mode pilot \
  --output-dir /kaggle/working/kiaomni_kv_results/pilot
```

If the Kaggle session stops, rerun the same cell with `--resume`:

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_qwen25.py \
  --mode pilot \
  --output-dir /kaggle/working/kiaomni_kv_results/pilot \
  --resume
```

## Cell 8 — Inspect pilot

```python
import pandas as pd
from pathlib import Path

out = Path("/kaggle/working/kiaomni_kv_results/pilot")
summary = pd.read_csv(out / "summary.csv")
paired = pd.read_csv(out / "paired_summary.csv")
display(summary.sort_values(["context_tokens", "budget_label", "method"]))
display(paired.sort_values(["context_tokens", "budget_label", "method"]))
```

## Cell 9 — Final 4K + 8K sweep

Only run this after the smoke and pilot complete without cache/position errors.

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_qwen25.py \
  --mode final \
  --output-dir /kaggle/working/kiaomni_kv_results/final
```

Resume form:

```python
!python experiments/kv_cache_compression_v1/real_model/run_kaggle_qwen25.py \
  --mode final \
  --output-dir /kaggle/working/kiaomni_kv_results/final \
  --resume
```

## Cell 10 — Final comparison table

```python
import pandas as pd
from pathlib import Path

out = Path("/kaggle/working/kiaomni_kv_results/final")
summary = pd.read_csv(out / "summary.csv")
paired = pd.read_csv(out / "paired_summary.csv")

cols = [
    "context_tokens", "method", "budget_label", "n", "accuracy",
    "answer_recall", "kv_reduction_ratio", "peak_allocated_gb",
    "ttft_seconds", "tokens_per_second", "greedy_self_ppl",
]
display(summary[cols].sort_values(["context_tokens", "budget_label", "method"]))

display(
    paired.sort_values(["context_tokens", "budget_label", "method"])
)
```

## Cell 11 — Archive outputs

```python
%cd /kaggle/working
!zip -qr kiaomni_true_kv_qwen25_results.zip kiaomni_kv_results
!ls -lh kiaomni_true_kv_qwen25_results.zip
```

## Interpretation boundary

The controlled benchmark is deterministic and long-context, with single-needle, multi-key, multi-value, multi-query, variable-tracking, and summary-fact tasks. It is **RULER-style but not the official NVIDIA RULER benchmark**.

The first free answer token is produced only after a forced post-compression bridge token. Therefore `kv_global` and `kv_layerwise` cannot receive a free full-context first answer token.

Promotion to a large MoE experiment should happen only after:

1. physical KV reduction is confirmed on Qwen2.5-7B;
2. both true-KV variants decode without cache-position failures;
3. paired quality results are available against `full_kv`;
4. actual KV bytes and system metrics are recorded.
