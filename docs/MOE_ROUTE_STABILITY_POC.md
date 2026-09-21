# Kaggle POC — KiaOmni + Adaptive MoE Route Stability

This experiment validates two inference-time ideas on a **single Kaggle T4**:

1. **KiaOmni prompt/context reduction**
2. **Adaptive low-jitter MoE routing**, derived from the earlier EG-MoT causal-inertia work

The model weights remain frozen. No expert weights, router weights, or task data are trained.

## Default model

`facebook/MobileMoE-M-SFT`

Why this target:

- Released by Meta in **August 2026**
- Instruction-tuned / chat-ready
- **2.8B total parameters / 528M active**
- **60 routed experts, Top-4 active, plus one shared expert**
- 26 layers, QK-Norm, 8,192-token context
- Official BF16 checkpoint; the POC loads it directly as **FP16 on one T4**
- No AWQ, GPTQModel, bitsandbytes, GGUF, or custom quantization loader
- The routing experiment is therefore not confounded by a quantization runtime

The model is gated on Hugging Face under the FAIR Noncommercial Research License.
Accept the model terms once and provide an `HF_TOKEN` with read access.

## Kaggle setup

Use:

- Accelerator: **GPU T4 x1** (T4 x2 is fine; the POC intentionally uses GPU0 only)
- Internet: **On**
- Hugging Face: accept access for `facebook/MobileMoE-M-SFT`

The model card requires only PyTorch, Transformers, safetensors, and Accelerate.
The existing experiment branch already uses a compatible Transformers release.

If the current notebook already has the experiment virtualenv created during
earlier attempts, it can be reused; GPTQModel is no longer imported or used.

Update the branch:

```bash
cd /kaggle/working/kiaomni
git pull
git rev-parse HEAD
```

Install the branch into the interpreter you will run:

```bash
/kaggle/working/kia-awq-venv/bin/pip install -q -e /kaggle/working/kiaomni --no-deps
```

The directory name `kia-awq-venv` is historical only. The MobileMoE POC does
not use AWQ/GPTQ.

## Hugging Face token

Store a read token in Kaggle Secrets as `HF_TOKEN`, then expose it to child
processes before running the experiment:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF_TOKEN available:", bool(os.environ.get("HF_TOKEN")))
```

Do not print the token value.

## Fast smoke run

```bash
/kaggle/working/kia-awq-venv/bin/python /kaggle/working/kiaomni/experiments/kaggle_moe4bit_poc.py \
  --budget 512 \
  --alpha-max 0.10 \
  --long-tokens 1200 \
  --output /kaggle/working/moe_route_stability_smoke.json
```

The script filename is retained for continuity; this version of the POC is
**not a 4-bit experiment**.

The runner prints the actual CUDA residency after load and fails if GPU0 exceeds
12.5 GiB before the benchmark begins.

## Main POC

After the smoke passes:

```bash
/kaggle/working/kia-awq-venv/bin/python /kaggle/working/kiaomni/experiments/kaggle_moe4bit_poc.py \
  --budget 768 \
  --alpha-max 0.10 \
  --long-tokens 1800 \
  --output /kaggle/working/moe_route_stability_poc.json
```

The value `alpha_max=0.10` is deliberately tied to the earlier EG-MoT ablation
sweet spot. The controller does **not** apply 0.10 constantly:

```text
alpha_t = alpha_max * router_uncertainty_t * hidden_similarity_t
```

For sigmoid-routing models such as MobileMoE, uncertainty is computed from
normalized sigmoid router scores rather than softmax probabilities.

## Controlled arms

| Arm | KiaOmni | Adaptive routing |
|---|---:|---:|
| baseline | no | no |
| route_only | no | yes |
| kiaomni_only | yes | no |
| kiaomni_plus_route | yes | yes |

No arm reloads or retrains the weights.

## What the POC records

Per case:

- prompt token count
- generated token count and text
- deterministic token agreement with baseline
- needle retrieval pass/fail
- generation time and tokens/sec
- peak allocated CUDA VRAM

For routing arms:

- raw vs stabilized top-1 expert transition rate
- raw vs stabilized Top-K Jaccard
- intervention rate
- mean/max adaptive alpha
- router uncertainty
- adjacent hidden-state similarity
- discovered router count and names
- router score function

## First scientific gate

The POC is useful only if:

1. `route_only` lowers unnecessary route transitions.
2. Expert diversity and quality do not collapse.
3. `route_only` remains close to baseline output quality.
4. `kiaomni_only` preserves the needle tests under context reduction.
5. `kiaomni_plus_route` does not add a large quality regression.
6. Actual peak VRAM remains inside one T4.

If routing stability improves but quality falls materially, do **not** simply
increase inertia. Earlier EG-MoT ablations already showed stale-routing failure
at excessive inertia.

## Identity check

```bash
/kaggle/working/kia-awq-venv/bin/python /kaggle/working/kiaomni/experiments/kaggle_moe4bit_poc.py \
  --alpha-max 0.0 \
  --budget 768 \
  --long-tokens 1200 \
  --output /kaggle/working/moe_route_identity.json
```

At `alpha_max=0`, the route wrapper must preserve raw router logits exactly.
