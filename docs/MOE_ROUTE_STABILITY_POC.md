# Kaggle POC — KiaOmni + Adaptive MoE Route Stability

This experiment validates two inference-time ideas on a **single 16 GB Kaggle GPU** using a **4-bit quantized MoE**:

1. **KiaOmni prompt/context reduction**
2. **Adaptive low-jitter MoE routing**, derived from the earlier EG-MoT causal-inertia work

The model weights remain frozen. No expert weights, router weights, or task data are trained.

## Default model

`cyankiwi/LFM2.5-8B-A1B-AWQ-INT4`

Why this target:

- Modern LFM2.5 MoE family, 8.3B total / 1.5B active parameters
- 32 experts, Top-4 active
- Hybrid 18 convolution + 6 GQA attention layers
- **Pre-quantized AWQ INT4** checkpoint, about 5.37 GB on disk
- Expert weights are already quantized before loading, avoiding the FP16-on-load peak
- Router gates remain directly observable for the routing POC
- On Kaggle T4 x2, the quantized model is balanced across both GPUs with GPU 0 kept lighter for generation headroom

The runner also checks actual CUDA allocation immediately after load and fails closed if it is already above 14.5 GB.

## Kaggle settings

Use:

- Accelerator: **GPU T4 x1** (or another single CUDA GPU with at least 16 GB)
- Internet: **On** for the first model download

## Install cell

Use a fresh Kaggle session. Keep the notebook's base environment untouched and
create an isolated virtual environment for the AWQ runtime:

```bash
python -m pip install -q -U virtualenv
python -m virtualenv --system-site-packages /kaggle/working/kia-awq-venv

/kaggle/working/kia-awq-venv/bin/pip install -q -U pip setuptools wheel
/kaggle/working/kia-awq-venv/bin/pip install -q -U accelerate optimum ninja "transformers>=5.10,<6"
/kaggle/working/kia-awq-venv/bin/pip install -q -U "gptqmodel==7.5.0" --no-build-isolation
/kaggle/working/kia-awq-venv/bin/pip install -q -U "numpy==2.2.6" "scipy==1.15.3"
```

Because this is a virtualenv created with `--system-site-packages`, Kaggle's CUDA/PyTorch
installation remains available without downloading another multi-gigabyte Torch
stack. Any NumPy/Protobuf versions required by GPTQModel are shadowed only inside
the venv and do not mutate the notebook kernel's base environment.

Verify the isolated runtime with a separate process:

```bash
/kaggle/working/kia-awq-venv/bin/python - <<'PY'
import torch, transformers, numpy, scipy, gptqmodel
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("gptqmodel", getattr(gptqmodel, "__version__", "installed"))
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory / 1024**3)
PY
```

Then clone and install this branch into the same venv:

```bash
git clone -b exp/moe-route-stability-v1 https://github.com/Aliw02/kiaomni.git
/kaggle/working/kia-awq-venv/bin/pip install -q -e /kaggle/working/kiaomni --no-deps
```

Do not install AutoAWQ and do not run the experiment with the notebook kernel's
`python`; use the venv interpreter shown below.

## Fast smoke run

Start with a smaller long-context case to prove that model loading, KiaOmni probing, hybrid-layer saliency, and MoE gate hooks all work:

```bash
/kaggle/working/kia-awq-venv/bin/python /kaggle/working/kiaomni/experiments/kaggle_moe4bit_poc.py \
  --budget 512 \
  --alpha-max 0.10 \
  --long-tokens 1200 \
  --output /kaggle/working/moe_route_stability_smoke.json
```

## Main POC

After the smoke passes:

```bash
/kaggle/working/kia-awq-venv/bin/python /kaggle/working/kiaomni/experiments/kaggle_moe4bit_poc.py \
  --budget 768 \
  --alpha-max 0.10 \
  --long-tokens 1800 \
  --output /kaggle/working/moe_route_stability_poc.json
```

The value `alpha_max=0.10` is deliberately tied to the earlier EG-MoT ablation sweet spot. The new controller does **not** apply 0.10 constantly. It computes:

```text
alpha_t = alpha_max * router_uncertainty_t * hidden_similarity_t
```

so confident route changes remain nearly untouched, while near-tie routes between semantically similar adjacent tokens receive more inertia.

## Controlled arms

The same frozen checkpoint is evaluated under four arms:

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
- generated token count
- generated text
- deterministic token agreement with the baseline
- needle-in-haystack retrieval pass/fail
- wall-clock generation time
- generated tokens/sec
- peak allocated CUDA VRAM

For routing arms:

- raw top-1 expert transition rate
- stabilized top-1 expert transition rate
- raw/stabilized Top-K Jaccard
- intervention rate
- mean and maximum adaptive alpha
- mean router uncertainty
- mean adjacent hidden-state similarity
- discovered router count and names

The runner includes short deterministic prompts and three long needle tests with the critical record near the beginning, middle, and end.

## First scientific gate

The POC is useful only if all of these are true:

1. `route_only` lowers route transition rate compared with its raw route trace.
2. The effect is not obtained by forcing one expert: Top-K diversity and needle quality remain meaningful.
3. `route_only` keeps deterministic output reasonably close to baseline.
4. `kiaomni_only` demonstrates context reduction without collapsing the needle tests.
5. `kiaomni_plus_route` does not introduce a large additional quality regression over KiaOmni alone.
6. Actual peak VRAM remains inside the 16 GB envelope.

If routing stability improves but quality falls materially, do **not** increase inertia. The earlier EG-MoT ablation already showed that excessive fixed inertia produces stale routing. The next experiment should instead tune the adaptive confidence gate.

## Optional identity check

To prove that the route wrapper itself is not perturbing the model, run:

```bash
/kaggle/working/kia-awq-venv/bin/python /kaggle/working/kiaomni/experiments/kaggle_moe4bit_poc.py \
  --alpha-max 0.0 \
  --budget 768 \
  --long-tokens 1200 \
  --output /kaggle/working/moe_route_identity.json
```

For the `route_only` arm, `alpha-max=0` should preserve router logits exactly.

## Result file

After the run, download or keep:

```text
/kaggle/working/moe_route_stability_poc.json
```

That artifact is sufficient for the next analysis step; no screenshots are required.
