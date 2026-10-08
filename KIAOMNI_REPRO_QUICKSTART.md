# KiaOmni Repro SDK v0.3.1 — private evaluation workflow

Branch: `exp/kiaomni-repro-sdk-v1`. Keep this branch separate from the published main package.

## Kaggle quick start

Enable an NVIDIA GPU and Internet. Run:

```python
%pip install -q "transformers==4.57.6"
%pip install -q "kiaomni[repro] @ git+https://github.com/Aliw02/kiaomni.git@exp/kiaomni-repro-sdk-v1"
```

Enter the private code without saving it in the notebook:

```python
import os, getpass
os.environ["KIAOMNI_LICENSE_KEY"] = getpass.getpass("Evaluation license: ")
```

```python
!kiaomni-repro check-model --model-id Qwen/Qwen2.5-7B-Instruct
!kiaomni-repro demo --mode smoke --methods full_kv prompt_selection kv_layerwise --output-dir /kaggle/working/kiaomni_results
```

Results are saved as JSONL/CSV and protocol files. Use `--mode pilot` only after smoke passes.

## Python SDK

```python
from kiaomni import KiaOmniModel

model = KiaOmniModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")
result = model.generate("Summarize GPU memory.", mode="true_kv", budget=256)
print(result.text)
print(result.metrics)
```

Valid modes are `full_kv`, `prompt_selection`, `true_kv` — no hybrid.

## Model-family support

- Qwen2/Qwen2.5 full attention: candidate for all three modes, **still requires Kaggle GPU test**.
- Llama, Mistral, Gemma, Phi and additional HF causal LMs: original `full_kv`; existing `prompt_selection` adapter where compatible. These are not universally tested.
- Non-Qwen2 `true_kv`: deliberately raises `UnsupportedArchitecture`; adapters are not yet validated.
- `check-model` checks static configuration without downloading weights. It does not certify runtime correctness.
- The bundled paired scientific benchmark currently targets Qwen2.5 only and pins Transformers 4.57.6.

## Scientific caveats

Post-prefill compaction decreases the *stored* KV footprint, but does not eliminate peak full-context prefill memory. Prompt selection may require a separate full-context saliency pass; measure total time. Physical KV bytes are not total GPU VRAM. Errors should never be reported as valid compression results.

The benchmark is a deterministic, synthetic, small-N test, not an official RULER run. The first decoded answer token follows a bridge token in the benchmark to avoid quality inflation from full-prefill logits.

## Licensing caveat

The evaluation gate stores only the SHA256 hash of a hardcoded code and accepts a private `KIAOMNI_LICENSE_KEY`. It is **not real copy protection** in public source code; it is trivially removable by someone with Python access. The pre-existing public MIT license cannot be revoked retroactively. Move private implementation to a private repo/server and implement signed expirations or activation once commercialization is approved.
