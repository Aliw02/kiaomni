# Plan: 037 Falcon3-7B Benchmark Script

## Context
Phi-3 blocked by bitsandbytes/device_map incompatibility. Switching to Falcon3-7B-Instruct
as the 3rd architecture for cross-architecture KV-cache claims in the paper.

**Why Falcon3-7B:**
- Fully ungated (no HuggingFace auth needed)
- 32K native context — exceeds 16K requirement
- GQA: 12 query heads / 4 KV heads — architecturally distinct from Qwen2.5-7B (32Q/8KV) and Mistral-7B (32Q/8KV)
- Citable: arXiv 2311.16867, Technology Innovation Institute (TII)
- 7B params → fits L4 24GB in NF4 4-bit

**Cross-arch taxonomy after this:**
| Model | Attention | Q heads | KV heads | head_dim |
|-------|-----------|---------|---------|---------|
| Qwen2.5-7B | GQA | 28 | 4 | 128 |
| Mistral-7B-v0.3 | GQA | 32 | 8 | 128 |
| Falcon3-7B | GQA | 12 | 4 | 256 |

## Files to Create

### 1. `notebook/kv_cache_benchmark/037_falcon3_comparison.py`
Clone structure from `036_phi3_comparison.py` with these changes:

**CONFIG section:**
```python
MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
CTX_LENS   = [4096, 8192, 16384]
BUDGETS    = [96, 128, 256, 512]
SIGMA_MAX  = 64   # tune after run — Falcon3 is new architecture
OUT_DIR    = Path("/vol/kv_results/037_falcon3_results") if IN_MODAL else Path("037_falcon3_results")
```

**load_model() — apply the confirmed fix:**
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=cfg,   # no device_map, no torch_dtype — bnb handles CUDA
    trust_remote_code=False,   # Falcon3 uses standard transformers arch
    attn_implementation="eager",
)
```

**extract_all_saliency():**
- Falcon3 uses standard separate `q_proj` / `k_proj` — NO fused qkv_proj
- Remove `_fused_qkv` detection entirely (always use the standard path)
- Set `nh=12`, `nk=4`, `hd=256` from config (or read from `model.config`)

**Everything else** (RULER tasks, LB tasks, policies, metrics, checkpointing) stays identical to 036.

### 2. `notebook/kv_cache_benchmark/modal_run_037_falcon3.py`
Clone `modal_run_036_phi3.py` with:
```python
app  = modal.App("falcon3-037-eval")
# transformers>=4.40 sufficient — Falcon3 is standard arch, no special version pin needed
image = modal.Image...pip_install("transformers>=4.40.0", ...)
       .add_local_file("notebook/kv_cache_benchmark/037_falcon3_comparison.py",
                       "/root/037_falcon3_comparison.py")

# inside run_benchmark():
result = subprocess.run(["python", "037_falcon3_comparison.py"], ...)
```

## Verification
```bash
modal run --detach notebook/kv_cache_benchmark/modal_run_037_falcon3.py
```
Expected: `Model ready.` → saliency hooks register → RULER trials start printing.
