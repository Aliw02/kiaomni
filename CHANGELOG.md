# Changelog

All notable changes to **kiaomni** are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.5] — 2026-05-21

### Changed
- **Softened `attn_implementation` gate (`adapters/probe.py`).** Previous versions
  raised `KiaomniConfigError` when the model was loaded with `sdpa` or
  `flash_attention_2`. Validation against `notebook/kv_cache_benchmark/039_swap_experiment.py`
  on Qwen2.5-7B 4-bit confirmed that forward hooks on the **standalone**
  `q_proj`/`k_proj` linear modules fire **before** the fused attention kernel
  runs — so saliency extraction remains observable. The gate now downgrades to
  a warning instead of refusing to attach.
- **Saliency hooks now cast Q/K to CPU + fp32 immediately (`adapters/saliency.py`).**
  Under 4-bit NF4 with `bnb_4bit_compute_dtype=bfloat16`, projection outputs are
  bf16; the downstream `softmax(QK^T/√d)` accumulates error and emits NaN/Inf on
  long sequences. Both the separate-projection path and the fused-QKV path now
  use `.detach().cpu().to(torch.float32)` — the same numerical refuge proven by
  the 039 experiment.
- **Reverted `_orig` capture to the instance attribute (`monkey_patch.py`).**
  v0.2.3 went through `type(model).generate.__get__(model, type(model))` to
  bypass a stale Accelerate-installed instance attribute. That fix was correct
  for the symptom but wrong for the cause: it permanently sidesteps Accelerate's
  device-placement and bitsandbytes dequantization hooks, which **are** legitimate
  instance wrappers on quantized / multi-device models. v0.2.5 restores
  `_orig = model.generate` and relies on the idempotent unwind (added in v0.2.2)
  + delattr-only `remove_kiaomni` (added in v0.2.4) to prevent stacked wrappers.

### Why
- Issue raised by architectural review against `039_swap_experiment.py`: three
  defenses in the repo were over-restrictive relative to the experimentally-validated
  code path. This release aligns the public package with the proven experiment.

---

## [0.2.4] — 2026-05-17

### Fixed
- **`remove_kiaomni` over-unwound past the bound `generate` method.** The previous
  walker followed `__wrapped__` through `@torch.no_grad()`'s `functools.wraps`
  pointer down to the raw unbound `generate(self, inputs, ...)` function, then
  reassigned that as an instance attribute. The next `model.generate(ids, ...)`
  call became `generate(self=ids)` — Tensor-as-self crash. Fix: drop the chain
  walk entirely and just `delattr(model, "generate")`, letting Python's
  descriptor protocol re-bind the class-level method on every access.

### Validated
- End-to-end hard test on Qwen2.5-7B-Instruct (4-bit NF4) — see
  `kiaomni_kaggle_validation.md`. Headline: `kiaomni_s8 @ budget=256` scored
  **1.000** on NIAH vs FullContext 0.333 across 3 trials at ~1500-token prompts.

---

## [0.2.3] — 2026-05-17

### Fixed
- **`apply_kiaomni` captured an unbound function when Accelerate hooks were
  active.** Loading a model with `device_map="auto"` (required for multi-GPU
  and 4-bit NF4) can cause Accelerate to override `model.generate` as an
  instance attribute. The previous `_orig = model.generate` then held a raw
  function with no `self` binding. Fix: resolve via
  `type(model).generate.__get__(model, type(model))` — explicit descriptor
  invocation that bypasses any instance attribute.

---

## [0.2.2] — 2026-05-17

### Added
- **Idempotent `apply_kiaomni`.** Calling `apply_kiaomni` multiple times on the
  same model now auto-unwinds the prior patch before installing a new one.
  Prevents wrapper-stacking corruption in multi-policy test loops.

---

## [0.2.1] — 2026-05-17

### Fixed
- **HuggingFace `generate` contract restoration.** Prompt-side eviction
  internally shortens `input_ids`, breaking the caller's assumption that
  `out[:, input_ids.shape[1]:]` yields the new tokens. The patched `generate`
  now `torch.cat([original_input_ids, new_tokens], dim=1)` before returning,
  so downstream slicing works unchanged.

---

## [0.2.0] — 2026-05-17 — **BREAKING (algorithm)**

### Changed
- **Switched from cache-side to prompt-side eviction.** The v0.1.x algorithm
  gathered KV tensors and resumed `generate` with `past_key_values`, which
  required exact alignment of `cache_position`, `position_ids`, and RoPE
  rotation positions — contracts that drift between `transformers` releases.
  v0.2.0 instead slices `input_ids` by the kept positions and re-invokes
  `model.generate` on the shorter prompt as a fresh sequence. The model
  handles its own cache, position encoding, and attention masking — kiaomni
  stays out of the way.
- This matches the paper-evaluated algorithm in
  `notebook/kv_cache_benchmark/033_full_comparison.py` and has been validated
  on Qwen2.5-7B, Mistral, BioMistral, Llama-3.1, and TinyLlama.

### Removed
- `kiaomni/adapters/cache.py` — no longer needed under prompt-side eviction.

---

## [0.1.2] — pre-history

### Fixed
- Dropped `L-1` from `keep_indices` to avoid double-insertion of the last
  token during cache-side eviction (caused "the following the following…"
  degenerate repetition on TinyLlama).

---

## [0.1.1] — pre-history

### Fixed
- Injected `cache_position=torch.arange(max_keep, max_keep + 1, ...)` when
  resuming with `past_key_values` (cache-side path). Without it, TinyLlama
  raised `IndexError: cache_position size 0`.

---

## [0.1.0] — pre-history

### Added
- Initial public release. Cache-side eviction via `past_key_values`. Worked
  on TinyLlama in isolation; broke under version-drift across larger models.
