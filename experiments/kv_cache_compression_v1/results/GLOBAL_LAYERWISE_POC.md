# Global vs Layerwise KV Mask POC

## Status

PASS for implementation correctness and equal-budget plumbing.

This is **not** a model-quality benchmark. No claim about downstream LLM accuracy is allowed from this result.

## Isolation

The frozen prompt-selection implementation remains unchanged. The existing KV-specific Gaussian copy also remains unchanged.

Two new strategies are implemented separately:

- `global_mask_strategy.py`: aggregate layer saliency and apply one shared position mask to every layer.
- `layerwise_mask_strategy.py`: compute one position mask per layer.

Both use the same KV Gaussian selector and the same sink/recency protection rules.

## Fairness constraint

Global and layerwise runs receive the same retained-position budget **per layer**. Therefore total compressed KV bytes are equal by construction and verified in the tests.

## Test result

- Correctness tests: **4/4 passed**
- Global decode after physical compaction: **PASS**
- Layerwise decode after physical compaction: **PASS**
- Equal compressed memory: **PASS**

Observed physical reductions:

| Context | Budget/layer | Full KV bytes | Global bytes | Layerwise bytes | Reduction |
|---:|---:|---:|---:|---:|---:|
| 64 | 24 | 65,536 | 24,576 | 24,576 | 2.67x |
| 96 | 32 | 98,304 | 32,768 | 32,768 | 3.00x |
| 128 | 40 | 131,072 | 40,960 | 40,960 | 3.20x |

## Specialization plumbing check

The POC injects a different known-important policy signal into each layer. This is intentionally synthetic and exists only to test whether the layerwise implementation can preserve layer-specific positions under the same memory budget.

- Mean Global layer-specific signal survival: **83.3%**
- Mean Layerwise layer-specific signal survival: **100%**

This supports the implementation hypothesis that a layerwise mask can preserve different positions for different layers. It does **not** prove that layerwise is better on a real pretrained model.

## What is not promoted

- CPU micro-timings are not treated as meaningful performance evidence.
- L2 output distance is diagnostic only, not a quality metric.
- Synthetic signal survival is not task accuracy.

## Next experiment

Move both strategies to a real pretrained Hugging Face decoder cache:

1. Run `full_kv`.
2. Run `kv_global`.
3. Run `kv_layerwise`.
4. Use the same examples, generation settings, and actual KV-byte budget.
5. Evaluate real Passkey/NIAH accuracy and paired regressions.
6. Only then compare quality retention versus memory reduction.

The real-model experiment should test Global first as the simpler baseline, then Layerwise using the same protocol.
