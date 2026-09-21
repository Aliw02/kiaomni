# KiaOmni × MoE Model Lab

This directory is the canonical result archive for experiments that combine
KiaOmni prompt/context compression with Mixture-of-Experts (MoE) models.

Each model or experiment phase gets its own subdirectory so routing,
compression, quality, latency, memory, and ablation results remain isolated and
reproducible.

## Layout

- `phase_01_mobilemoe_route_stability/` — frozen MobileMoE-M-SFT study:
  KiaOmni budget sweep + counterfactual adaptive route-stability measurements.
- `phase_02_mobilemoe_multineedle_baselines/` — 4K MobileMoE benchmark:
  single/multi/hard-multi/reasoning tasks with FullContext, KiaOmni, BlockSal,
  and validation-gated NVIDIA SnapKV / StreamingLLM baselines.

Future MoE model tests should be added as new phase/model directories here
instead of modifying frozen result artifacts.
