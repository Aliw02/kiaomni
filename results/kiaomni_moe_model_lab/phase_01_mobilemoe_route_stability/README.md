# Phase 01 — MobileMoE Route Stability

## Scope

Model: `facebook/MobileMoE-M-SFT`

This phase combines:
- KiaOmni prompt-side saliency compression.
- MobileMoE Top-4 routing observation.
- Adaptive causal route-stability logic.
- A 4-arm comparison: baseline, route-only, KiaOmni-only, KiaOmni+route.
- KiaOmni budgets: 512 / 256 / 128 / 98.
- Requested long-context scale: 3200+ tokens.
- Measured needle prompt length: 4294 tokens.
- Router scoring: sigmoid.
- Route-stability mode in this artifact: `shadow_counterfactual`.
- `active_intervention=false`: routing stabilization was measured
  counterfactually and did not mutate MobileMoE's real expert dispatch.

## Frozen result

Primary artifact:
`moe_route_budget_sweep_3200.json`

Uploaded artifact SHA-256:
`c4a0c6a60f5fd5e67d77823f68312910fce6bb64a736ef07cfd5ca14fa897a9a`

### KiaOmni result

All tested KiaOmni budgets preserved:
- `needle_pass_rate = 1.0`
- `mean_baseline_token_lcp_ratio = 1.0`

At budget 98, the 4294-token prompt was reduced to 98 retained tokens
(~43.82x compression) while all three early/middle/late needle cases passed.

Baseline peak allocated VRAM: ~8.016 GB.
KiaOmni-only peak allocated VRAM: ~6.292 GB.

### Counterfactual routing result at alpha_max=0.10

Route-only aggregate:
- Raw Top-1 transition rate: 0.8408916807
- Stable Top-1 transition rate: 0.8074212443
- Absolute reduction: 0.0334704364
- Relative reduction: ~3.98%
- Raw Top-K Jaccard: 0.2177643125
- Stable Top-K Jaccard: 0.2474935419
- Intervention rate: 0.4192987932
- Mean alpha: 0.0752742059
- Routed token observations: 339586

These routing values are counterfactual diagnostics only. They are not yet
evidence that changing MobileMoE's real expert dispatch preserves output quality.

## Provenance

The uploaded result was produced before the later vectorized shadow-probe
performance optimization. The exact run SHA was not embedded inside the result
artifact, so the run-producing commit is recorded in `FREEZE_MANIFEST.json`
as session-derived provenance rather than artifact-verified provenance.

Do not overwrite this phase directory with future reruns. Create a new phase or
run directory instead.
