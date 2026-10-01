# Phase 03 Adjudication + Actual Routing V1 — Runbook

Branch: `exp/kiaomni-qwen3-30b-adjudication-routing-v1`

This run preserves the frozen KiaOmni policies and legacy fixed budgets:

- `kiaomni_s8`
- `kiaomni_gaussian`
- budgets: `512, 256, 128, 98`

The parent Phase-03 results remain untouched on commit
`aa5d0d5c0eff8809398af1d53a29ed7d83ab5cfb`.

## 1. Sync branch

```powershell
git fetch origin
git switch exp/kiaomni-qwen3-30b-adjudication-routing-v1
git pull --ff-only
git rev-parse HEAD
```

## 2. Preflight

The existing `kiaomni-qwen3-assets` volume is reused. Do not delete it.

```powershell
modal run --detach modal/qwen3_30b_adjudication_modal.py --stage preflight --gpu A100-80GB --prepare-index
```

Follow logs using the App ID printed by Modal:

```powershell
modal app logs APP_ID --follow
```

Preflight must end with:

```text
"status": "PASS"
```

Download:

```powershell
$out = ".\results\kiaomni_moe_model_lab\phase_03_adjudication_routing_v1"
New-Item -ItemType Directory -Force -Path $out | Out-Null
modal volume get kiaomni-qwen3-adjudication-results phase_03_adjudication_routing_v1/preflight.json "$out\preflight.json"
modal volume get kiaomni-qwen3-adjudication-results phase_03_adjudication_routing_v1/preflight.log "$out\preflight.log"
```

Do not run final if preflight is not PASS.

## 3. Final

```powershell
modal run --detach modal/qwen3_30b_adjudication_modal.py --stage final --gpu A100-80GB
```

Follow the new App ID:

```powershell
modal app logs APP_ID --follow
```

The log prints every raw model answer and its metrics live.

Download:

```powershell
modal volume get kiaomni-qwen3-adjudication-results phase_03_adjudication_routing_v1/final.json "$out\final.json"
modal volume get kiaomni-qwen3-adjudication-results phase_03_adjudication_routing_v1/final.log "$out\final.log"
```

## 4. Freeze outputs in Git

The `results/` tree is ignored globally, so force-add only these run artifacts:

```powershell
git add -f -- "$out\preflight.json"
git add -f -- "$out\preflight.log"
git add -f -- "$out\final.json"
git add -f -- "$out\final.log"
git status --short
git commit -m "results: freeze Qwen3 30B adjudication routing V1"
git push origin exp/kiaomni-qwen3-30b-adjudication-routing-v1
git status
git rev-parse HEAD
```

Do not delete `kiaomni-qwen3-assets` until the results have been reviewed.

## What the final run measures

Each case runs:

- FullContext
- `kiaomni_s8` at B512, B256, B128, B98
- `kiaomni_gaussian` at B512, B256, B128, B98

Real set: all 27 frozen LongBench-v2 parent IDs that remain within 8K-16K under the official zero-shot prompt.

Diagnostics include:

- raw answer
- parsed multiple-choice accuracy
- FC↔Kia agreement / regression / rescue
- gold-answer NLL and PPL
- generation peak VRAM
- routing/PPL teacher-forward peak VRAM
- saliency peak VRAM
- pipeline peak VRAM
- generated tokens
- output tokens/s
- 256-token output cap and cap-hit flag
- actual MoE routing metrics on matched retained source tokens:
  - top-1 expert agreement
  - top-8 expert-set Jaccard
  - dispatch-weight cosine
  - dispatch-entropy delta
  - expert-load JSD
  - layer-by-layer values

Routing is captured from the executed Transformers 4.57.6 Qwen3-MoE gate. The run independently hooks every expert module and requires the observed per-expert dispatch counts to match the gate-implied assignments exactly.
