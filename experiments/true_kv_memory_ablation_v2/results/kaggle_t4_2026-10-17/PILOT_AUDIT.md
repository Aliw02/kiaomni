# KiaOmni Original — True KV Memory Ablation V2 — 4K Pilot Audit

**Date:** 2026-10-17  | **Scope:** Original text-only KiaOmni / Qwen2.5-7B NF4; NOT the 1300-run OCR study.
**Evidence:** User-uploaded `KiaOmni_TrueKV_V2_RESEARCH_Results (2).zip`, `pilot/` raw JSONL, protocol, manifest, preflight and derived CSV. This report independently recomputes figures from the raw runs.

## Executive finding
- **Pilot executed completely:** 174/174 GPU rows have status `ok`, zero OOM, zero runtime errors, six 4096-token tasks, one seed/repetition per task.
- **Both improvements are real, but separate:** chunked full prefill lowers 4K PyTorch peak allocated 9.7509 -> ~5.60 GiB; True-KV physically compacts a full 4096-token cache 224 MiB -> 14 MiB at B256, saving 210 MiB *after prefill*.
- **B256 and larger preserves the measured strict first-line accuracy at this tiny pilot size:** 5/6 for FC and every KiaOmni K0–K4; B128 4/6; B98 2/6. This does not establish general accuracy parity.
- **No capacity/OOM breakthrough is demonstrated:** F0 succeeded at 4K. K2–K4 still build and retain the entire KV cache until post-prefill; this method may hit OOM at sufficiently large context due to the accumulated cache.
- **Generation validity caveat:** all 174 outputs hit the 128-token cap (none ended early under this harness); most outputs include irrelevant continuation. The harness scores first answer line and its `status=ok` does not mean graceful EOS termination.

## Frozen environment / provenance
- Research Git commit: `f578af4187901e9359d546ede9d8f6663da01bc5` on `exp/kiaomni-true-kv-memory-ablation-v2-research`.
- Protocol digest SHA256: `2c3fc97358c1481f835c12ac87e126b702c75ad4df22416902532996bc51eaec`; NVIDIA Tesla T4 14.562 GiB; Python `3.13.15`; torch `2.11.0+cu128`; SDPA; Qwen2.5-7B NF4 4-bit / FP16.
- 28 Qwen decoder layers, 4 KV heads and 128 head dimensions. Text cases: single_needle, multi_key, multi_value, multi_query, variable_tracking, summary_facts.
- Deduplicated budgets at 4K are 98, 128, 256, 512, 1024. Aliases: 256=B256=r0.0625, 512=B512=r0.125, 1024=r0.25.
- `manifest.json`: `COMPLETED`, expected rows=174, successes=174, errors=0, oom=0. `scientific_status=GPU_EXECUTED_PENDING_REVIEW` indicates results still need interpretation.
- `preflight.json`: `PASS` on 1024 tokens, 128/256/512 chunks share same cache lengths and next-token argmax; exact logit values differ by up to 0.078125, 0.0546875, 0.048828125 respectively. Legacy vs deferred saliency masks match for B128/B256/B512 at 1K.

## 4K full vs chunked vs True KV at B256
| Method | Meaning | Accuracy | Peak allocated GiB | Peak reserved GiB | TTFT sec | Total sec | Generated tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| F0 | Full Context / one shot | 5/6 | 9.751 | 10.154 | 7.447 | 21.605 | 9.051 |
| F1 | Full KV / chunks 128 | 5/6 | 5.598 | 6.725 | 8.756 | 22.923 | 9.043 |
| F2 | Full KV / chunks 256 | 5/6 | 5.598 | 6.713 | 6.276 | 20.465 | 9.028 |
| F3 | Full KV / chunks 512 | 5/6 | 5.598 | 6.709 | 4.746 | 18.967 | 9.006 |
| K0 | True KV, legacy QK hooks, one shot | 5/6 | 9.751 | 10.180 | 7.792 | 16.678 | 14.411 |
| K1 | True KV, deferred QK, one shot | 5/6 | 9.751 | 10.180 | 7.740 | 16.626 | 14.411 |
| K2 | True KV, deferred QK, chunks 128 | 5/6 | 5.546 | 6.725 | 9.137 | 18.024 | 14.408 |
| K3 | True KV, deferred QK, chunks 256 | 5/6 | 5.559 | 6.713 | 6.505 | 15.444 | 14.324 |
| K4 | True KV, deferred QK, chunks 512 | 5/6 | 5.587 | 6.709 | 4.838 | 13.683 | 14.476 |

### Causal attribution and VRAM accounting
- F0 -> K4(B256): peak allocated decreases **4.164 GiB (42.7%)**; peak reserved decreases **3.445 GiB (33.9%)** on this 4K test.
- F3 vs K4 peak allocated: F3 5.598 vs K4 5.587 GiB — practically the **same prefill peak**, despite K4 retaining less KV after prefill. Therefore the 42.7% peak reduction is **chunking**, not evidence that saliency-compaction itself reduced prefill peak.
- Physical KV after B256: **224 MiB -> 14 MiB**, **16x** reduction; approximately 210 MiB freed from the cache after prefill.
- Decoder-stage allocated-after: F3 **5.411 GiB** vs K4(B256) **5.206 GiB**, ~**210 MiB** less for K4. This is consistent with actual KV shrinkage, distinct from peak prefill savings.
- Chunking memory was measured on this specific implementation using a 4D Boolean causal mask with SDPA, which may trigger a different attention kernel/memory plan than the one-shot path. Hardware/framework/attention backend must be frozen for further comparison.
- `peak_reserved` is allocator-reserved memory, not equivalent to the physical KV tensor size or peak allocated memory; driver-free memory snapshots are a separate quantity.

## Budget-quality outcomes across all five K methods
| Retained KV tokens per layer | Compression ratio vs 4K full KV | Strict correct cases (of 6) | Mean first-line recall | Compared with FC |
|---:|---:|---:|---:|---|
| 98 | 41.80x | 2/6 | 0.4444 | 3 regressions (multi_key, multi_value, summary_facts), 0 rescues |
| 128 | 32.00x | 4/6 | 0.7917 | 1 regressions (summary_facts), 0 rescues |
| 256 | 16.00x | 5/6 | 0.8333 | No correctness regressions or rescues |
| 512 | 8.00x | 5/6 | 0.8333 | No correctness regressions or rescues |
| 1024 | 4.00x | 5/6 | 0.8333 | No correctness regressions or rescues |
- The above per-budget correctness and recall are identical for K0, K1, K2, K3 and K4; this does **not** imply their full decoded text or KV masks match.
- At B256, F0 and K4 each answered five tasks correctly. Both missed `variable_tracking`: expected `VX-29317`, first answer was `DX-61022`. No quality improvement over FC is observed on this case set.

### Case-level comparison (FC vs K4)
| Task | Reference | FC first-line | K4 B98 | K4 B128 | K4 B256 | K4 B512 | K4 B1024 |
|---|---|---|---|---|---|---|---|
| multi_key | `285629, 468794, 376828` | `285629, 468794, 376828` | FAIL | PASS | PASS | PASS | PASS |
| multi_query | `663501, 749793` | `663501 749793` | PASS | PASS | PASS | PASS | PASS |
| multi_value | `543, 1697, 68204` | `543, 1697, 68204` | FAIL | PASS | PASS | PASS | PASS |
| single_needle | `206344` | `206344` | PASS | PASS | PASS | PASS | PASS |
| summary_facts | `9490, 6586, 3454, 3072` | `9490 6586 3454 3072` | FAIL | FAIL | PASS | PASS | PASS |
| variable_tracking | `VX-29317` | `DX-61022` | FAIL | FAIL | FAIL | FAIL | FAIL |

## Saliency masks, answer stability and decode format
| K variant vs K0 | Equal SHA-256 selection masks across 6 tasks × 5 budgets | Identical complete decoded text vs K0 |
|---|---:|---:|
| K1 | 30/30 | 30/30 |
| K2 | 1/30 | 23/30 |
| K3 | 2/30 | 23/30 |
| K4 | 1/30 | 21/30 |
- K1 is a clean refactor of K0 saliency according to this tested hash evidence (30/30 equal masks at 4K), with no accuracy differences. The chunked variants produce different masks in the large majority of tested cases; exact overlap/Jaccard and per-layer saliency deltas were not logged.
- **All 174 `generated_tokens` = 128**, the configured cap. The model did not stop early. The runner reports `status=ok` whenever the code path returns successfully even if EOS was never generated. This is a benchmark-completion status, not response-completion status.
- In **120/174** decoded outputs, text contains `Human:` or `User:` after the answer, evidence of uncontrolled continuation/role-like material. This does not invalidate first-line scores, but invalidates treating whole-output generation as clean short-answer completion.
- `score_output` scores reference strings that occur in the first non-empty output line, not an exact full answer. It should be accompanied by exact numeric tuple extraction and an EOS/completion flag in a future revision.

## Speed interpretation
- F0 vs K4(B256): mean end-to-end time **21.605s -> 13.683s**, reduction **36.7%**. TTFT **7.447s -> 4.838s**. Token decoding throughput **9.051 -> 14.476 tokens/s** (about **59.9%** higher).
- These timings reflect generating exactly 128 tokens for all six cases, often continuing past the user-requested answer. They do **not** estimate production answer completion latency, especially if future EOS stopping behavior changes.
- One repetition per task, hence no confidence intervals or power claims are justified.

## Scientific conclusion and gate decision
**PASS, engineering pilot:** 174 rows, physical True KV compression, 1K preflight, repeatable within the supplied raw evidence, and a large 4K peak memory reduction using chunked prefill.
**CONDITIONAL quality signal:** no *observed* strict first-line accuracy regression at B256/B512/B1024 vs FC on six synthetic tasks, but small sample size, one seed, permissive first-line metric and generation truncation demand stronger validation.
**NOT YET PROVEN:** KiaOmni works at an input length where one-shot Full Context actually experiences CUDA OOM while preserving answer quality; method currently retains all historical KV until prefill finishes.
**DO NOT run the 756-job Final automatically** from this pilot. Next: targeted 8K/16K capacity gate on the same T4 with isolated GPU processes for FC and KiaOmni to avoid allocator contamination after any OOM; preserve entire context and benchmark cases.

## Recommended immediate experiments (in order)
1. **Capacity gate, 8K first.** Same model revision, torch/transformers/SDPA, T4, fixed inputs, seed, budget B256/B512. Compare `F0` one-shot FC, `F3` no-eviction 512 chunks, and `K4` chunks+True KV. For clean failure accounting, execute each in a fresh process and capture `error_stage`, CUDA OOM, physical KV bytes, peak allocated/reserved, and driver-free memory. Try 16K and then 32K only if the 8K evidence justifies it; choose protected batch/task sizes.
2. **Quality validity gate.** Use an EOS/completion flag, a stop-on-answer evaluator or exact first-line tuple metric, and several independent seeds; focus on multi_key, multi_query, variable_tracking, summary_facts. Do not count repetitive 128-token continuations as successful finished answers.
3. **Mask/roPE equivalence audit.** At matching input/positions, measure per-layer Jaccard overlap of selected KV indices and saliency deltas between K0/K1 and K2–K4. Confirm 8K preflight next-token/full-cache parity where feasible, not only 1K argmax.
4. **Capacity design after measuring failure mode.** If FC is OOM solely because of one-shot attention intermediates but F3/K4 complete, document this as a *chunked-prefill* capacity improvement, not proof that True KV fixed peak. If F3 later OOMs from full KV growth, design/test a bounded prefill that does not permanently discard potentially future-relevant KV using the frozen previous streaming-eviction method; distinguish new designs and benchmark their accuracy independently.
5. **Do not conflate weight offload (AirLLM), multimodal OCR, Prompt Selection, and True KV V2**. No 8K FC OOM has yet been directly observed in this uploaded 4K pilot.

## Artifacts and limitations
- Original user-supplied ZIP: `KiaOmni_TrueKV_V2_RESEARCH_Results (2).zip`. Contains `/pilot/{runs.jsonl, protocol.json, manifest.json, preflight.json, summary.csv, paired.csv, ablation_contrasts.csv, mask_equivalence.csv, model_info.json}` and corresponding Smoke outputs. Preserve its bytes unmodified.
- Source URLs: https://github.com/Aliw02/kiaomni/tree/exp/kiaomni-true-kv-memory-ablation-v2-research and the original experiment branch `exp/kiaomni-true-kv-memory-ablation-v2`. `run_ablation.py`, `memory_runtime.py`, `kiaomni/repro/qwen25_kv_runtime.py` are the primary audit entry points.
- Measured scope is a controlled six-task synthetic 4K pilot. Do not extrapolate 83.3% to general language understanding, LongBench, RULER, or 8K OOM capacity.

**END OF AUDIT — no GPU processes were executed during this offline analysis.**
