# KiaOmni multimodal OCR: falsifiable experiment protocol
Date: 2026-10-11. Status: **PLAN, not executed**.
Do not conflate image whitening / OCR text truncation proxies with internal visual-token pruning or KV cache eviction.

## 0. Freeze and data integrity
- Keep the 5 existing synthetic samples as debugging/smoke only. S03 Latin ID glyphs and S05 date need human visual verification.
- Build 20-30 annotated development docs and 50-100+ strictly held-out evaluation docs, with template/source split and field-to-bbox mapping.
- Ground truth must include exact amounts/IDs, semantic dates, row/column associations, and abstention labels.
- Freeze original images, annotation revisions, image preprocessing versions, prompts, processor and tokenizer versions, seeds, and model revisions/commits.
- Preserve all raw outputs including repeated/truncated responses; never equate `status=ok` with a correct prediction.

## 1. Full-context baseline + observational hooks
- Models separately: Qwen3-VL-4B-Instruct and Gemma 4 E4B; retain the native official visual budget baseline. Run deterministic decode, identical documents and extraction JSON target.
- Instrument before/after vision projector; log *actual* sequence length, image patch/embedding mapping, spatial coordinates and special image/text token spans, positional embeddings, attention backend and Q/K tensor shapes.
- Record per-layer and per-head cross-attention/saliency to visual spans and the decoder task-relevant query window. Preserve both the raw attention and any derived smoothed score.
- Confirm batch=1 no-op at 100% retention: generated output parity and logits numerical tolerance versus unmodified model. Confirm trace consistency across repeated runs.
- If model internal APIs cannot support this mapping/position integrity, label `UNSUPPORTED`; do not emulate with image whitening and call it internal compression.

## 2. Independent intervention families
T: Text-token selection *inside* the multimodal prompt/context; keep field labels/values linked.
V: Actual vision-embedding selection *after vision encoder/projector*, not image masking. Respect visual spatial mapping, multi-axis/rotary positions and multimodal separators; implement safe fallback.
K: True KV gather/prune after prefill, remap/validate cache metadata and continue decode. Do not reset positions to a false short sequence. Log retained KV elements per layer and prefill peak separately.
V/K: test each independently before joint combinations. Joint only after single-family parity passes.
Negative controls: random keep, unsmoothed top-k, naive image masking, native Gemma budget, simple ROI crop.

## 3. KiaOmni signal ablations
- Faithful 1D log1p saliency + Gaussian sigma 4 and Boxcar radius 8, plus unsmoothed top-k.
- Compare decoder last-query vs 32-query max pooling; current repository documents differ in these descriptions. Include mean-head and per-layer variants.
- Test 2D neighborhood smoothing only after visual patch token mapping is validated; keep the same token budget.
- No ground-truth field positions or annotations may enter the inference-time selector. If using OCR detections as a policy input, declare this cost and input explicitly.
- Protect special/control tokens and essential text; count them in budgets, not outside them.

## 4. Budget grid, feasibility and measurements
Fixed retained positions: B=98,128,256,512; proportional: r=0.25,0.125,0.0625; full-context reference and a no-pruning 100% parity test.
For vision-specific budgets also report the *actual* retained visual patch tokens, and compare same retained-token counts. Mark a budget infeasible if mandatory protected tokens exceed it.
Timing: full document end-to-end, preprocessing, vision encoder, prefill, decode, postprocessing and any reload; TTFT/p50/p95.
Memory: CUDA peak allocated/reserved, GPU process peak, model+KV occupancy, final KV size; do not claim improved prefill memory from post-prefill pruning.
Quality: field exact/semantic accuracy, all-fields-correct document rate, monetary critical-field errors, table cell-linkage, valid abstention/false-accept, correction harms, runtime failure/truncation.
Cost: compute $ per correctly auto-accepted document using measured hardware runtime and honest reviewer cost; no savings claim from theoretical token count only.
Repeat each condition with multiple run seeds/environment provenance. Keep failed/skipped jobs in denominator as explicit status.

## 5. Go/no-go gates
Gate 1: Untouched 100% parity + correct visual span mapping, no silent fallback.
Gate 2: True retained KV/vision tokens demonstrably decrease at requested budget; memory and timing measured, not assumed.
Gate 3: At least equivalent financial correctness to the full uncompressed baseline on held-out examples, with no observed regression in critical auto-accepted fields (not a guarantee of zero risk).
Gate 4: Measured end-to-end savings vs both native-token-budget and ROI controls, accounting for selection overhead.
No winner is declared from the original five synthetic docs or from the proxy runs.

## 6. Archival and reproducibility
Each job should persist:
`run_id, model_id, model_revision, git_sha, dataset_sha256, sample_id, run_family, score_policy, budget_type, budget_value, selected_token_indices, selected_token_coords, prompt_hash, processor_config, seed, environment_lock, raw_output, status, timing, memory, correctness`.
Persist raw JSONL with one document/condition per record plus a reproducibility script, dataset manifest, data licensing, SHA256SUMS and a human-readable findings report. Publish a PR against `main`, never mutate historical KiaOmni benchmarks.
