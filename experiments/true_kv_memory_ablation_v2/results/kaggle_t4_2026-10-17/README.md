# KiaOmni Original — True KV Memory Ablation V2 — Kaggle T4 GPU evidence

**Experiment date:** 2026-10-17.  **Scope:** Text-only Qwen2.5-7B-Instruct / NF4 / Tesla T4 (14.562 GiB). **NOT** the Official20 OCR 1300-job experiment.

## Immutable input/provenance

- Code execution commit: `f578af4187901e9359d546ede9d8f6663da01bc5` on `exp/kiaomni-true-kv-memory-ablation-v2-research`.
- Smoke: `smoke/` directory; SHA256 protocol `152dc9c7c5c8ec874d5ad52eb7fffb5c2402dddbac0e7f87065df9be08f55`; 28/28 attempts, all `ok`, no OOM/errors.
- Pilot: `pilot/` directory; SHA256 protocol `2c3fc97358c1481f835c12ac87e126b702c75ad4df22416902532996bc51eaec`; 174/174 attempts, all `ok`, no OOM/errors; six 4096-token synthetic tasks, one repetition.
- Runtime: PyTorch `2.11.0+cu128`, Transformers `4.57.6`, Python `3.13.15`, NF4 quantized weights, SDPA.
- Data copied **unmodified** from the user's GPU-export ZIP (except this README), with `runs.jsonl` as the raw authoritative record. The commit containing the results is later than the frozen code execution commit; do not confuse them.

## Paths / contents

- `pilot/runs.jsonl`: every executed 4K case/method/budget, per-phase memory/timing, output text and strict first-line answers.
- `pilot/summary.csv`: aggregate accuracy/recall, GPU allocated/reserved, throughput, timing, physical KV ratio.
- `pilot/paired.csv`, `pilot/ablation_contrasts.csv`, `pilot/mask_equivalence.csv`: paired controls and mask-digest comparisons.
- `pilot/manifest.json`, `pilot/protocol.json`, `pilot/preflight.json`, `pilot/model_info.json`: run gates and provenance.
- `smoke/`: corresponding 2K smoke evidence.

## Verified findings (4K pilot)

| Setting | Strict correct | Peak allocated GiB | Peak reserved GiB | Mean end-to-end sec |
|---|---:|---:|---:|---:|
| F0 — full, one-shot | 5/6 | 9.751 | 10.154 | 21.605 |
| F3 — full, 512 chunks | 5/6 | 5.598 | 6.709 | 18.967 |
| K4 — 512 chunks + True KV B256 | 5/6 | 5.587 | 6.709 | 13.683 |

- At **B256**, physical KV cache shrank **224 MiB -> 14 MiB (16x)** after full prefill. Correct cases at B256/B512/B1024 = 5/6 for each K0–K4; B128=4/6, B98=2/6.
- Most of the 4K prefill peak reduction comes from **chunking**, not from post-prefill KV selection; F3 and K4 share essentially the same peak memory.
- All 174 decoded outputs reached the 128-token cap, rather than ending naturally; the first-line accuracy metric does not imply completion quality. 1K preflight checked same argmax, not exact logits equality.
- Full Context **did not OOM at 4K**. Neither this data nor a historical 8K safety skip establishes a real OOM capacity breakthrough.
- Do not claim broader accuracy guarantees from one seed per task. Do not overwrite these raw files on reruns; use another run-id/subdirectory.

## Next step

Conduct a **small, isolated 8K capacity boundary gate** with F0 (one-shot Full Context), F3 (no-eviction 512-token chunked Full KV), and K4 (512-token chunks + true KV B256/B512), identical model/settings and paired prompts. Use fresh CUDA process per condition, capture actual OOM vs completed output, allocator-allocated/reserved, driver memory, KV bytes, EOS status and exact answers. Gradually expand to 16K only if warranted. Do **not** spend on a 756-run Final yet.

The user's primary goal is True KV inference across a context that **actually OOMs for FC on the same GPU**, without re-feeding a shortened prompt or misattributing chunk-only memory savings to compression. V2 still retains the full KV until prefill completes; a bounded-prefill design is a separate future effort and must protect recall. **Do not silently reintroduce the previously rejected streaming-eviction approach.**
