# Financial OCR experiment evidence — 2026-10-11

This is a **review/staging branch**, not a production implementation or a claim that internal KiaOmni vision/KV compression was benchmarked successfully.

## Provenance: original artifacts retained outside this branch
- `KiaOmni_ALL16_Results.zip`, 407 KB, SHA256 `0ab3abb09c707e19e68b58354758968041ebf742110bfd56ff5df21bcc980d8b`.
- `KiaOmni_Financial_OCR_V3_ALL16_Complete.zip`, SHA256 `128d465c14e2ce87a51218024a56f9b38fafb42ed232093cd3ecb00bc0b61491`.
- `KiaOmni_ALL16_Forensic_Review_2026-10-11.md`, SHA256 `fe3a5bf0834dc889c0a537410065b72fef1fe32c024462615afd59eaaeaede65`.

**Important: the ZIPs and full raw per-case outputs are not uploaded to GitHub in this commit.** The branch contains a compact extracted route summary, the two exact KV read-only status records, and a protocol. Complete independent reproduction requires archiving the full raw ZIP and the exact executed Notebook/source code or exporting its raw cases into versioned text files.

## Audit facts (not extrapolated)
- Five synthetic docs; 35 scored fields.
- 250 jobs planned; 200 actual Qwen/Gemma GPU inference jobs with `ok`; 45 text-budget skips; 5 Falcon jobs not run (historic Falcon text reused).
- Six Qwen responses hit 1300 output-token cap yet got `ok` status.
- Route A (Qwen + parser) scored 33/35 fields, 3/5 all-correct docs.
- Route P2/S2 ties A but is the same underlying spatial-proxy run, not independent evidence; no measured VRAM/token saving.
- Read-only `DynamicCache` probes: Qwen S01 1595 tokens, Gemma S01 273; `true_eviction_executed=false` for both.
- Reported stage-time is NOT full end-to-end time for multi-model routes.
- **None** of the 16 routes proved real Vision Token pruning or true KV-cache compression.

## Files
- `architecture_summary.csv`: 16 route rows transcribed from original evaluation JSON.
- `kv_probe_*.json`: original two minimal diagnostics.
- `VALIDATION_PROTOCOL.md`: preregistered experiment gates and baselines.

Please keep this separate from previously published LongBench/RULER/LLM text-cache experiments.
