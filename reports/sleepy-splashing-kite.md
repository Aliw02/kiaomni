# Plan: Mistral 32K Modal Run

## Context
Qwen2.5-7B was already benchmarked at 32K (035_v1_qwen_longctx_results). To validate
long-context KV-cache eviction cross-model, we need Mistral-7B-v0.3 at 32K as well.
Mistral natively supports 32K via sliding window + RoPE — no scaling hack needed.
The existing `034_v3_mistral_comparison.py` needs 5 targeted edits, then runs on Modal.

---

## File to Modify
`notebook/kv_cache_benchmark/034_v3_mistral_comparison.py`

---

## Changes (in order)

### 1. CTX_LENS — line 39
```python
# Before
CTX_LENS = [8192, 16384]
# After
CTX_LENS = [32768]
```

### 2. OUT_DIR (Modal branch) — line 60
```python
# Before
OUT_DIR = Path("/vol/kv_results/035_v1_mistral_results")
# After
OUT_DIR = Path("/vol/kv_results/035_v1_mistral_32k_results")
```

### 3. FILLER_SENTENCES — lines 642–663
Expand from 20 → 80 diverse sentences (same fix applied to Qwen 035).
At 32K with only 20 sentences the haystack repeats ~445× — statistically invalid.
Replace the 20-sentence list with 80 sentences covering varied domains.

### 4. Remove CAKE from POLICIES — line 500
```python
# Remove this line entirely:
"CAKE": ("sal_all", lambda s, B, L: cake_keep(s["sal_mean"], s["sal_std"], B, L)),
```
CAKE is a decode-time policy — excluded by design (documented in paper appendix).

### 5. LB is auto-skipped — no change needed
`LB_CTX_LENS = [c for c in CTX_LENS if c <= 4096]` returns `[]` when CTX_LENS=[32768].
LongBench will not run. Correct — short documents are scientifically useless at 32K.

---

## Modal Run Command
```bash
modal run 034_v3_mistral_comparison.py
```
Output lands in `/vol/kv_results/035_v1_mistral_32k_results/`.
Download and keep in `035_v1_mistral_results/` locally (032K-only, separate from merged 034).

---

## Verification
- Sanity tests print 12/12 passed (CAKE removed, so 13→12 policies)
- First trial log shows `ctx=32768`
- No `LongBench` lines in output log
- `predictions.csv` ctx column = 32768 only
- RatioAdaptive will score near 0 at B=96 (compression_ratio=341 → σ explodes) — expected, disclose in paper
