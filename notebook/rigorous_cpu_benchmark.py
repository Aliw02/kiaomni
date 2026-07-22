"""
Rigorous CPU Benchmark — KV-Cache Eviction Policies
=====================================================
Scientific design principles:
  1. Context >= 2000 tokens  (tests 'lost in the middle' regime)
  2. Qwen2-0.5B architecture (modern attention sinks, GQA-like patterns)
  3. Real Semantic Needle    (zero attention manipulation — ground truth is positional)
  4. Wilcoxon signed-rank    (non-parametric, paired, one-sided significance test)

Pre-registration hash must be printed and locked before any run.
All policies receive IDENTICAL attention tensors from a SINGLE forward pass.
"""

import numpy as np
import hashlib, json, datetime, sys, re
sys.stdout.reconfigure(encoding='utf-8')
from scipy import stats

# ══════════════════════════════════════════════════════════════════════
# 0. PRE-REGISTRATION PROTOCOL
# ══════════════════════════════════════════════════════════════════════
PROTOCOL = {
    "version":              "rigorous-v1.0",
    "date":                 str(datetime.date.today()),
    "model":                "Qwen/Qwen2-0.5B",      # ~1GB RAM on CPU, modern arch
    "cache_budget_ratio":   0.35,                   # keep 35% — real eviction pressure
    "context_len":          2000,                   # minimum for lost-in-the-middle
    "n_samples_per_domain": 20,                     # 20 paired samples per domain
    "random_seed":          2025,
    "needle_manipulation":  "NONE",                 # critical: no attn boosting
    "hyperparams": {
        # TFCache
        "TFCACHE_TAU": 800,   "TFCACHE_ALPHA": 1.5,
        # DualTime
        "DT_TAU_FAST": 200,   "DT_TAU_SLOW": 40000, "DT_LAM": 0.2,  "DT_ALPHA": 1.5,
        # VFCA
        "VFCA_TAU":    800,   "VFCA_ALPHA": 1.5,
        # GRACERank
        "GR_TAU":      800,   "GR_ALPHA": 1.5,  "GR_THETA": 0.3,
        # TurboCD
        "TCD_TAU_FAST": 400,  "TCD_TAU_SLOW": 50000, "TCD_LAM": 0.15,
        "TCD_ALPHA": 2.0,     "TCD_THETA": -0.5,
    },
    "primary_metric":    "needle_recall",
    "secondary_metrics": ["attn_coverage", "mean_rank_of_kept"],
    "significance_test": "wilcoxon_signed_rank_one_sided_greater",
    "alpha_level":       0.05,
}

proto_hash = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True).encode()
).hexdigest()[:16]
print(f"╔══════════════════════════════════════════╗")
print(f"  PROTOCOL HASH: {proto_hash}")
print(f"  Lock this value before running.")
print(f"╚══════════════════════════════════════════╝\n")

# ══════════════════════════════════════════════════════════════════════
# 1. MODEL LOAD — Qwen2-0.5B on CPU
# ══════════════════════════════════════════════════════════════════════
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME  = PROTOCOL["model"]
CONTEXT_LEN = PROTOCOL["context_len"]
BUDGET      = int(CONTEXT_LEN * PROTOCOL["cache_budget_ratio"])
HP          = PROTOCOL["hyperparams"]

print(f"Loading {MODEL_NAME} on CPU ...")
print(f"Context: {CONTEXT_LEN} tokens | Budget: {BUDGET} tokens ({PROTOCOL['cache_budget_ratio']*100:.0f}%)\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_attentions=True,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
model.eval()
print("Model ready.\n")

# ══════════════════════════════════════════════════════════════════════
# 2. DATASET — CNN/DailyMail (real long documents)
# ══════════════════════════════════════════════════════════════════════
from datasets import load_dataset

_RNG = np.random.default_rng(PROTOCOL["random_seed"])

def _get_texts(n: int):
    """Stream CNN/DailyMail and return n articles with >= 2000 tokens."""
    ds   = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
    long_docs = []
    for item in ds:
        text = item["article"]
        ids  = tokenizer.encode(text)
        if len(ids) >= CONTEXT_LEN:
            long_docs.append(text)
        if len(long_docs) >= n * 4:   # buffer for domain splits
            break
    print(f"  Collected {len(long_docs)} long documents (>= {CONTEXT_LEN} tokens).")
    idxs = _RNG.choice(len(long_docs), size=min(n, len(long_docs)), replace=False)
    return [long_docs[i] for i in idxs]

# ══════════════════════════════════════════════════════════════════════
# 3. ATTENTION EXTRACTOR
# ══════════════════════════════════════════════════════════════════════
def get_attention_weights(text: str):
    """
    Returns:
      attn_sum  : np.ndarray [CONTEXT_LEN]  — sum of attention received per token
      layer_attns: list of np.ndarray [heads, seq, seq]
    No manipulation. Raw model output only.
    """
    ids = tokenizer.encode(
        text, return_tensors="pt",
        max_length=CONTEXT_LEN, truncation=True
    )
    if ids.shape[1] < CONTEXT_LEN:
        return None

    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)

    n_layers    = len(out.attentions)
    attn_sum    = np.zeros(CONTEXT_LEN, dtype=np.float64)
    layer_attns = []

    for la in out.attentions:                        # [1, heads, seq, seq]
        arr = la[0].float().numpy()                  # [heads, seq, seq]
        # sum attention RECEIVED by each key position (column sum, mean over heads)
        col_sum = arr.mean(axis=0).sum(axis=0)       # [seq]
        attn_sum += col_sum[:CONTEXT_LEN]
        layer_attns.append(arr[:, :CONTEXT_LEN, :CONTEXT_LEN])

    attn_sum /= n_layers   # normalize by depth
    return attn_sum, layer_attns

# ══════════════════════════════════════════════════════════════════════
# 4. REAL SEMANTIC NEEDLE INJECTION
# ══════════════════════════════════════════════════════════════════════
# Patterns that identify factual spans (names, numbers, dates).
# We locate these in the FIRST QUARTER of the tokenized text —
# the hardest zone for recency-biased policies.
_NEEDLE_PATTERNS = [
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',           # Proper names
    r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',       # Numbers / figures
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4}\b',  # Dates
    r'\b[A-Z]{2,}\b',                           # Acronyms
]

def find_needle_token_span(text: str):
    """
    Locate a factual span in the FIRST 25% of the text.
    Returns (start_token, end_token) indices in the tokenized sequence,
    or None if no candidate found.
    NO attention weight modification. Ever.
    """
    # Tokenize full text
    enc = tokenizer(text, return_offsets_mapping=True,
                    max_length=CONTEXT_LEN, truncation=True)
    token_ids    = enc["input_ids"]
    offsets      = enc["offset_mapping"]   # (char_start, char_end) per token

    if len(token_ids) < CONTEXT_LEN:
        return None

    # Only search in first 25% of character span — the hard zone
    total_chars  = offsets[-1][1]
    hard_zone_end = int(total_chars * 0.25)

    for pattern in _NEEDLE_PATTERNS:
        for m in re.finditer(pattern, text):
            if m.end() > hard_zone_end:
                continue
            # Map char span → token span
            tok_start, tok_end = None, None
            for i, (cs, ce) in enumerate(offsets):
                if cs <= m.start() < ce and tok_start is None:
                    tok_start = i
                if cs < m.end() <= ce:
                    tok_end = i
                    break
            if tok_start is not None and tok_end is not None and tok_end < CONTEXT_LEN:
                # needle_positions: all token indices covering the span
                return list(range(tok_start, tok_end + 1))

    return None   # no needle found in this document

# ══════════════════════════════════════════════════════════════════════
# 5. POSITION ARRAYS
# ══════════════════════════════════════════════════════════════════════
positions = np.arange(CONTEXT_LEN, dtype=np.float64)
dt        = (CONTEXT_LEN - 1) - positions    # distance from the last token

# ══════════════════════════════════════════════════════════════════════
# 6. POLICY IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════
def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    top = np.argpartition(scores, -k)[-k:]
    return np.sort(top)

def _cv(layer_attns) -> np.ndarray:
    """Coefficient of variation across layers (VFCA burstiness signal)."""
    stacked = np.stack([la.mean(axis=0).sum(axis=0) for la in layer_attns])  # [L, seq]
    mu  = stacked.mean(0) + 1e-9
    std = stacked.std(0)
    return std / mu

def policy_recency(attn, layers, k):
    return np.arange(CONTEXT_LEN - k, CONTEXT_LEN)

def policy_h2o(attn, layers, k):
    return top_k(attn, k)

def policy_tfcache(attn, layers, k,
                   tau=HP["TFCACHE_TAU"], alpha=HP["TFCACHE_ALPHA"]):
    freq  = np.power(np.log1p(attn), alpha)
    decay = np.exp(-dt / tau)
    return top_k(freq * decay, k)

def policy_dualtime(attn, layers, k,
                    tau_fast=HP["DT_TAU_FAST"], tau_slow=HP["DT_TAU_SLOW"],
                    lam=HP["DT_LAM"],           alpha=HP["DT_ALPHA"]):
    freq  = np.power(np.log1p(attn), alpha)
    decay = lam * np.exp(-dt / tau_fast) + (1 - lam) * np.exp(-dt / tau_slow)
    return top_k(freq * decay, k)

def policy_vfca(attn, layers, k,
                tau=HP["VFCA_TAU"], alpha=HP["VFCA_ALPHA"]):
    freq  = np.power(np.log1p(attn), alpha)
    decay = np.exp(-dt / tau)
    cv    = _cv(layers)
    return top_k(freq / (cv + 1.0) * decay, k)

def policy_gracerank(attn, layers, k,
                     tau=HP["GR_TAU"], alpha=HP["GR_ALPHA"], theta=HP["GR_THETA"]):
    freq     = np.power(np.log1p(attn), alpha)
    decay    = np.exp(-dt / tau)
    raw      = freq * decay
    z        = (raw - raw.mean()) / (raw.std() + 1e-9)
    admitted = np.where(z >= theta)[0]
    if len(admitted) >= k:
        chosen = admitted[np.argsort(raw[admitted])[-k:]]
    elif len(admitted) == 0:
        chosen = top_k(raw, k)
    else:
        non_adm = np.setdiff1d(np.arange(CONTEXT_LEN), admitted)
        fill    = non_adm[np.argsort(raw[non_adm])[-(k - len(admitted)):]]
        chosen  = np.concatenate([admitted, fill])
    return np.sort(chosen)

def policy_turbocd(attn, layers, k,
                   tau_fast=HP["TCD_TAU_FAST"], tau_slow=HP["TCD_TAU_SLOW"],
                   lam=HP["TCD_LAM"],           alpha=HP["TCD_ALPHA"],
                   theta=HP["TCD_THETA"]):
    freq     = np.power(np.log1p(attn), alpha)
    decay    = lam * np.exp(-dt / tau_fast) + (1 - lam) * np.exp(-dt / tau_slow)
    cv       = _cv(layers)
    raw      = (freq / (cv + 1.0)) * decay
    z        = (raw - raw.mean()) / (raw.std() + 1e-9)
    admitted = np.where(z >= theta)[0]
    if len(admitted) >= k:
        chosen = admitted[np.argsort(raw[admitted])[-k:]]
    elif len(admitted) == 0:
        chosen = top_k(raw, k)
    else:
        non_adm = np.setdiff1d(np.arange(CONTEXT_LEN), admitted)
        fill    = non_adm[np.argsort(raw[non_adm])[-(k - len(admitted)):]]
        chosen  = np.concatenate([admitted, fill])
    return np.sort(chosen)

ALL_POLICIES = [
    ("Recency",   policy_recency),
    ("H2O",       policy_h2o),
    ("TFCache",   policy_tfcache),
    ("DualTime",  policy_dualtime),
    ("VFCA",      policy_vfca),
    ("GRACERank", policy_gracerank),
    ("TurboCD",   policy_turbocd),
]

# ══════════════════════════════════════════════════════════════════════
# 7. METRICS
# ══════════════════════════════════════════════════════════════════════
def needle_recall(kept_idx: np.ndarray, needle_positions: list) -> float:
    """
    Fraction of needle tokens retained.
    For a single-token needle this is 0 or 1.
    For multi-token spans this is partial credit.
    NO attention manipulation involved.
    """
    kept_set = set(kept_idx.tolist())
    hits = sum(1 for p in needle_positions if p in kept_set)
    return hits / len(needle_positions)

def attn_coverage(kept_idx: np.ndarray, attn: np.ndarray) -> float:
    """Fraction of total attention mass captured by kept tokens."""
    return float(attn[kept_idx].sum() / (attn.sum() + 1e-9))

def mean_rank_of_kept(kept_idx: np.ndarray, attn: np.ndarray) -> float:
    """Mean percentile rank (by attention) of kept tokens. 1.0 = perfect."""
    sorted_idx = np.argsort(attn)
    ranks = np.empty(CONTEXT_LEN)
    ranks[sorted_idx] = np.arange(CONTEXT_LEN) / CONTEXT_LEN
    return float(ranks[kept_idx].mean())

# ══════════════════════════════════════════════════════════════════════
# 8. DOMAINS
#    needle_hard  — factual span in first 25% of 2K context
#    needle_mid   — factual span in 25–50% zone  (softer)
#    coverage     — no needle; tests attention mass preservation
# ══════════════════════════════════════════════════════════════════════
DOMAINS = {
    "needle_hard": "first-25%-of-2K-context",
    "coverage":    "no-needle-attn-mass",
}

# ══════════════════════════════════════════════════════════════════════
# 9. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════
print("Fetching long documents (>= 2000 tokens) ...")
N = PROTOCOL["n_samples_per_domain"]
texts = _get_texts(N * 6)   # large buffer
print()

pol_names = [n for n, _ in ALL_POLICIES]

# results[domain][policy] = {"recall": [], "cov": [], "rank": []}
results = {
    d: {p: {"recall": [], "cov": [], "rank": []} for p in pol_names}
    for d in DOMAINS
}

# ── needle_hard domain ───────────────────────────────────────────────
print("=" * 68)
print("  DOMAIN: NEEDLE_HARD  (real semantic needle, first 25% of 2K)")
print("=" * 68)

collected   = 0
text_cursor = 0
while collected < N and text_cursor < len(texts):
    text        = texts[text_cursor]
    text_cursor += 1

    aw = get_attention_weights(text)
    if aw is None:
        continue
    attn, layers = aw

    needle_positions = find_needle_token_span(text)
    if needle_positions is None:
        continue   # no factual span found in hard zone — skip

    for pol_name, pol_fn in ALL_POLICIES:
        kept = pol_fn(attn, layers, BUDGET)
        r    = results["needle_hard"][pol_name]
        r["recall"].append(needle_recall(kept, needle_positions))
        r["cov"].append(attn_coverage(kept, attn))
        r["rank"].append(mean_rank_of_kept(kept, attn))

    collected += 1
    print(f"  sample {collected}/{N}  needle_pos={needle_positions[:3]}...", end="\r")

print(f"\n  Done: {collected} samples.\n")

# ── coverage domain ──────────────────────────────────────────────────
print("=" * 68)
print("  DOMAIN: COVERAGE  (no needle — pure attention mass)")
print("=" * 68)

collected = 0
while collected < N and text_cursor < len(texts):
    text        = texts[text_cursor]
    text_cursor += 1

    aw = get_attention_weights(text)
    if aw is None:
        continue
    attn, layers = aw

    for pol_name, pol_fn in ALL_POLICIES:
        kept = pol_fn(attn, layers, BUDGET)
        r    = results["coverage"][pol_name]
        r["cov"].append(attn_coverage(kept, attn))
        r["rank"].append(mean_rank_of_kept(kept, attn))

    collected += 1
    print(f"  sample {collected}/{N}", end="\r")

print(f"\n  Done: {collected} samples.\n")

# ══════════════════════════════════════════════════════════════════════
# 10. STATISTICAL REPORT
#     Wilcoxon signed-rank test (paired, one-sided: policy > H2O)
#     N=20 satisfies minimum sample requirement for Wilcoxon
# ══════════════════════════════════════════════════════════════════════
def wilcoxon_vs_baseline(arr_policy, arr_baseline, alternative="greater"):
    """Paired Wilcoxon signed-rank. Returns (statistic, p_value)."""
    a = np.array(arr_policy)
    b = np.array(arr_baseline)
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 5:
        return float("nan"), float("nan")
    diff = a[valid] - b[valid]
    if np.all(diff == 0):
        return float("nan"), 1.0
    try:
        stat, p = stats.wilcoxon(a[valid], b[valid], alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")

print("\n\n" + "═" * 68)
print("  STATISTICAL RESULTS")
print("  Baseline: H2O  |  Test: Wilcoxon signed-rank (one-sided, p<0.05)")
print("═" * 68)

for domain in DOMAINS:
    h2o_cov    = np.array(results[domain]["H2O"]["cov"])
    h2o_recall = np.array(results[domain]["H2O"].get("recall", []))
    has_needle = len(results[domain]["H2O"]["recall"]) > 0

    print(f"\n{'─'*68}")
    print(f"  {domain.upper()}")
    print(f"{'─'*68}")

    if has_needle:
        print(f"  {'Policy':<14} {'Recall%':>8} {'AttnCov%':>10} "
              f"{'MeanRank':>10} {'p(cov>H2O)':>12} {'p(rec>H2O)':>12}")
    else:
        print(f"  {'Policy':<14} {'AttnCov%':>10} "
              f"{'MeanRank':>10} {'p(cov>H2O)':>12}")
    print("  " + "·" * 58)

    rows = []
    for pol in pol_names:
        r           = results[domain][pol]
        cov_arr     = np.array(r["cov"])
        rnk_arr     = np.array(r["rank"])
        rec_arr     = np.array(r["recall"]) if r["recall"] else np.array([])

        _, p_cov = wilcoxon_vs_baseline(cov_arr, h2o_cov)
        p_rec    = float("nan")
        if has_needle and pol != "H2O" and len(rec_arr) >= 5:
            _, p_rec = wilcoxon_vs_baseline(rec_arr, h2o_recall)

        rec_pct = np.mean(rec_arr) * 100 if len(rec_arr) > 0 else float("nan")
        rows.append((pol, rec_pct, cov_arr.mean()*100, rnk_arr.mean(), p_cov, p_rec))

    rows.sort(key=lambda x: -x[2])
    for pol, rec, cov, rnk, p_cov, p_rec in rows:
        sig_cov = " ★" if (not np.isnan(p_cov) and p_cov < 0.05) else "  "
        sig_rec = " ★" if (not np.isnan(p_rec) and p_rec < 0.05) else "  "
        if has_needle:
            print(f"  {pol:<14} {rec:>7.1f}%{sig_rec} {cov:>9.1f}%{sig_cov} "
                  f"{rnk:>10.4f} {p_cov:>12.4f} {p_rec:>12.4f}")
        else:
            print(f"  {pol:<14} {cov:>9.1f}%{sig_cov} {rnk:>10.4f} {p_cov:>12.4f}")

    print(f"  ★ = significantly better than H2O (Wilcoxon p < {PROTOCOL['alpha_level']})")

# ══════════════════════════════════════════════════════════════════════
# 11. OVERALL RANKING — primary: needle_recall | secondary: attn_coverage
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*68}")
print("  OVERALL RANKING")
print(f"  Primary: needle_recall (needle_hard) | Secondary: attn_coverage")
print(f"{'═'*68}")

agg = {p: {"recall": [], "cov": []} for p in pol_names}
for domain in DOMAINS:
    for pol in pol_names:
        r = results[domain][pol]
        agg[pol]["cov"].extend(r["cov"])
        if r["recall"]:
            agg[pol]["recall"].extend(r["recall"])

ranked = sorted(
    pol_names,
    key=lambda p: (
        -np.mean(agg[p]["recall"]) if agg[p]["recall"] else 0,
        -np.mean(agg[p]["cov"])
    )
)

for rank, pol in enumerate(ranked, 1):
    rec = np.mean(agg[pol]["recall"]) * 100 if agg[pol]["recall"] else float("nan")
    cov = np.mean(agg[pol]["cov"])    * 100
    print(f"  {rank:>2}. {pol:<14}  recall={rec:5.1f}%  attn_cov={cov:.1f}%")

print(f"\nProtocol hash: {proto_hash}  (must match pre-registration)")
print("Done.")
