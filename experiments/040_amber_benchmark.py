"""
040_amber_benchmark.py — Full KV-Cache Benchmark on LLM360/Amber (7B MHA)
===========================================================================
Model    : LLM360/Amber — fully open LLaMA-style 7B, standard MHA
Policies : SnapKV_Modified, RealSnapKV, Ada-SnapKV, H2O,
           KiaOmni_σ8, KiaOmni_Adaptive, KiaOmni_RatioAdaptive,
           KiaOmni_Quest, KiaOmni_Gaussian, KiaOmni_AnchorExp,
           KiaOmni_Scissorhands
Tasks    : RULER (niah_single, niah_multikey, vt)
           + LongBench (narrativeqa, qasper, multifieldqa_en,
                        hotpotqa, 2wikimqa, musique, gov_report, qmsum)
Contexts : {512, 1024, 2048}
Budgets  : {98, 128, 256, 512}
N        : 15 RULER trials / 15 LongBench samples per cell

Install:
    pip install -U -q transformers bitsandbytes accelerate rouge-score scipy

Run:
    python experiments/040_amber_benchmark.py

Outputs:
    experiments/results/040_amber_results/results.json
    experiments/results/040_amber_results/predictions.csv
    experiments/results/040_amber_results/speed_vram.csv
    experiments/results/040_amber_results/eviction_coherence_loss.csv
    experiments/results/040_amber_results/checkpoints/
"""

import csv, gc, json, math, os, random, re, string, collections, time
import urllib.request, zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME   = "LLM360/Amber"
CTX_LENS     = [512, 1024, 2048]
BUDGETS      = [98, 128, 256, 512]
N_TRIALS     = 15
LB_SAMPLES   = 15
RULER_TASKS  = ["niah_single", "niah_multikey", "vt"]
LB_TASKS     = ["narrativeqa", "qasper", "multifieldqa_en",
                 "hotpotqa", "2wikimqa", "musique",
                 "gov_report", "qmsum"]
SEED         = 42
MAX_NEW      = 96

N_SINK      = 16
RECENCY     = 32
BLOCK_SIZE  = 16
SIGMA_FIXED = 8
SIGMA_MAX   = 64
SNAP_POOL_K = 5
SNAP_OBS_W  = 32

N_KEYS      = 4
CHAIN_LEN   = 5

OUT_DIR  = Path(os.environ.get("OUT_DIR", "results/040_amber_results"))
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

PRED_CSV_PATH      = OUT_DIR / "predictions.csv"
SPEED_CSV_PATH     = OUT_DIR / "speed_vram.csv"
COHERENCE_CSV_PATH = OUT_DIR / "eviction_coherence_loss.csv"

PRED_COLS  = ["source", "task", "ctx", "trial_or_sample", "policy", "budget",
              "ground_truth", "prediction", "f1", "em", "rouge_l", "contains"]
SPEED_COLS = ["source", "task", "ctx", "trial_or_sample", "policy", "budget",
              "sal_ms", "gen_ms", "tokens_per_sec",
              "vram_sal_mb", "vram_gen_mb"]
COHERENCE_COLS = ["source", "task", "ctx", "trial_or_sample", "policy", "budget",
                  "eviction_coherence_loss"]

METRIC_KEYS = ["f1", "em", "rouge_l", "contains"]


def load_model():
    print(f"Loading {MODEL_NAME} (4-bit NF4, bfloat16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=cfg,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            print(f"  attn_implementation={attn_impl}", flush=True)
            break
        except Exception as e:
            print(f"  {attn_impl} failed: {e}", flush=True)
    model.eval()
    print("Model ready.", flush=True)
    return model, tok


def extract_all_saliency(ids: torch.Tensor, model) -> dict:
    c        = model.config
    nh       = c.num_attention_heads
    nk       = getattr(c, "num_key_value_heads", nh)
    hd       = c.hidden_size // nh
    n_layers = len(model.model.layers)

    last_k_buf: dict = {}
    sal_per_layer_list: list = [None] * n_layers
    hooks = []

    for l_idx, layer in enumerate(model.model.layers):

        def _make_k_hook(layer_idx, is_last, q_store):
            def _h(m, inp, out):
                k_raw = out.detach().cpu().to(torch.float32)
                q_raw = q_store.get("q")
                if q_raw is None:
                    return
                _L   = ids.shape[1]
                _nh2 = c.num_attention_heads
                _nk2 = getattr(c, "num_key_value_heads", _nh2)
                _hd2 = c.hidden_size // _nh2
                q2 = q_raw.view(1, _L, _nh2, _hd2).transpose(1, 2)
                k2 = k_raw.view(1, _L, _nk2, _hd2).transpose(1, 2)
                if _nk2 != _nh2:
                    k2 = k2.repeat_interleave(_nh2 // _nk2, dim=1)
                sc2 = torch.matmul(q2[:, :, -1:, :], k2.transpose(-2, -1)) * (_hd2 ** -0.5)
                sal_heads = torch.softmax(sc2, dim=-1)[0, :, 0, :]
                sal_mean_l = sal_heads.mean(0).numpy()
                sal_per_layer_list[layer_idx] = sal_mean_l.astype(np.float32)
                if is_last:
                    last_k_buf["sal_heads"] = sal_heads.numpy()
                    obs_w = min(SNAP_OBS_W, _L)
                    q_obs = q2[:, :, -obs_w:, :]
                    sc_obs = torch.matmul(q_obs, k2.transpose(-2, -1)) * (_hd2 ** -0.5)
                    prefix_len = max(1, _L - obs_w)
                    attn_prefix = torch.softmax(sc_obs[..., :prefix_len], dim=-1)
                    votes = attn_prefix.sum(dim=-2)
                    max_v = votes.max(dim=-1, keepdim=True).values
                    pad   = max_v.expand(1, _nh2, obs_w)
                    sal_snapkv_h = torch.cat([votes, pad], dim=-1)[0]
                    last_k_buf["sal_snapkv"] = sal_snapkv_h.numpy()
                    del q_obs, sc_obs, attn_prefix, votes, max_v, pad, sal_snapkv_h
                del q2, k2, sc2, sal_heads
            return _h

        is_last = (l_idx == n_layers - 1)
        _temp_q_store: dict = {}

        def _q_capture(m, inp, out, _store=_temp_q_store):
            _store["q"] = out.detach().cpu().to(torch.float32)

        hooks.append(layer.self_attn.q_proj.register_forward_hook(_q_capture))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            _make_k_hook(l_idx, is_last, _temp_q_store)))

    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    sal_heads_last  = last_k_buf.get("sal_heads")
    sal_snapkv_last = last_k_buf.get("sal_snapkv")

    if sal_heads_last is None:
        sal_mean = _last_layer_saliency(ids, model)
        sal_per_head = np.tile(sal_mean, (nh, 1))
    else:
        sal_per_head = sal_heads_last.astype(np.float32)
        sal_mean     = sal_per_head.mean(0)

    sal_snapkv = (
        sal_snapkv_last.astype(np.float32)
        if sal_snapkv_last is not None
        else sal_per_head
    )

    sal_per_layer = np.stack(
        [(x if x is not None else sal_mean) for x in sal_per_layer_list], axis=0
    ).astype(np.float32)

    n_lay = sal_per_layer.shape[0]
    sal_scissor = (
        sal_per_layer[n_lay // 4] +
        sal_per_layer[n_lay // 2] +
        sal_per_layer[-1]
    ) / 3.0

    del last_k_buf
    return {
        "sal_mean":      sal_mean,
        "sal_snapkv":    sal_snapkv,
        "sal_per_head":  sal_per_head,
        "sal_per_layer": sal_per_layer,
        "sal_scissor":   sal_scissor,
    }


def _last_layer_saliency(ids: torch.Tensor, model) -> np.ndarray:
    buf: dict = {}
    last = model.model.layers[-1].self_attn
    hq = last.q_proj.register_forward_hook(
        lambda *a: buf.update({"q": a[2].detach().cpu().to(torch.float32)}))
    hk = last.k_proj.register_forward_hook(
        lambda *a: buf.update({"k": a[2].detach().cpu().to(torch.float32)}))
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        hq.remove(); hk.remove()
    q, k = buf["q"], buf["k"]
    L = ids.shape[1]; c = model.config
    nh = c.num_attention_heads
    nk = getattr(c, "num_key_value_heads", nh)
    hd = c.hidden_size // nh
    q = q.view(1, L, nh, hd).transpose(1, 2)
    k = k.view(1, L, nk, hd).transpose(1, 2)
    if nk != nh:
        k = k.repeat_interleave(nh // nk, dim=1)
    sc  = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) * (hd ** -0.5)
    sal = torch.softmax(sc, dim=-1)[0, :, 0, :].mean(0).numpy()
    del q, k, sc
    return sal


def _protected(n: int) -> set:
    return set(range(min(N_SINK, n))) | set(range(max(0, n - RECENCY), n))


def _boxcar(x: np.ndarray, sigma: int) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    ps = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
    lo = np.maximum(0, np.arange(len(x)) - sigma)
    hi = np.minimum(len(x), np.arange(len(x)) + sigma + 1)
    return ((ps[hi] - ps[lo]) / (hi - lo)).astype(np.float32)


def snapkv_modified_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    prot_mask = np.zeros(seq_len, dtype=bool)
    prot_mask[:N_SINK] = True
    prot_mask[max(0, seq_len - RECENCY):] = True
    evict_idx = np.where(~prot_mask)[0]
    if len(evict_idx) == 0 or budget >= seq_len:
        return set(range(seq_len))
    page_ids     = evict_idx // BLOCK_SIZE
    sal_evict    = sal[evict_idx]
    unique_pages = np.unique(page_ids)
    page_scores  = np.array(
        [sal_evict[page_ids == pg].mean() for pg in unique_pages], dtype=np.float32)
    order         = np.argsort(page_scores)
    evicted_mask  = np.zeros(seq_len, dtype=bool)
    tokens_evicted = 0
    target_evict  = max(0, seq_len - budget)
    for pi in order:
        if tokens_evicted >= target_evict:
            break
        pg_mask = page_ids == unique_pages[pi]
        evicted_mask[evict_idx[pg_mask]] = True
        tokens_evicted += int(pg_mask.sum())
    return set(np.where(~evicted_mask)[0].tolist())


def snapkv_real_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    from scipy.ndimage import maximum_filter1d
    if budget >= seq_len:
        return set(range(seq_len))
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if eff <= 0:
        return prot
    if sal.ndim == 1:
        sal = sal[np.newaxis, :]
    n_heads = sal.shape[0]
    k_per_h = max(1, eff // n_heads)
    free    = np.array([i for i in range(seq_len) if i not in prot])
    if len(free) == 0:
        return prot
    kept: set = set(prot)
    for h in range(n_heads):
        pooled = maximum_filter1d(sal[h, :seq_len].astype(np.float32), size=SNAP_POOL_K)
        k      = min(k_per_h, len(free))
        top    = np.argpartition(pooled[free], -k)[-k:]
        kept  |= set(free[top].tolist())
    if len(kept) > budget:
        mean_sal = sal.mean(0)
        kept_arr = np.array(sorted(kept))
        trim_k   = np.argpartition(mean_sal[kept_arr], -budget)[-budget:]
        kept     = set(kept_arr[trim_k].tolist())
    return kept


def h2o_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    prot  = _protected(seq_len)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-sal[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot


def get_adaptive_sigma(sal: np.ndarray, budget: int, seq_len: int) -> int:
    p = sal / (np.sum(sal) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    h_norm = entropy / np.log(max(seq_len, 2))
    peakiness = max(0.0, 1.0 - h_norm)
    adaptive_sigma = SIGMA_MAX * peakiness * np.sqrt(budget / seq_len)
    return int(max(1, round(adaptive_sigma)))


def get_ratio_adaptive_sigma(sal: np.ndarray, budget: int, seq_len: int) -> int:
    p = sal / (sal.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    h_norm = entropy / np.log(max(seq_len, 2))
    peakiness = max(0.0, 1.0 - h_norm)
    compression_ratio = seq_len / max(1, budget)
    adaptive_sigma = compression_ratio * peakiness
    return int(max(1, round(adaptive_sigma)))


def ada_snapkv_keep(sal_mean, budget, seq_len, n_sink=N_SINK, recency=RECENCY, obs_window=64):
    start = max(0, seq_len - recency - obs_window)
    end   = seq_len - recency
    obs = sal_mean[start:end]
    if len(obs) == 0:
        ratio = 0
    else:
        p = obs / (obs.sum() + 1e-9)
        H = -(p * np.log(p + 1e-9)).sum()
        H_max = np.log(obs_window + 1e-9)
        ratio = H / H_max
    dynamic_budget = int(budget * (1.0 + 0.5 * ratio))
    dynamic_budget = min(dynamic_budget, seq_len)
    protected = _protected(seq_len)
    eff = max(0, dynamic_budget - len(protected))
    sal = sal_mean.copy()
    sal[list(protected)] = -np.inf
    topk = np.argpartition(sal, -eff)[-eff:] if eff > 0 else np.array([], dtype=int)
    return set(topk.tolist()) | protected


def _quest_envelope(x: np.ndarray, sigma: int) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(x.astype(np.float32), size=(2 * sigma) + 1)


def kiaomni_quest_keep(sal: np.ndarray, budget: int, seq_len: int, sigma: int = 8) -> set:
    prot  = _protected(seq_len)
    F     = _quest_envelope(np.log1p(sal), sigma)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x.astype(np.float32), sigma=sigma)


def kiaomni_gaussian_keep(sal: np.ndarray, budget: int, seq_len: int, sigma: float = 4.0) -> set:
    prot  = _protected(seq_len)
    F     = _gaussian_smooth(np.log1p(sal), sigma)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot


def kiaomni_anchor_expand_keep(sal: np.ndarray, budget: int, seq_len: int, radius: int = 5) -> set:
    prot = _protected(seq_len)
    free = max(0, budget - len(prot))
    if free <= 0:
        return prot
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if len(cands) == 0:
        return prot
    sorted_idx = cands[np.argsort(-sal[cands])]
    keep_set = set(prot)
    for anchor in sorted_idx:
        if len(keep_set) >= budget:
            break
        lo, hi = max(0, anchor - radius), min(seq_len, anchor + radius + 1)
        for j in range(lo, hi):
            if len(keep_set) >= budget:
                break
            keep_set.add(j)
    return keep_set


def kiaomni_fixed_keep(sal: np.ndarray, budget: int, seq_len: int, sigma: int = 8) -> set:
    prot  = _protected(seq_len)
    F     = _boxcar(np.log1p(sal), sigma)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot


def kiaomni_adaptive_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    sigma = get_adaptive_sigma(sal, budget, seq_len)
    return kiaomni_fixed_keep(sal, budget, seq_len, sigma)


def kiaomni_ratio_adaptive_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    sigma = get_ratio_adaptive_sigma(sal, budget, seq_len)
    print(f"Ratio-Adaptive sigma: {sigma}", flush=True)
    return kiaomni_fixed_keep(sal, budget, seq_len, sigma)


POLICIES: dict = {
    "SnapKV_Modified":       ("sal_mean",    lambda sals, B, L: snapkv_modified_keep(sals["sal_mean"], B, L)),
    "RealSnapKV":            ("sal_snapkv",  lambda sals, B, L: snapkv_real_keep(sals["sal_snapkv"], B, L)),
    "Ada-SnapKV":            ("sal_mean",    lambda sals, B, L: ada_snapkv_keep(sals["sal_mean"], B, L)),
    "H2O":                   ("sal_mean",    lambda sals, B, L: h2o_keep(sals["sal_mean"], B, L)),
    "KiaOmni_\u03c38":       ("sal_mean",    lambda sals, B, L: kiaomni_fixed_keep(sals["sal_mean"], B, L)),
    "KiaOmni_Adaptive":      ("sal_mean",    lambda sals, B, L: kiaomni_adaptive_keep(sals["sal_mean"], B, L)),
    "KiaOmni_RatioAdaptive": ("sal_mean",    lambda sals, B, L: kiaomni_ratio_adaptive_keep(sals["sal_mean"], B, L)),
    "KiaOmni_Quest":         ("sal_mean",    lambda sals, B, L: kiaomni_quest_keep(sals["sal_mean"], B, L)),
    "KiaOmni_Gaussian":      ("sal_mean",    lambda sals, B, L: kiaomni_gaussian_keep(sals["sal_mean"], B, L)),
    "KiaOmni_AnchorExp":     ("sal_mean",    lambda sals, B, L: kiaomni_anchor_expand_keep(sals["sal_mean"], B, L)),
    "KiaOmni_Scissorhands":  ("sal_scissor", lambda sals, B, L: kiaomni_fixed_keep(sals["sal_scissor"], B, L)),
}


@torch.no_grad()
def gen_evict(model, tok, ids: torch.Tensor, keep: set, max_new: int = MAX_NEW) -> str:
    keep_t = torch.tensor(sorted(keep), device=ids.device, dtype=torch.long)
    p      = ids[:, keep_t]
    out    = model.generate(p, attention_mask=torch.ones_like(p),
                            max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, p.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def gen_full(model, tok, ids: torch.Tensor, max_new: int = MAX_NEW) -> str:
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _token_f1(pred: str, truth: str) -> float:
    p, t = _norm(pred).split(), _norm(truth).split()
    if not p or not t:
        return float(p == t)
    common = sum((collections.Counter(p) & collections.Counter(t)).values())
    if common == 0:
        return 0.0
    return 2 * common / (len(p) + len(t))


def _rouge_l(pred: str, truth: str) -> float:
    p, t = _norm(pred).split(), _norm(truth).split()
    if not p or not t:
        return 0.0
    m, n = len(p), len(t)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == t[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec)

try:
    from rouge_score import rouge_scorer as _rs
    _rouge_lib = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    def _rouge_l(pred: str, truth: str) -> float:  # noqa: F811
        return _rouge_lib.score(_norm(truth), _norm(pred))["rougeL"].fmeasure
except ImportError:
    pass


def compute_metrics(pred: str, ground_truth) -> dict:
    if isinstance(ground_truth, list):
        answers = ground_truth
    else:
        answers = [str(ground_truth)]
    best: dict = {"f1": 0.0, "em": 0.0, "rouge_l": 0.0, "contains": 0.0}
    pn = _norm(pred)
    for a in answers:
        an = _norm(str(a))
        best["f1"]       = max(best["f1"],      _token_f1(pred, a))
        best["em"]       = max(best["em"],       float(pn == an))
        best["rouge_l"]  = max(best["rouge_l"],  _rouge_l(pred, a))
        best["contains"] = max(best["contains"], float(an in pn))
    return best


@torch.no_grad()
def measure_coherence_loss(model, ids: torch.Tensor, keep: set) -> float:
    if not keep:
        return float("inf")
    keep_t    = torch.tensor(sorted(keep), device=ids.device, dtype=torch.long)
    ids_evict = ids[:, keep_t]
    if ids_evict.shape[1] < 2:
        return float("inf")
    try:
        out = model(ids_evict, labels=ids_evict)
        return float(torch.exp(out.loss).item())
    except Exception:
        return float("nan")


def _vram_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


FILLER_SENTENCES = [
    "The history of astronomy dates back thousands of years to ancient civilizations.",
    "Ocean currents are driven by wind patterns and differences in water temperature.",
    "The periodic table organizes elements by atomic number and chemical properties.",
    "Languages evolve over time, borrowing words and structures from neighboring tongues.",
    "Tectonic plates move at roughly the same speed as human fingernails grow.",
    "The Amazon rainforest produces about 20 percent of the world's oxygen supply.",
    "Medieval castles were built with thick stone walls to withstand siege weapons.",
    "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
    "Neurons communicate through electrochemical signals called action potentials.",
    "Trade routes like the Silk Road connected civilizations across thousands of miles.",
    "Climate zones are determined by latitude, altitude, and proximity to oceans.",
    "The printing press revolutionized the spread of information in the 15th century.",
    "Mitochondria generate most of the ATP used by eukaryotic cells for energy.",
    "Stars in a galaxy orbit around a central region containing a supermassive black hole.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Plate tectonics explains volcanic activity, earthquakes, and mountain formation.",
    "The Renaissance was a cultural movement emphasizing art, science, and philosophy.",
    "Quantum entanglement allows two particles to affect each other instantly over distance.",
    "Coral reefs are among the most biodiverse ecosystems on the planet.",
    "The moon's gravitational pull is responsible for Earth's ocean tides.",
]


def _rand_code(rng: random.Random, k: int = 8) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=k))


def _build_haystack(rng: random.Random, target_chars: int) -> str:
    sentences = FILLER_SENTENCES[:]
    hay = ""
    while len(hay) < target_chars:
        rng.shuffle(sentences)
        hay += " ".join(sentences) + " "
    return hay[:target_chars]


def build_niah_single(rng: random.Random, tok, target_toks: int, depth: float):
    key    = _rand_code(rng, 8)
    needle = f"The secret passphrase is: {key}."
    q_text = "\n\nWhat is the secret passphrase? Answer with only the passphrase value."
    pre    = "Read the following text carefully.\n\n"
    hay    = _build_haystack(rng, target_toks * 5)
    for _ in range(20):
        split = int(len(hay) * depth)
        full  = pre + hay[:split] + "\n\n" + needle + "\n\n" + hay[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, key


def build_niah_multikey(rng: random.Random, tok, target_toks: int):
    keys    = [_rand_code(rng, 8) for _ in range(N_KEYS)]
    depths  = [i / (N_KEYS + 1) for i in range(1, N_KEYS + 1)]
    needles = [f"Secret key {i+1} is: {k}." for i, k in enumerate(keys)]
    q_text  = "\n\nList all secret keys in order (key1, key2, key3, key4):"
    pre     = "Read the following text carefully.\n\n"
    hay     = _build_haystack(rng, target_toks * 5)
    for _ in range(20):
        parts: list = [pre]
        prev = 0
        for depth, needle in zip(depths, needles):
            pos = int(len(hay) * depth)
            parts.append(hay[prev:pos] + "\n\n" + needle + "\n\n")
            prev = pos
        parts.append(hay[prev:] + q_text)
        full = "".join(parts)
        ids  = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, keys


def build_vt(rng: random.Random, tok, target_toks: int):
    vars_     = [f"var_{_rand_code(rng, 4)}" for _ in range(CHAIN_LEN + 1)]
    final_val = _rand_code(rng, 6)
    assigns   = ([f"{vars_[0]} = {final_val}"] +
                 [f"{vars_[i]} = {vars_[i-1]}" for i in range(1, CHAIN_LEN + 1)])
    rng.shuffle(assigns)
    question_var = vars_[-1]
    hay      = _build_haystack(rng, target_toks * 5)
    q_text   = f"\n\nWhat is the value of {question_var}? Answer with only the value."
    pre      = "Given the following variable assignments:\n\n"
    for _ in range(20):
        full = pre + "\n".join(assigns) + "\n\nAnd the following context:\n\n" + hay + q_text
        ids  = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, final_val


RULER_DEPTHS = [0.25, 0.5, 0.75]


LB_TASK_FILES: dict = {
    "narrativeqa":     "narrativeqa.jsonl",
    "qasper":          "qasper_e.jsonl",
    "multifieldqa_en": "multifieldqa_en_e.jsonl",
    "hotpotqa":        "hotpotqa_e.jsonl",
    "2wikimqa":        "2wikimqa_e.jsonl",
    "musique":         "musique.jsonl",
    "gov_report":      "gov_report_e.jsonl",
    "qmsum":           "qmsum.jsonl",
}
LB_DATA_ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"


def _ensure_lb_zip() -> Path:
    zip_path = OUT_DIR / "longbench_data.zip"
    if zip_path.exists():
        return zip_path
    print("Downloading LongBench data.zip (~250 MB)...", flush=True)
    req  = urllib.request.Request(LB_DATA_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=300)
    zip_path.write_bytes(resp.read())
    print(f"  Saved to {zip_path}", flush=True)
    return zip_path


def _ensure_lb_data() -> None:
    all_present = all((OUT_DIR / fname).exists() for fname in LB_TASK_FILES.values())
    if all_present:
        return
    zip_path = _ensure_lb_zip()
    print("  Extracting LongBench JSONL files...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        for fname in LB_TASK_FILES.values():
            for member in zf.namelist():
                if member.endswith(fname):
                    (OUT_DIR / fname).write_bytes(zf.open(member).read())
                    break


def load_lb_task(task: str, n: int) -> list:
    _ensure_lb_data()
    jpath = OUT_DIR / LB_TASK_FILES[task]
    samples = []
    with open(jpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


LB_PROMPTS: dict = {
    "narrativeqa":
        "Answer the question based on the story. Be concise.\n\nStory: {context}\n\nQuestion: {input}\n\nAnswer:",
    "qasper":
        "Answer the question based on the given paper. Be concise.\n\nPaper: {context}\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en":
        "Answer the question based on the given context. Be concise.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:",
    "hotpotqa":
        "Answer the question based on the given passages. Be concise.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "2wikimqa":
        "Answer the question based on the given passages. Be concise.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "musique":
        "Answer the question based on the given passages. Be concise.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "gov_report":
        "Write a one-page summary of the following government report.\n\nReport: {context}\n\nSummary:",
    "qmsum":
        "Based on the meeting transcript, answer the query.\n\nTranscript: {context}\n\nQuery: {input}\n\nAnswer:",
}


def build_lb_ids(sample: dict, task: str, tok, max_ctx: int) -> torch.Tensor:
    tmpl    = LB_PROMPTS[task]
    context = sample.get("context", "")
    input_  = sample.get("input", "")
    full    = tmpl.format(context=context, input=input_)
    ids     = tok(full, return_tensors="pt").input_ids
    if ids.shape[1] > max_ctx:
        half    = int(len(context) * (max_ctx / max(ids.shape[1], 1)) * 0.85) // 2
        context = context[:half] + context[len(context) - half:]
        ids     = tok(tmpl.format(context=context, input=input_), return_tensors="pt").input_ids
    return ids


def get_lb_answers(sample: dict) -> list:
    ans = sample.get("answers", sample.get("answer", ""))
    if isinstance(ans, list):
        return ans
    return [str(ans)]


def ckpt_key(source: str, task: str, ctx: int, idx: int) -> str:
    return f"{source}_{task}_ctx{ctx}_i{idx:04d}"


def load_checkpoints() -> dict:
    done: dict = {}
    for f in CKPT_DIR.glob("*.json"):
        done[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return done


def save_checkpoint(key: str, data: dict) -> None:
    (CKPT_DIR / f"{key}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _init_csv(path: Path, cols: list) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()


def _append_csv(path: Path, cols: list, rows: list[dict]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        for r in rows:
            w.writerow(r)


def _empty_scores() -> dict:
    return {m: [] for m in METRIC_KEYS}


def _make_results_skeleton(pol_names: list) -> dict:
    res: dict = {}
    for src in ("ruler", "longbench"):
        res[src] = {}
        tasks = RULER_TASKS if src == "ruler" else LB_TASKS
        for t in tasks:
            res[src][t] = {}
            for ctx in CTX_LENS:
                res[src][t][ctx] = {}
                for pol in pol_names:
                    res[src][t][ctx][pol] = {B: _empty_scores() for B in BUDGETS}
    return res


def _reload_results_from_checkpoints(completed: dict, results: dict,
                                     pol_names: list) -> None:
    for key, data in completed.items():
        try:
            parts = key.split("_ctx")
            pre, rest = parts[0], parts[1]
            ctx_s, _ = rest.split("_i")
            ctx = int(ctx_s)
            src_task = pre.split("_", 1)
            source = src_task[0]
            task   = src_task[1]
            if source not in results or task not in results[source]:
                continue
            if ctx not in results[source][task]:
                continue
            for pol in pol_names:
                for B in BUDGETS:
                    for m in METRIC_KEYS:
                        v = data.get(pol, {}).get(str(B), {}).get(m)
                        if v is not None:
                            results[source][task][ctx][pol][B][m].append(v)
        except Exception:
            pass


def run_trial(
    source: str,
    task: str,
    ctx: int,
    trial_idx: int,
    ids: torch.Tensor,
    ground_truth,
    model,
    tok,
    results: dict,
    pol_names: list,
) -> tuple[dict, list, list, list]:
    device  = next(model.parameters()).device
    ids     = ids.to(device)
    seq_len = ids.shape[1]

    trial_data: dict = {}
    pred_rows:  list = []
    speed_rows: list = []
    coherence_rows: list = []

    _vram_reset()
    t_sal0 = time.perf_counter()
    try:
        sals = extract_all_saliency(ids, model)
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM saliency ctx={ctx} trial={trial_idx}", flush=True)
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return trial_data, pred_rows, speed_rows, coherence_rows
    sal_ms   = (time.perf_counter() - t_sal0) * 1000
    vram_sal = _vram_peak_mb()

    _vram_reset()
    t0 = time.perf_counter()
    try:
        pred_fc = gen_full(model, tok, ids)
    except Exception as e:
        pred_fc = ""
        print(f"  FullContext gen error: {e}", flush=True)
    gen_ms_fc   = (time.perf_counter() - t0) * 1000
    vram_gen_fc = _vram_peak_mb()
    new_toks_fc = len(tok.encode(pred_fc)) if pred_fc else 0
    tps_fc      = new_toks_fc / max(gen_ms_fc / 1000, 1e-6)
    mets_fc     = compute_metrics(pred_fc, ground_truth)

    trial_data["FullContext"] = {str(B): mets_fc for B in BUDGETS}
    for B in BUDGETS:
        for m in METRIC_KEYS:
            results[source][task][ctx]["FullContext"][B][m].append(mets_fc[m])
        pred_rows.append({
            "source": source, "task": task, "ctx": ctx,
            "trial_or_sample": trial_idx, "policy": "FullContext", "budget": B,
            "ground_truth": str(ground_truth), "prediction": pred_fc,
            **mets_fc,
        })
        speed_rows.append({
            "source": source, "task": task, "ctx": ctx,
            "trial_or_sample": trial_idx, "policy": "FullContext", "budget": B,
            "sal_ms": round(sal_ms, 1), "gen_ms": round(gen_ms_fc, 1),
            "tokens_per_sec": round(tps_fc, 2),
            "vram_sal_mb": round(vram_sal, 1), "vram_gen_mb": round(vram_gen_fc, 1),
        })
        coherence_rows.append({
            "source": source, "task": task, "ctx": ctx,
            "trial_or_sample": trial_idx, "policy": "FullContext", "budget": B,
            "eviction_coherence_loss": measure_coherence_loss(model, ids, set(range(seq_len))),
        })

    for pol_name, (sig_key, pol_fn) in POLICIES.items():
        trial_data[pol_name] = {}
        for B in BUDGETS:
            try:
                if B >= seq_len:
                    pred = gen_full(model, tok, ids)
                    keep = set(range(seq_len))
                    gen_ms = gen_ms_fc
                    vram_gen = vram_gen_fc
                else:
                    keep = pol_fn(sals, B, seq_len)
                    _vram_reset()
                    t_g0 = time.perf_counter()
                    pred = gen_evict(model, tok, ids, keep)
                    gen_ms   = (time.perf_counter() - t_g0) * 1000
                    vram_gen = _vram_peak_mb()

                new_toks = len(tok.encode(pred)) if pred else 0
                tps      = new_toks / max(gen_ms / 1000, 1e-6) if B < seq_len else tps_fc
                mets     = compute_metrics(pred, ground_truth)
                coh_val  = measure_coherence_loss(model, ids, keep)

                trial_data[pol_name][str(B)] = mets
                for m in METRIC_KEYS:
                    results[source][task][ctx][pol_name][B][m].append(mets[m])

                pred_rows.append({
                    "source": source, "task": task, "ctx": ctx,
                    "trial_or_sample": trial_idx, "policy": pol_name, "budget": B,
                    "ground_truth": str(ground_truth), "prediction": pred,
                    **mets,
                })
                speed_rows.append({
                    "source": source, "task": task, "ctx": ctx,
                    "trial_or_sample": trial_idx, "policy": pol_name, "budget": B,
                    "sal_ms": round(sal_ms, 1),
                    "gen_ms": round(gen_ms if B < seq_len else gen_ms_fc, 1),
                    "tokens_per_sec": round(tps, 2),
                    "vram_sal_mb": round(vram_sal, 1),
                    "vram_gen_mb": round(vram_gen if B < seq_len else vram_gen_fc, 1),
                })
                coherence_rows.append({
                    "source": source, "task": task, "ctx": ctx,
                    "trial_or_sample": trial_idx, "policy": pol_name, "budget": B,
                    "eviction_coherence_loss": round(coh_val, 4),
                })

            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                print(f"  OOM {pol_name} B={B}", flush=True)
            except Exception as e:
                print(f"  {pol_name} B={B} error: {e}", flush=True)

    return trial_data, pred_rows, speed_rows, coherence_rows


def main() -> None:
    _init_csv(PRED_CSV_PATH,      PRED_COLS)
    _init_csv(SPEED_CSV_PATH,     SPEED_COLS)
    _init_csv(COHERENCE_CSV_PATH, COHERENCE_COLS)

    completed  = load_checkpoints()
    print(f"Resuming: {len(completed)} trials already done.", flush=True)

    model, tok = load_model()

    pol_names  = list(POLICIES) + ["FullContext"]
    results    = _make_results_skeleton(pol_names)
    _reload_results_from_checkpoints(completed, results, pol_names)

    t0_total = time.time()
    done     = len(completed)
    total_ruler = len(RULER_TASKS) * len(CTX_LENS) * N_TRIALS
    total_lb    = len(LB_TASKS) * len(CTX_LENS) * LB_SAMPLES
    total       = total_ruler + total_lb

    for task_name in RULER_TASKS:
        for ctx_len in CTX_LENS:
            print(f"\n{'='*60}", flush=True)
            print(f"[RULER] {task_name} @ ctx={ctx_len}", flush=True)
            rng_task = random.Random(ctx_len * 7 + hash(task_name) % 1000)

            for trial in range(N_TRIALS):
                key = ckpt_key("ruler", task_name, ctx_len, trial)
                if key in completed:
                    continue

                rng_t = random.Random(trial * 31337 + ctx_len + SEED)
                try:
                    if task_name == "niah_single":
                        depth = rng_task.choice(RULER_DEPTHS)
                        ids, gt = build_niah_single(rng_t, tok, ctx_len, depth)
                    elif task_name == "niah_multikey":
                        ids, gt = build_niah_multikey(rng_t, tok, ctx_len)
                    else:
                        ids, gt = build_vt(rng_t, tok, ctx_len)
                except Exception as e:
                    print(f"  build error trial={trial}: {e}", flush=True)
                    continue

                trial_data, pr, sr, cr = run_trial(
                    "ruler", task_name, ctx_len, trial,
                    ids, gt, model, tok, results, pol_names)

                save_checkpoint(key, trial_data)
                _append_csv(PRED_CSV_PATH,      PRED_COLS,      pr)
                _append_csv(SPEED_CSV_PATH,     SPEED_COLS,     sr)
                _append_csv(COHERENCE_CSV_PATH, COHERENCE_COLS, cr)
                done += 1

                if done % 5 == 0:
                    elapsed = (time.time() - t0_total) / 60
                    ki = np.mean(results["ruler"][task_name][ctx_len]["KiaOmni_\u03c38"][256]["f1"] or [0])
                    sk = np.mean(results["ruler"][task_name][ctx_len]["SnapKV_Modified"][256]["f1"] or [0])
                    print(f"  [{done}/{total}] {elapsed:.1f}min | KiaOmni_\u03c38={ki:.3f} SnapKV_Modified={sk:.3f}", flush=True)

    for task_name in LB_TASKS:
        samples = load_lb_task(task_name, LB_SAMPLES)
        for ctx_len in CTX_LENS:
            print(f"\n{'='*60}", flush=True)
            print(f"[LongBench] {task_name} @ ctx={ctx_len}", flush=True)

            for si, sample in enumerate(samples):
                key = ckpt_key("longbench", task_name, ctx_len, si)
                if key in completed:
                    continue

                try:
                    ids = build_lb_ids(sample, task_name, tok, ctx_len)
                    gt  = get_lb_answers(sample)
                except Exception as e:
                    print(f"  build error sample={si}: {e}", flush=True)
                    continue

                trial_data, pr, sr, cr = run_trial(
                    "longbench", task_name, ctx_len, si,
                    ids, gt, model, tok, results, pol_names)

                save_checkpoint(key, trial_data)
                _append_csv(PRED_CSV_PATH,      PRED_COLS,      pr)
                _append_csv(SPEED_CSV_PATH,     SPEED_COLS,     sr)
                _append_csv(COHERENCE_CSV_PATH, COHERENCE_COLS, cr)
                done += 1

                if done % 5 == 0:
                    elapsed = (time.time() - t0_total) / 60
                    print(f"  [{done}/{total}] {elapsed:.1f}min", flush=True)

    summary: dict = {}
    for src in ("ruler", "longbench"):
        summary[src] = {}
        tasks = RULER_TASKS if src == "ruler" else LB_TASKS
        for t in tasks:
            summary[src][t] = {}
            for ctx in CTX_LENS:
                summary[src][t][ctx] = {}
                for pol in pol_names:
                    summary[src][t][ctx][pol] = {}
                    for B in BUDGETS:
                        sc = results[src][t][ctx][pol][B]
                        summary[src][t][ctx][pol][B] = {
                            m: (float(np.mean(sc[m])) if sc[m] else None)
                            for m in METRIC_KEYS
                        }
                        summary[src][t][ctx][pol][B]["n"] = len(sc["f1"])

    macro: dict = {pol: {B: [] for B in BUDGETS} for pol in pol_names}
    for src in ("ruler", "longbench"):
        tasks = RULER_TASKS if src == "ruler" else LB_TASKS
        for t in tasks:
            for ctx in CTX_LENS:
                for pol in pol_names:
                    for B in BUDGETS:
                        v = summary[src][t][ctx][pol][B]["f1"]
                        if v is not None:
                            macro[pol][B].append(v)

    macro_avg = {
        pol: {B: (float(np.mean(macro[pol][B])) if macro[pol][B] else None)
              for B in BUDGETS}
        for pol in pol_names
    }

    print(f"\n{'='*60}", flush=True)
    print("[040] MACRO AVG F1 (all tasks + contexts):", flush=True)
    for pol, bdict in macro_avg.items():
        row = "  ".join(f"B={B}:{v:.3f}" for B, v in bdict.items() if v is not None)
        print(f"  {pol:<22}  {row}", flush=True)

    out = {
        "experiment": "040_amber_comparison",
        "model": MODEL_NAME,
        "policies": list(POLICIES.keys()) + ["FullContext"],
        "ruler_tasks": RULER_TASKS,
        "lb_tasks": LB_TASKS,
        "ctx_lens": CTX_LENS,
        "budgets": BUDGETS,
        "n_trials_ruler": N_TRIALS,
        "n_samples_lb": LB_SAMPLES,
        "macro_avg_f1": macro_avg,
        "per_source_task_ctx": summary,
    }
    rpath = OUT_DIR / "results.json"
    rpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults     \u2192 {rpath}", flush=True)
    print(f"Predictions \u2192 {PRED_CSV_PATH}", flush=True)
    print(f"Speed/VRAM  \u2192 {SPEED_CSV_PATH}", flush=True)
    print(f"Coherence   \u2192 {COHERENCE_CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
