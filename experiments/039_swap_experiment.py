"""
039_swap_experiment.py — Signal Swap Experiment
================================================
Purpose: Isolate selector contribution from signal contribution.
         Tests KiaOmni_σ8 and RealSnapKV each under their own signal
         AND under the other's signal.

4 conditions:
  KiaOmni_σ8   + sal_mean     (natural)
  KiaOmni_σ8   + sal_snapkv   (swapped)
  RealSnapKV   + sal_snapkv   (natural)
  RealSnapKV   + sal_mean     (swapped)

Model   : Qwen/Qwen2.5-7B-Instruct (4-bit NF4)
Context : 4096
Budgets : [98, 128]  (2.4% and 3.1% compression)
Tasks   : niah_single, niah_multikey, vt
Trials  : 15 per cell

Run:
    python experiments/039_swap_experiment.py

Outputs:
    experiments/results/039_swap_results/results.json
    experiments/results/039_swap_results/predictions.csv
    experiments/results/039_swap_results/checkpoints/
"""

import csv, gc, json, os, random, re, string, collections, time
import urllib.request, zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
CTX_LEN     = 4096
BUDGETS     = [98, 128]
N_TRIALS    = 15
RULER_TASKS = ["niah_single", "niah_multikey", "vt"]
SEED        = 42
MAX_NEW     = 96

N_SINK      = 16
RECENCY     = 32
BLOCK_SIZE  = 16
SIGMA_FIXED = 8
SNAP_POOL_K = 5
SNAP_OBS_W  = 32
N_KEYS      = 4
CHAIN_LEN   = 5

OUT_DIR  = Path("results/039_swap_results")
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

PRED_CSV_PATH = OUT_DIR / "predictions.csv"
PRED_COLS = ["task", "trial", "budget", "policy", "signal", "condition",
             "ground_truth", "prediction", "f1", "em", "rouge_l", "contains"]
METRIC_KEYS = ["f1", "em", "rouge_l", "contains"]

CONDITIONS = {
    "KiaOmni_NaturalSignal":  ("kiaomni", "sal_mean"),
    "KiaOmni_SwappedSignal":  ("kiaomni", "sal_snapkv"),
    "SnapKV_NaturalSignal":   ("snapkv",  "sal_snapkv"),
    "SnapKV_SwappedSignal":   ("snapkv",  "sal_mean"),
}


def load_model():
    print(f"Loading {MODEL_NAME} (4-bit NF4)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            print(f"  attn_implementation={attn_impl}", flush=True)
            break
        except Exception as e:
            print(f"  {attn_impl} failed: {e}", flush=True)
    model.eval()
    return model, tok


def extract_all_saliency(ids: torch.Tensor, model) -> dict:
    L_seq    = ids.shape[1]
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
                sc2     = torch.matmul(q2[:, :, -1:, :], k2.transpose(-2, -1)) * (_hd2 ** -0.5)
                sal_heads = torch.softmax(sc2, dim=-1)[0, :, 0, :]
                sal_mean_l = sal_heads.mean(0).numpy()
                sal_per_layer_list[layer_idx] = sal_mean_l.astype(np.float32)
                if is_last:
                    last_k_buf["sal_heads"] = sal_heads.numpy()
                    obs_w      = min(SNAP_OBS_W, _L)
                    q_obs      = q2[:, :, -obs_w:, :]
                    sc_obs     = torch.matmul(q_obs, k2.transpose(-2, -1)) * (_hd2 ** -0.5)
                    prefix_len = max(1, _L - obs_w)
                    attn_pfx   = torch.softmax(sc_obs[..., :prefix_len], dim=-1)
                    votes      = attn_pfx.sum(dim=-2)
                    max_v      = votes.max(dim=-1, keepdim=True).values
                    pad        = max_v.expand(1, _nh2, obs_w)
                    sal_snapkv_h = torch.cat([votes, pad], dim=-1)[0]
                    last_k_buf["sal_snapkv"] = sal_snapkv_h.numpy()
                    del q_obs, sc_obs, attn_pfx, votes, max_v, pad, sal_snapkv_h
                del q2, k2, sc2, sal_heads
            return _h

        is_last       = (l_idx == n_layers - 1)
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
        sal_mean     = _last_layer_saliency(ids, model)
        sal_per_head = np.tile(sal_mean, (nh, 1))
    else:
        sal_per_head = sal_heads_last.astype(np.float32)
        sal_mean     = sal_per_head.mean(0)

    sal_snapkv = (
        sal_snapkv_last.astype(np.float32)
        if sal_snapkv_last is not None
        else sal_per_head
    )

    del last_k_buf
    return {"sal_mean": sal_mean, "sal_snapkv": sal_snapkv}


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


def select_kiaomni(sal: np.ndarray, budget: int, seq_len: int) -> set:
    prot  = _protected(seq_len)
    F     = _boxcar(np.log1p(sal if sal.ndim == 1 else sal.mean(0)), SIGMA_FIXED)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot


def select_snapkv(sal: np.ndarray, budget: int, seq_len: int) -> set:
    from scipy.ndimage import maximum_filter1d
    if budget >= seq_len:
        return set(range(seq_len))
    prot     = _protected(seq_len)
    eff      = max(0, budget - len(prot))
    if eff <= 0:
        return prot
    if sal.ndim == 1:
        sal = sal[np.newaxis, :]
    n_heads  = sal.shape[0]
    k_per_h  = max(1, eff // n_heads)
    free     = np.array([i for i in range(seq_len) if i not in prot])
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


SELECTORS = {"kiaomni": select_kiaomni, "snapkv": select_snapkv}


@torch.no_grad()
def gen_evict(model, tok, ids: torch.Tensor, keep: set) -> str:
    keep_t = torch.tensor(sorted(keep), device=ids.device, dtype=torch.long)
    p      = ids[:, keep_t]
    out    = model.generate(p, attention_mask=torch.ones_like(p),
                            max_new_tokens=MAX_NEW, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, p.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def gen_full(model, tok, ids: torch.Tensor) -> str:
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=MAX_NEW, do_sample=False,
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
    return 0.0 if common == 0 else 2 * common / (len(p) + len(t))


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
    answers = ground_truth if isinstance(ground_truth, list) else [str(ground_truth)]
    best = {"f1": 0.0, "em": 0.0, "rouge_l": 0.0, "contains": 0.0}
    pn   = _norm(pred)
    for a in answers:
        an = _norm(str(a))
        best["f1"]       = max(best["f1"],      _token_f1(pred, a))
        best["em"]       = max(best["em"],       float(pn == an))
        best["rouge_l"]  = max(best["rouge_l"],  _rouge_l(pred, a))
        best["contains"] = max(best["contains"], float(an in pn))
    return best


FILLER_SENTENCES = [
    "The history of astronomy dates back thousands of years to ancient civilizations.",
    "Ocean currents are driven by wind patterns and differences in water temperature.",
    "The periodic table organizes elements by atomic number and chemical properties.",
    "Languages evolve over time, borrowing words and structures from neighboring tongues.",
    "Tectonic plates move at roughly the same speed as human fingernails grow.",
    "The Amazon rainforest produces about 20 percent of the world\'s oxygen supply.",
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
    "The moon\'s gravitational pull is responsible for Earth\'s ocean tides.",
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


def build_niah_single(rng, tok, target_toks: int, depth: float):
    key    = _rand_code(rng, 8)
    needle = f"The secret passphrase is: {key}."
    q_text = "\n\nWhat is the secret passphrase? Answer with only the passphrase value."
    pre    = "Read the following text carefully.\n\n"
    hay    = _build_haystack(rng, target_toks * 5)
    for _ in range(20):
        split = int(len(hay) * depth)
        full  = pre + hay[:split] + "\n\n" + needle + "\n\n" + hay[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] >= target_toks:
            return ids[:, :target_toks], key
    return tok(full, return_tensors="pt").input_ids[:, :target_toks], key


def build_niah_multikey(rng, tok, target_toks: int, n_keys: int = N_KEYS):
    keys    = [_rand_code(rng, 8) for _ in range(n_keys)]
    needles = [f"Secret key {i+1} is: {k}." for i, k in enumerate(keys)]
    q_text  = "\n\nList all secret keys in order, comma-separated."
    pre     = "Read the following text carefully.\n\n"
    hay     = _build_haystack(rng, target_toks * 6)
    chunk   = max(1, len(hay) // (n_keys + 1))
    full    = pre
    for i, nd in enumerate(needles):
        full += hay[i * chunk:(i + 1) * chunk] + "\n\n" + nd + "\n\n"
    full += hay[n_keys * chunk:] + q_text
    ids = tok(full, return_tensors="pt").input_ids
    return ids[:, :target_toks], ", ".join(keys)


def build_vt(rng, tok, target_toks: int, chain_len: int = CHAIN_LEN):
    names  = [_rand_code(rng, 6) for _ in range(chain_len + 1)]
    facts  = [f"{names[i]} passed the token to {names[i+1]}." for i in range(chain_len)]
    q_text = f"\n\nWho finally received the token? Answer with only the name."
    pre    = "Track the token as it passes between people.\n\n"
    hay    = _build_haystack(rng, target_toks * 5)
    chunk  = max(1, len(hay) // (chain_len + 1))
    full   = pre
    for i, fact in enumerate(facts):
        full += hay[i * chunk:(i + 1) * chunk] + "\n\n" + fact + "\n\n"
    full += hay[chain_len * chunk:] + q_text
    ids = tok(full, return_tensors="pt").input_ids
    return ids[:, :target_toks], names[-1]


def build_task(task: str, rng, tok, ctx: int):
    if task == "niah_single":
        return build_niah_single(rng, tok, ctx, rng.uniform(0.1, 0.9))
    if task == "niah_multikey":
        return build_niah_multikey(rng, tok, ctx)
    if task == "vt":
        return build_vt(rng, tok, ctx)
    raise ValueError(f"Unknown task: {task}")


def _ckpt_key(task: str, trial: int) -> str:
    return f"{task}_t{trial:03d}"


def load_completed() -> dict:
    ckpt_file = CKPT_DIR / "completed.json"
    if ckpt_file.exists():
        return json.loads(ckpt_file.read_text())
    return {}


def save_completed(completed: dict) -> None:
    (CKPT_DIR / "completed.json").write_text(json.dumps(completed, indent=2))


def main():
    random.seed(SEED)
    rng = random.Random(SEED)

    model, tok = load_model()
    device     = next(model.parameters()).device

    completed  = load_completed()

    results: dict = {
        task: {
            B: {cond: {m: [] for m in METRIC_KEYS} for cond in CONDITIONS}
            for B in BUDGETS
        }
        for task in RULER_TASKS
    }

    for key, data in completed.items():
        try:
            task, t_str = key.rsplit("_t", 1)
            if task not in results:
                continue
            for B in BUDGETS:
                for cond in CONDITIONS:
                    for m in METRIC_KEYS:
                        v = data.get(str(B), {}).get(cond, {}).get(m)
                        if v is not None:
                            results[task][B][cond][m].append(v)
        except Exception:
            pass

    if not PRED_CSV_PATH.exists():
        with open(PRED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PRED_COLS).writeheader()

    first_cond  = list(CONDITIONS)[0]
    first_B     = BUDGETS[0]
    total_done  = sum(len(v[first_B][first_cond]["f1"]) for v in results.values())
    total_cells = len(RULER_TASKS) * N_TRIALS
    print(f"Starting swap experiment. {total_done}/{total_cells} trial-tasks already done.",
          flush=True)

    for task in RULER_TASKS:
        for trial in range(N_TRIALS):
            key = _ckpt_key(task, trial)
            if key in completed:
                print(f"  [skip] {task} trial={trial}", flush=True)
                continue

            print(f"\n[{task}] trial={trial+1}/{N_TRIALS}", flush=True)

            ids, ground_truth = build_task(task, rng, tok, CTX_LEN)
            ids = ids.to(device)
            seq_len = ids.shape[1]

            try:
                sals = extract_all_saliency(ids, model)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on saliency — skipping trial", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            trial_data: dict = {}
            pred_rows:  list = []

            for B in BUDGETS:
                trial_data[str(B)] = {}
                for cond_name, (selector_name, signal_key) in CONDITIONS.items():
                    selector = SELECTORS[selector_name]
                    signal   = sals[signal_key]

                    try:
                        keep = selector(signal, B, seq_len)
                        pred = gen_evict(model, tok, ids, keep)
                        mets = compute_metrics(pred, ground_truth)
                    except Exception as e:
                        print(f"  Error {cond_name} B={B}: {e}", flush=True)
                        pred = ""
                        mets = {m: 0.0 for m in METRIC_KEYS}

                    trial_data[str(B)][cond_name] = mets
                    for m in METRIC_KEYS:
                        results[task][B][cond_name][m].append(mets[m])

                    pred_rows.append({
                        "task": task, "trial": trial, "budget": B,
                        "policy": selector_name, "signal": signal_key,
                        "condition": cond_name,
                        "ground_truth": str(ground_truth), "prediction": pred,
                        **mets,
                    })
                    print(f"    B={B:<4} {cond_name:35s}  F1={mets['f1']:.3f}  EM={mets['em']:.3f}",
                          flush=True)

            with open(PRED_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=PRED_COLS)
                for row in pred_rows:
                    w.writerow(row)

            completed[key] = trial_data
            save_completed(completed)

            del ids, sals
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary: dict = {}
    for task in RULER_TASKS:
        summary[task] = {}
        for B in BUDGETS:
            summary[task][str(B)] = {}
            for cond in CONDITIONS:
                cond_data = results[task][B][cond]
                summary[task][str(B)][cond] = {
                    m: {
                        "mean": float(np.mean(cond_data[m])) if cond_data[m] else 0.0,
                        "std":  float(np.std(cond_data[m]))  if cond_data[m] else 0.0,
                        "n":    len(cond_data[m]),
                    }
                    for m in METRIC_KEYS
                }

    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("SWAP EXPERIMENT RESULTS — F1 Score (mean ± std)")
    print("=" * 80)
    for B in BUDGETS:
        print(f"\n--- Budget = {B} ({B/CTX_LEN*100:.1f}% of context) ---")
        print(f"{'Task':<18} {'Condition':<35} {'F1':>6}  {'±':>5}")
        print("-" * 68)
        for task in RULER_TASKS:
            for cond in CONDITIONS:
                d = summary[task][str(B)][cond]["f1"]
                print(f"{task:<18} {cond:<35} {d['mean']:>6.3f}  {d['std']:>5.3f}")
            print()

    print("SWAP DELTA ANALYSIS")
    print("=" * 80)
    for B in BUDGETS:
        print(f"\n=== Budget = {B} ===")
        for task in RULER_TASKS:
            kia_nat  = summary[task][str(B)]["KiaOmni_NaturalSignal"]["f1"]["mean"]
            kia_swap = summary[task][str(B)]["KiaOmni_SwappedSignal"]["f1"]["mean"]
            snp_nat  = summary[task][str(B)]["SnapKV_NaturalSignal"]["f1"]["mean"]
            snp_swap = summary[task][str(B)]["SnapKV_SwappedSignal"]["f1"]["mean"]

            kia_delta = kia_nat - kia_swap
            snp_delta = snp_nat - snp_swap
            kia_wins_nat  = kia_nat  > snp_nat
            kia_wins_swap = kia_swap > snp_swap

            verdict = (
                "✅ SELECTOR WINS — Kia beats SnapKV on BOTH signals"
                if kia_wins_nat and kia_wins_swap
                else "⚠️  SIGNAL-DEPENDENT — Kia only wins on its natural signal"
                if kia_wins_nat and not kia_wins_swap
                else "❌ SnapKV dominates on both signals"
                if not kia_wins_nat and not kia_wins_swap
                else "🔀 Mixed result"
            )

            print(f"\n  {task}:")
            print(f"    KiaOmni  natural→swap delta : {kia_delta:+.3f}  "
                  f"(nat={kia_nat:.3f} swap={kia_swap:.3f})")
            print(f"    SnapKV   natural→swap delta : {snp_delta:+.3f}  "
                  f"(nat={snp_nat:.3f} swap={snp_swap:.3f})")
            print(f"    Verdict : {verdict}")

    print("\nResults saved to:", OUT_DIR / "results.json")
    print("Predictions  at:", PRED_CSV_PATH)


if __name__ == "__main__":
    main()
