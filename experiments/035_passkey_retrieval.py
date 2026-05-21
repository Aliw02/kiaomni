"""
035_passkey_retrieval.py — Passkey Retrieval Benchmark
=======================================================
Platform : Kaggle T4×2
Model    : configurable via MODEL_NAME (default: Qwen/Qwen2.5-7B-Instruct)
Task     : Hide a random passkey at varying depths in noise text.
           Measure retrieval accuracy under KV-cache compression.
Policies : All KiaOmni variants + SnapKV baselines + FullContext
Contexts : {4096, 8192, 16384}
Budgets  : {98, 128, 256, 512}
Depths   : {0.1, 0.25, 0.5, 0.75, 0.9}
N        : 20 trials per (ctx, depth, budget, policy) cell

Run:
    python experiments/035_passkey_retrieval.py

Outputs:
    experiments/results/035_passkey_results/results.json
    experiments/results/035_passkey_results/accuracy_table.csv
    experiments/results/035_passkey_results/checkpoints/
"""

import gc, json, math, os, random, re, string, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
CTX_LENS    = [4096, 8192, 16384]
BUDGETS     = [98, 128, 256, 512]
DEPTHS      = [0.1, 0.25, 0.5, 0.75, 0.9]
N_TRIALS    = 20
SEED        = 42
MAX_NEW     = 32

N_SINK      = 16
RECENCY     = 32
SNAP_POOL_K = 5
SNAP_OBS_W  = 32
SIGMA_FIXED = 8

OUT_DIR  = Path("results/035_passkey_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()
    return model, tok


FILLER = [
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
    "Quantum entanglement allows two particles to affect each other instantly over distance.",
    "Coral reefs are among the most biodiverse ecosystems on the planet.",
    "The moon's gravitational pull is responsible for Earth's ocean tides.",
]

def _rand_key(rng: random.Random, digits: int = 9) -> str:
    return "".join(rng.choices(string.digits, k=digits))

def _build_noise(rng: random.Random, target_chars: int) -> str:
    sents = FILLER[:]
    noise = ""
    while len(noise) < target_chars:
        rng.shuffle(sents)
        noise += " ".join(sents) + " "
    return noise[:target_chars]

def build_passkey(rng: random.Random, tok, target_toks: int, depth: float):
    key    = _rand_key(rng)
    needle = f"The secret passkey is: {key}. Remember this passkey."
    q_text = "\n\nWhat is the secret passkey? Respond with only the digits."
    pre    = "Read the following text carefully and remember any passkeys.\n\n"
    noise  = _build_noise(rng, target_toks * 6)
    for _ in range(25):
        split = int(len(noise) * depth)
        full  = pre + noise[:split] + "\n\n" + needle + "\n\n" + noise[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 150:
            break
        noise = noise[:int(len(noise) * 0.90)]
    return ids, key


def _split_qkv(raw: torch.Tensor, nh: int, nk: int, hd: int, L: int):
    q = raw[..., : nh * hd].reshape(1, L, nh, hd).transpose(1, 2)
    k = raw[..., nh * hd: nh * hd + nk * hd].reshape(1, L, nk, hd).transpose(1, 2)
    return q, k

def extract_saliency(ids: torch.Tensor, model) -> dict:
    c    = model.config
    nh   = c.num_attention_heads
    nk   = getattr(c, "num_key_value_heads", nh)
    hd   = c.hidden_size // nh
    L    = ids.shape[1]
    buf: dict = {}
    layer_sals: dict = {}
    hooks = []

    layers = (model.model.layers if hasattr(model, "model") else
              model.transformer.h if hasattr(model, "transformer") else [])
    if not layers:
        return {"sal_snapkv": None, "sal_mean": None, "sal_scissor": None}

    n_lay = len(layers)
    scissor_idxs = {n_lay // 4, n_lay // 2, n_lay - 1}

    def _make_layer_hook(l_idx: int, is_last: bool, fused: bool):
        def _h(m, inp, out):
            raw = out.detach().cpu().to(torch.float32)
            if fused:
                q, k = _split_qkv(raw, nh, nk, hd, L)
            else:
                k = raw.view(1, L, nk, hd).transpose(1, 2)
                q = buf.get(f"q_{l_idx}")
                if q is None:
                    return
            if nk != nh:
                k = k.repeat_interleave(nh // nk, dim=1)
            sc    = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) * (hd ** -0.5)
            sal_h = torch.softmax(sc, dim=-1)[0, :, 0, :]
            layer_sals[l_idx] = sal_h.mean(0).numpy().astype(np.float32)
            if is_last:
                buf["sal_mean"] = layer_sals[l_idx]
                buf["sal_heads"] = sal_h.numpy().astype(np.float32)
                obs_w      = min(SNAP_OBS_W, L)
                q_obs      = q[:, :, -obs_w:, :]
                sc_obs     = torch.matmul(q_obs, k.transpose(-2, -1)) * (hd ** -0.5)
                prefix_len = max(1, L - obs_w)
                attn_pre   = torch.softmax(sc_obs[..., :prefix_len], dim=-1)
                votes      = attn_pre.sum(dim=-2)
                max_v      = votes.max(dim=-1, keepdim=True).values
                pad        = max_v.expand(1, nh, obs_w)
                buf["sal_snapkv"] = torch.cat([votes, pad], dim=-1)[0].numpy().astype(np.float32)
        return _h

    for l_idx, layer in enumerate(layers):
        if l_idx not in scissor_idxs:
            continue
        is_last = (l_idx == n_lay - 1)
        if hasattr(layer.self_attn, "qkv_proj"):
            hooks.append(layer.self_attn.qkv_proj.register_forward_hook(
                _make_layer_hook(l_idx, is_last, True)))
        else:
            def _q_cap(m, inp, out, _li=l_idx):
                buf[f"q_{_li}"] = out.detach().cpu().to(torch.float32).view(1, L, nh, hd).transpose(1, 2)
            hooks.append(layer.self_attn.q_proj.register_forward_hook(_q_cap))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(
                _make_layer_hook(l_idx, is_last, False)))

    device = next(model.parameters()).device
    ids = ids.to(device)
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    sal_mean   = buf.get("sal_mean")
    sal_snapkv = buf.get("sal_snapkv")
    scissor_layers = [layer_sals[i] for i in sorted(scissor_idxs) if i in layer_sals]
    sal_scissor = np.mean(scissor_layers, axis=0).astype(np.float32) if scissor_layers else sal_mean

    if sal_snapkv is None and sal_mean is not None:
        sal_snapkv = np.tile(sal_mean, (nh, 1))
    if sal_mean is None:
        sal_mean = np.zeros(L, dtype=np.float32)
        sal_snapkv = np.zeros((nh, L), dtype=np.float32)
        sal_scissor = np.zeros(L, dtype=np.float32)
    return {"sal_snapkv": sal_snapkv, "sal_mean": sal_mean, "sal_scissor": sal_scissor}


def _protected(seq_len: int) -> set:
    sink    = set(range(min(N_SINK, seq_len)))
    recency = set(range(max(0, seq_len - RECENCY), seq_len))
    return sink | recency

def _top_free(sal_1d: np.ndarray, budget: int, seq_len: int) -> set:
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if budget >= seq_len:
        return set(range(seq_len))
    free = np.array([i for i in range(seq_len) if i not in prot])
    if len(free) == 0 or eff == 0:
        return prot
    k   = min(eff, len(free))
    top = np.argpartition(sal_1d[free], -k)[-k:]
    return prot | set(free[top].tolist())

def snapkv_keep(sals: dict, budget: int, seq_len: int) -> set:
    from scipy.ndimage import maximum_filter1d
    sal = sals["sal_snapkv"]
    if budget >= seq_len:
        return set(range(seq_len))
    prot   = _protected(seq_len)
    eff    = max(0, budget - len(prot))
    if eff <= 0:
        return prot
    if sal.ndim == 1:
        sal = sal[np.newaxis, :]
    n_heads = sal.shape[0]
    k_per_h = max(1, eff // n_heads)
    free    = np.array([i for i in range(seq_len) if i not in prot])
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

def kiaomni_keep(sals: dict, budget: int, seq_len: int, sigma: int = SIGMA_FIXED) -> set:
    from scipy.ndimage import maximum_filter1d
    sal  = sals["sal_mean"]
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if budget >= seq_len:
        return set(range(seq_len))
    if eff <= 0:
        return prot
    smoothed = maximum_filter1d(sal[:seq_len].astype(np.float32), size=(2 * sigma) + 1)
    free     = np.array([i for i in range(seq_len) if i not in prot])
    k        = min(eff, len(free))
    top      = np.argpartition(smoothed[free], -k)[-k:]
    return prot | set(free[top].tolist())

def kiaomni_gaussian_keep(sals: dict, budget: int, seq_len: int) -> set:
    from scipy.ndimage import gaussian_filter1d
    sal  = np.log1p(sals["sal_mean"])
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if budget >= seq_len:
        return set(range(seq_len))
    if eff <= 0:
        return prot
    smoothed = gaussian_filter1d(sal[:seq_len].astype(np.float32), sigma=SIGMA_FIXED)
    free     = np.array([i for i in range(seq_len) if i not in prot])
    k        = min(eff, len(free))
    top      = np.argpartition(smoothed[free], -k)[-k:]
    return prot | set(free[top].tolist())

def kiaomni_scissorhands_keep(sals: dict, budget: int, seq_len: int) -> set:
    sal  = sals["sal_scissor"]
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if budget >= seq_len:
        return set(range(seq_len))
    if eff <= 0:
        return prot
    free = np.array([i for i in range(seq_len) if i not in prot])
    k    = min(eff, len(free))
    top  = np.argpartition(sal[free], -k)[-k:]
    return prot | set(free[top].tolist())

def snapkv_modified_keep(sals: dict, budget: int, seq_len: int) -> set:
    sal       = sals["sal_mean"]
    prot_mask = np.zeros(seq_len, dtype=bool)
    prot_mask[:N_SINK] = True
    prot_mask[max(0, seq_len - RECENCY):] = True
    evict_idx = np.where(~prot_mask)[0]
    if len(evict_idx) == 0 or budget >= seq_len:
        return set(range(seq_len))
    BLOCK_SIZE   = 8
    page_ids     = evict_idx // BLOCK_SIZE
    sal_evict    = sal[evict_idx]
    unique_pages = np.unique(page_ids)
    page_scores  = np.array([sal_evict[page_ids == pg].mean() for pg in unique_pages], dtype=np.float32)
    order        = np.argsort(page_scores)
    evicted_mask = np.zeros(seq_len, dtype=bool)
    tokens_evicted = 0
    target_evict = max(0, seq_len - budget)
    for pi in order:
        if tokens_evicted >= target_evict:
            break
        pg_mask = page_ids == unique_pages[pi]
        evicted_mask[evict_idx[pg_mask]] = True
        tokens_evicted += int(pg_mask.sum())
    return set(np.where(~evicted_mask)[0].tolist())

def h2o_keep(sals: dict, budget: int, seq_len: int) -> set:
    sal  = sals["sal_mean"]
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if budget >= seq_len:
        return set(range(seq_len))
    if eff <= 0:
        return prot
    free = np.array([i for i in range(seq_len) if i not in prot])
    k    = min(eff, len(free))
    top  = np.argpartition(sal[free], -k)[-k:]
    return prot | set(free[top].tolist())

POLICIES = {
    "FullContext":          None,
    "KiaOmni_sigma8":       lambda s, B, L: kiaomni_keep(s, B, L, sigma=8),
    "KiaOmni_Gaussian":     kiaomni_gaussian_keep,
    "KiaOmni_Scissorhands": kiaomni_scissorhands_keep,
    "SnapKV_Modified":      snapkv_modified_keep,
    "H2O":                  h2o_keep,
    "RealSnapKV":           snapkv_keep,
}


@torch.no_grad()
def generate(model, tok, ids: torch.Tensor, keep: set | None = None) -> str:
    device = next(model.parameters()).device
    ids    = ids.to(device)
    if keep is not None:
        keep_t = torch.tensor(sorted(keep), device=device, dtype=torch.long)
        ids    = ids[:, keep_t]
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def score(pred: str, key: str) -> bool:
    return key in re.sub(r"\s+", "", pred)


def ckpt_key(ctx: int, depth: float, trial: int) -> str:
    return f"ctx{ctx}_d{int(depth*100):03d}_t{trial:04d}"

def load_checkpoints() -> dict:
    done: dict = {}
    for f in CKPT_DIR.glob("*.json"):
        done[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return done

def save_checkpoint(key: str, data: dict) -> None:
    (CKPT_DIR / f"{key}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    model, tok = load_model()
    completed  = load_checkpoints()
    print(f"Resuming: {len(completed)} trials done.", flush=True)

    results: dict = {
        ctx: {d: {p: {B: [] for B in BUDGETS} for p in POLICIES}
              for d in DEPTHS}
        for ctx in CTX_LENS
    }

    for key, data in completed.items():
        try:
            parts = key.split("_")
            ctx   = int(parts[0].replace("ctx", ""))
            depth = int(parts[1].replace("d", "")) / 100
            for pol in POLICIES:
                for B in BUDGETS:
                    v = data.get(pol, {}).get(str(B))
                    if v is not None:
                        results[ctx][depth][pol][B].append(v)
        except Exception:
            pass

    total_trials = len(CTX_LENS) * len(DEPTHS) * N_TRIALS
    done_count   = len(completed)

    for ctx in CTX_LENS:
        for depth in DEPTHS:
            for trial in range(N_TRIALS):
                key = ckpt_key(ctx, depth, trial)
                if key in completed:
                    continue
                rng = random.Random(SEED + trial * 997 + ctx + int(depth * 100))
                try:
                    ids, passkey = build_passkey(rng, tok, ctx, depth)
                except Exception as e:
                    print(f"  build error ctx={ctx} d={depth} t={trial}: {e}", flush=True)
                    continue

                device = next(model.parameters()).device
                ids    = ids.to(device)
                seq_len = ids.shape[1]

                try:
                    sals = extract_saliency(ids.cpu(), model)
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM saliency ctx={ctx}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue

                trial_data: dict = {}
                for pol_name, pol_fn in POLICIES.items():
                    trial_data[pol_name] = {}
                    for B in BUDGETS:
                        try:
                            keep = None if pol_fn is None else pol_fn(sals, B, seq_len)
                            pred = generate(model, tok, ids.cpu(), keep)
                            hit  = score(pred, passkey)
                            trial_data[pol_name][str(B)] = hit
                            results[ctx][depth][pol_name][B].append(hit)
                        except torch.cuda.OutOfMemoryError:
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        except Exception as e:
                            print(f"  {pol_name} B={B} error: {e}", flush=True)

                save_checkpoint(key, trial_data)
                done_count += 1
                print(f"  [{done_count}/{total_trials}] ctx={ctx} depth={depth:.2f} trial={trial} "
                      f"passkey={passkey}", flush=True)

    summary: dict = {}
    csv_rows: list = []
    for ctx in CTX_LENS:
        summary[ctx] = {}
        for depth in DEPTHS:
            summary[ctx][depth] = {}
            for pol in POLICIES:
                summary[ctx][depth][pol] = {}
                for B in BUDGETS:
                    vals = results[ctx][depth][pol][B]
                    acc  = float(np.mean(vals)) if vals else None
                    summary[ctx][depth][pol][B] = acc
                    csv_rows.append({
                        "ctx": ctx, "depth": depth, "policy": pol,
                        "budget": B, "accuracy": acc,
                        "n_trials": len(vals),
                    })

    (OUT_DIR / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    with open(OUT_DIR / "accuracy_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ctx", "depth", "policy", "budget", "accuracy", "n_trials"])
        w.writeheader()
        w.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("PASSKEY RETRIEVAL - ACCURACY SUMMARY (budget=256)")
    print("=" * 70)
    print(f"{'Policy':<25}", end="")
    for ctx in CTX_LENS:
        print(f"  ctx={ctx:>5}", end="")
    print()
    for pol in POLICIES:
        print(f"{pol:<25}", end="")
        for ctx in CTX_LENS:
            vals = [summary[ctx][d][pol][256] for d in DEPTHS if summary[ctx][d][pol][256] is not None]
            avg  = float(np.mean(vals)) if vals else float("nan")
            print(f"  {avg:>8.3f}", end="")
        print()
    print("=" * 70)
    print(f"\nResults saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
