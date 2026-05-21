"""
035_niah_heatmap.py — NIAH Full Heatmap (Context Length × Needle Depth)
=======================================================================
Platform : Kaggle T4×2
Model    : configurable via MODEL_NAME (default: Qwen/Qwen2.5-7B-Instruct)
Task     : Needle-in-a-Haystack grid evaluation.
            Rows = context lengths, Columns = needle depths.
            Each cell = accuracy over N_TRIALS trials.
Policies : FullContext, KiaOmni_sigma8, KiaOmni_Gaussian, KiaOmni_Scissorhands,
            RealSnapKV, SnapKV_Modified, H2O
Budgets  : [98, 128, 256, 512]  (heatmap per budget)
Contexts : [1024, 2048, 4096, 8192]
Depths   : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N        : 10 trials per cell

Output   : matplotlib heatmaps saved as PNG + raw CSV

Run:
    python experiments/035_niah_heatmap.py

Outputs:
    experiments/results/035_heatmap_results/heatmap_{policy}_B{budget}.png
    experiments/results/035_heatmap_results/grid_data.json
    experiments/results/035_heatmap_results/grid_data.csv
    experiments/results/035_heatmap_results/checkpoints/
"""

import gc, json, os, random, re, string
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
CTX_LENS    = [1024, 2048, 4096, 8192]
DEPTHS      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BUDGETS     = [98, 128, 256, 512]
N_TRIALS    = 10
SEED        = 42
MAX_NEW     = 32

HEATMAP_POLICIES = [
    "FullContext",
    "KiaOmni_sigma8",
    "KiaOmni_Gaussian",
    "KiaOmni_Scissorhands",
    "RealSnapKV",
    "SnapKV_Modified",
    "H2O",
]

N_SINK      = 16
RECENCY     = 32
SNAP_POOL_K = 5
SNAP_OBS_W  = 32
SIGMA_FIXED = 8

OUT_DIR  = Path("results/035_heatmap_results")
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
    "Stars orbit around a central region containing a supermassive black hole.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Quantum entanglement allows two particles to affect each other instantly over distance.",
    "Coral reefs are among the most biodiverse ecosystems on the planet.",
    "The moon's gravitational pull is responsible for Earth's ocean tides.",
]

def _rand_code(rng: random.Random, k: int = 8) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=k))

def _build_haystack(rng: random.Random, target_chars: int) -> str:
    sents = FILLER[:]
    hay   = ""
    while len(hay) < target_chars:
        rng.shuffle(sents)
        hay += " ".join(sents) + " "
    return hay[:target_chars]

def build_niah(rng: random.Random, tok, target_toks: int, depth: float):
    key    = _rand_code(rng, 8)
    needle = f"The secret passphrase is: {key}."
    q_text = "\n\nWhat is the secret passphrase? Answer with only the passphrase value."
    pre    = "Read the following text carefully.\n\n"
    hay    = _build_haystack(rng, target_toks * 6)
    for _ in range(25):
        split = int(len(hay) * depth)
        full  = pre + hay[:split] + "\n\n" + needle + "\n\n" + hay[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.90)]
    return ids, key


def _split_qkv(raw, nh, nk, hd, L):
    q = raw[..., : nh * hd].reshape(1, L, nh, hd).transpose(1, 2)
    k = raw[..., nh * hd: nh * hd + nk * hd].reshape(1, L, nk, hd).transpose(1, 2)
    return q, k

def extract_saliency(ids: torch.Tensor, model) -> dict:
    c   = model.config
    nh  = c.num_attention_heads
    nk  = getattr(c, "num_key_value_heads", nh)
    hd  = c.hidden_size // nh
    L   = ids.shape[1]
    buf: dict = {}
    layer_sals: dict = {}
    hooks = []

    layers = (model.model.layers if hasattr(model, "model") else
              model.transformer.h if hasattr(model, "transformer") else [])
    if not layers:
        dummy = np.zeros(L, dtype=np.float32)
        return {"sal_snapkv": np.tile(dummy, (nh, 1)), "sal_mean": dummy,
                "sal_scissor": dummy}

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

    sal_mean   = buf.get("sal_mean", np.zeros(L, dtype=np.float32))
    sal_snapkv = buf.get("sal_snapkv", np.tile(sal_mean, (nh, 1)))
    scissor_layers = [layer_sals[i] for i in sorted(scissor_idxs) if i in layer_sals]
    sal_scissor = np.mean(scissor_layers, axis=0).astype(np.float32) if scissor_layers else sal_mean
    return {"sal_snapkv": sal_snapkv, "sal_mean": sal_mean, "sal_scissor": sal_scissor}


def _protected(seq_len: int) -> set:
    return set(range(min(N_SINK, seq_len))) | set(range(max(0, seq_len - RECENCY), seq_len))

def snapkv_keep(sals: dict, budget: int, seq_len: int) -> set:
    from scipy.ndimage import maximum_filter1d
    sal  = sals["sal_snapkv"]
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
    sal  = sals["sal_mean"]
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

POLICY_FNS = {
    "FullContext":          None,
    "KiaOmni_sigma8":       lambda s, B, L: kiaomni_keep(s, B, L, sigma=8),
    "KiaOmni_Gaussian":     kiaomni_gaussian_keep,
    "KiaOmni_Scissorhands": kiaomni_scissorhands_keep,
    "RealSnapKV":           snapkv_keep,
    "SnapKV_Modified":      snapkv_modified_keep,
    "H2O":                  h2o_keep,
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

def score_niah(pred: str, key: str) -> bool:
    return key.upper() in pred.upper() or key.lower() in pred.lower()


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


def render_heatmap(grid: np.ndarray, ctx_lens: list, depths: list,
                   title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(len(depths) * 1.1 + 1, len(ctx_lens) * 0.9 + 1))
        im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d:.1f}" for d in depths], fontsize=9)
        ax.set_yticks(range(len(ctx_lens)))
        ax.set_yticklabels([str(c) for c in ctx_lens], fontsize=9)
        ax.set_xlabel("Needle Depth", fontsize=11)
        ax.set_ylabel("Context Length (tokens)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for r in range(len(ctx_lens)):
            for c in range(len(depths)):
                val = grid[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.8 else "white")
        plt.colorbar(im, ax=ax, label="Accuracy")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Heatmap saved: {out_path}", flush=True)
    except ImportError:
        print("  matplotlib not installed - skipping heatmap render", flush=True)


def main() -> None:
    model, tok = load_model()
    completed  = load_checkpoints()
    print(f"Resuming: {len(completed)} trials done.", flush=True)

    results: dict = {
        pol: {B: {ctx: {d: [] for d in DEPTHS} for ctx in CTX_LENS} for B in BUDGETS}
        for pol in HEATMAP_POLICIES
    }

    for key, data in completed.items():
        try:
            parts = key.split("_")
            ctx   = int(parts[0].replace("ctx", ""))
            depth = int(parts[1].replace("d", "")) / 100
            for pol in HEATMAP_POLICIES:
                for B in BUDGETS:
                    v = data.get(pol, {}).get(str(B))
                    if v is not None:
                        results[pol][B][ctx][depth].append(v)
        except Exception:
            pass

    total = len(CTX_LENS) * len(DEPTHS) * N_TRIALS
    done_count = len(completed)

    for ctx in CTX_LENS:
        for depth in DEPTHS:
            for trial in range(N_TRIALS):
                key = ckpt_key(ctx, depth, trial)
                if key in completed:
                    continue
                rng = random.Random(SEED + trial * 997 + ctx + int(depth * 100))
                try:
                    ids, needle_key = build_niah(rng, tok, ctx, depth)
                except Exception as e:
                    print(f"  build error ctx={ctx} d={depth} t={trial}: {e}", flush=True)
                    continue

                device  = next(model.parameters()).device
                seq_len = ids.shape[1]

                try:
                    sals = extract_saliency(ids, model)
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM saliency ctx={ctx}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue

                trial_data: dict = {}
                for pol_name in HEATMAP_POLICIES:
                    pol_fn = POLICY_FNS[pol_name]
                    trial_data[pol_name] = {}
                    for B in BUDGETS:
                        try:
                            keep = None if pol_fn is None else pol_fn(sals, B, seq_len)
                            pred = generate(model, tok, ids, keep)
                            hit  = score_niah(pred, needle_key)
                            trial_data[pol_name][str(B)] = hit
                            results[pol_name][B][ctx][depth].append(hit)
                        except torch.cuda.OutOfMemoryError:
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        except Exception as e:
                            print(f"  {pol_name} B={B} error: {e}", flush=True)

                save_checkpoint(key, trial_data)
                done_count += 1
                print(f"  [{done_count}/{total}] ctx={ctx} d={depth:.1f} t={trial} "
                      f"key={needle_key}", flush=True)

    import csv
    summary: dict = {}
    csv_rows: list = []

    for pol in HEATMAP_POLICIES:
        summary[pol] = {}
        for B in BUDGETS:
            summary[pol][B] = {}
            grid = np.full((len(CTX_LENS), len(DEPTHS)), np.nan)
            for ri, ctx in enumerate(CTX_LENS):
                for ci, depth in enumerate(DEPTHS):
                    vals = results[pol][B][ctx][depth]
                    acc  = float(np.mean(vals)) if vals else float("nan")
                    grid[ri, ci] = acc
                    summary[pol][B][ctx] = summary[pol][B].get(ctx, {})
                    summary[pol][B][ctx][depth] = acc
                    csv_rows.append({"policy": pol, "budget": B, "ctx": ctx,
                                     "depth": depth, "accuracy": acc,
                                     "n_trials": len(vals)})
            render_heatmap(
                grid, CTX_LENS, DEPTHS,
                title=f"NIAH - {pol} | Budget={B}",
                out_path=OUT_DIR / f"heatmap_{pol}_B{B}.png",
            )

    (OUT_DIR / "grid_data.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(OUT_DIR / "grid_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "budget", "ctx", "depth",
                                           "accuracy", "n_trials"])
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\nAll heatmaps and data saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
