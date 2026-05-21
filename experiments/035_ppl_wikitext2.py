"""
035_ppl_wikitext2.py — Perplexity on WikiText-2 Under KV-Cache Compression
===========================================================================
Platform : Kaggle T4×2
Model    : configurable via MODEL_NAME (default: Qwen/Qwen2.5-7B-Instruct)
Task     : Measure cross-entropy loss (PPL) on WikiText-2 test split
           with each KV-cache eviction policy active.
Method   : Sliding window evaluation — each chunk is a fixed context window.
Budgets  : {98, 128, 256, 512, FullContext}
Chunks   : 50 non-overlapping chunks of CTX_LEN tokens

Run:
    python experiments/035_ppl_wikitext2.py

Outputs:
    experiments/results/035_ppl_results/ppl_table.json
    experiments/results/035_ppl_results/ppl_table.csv
"""

import gc, json, math, os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CTX_LEN    = 4096
N_CHUNKS   = 50
BUDGETS    = [98, 128, 256, 512]
STRIDE     = CTX_LEN

N_SINK      = 16
RECENCY     = 32
SNAP_POOL_K = 5
SNAP_OBS_W  = 32
SIGMA_FIXED = 8

OUT_DIR = Path("results/035_ppl_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_wikitext2_tokens(tok, n_chunks: int, ctx_len: int) -> list[torch.Tensor]:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        import urllib.request
        url  = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
        text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")

    all_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total   = all_ids.shape[0]
    chunks  = []
    for start in range(0, total - ctx_len, STRIDE):
        chunks.append(all_ids[start: start + ctx_len].unsqueeze(0))
        if len(chunks) >= n_chunks:
            break
    print(f"WikiText-2: {total:,} tokens -> {len(chunks)} chunks of {ctx_len}", flush=True)
    return chunks


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
        return {"sal_snapkv": np.tile(dummy, (nh, 1)), "sal_mean": dummy, "sal_scissor": dummy}

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

POLICIES = {
    "FullContext":          None,
    "RealSnapKV":           snapkv_keep,
    "SnapKV_Modified":      snapkv_modified_keep,
    "H2O":                  h2o_keep,
    "KiaOmni_sigma8":       lambda s, B, L: kiaomni_keep(s, B, L, sigma=8),
    "KiaOmni_Gaussian":     kiaomni_gaussian_keep,
    "KiaOmni_Scissorhands": kiaomni_scissorhands_keep,
}


@torch.no_grad()
def compute_ppl(model, ids: torch.Tensor, keep: set | None = None) -> float:
    device = next(model.parameters()).device
    ids    = ids.to(device)
    if keep is not None:
        keep_t = torch.tensor(sorted(keep), device=device, dtype=torch.long)
        ids    = ids[:, keep_t]
    if ids.shape[1] < 2:
        return float("nan")
    try:
        out = model(ids, labels=ids)
        return float(torch.exp(out.loss).item())
    except Exception:
        return float("nan")


def main() -> None:
    model, tok = load_model()
    chunks     = load_wikitext2_tokens(tok, N_CHUNKS, CTX_LEN)

    ppl_scores: dict = {pol: {B: [] for B in BUDGETS + ["full"]} for pol in POLICIES}

    for ci, chunk in enumerate(chunks):
        seq_len = chunk.shape[1]
        print(f"\n[Chunk {ci+1}/{len(chunks)}] seq_len={seq_len}", flush=True)

        try:
            sals = extract_saliency(chunk, model)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM saliency chunk={ci}", flush=True)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue

        for pol_name, pol_fn in POLICIES.items():
            if pol_name == "FullContext":
                ppl = compute_ppl(model, chunk, keep=None)
                ppl_scores[pol_name]["full"].append(ppl)
                for B in BUDGETS:
                    ppl_scores[pol_name][B].append(ppl)
                print(f"  FullContext  PPL={ppl:.3f}", flush=True)
                continue

            for B in BUDGETS:
                try:
                    keep = pol_fn(sals, B, seq_len)
                    ppl  = compute_ppl(model, chunk, keep=keep)
                    ppl_scores[pol_name][B].append(ppl)
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    print(f"  {pol_name} B={B} error: {e}", flush=True)

            mean_b = {B: float(np.mean(ppl_scores[pol_name][B])) for B in BUDGETS
                      if ppl_scores[pol_name][B]}
            print(f"  {pol_name:<25} " +
                  "  ".join(f"B={B}:{v:.3f}" for B, v in mean_b.items()), flush=True)

    summary: dict = {}
    csv_rows: list = []
    for pol in POLICIES:
        summary[pol] = {}
        for B in BUDGETS:
            vals = ppl_scores[pol][B]
            mean_ppl = float(np.mean(vals)) if vals else None
            summary[pol][B] = mean_ppl
            csv_rows.append({"policy": pol, "budget": B,
                              "ppl_mean": mean_ppl, "n_chunks": len(vals)})
        full_vals = ppl_scores[pol].get("full", [])
        summary[pol]["full"] = float(np.mean(full_vals)) if full_vals else None

    (OUT_DIR / "ppl_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    with open(OUT_DIR / "ppl_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "budget", "ppl_mean", "n_chunks"])
        w.writeheader()
        w.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("PERPLEXITY ON WikiText-2 (lower = better)")
    print("=" * 70)
    print(f"{'Policy':<25}", end="")
    for B in BUDGETS:
        print(f"  B={B:>4}", end="")
    print("  Full")
    for pol in POLICIES:
        print(f"{pol:<25}", end="")
        for B in BUDGETS:
            v = summary[pol].get(B)
            print(f"  {v:>6.2f}" if v else "     N/A", end="")
        v = summary[pol].get("full")
        print(f"  {v:>6.2f}" if v else "     N/A")
    print("=" * 70)
    print(f"\nResults saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
