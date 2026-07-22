"""
diag_peak.py — measure peak VRAM at every stage of a kiaomni call.

Runs the same prompt against fullcontext, kiaomni_s8, kiaomni_gaussian
and reports memory at three checkpoints:
  - pre : before the call
  - peak: max_memory_allocated() since last reset (the actual spike)
  - post: memory_reserved() and memory_allocated() after the call

Output is printed and also saved to /tmp/peak_diag.txt inside the container
(modal run shows stdout; the file is for later inspection if needed).
"""
from __future__ import annotations

import os
import time
import json

import modal

# Reuse the same image as modal_app.py so we don't pay a rebuild.
APP_NAME = "kiaomni-chat"
GPU = "A100-40GB"


def _build_image():
    from pathlib import Path
    pkgs = (
        "torch==2.6.0",
        "transformers==4.46.3",
        "accelerate>=0.26.0",
        "scipy",
    )
    img = (
        modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11")
        .apt_install("git")
        .pip_install(*pkgs)
        .pip_install("kiaomni[gaussian] @ git+https://github.com/Aliw02/kiaomni.git")
        .add_local_dir(Path(__file__).parent / "kiaomni_chat", "/root/kiaomni_chat", copy=True)
        .env({"PYTHONPATH": "/root:/root/kiaomni_chat"})
    )
    return img


app = modal.App("kiaomni-peak-diag", image=_build_image())


@app.function(gpu=GPU, timeout=900, scaledown_window=120)
def measure_peak():
    import sys
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from kiaomni import apply_kiaomni, remove_kiaomni

    MODEL_ID = os.environ.get("KIAOMNI_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    PROMPT_TOKENS_TARGET = 2229
    BUDGET = 512
    MAX_NEW = 32  # tiny — we only care about the prefill spike, not decode

    out = {}

    def _mb(b):
        return b / 2**20

    def _snap(stage: str, reset_peak: bool = False):
        if reset_peak and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        return {
            "stage": stage,
            "alloc_mb": _mb(torch.cuda.memory_allocated()),
            "resrv_mb": _mb(torch.cuda.memory_reserved()),
            "max_alloc_mb": _mb(torch.cuda.max_memory_allocated()),
        }

    # ── Load model ─────────────────────────────────────────────────────
    print(f"[load] downloading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[load] tokenizer ok ({time.perf_counter() - t0:.1f}s), loading weights ...", flush=True)
    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", attn_implementation="sdpa", torch_dtype=torch.float16,
    )
    model.eval()
    print(f"[load] model loaded in {time.perf_counter() - t1:.1f}s (total {time.perf_counter() - t0:.1f}s)", flush=True)
    out["after_load"] = _snap("after_load", reset_peak=True)

    # ── Build a 2229-token input (use the Starship article as a stand-in) ─
    article_path = "/root/kiaomni_chat/static/sample_data/article.txt"
    if os.path.exists(article_path):
        with open(article_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 500
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's questions about the document."},
        {"role": "user", "content": f"Here is a document:\n\n{text}\n\nQ1: When did IFT-3 launch?"},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    L = ids.shape[1]
    print(f"[input] L = {L} tokens (target {PROMPT_TOKENS_TARGET})", flush=True)
    if L < PROMPT_TOKENS_TARGET:
        # pad with extra text to reach the target
        pad = tok.encode(" The " * 200, return_tensors="pt")[0].to("cuda")
        ids = torch.cat([ids, pad.repeat((PROMPT_TOKENS_TARGET - L) // pad.shape[0] + 1)[:PROMPT_TOKENS_TARGET - L].unsqueeze(0)], dim=1)
        L = ids.shape[1]
        print(f"[input] padded to L = {L} tokens", flush=True)
    out["L"] = L
    out["after_input"] = _snap("after_input")

    # ── Run each policy ────────────────────────────────────────────────
    results = {}
    for policy, budget in [
        ("fullcontext", 0),
        ("kiaomni_s8", BUDGET),
        ("kiaomni_gaussian", BUDGET),
    ]:
        print(f"\n[{policy}] swap policy", flush=True)
        if policy == "fullcontext":
            remove_kiaomni(model)
        else:
            apply_kiaomni(model, policy=policy, budget=budget)
        torch.cuda.synchronize()

        # Fresh peak tracker for this call
        before = _snap(f"{policy}_before", reset_peak=True)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0

        after = _snap(f"{policy}_after")

        n_new = out_ids.shape[1] - L
        print(f"[{policy}] L_in={L} n_new={n_new} wall={wall:.0f}ms", flush=True)
        print(f"  before: alloc={before['alloc_mb']:.0f} resrv={before['resrv_mb']:.0f}", flush=True)
        print(f"  after : alloc={after['alloc_mb']:.0f} resrv={after['resrv_mb']:.0f} peak={after['max_alloc_mb']:.0f}", flush=True)
        print(f"  delta_resrv = {after['resrv_mb'] - before['resrv_mb']:+.0f} MB", flush=True)
        print(f"  peak_spike  = {after['max_alloc_mb'] - before['alloc_mb']:+.0f} MB (above pre-call alloc)", flush=True)

        results[policy] = {
            "before": before,
            "after": after,
            "n_new": n_new,
            "wall_ms": wall,
        }

    out["per_policy"] = results

    # ── Direct measurement: what does the saliency pass actually use? ─
    print(f"\n[direct] measure kiaomni saliency pass without generation", flush=True)
    from kiaomni.adapters.saliency import SaliencyAdapter
    from kiaomni.adapters.probe import ArchitectureProbe
    apply_kiaomni(model, policy="kiaomni_gaussian", budget=BUDGET)
    probe = ArchitectureProbe.probe(model)
    sal = SaliencyAdapter(probe)
    before = _snap("saliency_before", reset_peak=True)
    t0 = time.perf_counter()
    arr = sal.extract(ids, model)  # this is the offending forward pass
    torch.cuda.synchronize()
    sal_wall = (time.perf_counter() - t0) * 1000.0
    after = _snap("saliency_after")
    print(f"[direct] saliency pass wall={sal_wall:.0f}ms", flush=True)
    print(f"  before: alloc={before['alloc_mb']:.0f} resrv={before['resrv_mb']:.0f}", flush=True)
    print(f"  after : alloc={after['alloc_mb']:.0f} resrv={after['resrv_mb']:.0f} peak={after['max_alloc_mb']:.0f}", flush=True)
    print(f"  peak_spike = {after['max_alloc_mb'] - before['alloc_mb']:+.0f} MB", flush=True)
    print(f"  arr shape = {arr.shape}", flush=True)
    out["saliency_only"] = {"before": before, "after": after, "wall_ms": sal_wall}

    # ── Final ──────────────────────────────────────────────────────────
    out["model_weights_mb"] = _mb(sum(p.numel() * p.element_size() for p in model.parameters()))
    print(f"\n[final] model weights = {out['model_weights_mb']:.0f} MB", flush=True)

    print("\n=== JSON ===")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            measure_peak.remote()
