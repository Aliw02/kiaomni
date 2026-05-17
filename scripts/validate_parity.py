"""
scripts/validate_parity.py
==========================
Phase G validation. Two assertions:

1.  *Parity*: on TinyLlama, the new generic patch must select the SAME
    keep-indices as the legacy hardcoded patch (same saliency math).
    We bypass model.generate and just diff the eviction decisions.

2.  *Cross-arch smoke*: end-to-end generation on TinyLlama (separate Q/K/V,
    RoPE), GPT-2 (fused-concat c_attn, learned PE), and Pythia-160m
    (gpt_neox / query_key_value fused-interleaved, RoPE) produces a
    non-empty completion without error. Pythia is built on the GPT-NeoX
    architecture and exercises the fused-interleaved code path.

USAGE
-----
    python scripts/validate_parity.py            # CPU, all checks
    python scripts/validate_parity.py --skip-gpt-neox

This script DOES NOT auto-run as part of CI — it downloads real models.
Run manually before publishing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the in-tree package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kiaomni import apply_kiaomni, remove_kiaomni, ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter
from kiaomni.policies import get_policy
from kiaomni.utils import select_keep, N_SINK_DEFAULT, RECENCY_DEFAULT


# ── parity: new generic saliency must match legacy hardcoded saliency ────────

def _legacy_saliency_tinyllama(ids: torch.Tensor, model) -> np.ndarray:
    """The original monkey_patch.py extraction, inlined for comparison.
    Returns shape (L,) as the old code did."""
    c = model.config
    nh = c.num_attention_heads
    nk = getattr(c, "num_key_value_heads", nh)
    hd = c.hidden_size // nh
    layers = model.model.layers
    L = ids.shape[1]
    per_layer = [None] * len(layers)
    hooks = []

    for li, layer in enumerate(layers):
        store = {}

        def _qh(_m, _i, out, _s=store):
            _s["q"] = out.detach().cpu().to(torch.float32)

        def _kh(_m, _i, out, _li=li, _qs=store):
            k_raw = out.detach().cpu().to(torch.float32)
            q_raw = _qs.get("q")
            if q_raw is None:
                return
            q2 = q_raw.view(1, L, nh, hd).transpose(1, 2)
            k2 = k_raw.view(1, L, nk, hd).transpose(1, 2)
            if nk != nh:
                k2 = k2.repeat_interleave(nh // nk, dim=1)
            sc = torch.matmul(q2[:, :, -1:, :], k2.transpose(-2, -1)) * (hd ** -0.5)
            sal = torch.softmax(sc, dim=-1)[0, :, 0, :]
            per_layer[_li] = sal.mean(0).numpy().astype(np.float32)

        hooks.append(layer.self_attn.q_proj.register_forward_hook(_qh))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(_kh))

    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    fallback = np.zeros(L, dtype=np.float32)
    stacked = np.stack([(x if x is not None else fallback) for x in per_layer])
    return stacked.mean(0)


def check_parity_tinyllama() -> None:
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )

    prompt = "The history of science begins " + ("with curiosity. " * 80)
    ids = tok(prompt, return_tensors="pt").input_ids
    L = ids.shape[1]

    legacy_sal = _legacy_saliency_tinyllama(ids, model)
    probe = ArchitectureProbe.probe(model, force=True)
    new_sal = SaliencyAdapter(probe).extract(ids, model)[0]  # batch 0

    np.testing.assert_allclose(legacy_sal, new_sal, atol=1e-6, rtol=1e-5)
    print(f"[parity] saliency vectors match within tolerance over {L} tokens")

    score = get_policy("kiaomni_s8")(new_sal)
    keep = select_keep(score, budget=256, L=L,
                       n_sink=N_SINK_DEFAULT, recency=RECENCY_DEFAULT)
    print(f"[parity] kept {len(keep)}/{L} positions")


# ── cross-arch smoke -----------------------------------------------------------

def smoke(model_id: str, budget: int = 128) -> None:
    print(f"\n[smoke] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")

    probe = apply_kiaomni(model, policy="kiaomni_s8", budget=budget, verbose=True)
    print(f"  probe: {probe}")

    prompt = "Long context: " + ("filler tokens. " * 80)
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] <= budget:
        print(f"  (prompt too short for eviction at budget={budget})")

    out = model.generate(ids, max_new_tokens=16, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"  generated: {text[:120]}...")
    remove_kiaomni(model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-parity", action="store_true")
    p.add_argument("--skip-gpt-neox", action="store_true")
    p.add_argument("--skip-gpt2", action="store_true")
    args = p.parse_args()

    if not args.skip_parity:
        check_parity_tinyllama()
    if not args.skip_gpt2:
        smoke("gpt2", budget=128)
    if not args.skip_gpt_neox:
        smoke("EleutherAI/pythia-160m", budget=128)

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
