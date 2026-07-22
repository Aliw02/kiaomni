#!/usr/bin/env python3
"""
kvpress SnapKV parity test — the decisive experiment for the "RealSnapKV is broken"
review objection.

WHY THIS EXISTS
---------------
The repo's `snapkv_real_keep` (033_full_comparison.py:319) runs SnapKV's *scoring*
inside a SHARED, page-granular KV index where the TOTAL budget is split across heads
(`k_per_h = budget // n_heads`, line 335). Official SnapKV instead keeps `budget`
tokens **per head** in a per-head cache. That difference (not a coding typo, but a
budget-semantics choice) starves the baseline at small budgets and on needle tasks
(passkey ~0.001). We cannot claim "KiaOmni > SnapKV" from that handicapped baseline.

This script runs OFFICIAL SnapKV (NVIDIA `kvpress.SnapKVPress`, per-head cache) on the
same models/tasks/budgets so we get an apples-to-apples number against the paper's
KiaOmni results. Three possible outcomes (all acceptable — report whatever the data says):

  1. kvpress SnapKV ~80-90% of FullContext  -> our shared-budget RealSnapKV is
     understated; relabel it and re-run the comparison fairly.
  2. kvpress SnapKV also scores low here     -> low number is regime-specific, not a
     bug; rebut the objection with evidence.
  3. KiaOmni still wins vs kvpress SnapKV     -> genuinely strong result.

STATUS
------
UNTESTED SCAFFOLD — written without a GPU, so NOT executed. Trust its numbers only
after (a) confirming the kvpress API version matches your install and (b) hand-checking
one task. Do not paste its output into the paper until it runs clean end-to-end.

REQUIREMENTS
------------
    pip install kvpress datasets transformers accelerate
    # kvpress: https://github.com/NVIDIA/kvpress  (tested API: >=0.2)

USAGE
-----
    python scripts/kvpress_snapkv_parity.py --model qwen --budgets 96 128 256 512
    python scripts/kvpress_snapkv_parity.py --model mistral --max-samples 40
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable

# --- Model registry (matches the paper's four-architecture suite) -------------
MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "falcon3": "tiiuae/Falcon3-7B-Instruct",
    "biomistral": "BioMistral/BioMistral-7B",
}

# LongBench tasks used in Experiment 033 (the 8-task subset).
# Loaded from THUDM/LongBench. Keep in sync with 033.
LONGBENCH_TASKS: tuple[str, ...] = (
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "triviaqa",
)


# --- Metrics: mirror the paper's primary signal (substring "contains") --------
def normalize(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra-space (SQuAD-style)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_score(pred: str, golds: Iterable[str]) -> int:
    """1 if any normalized gold answer is a substring of the normalized prediction.

    Reproduces the paper's `contains` column — the load-bearing CORRECT% metric
    (see project_hallucination_metric).
    """
    np_ = normalize(pred)
    return int(any(normalize(g) in np_ for g in golds if g))


def token_f1(pred: str, golds: Iterable[str]) -> float:
    """Max token-level F1 over reference answers (LongBench-style)."""
    best = 0.0
    pt = normalize(pred).split()
    for g in golds:
        gt = normalize(g).split()
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# --- budget <-> compression-ratio mapping -------------------------------------
def budget_to_ratio(budget: int, ctx_len: int) -> float:
    """kvpress compression_ratio = fraction of KV *removed*.

    To retain `budget` positions out of `ctx_len`, drop the rest. Clamped to
    [0, 0.95] to avoid an all-evicted cache. kvpress applies this per layer
    (uniform) — itself a SnapKV simplification; document it in the writeup.
    """
    return max(0.0, min(0.95, 1.0 - (budget / max(1, ctx_len))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Official kvpress SnapKV parity eval.")
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen")
    ap.add_argument("--budgets", type=int, nargs="+", default=[96, 128, 256, 512])
    ap.add_argument("--tasks", nargs="+", default=list(LONGBENCH_TASKS))
    ap.add_argument("--max-samples", type=int, default=45,
                    help="Per task. Paper N=360 total ~= 45 x 8 tasks.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=Path("scripts/kvpress_parity_results.json"))
    args = ap.parse_args()

    # Heavy imports deferred so --help works without the full stack installed.
    import torch
    from datasets import load_dataset
    from transformers import pipeline
    from kvpress import SnapKVPress

    model_id = MODELS[args.model]
    print(f"[parity] loading {model_id} ...", flush=True)
    pipe = pipeline(
        "kv-press-text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    results: dict = {"model": model_id, "metric": "contains/f1", "by_budget": {}}

    for budget in args.budgets:
        agg_contains: list[int] = []
        agg_f1: list[float] = []
        for task in args.tasks:
            ds = load_dataset("THUDM/LongBench", task, split="test",
                              trust_remote_code=True)
            for ex in ds.select(range(min(args.max_samples, len(ds)))):
                ctx: str = ex["context"]
                question: str = ex["input"]
                golds = ex["answers"]
                ctx_len = len(pipe.tokenizer(ctx)["input_ids"])
                press = SnapKVPress(compression_ratio=budget_to_ratio(budget, ctx_len))
                pred = pipe(ctx, question=question, press=press,
                            max_new_tokens=args.max_new_tokens)["answer"]
                agg_contains.append(contains_score(pred, golds))
                agg_f1.append(token_f1(pred, golds))

        n = max(1, len(agg_contains))
        correct_pct = 100.0 * sum(agg_contains) / n
        f1_mean = sum(agg_f1) / n
        results["by_budget"][budget] = {
            "n": n, "correct_pct": round(correct_pct, 1), "f1": round(f1_mean, 4),
        }
        print(f"[parity] B={budget:>4}  N={n:>4}  CORRECT={correct_pct:5.1f}%  "
              f"F1={f1_mean:.4f}", flush=True)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[parity] wrote {args.out}")
    print("[parity] Compare CORRECT% above against the paper's RealSnapKV row "
          "(~47-49% @ B=256). A large gap confirms the shared-budget handicap.")


if __name__ == "__main__":
    main()
