# -*- coding: utf-8 -*-
"""Build the comprehensive main results tables for the KiaOmni paper.

Aggregates the combined LLM-judge output (61,681 rows,
``reports/llm-judge/data/llm_judge_results.csv``) into:

  1. CORRECT% by model x budget x policy   (pooled over tasks and contexts)
  2. CORRECT% by model x context x policy  (pooled over tasks and budgets)
  3. Cross-model mean of %-of-FullContext per budget

Aggregation is micro-averaged CORRECT% — the fraction of judged rows
labelled CORRECT — which exactly reproduces the published Table 1 values
(e.g. Qwen2.5-7B KiaOmni-Gaussian 89.5% of FullContext at B=512).

Output: ``main_results_table.md`` (markdown tables embedded in the paper)
and ``main_results_table.csv`` (machine-readable long format).

Usage:
    python build_main_table.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

HERE = Path(__file__).parent

SOURCE = HERE.parent / "reports" / "llm-judge" / "data" / "llm_judge_results.csv"
MODELS: List[str] = ["Qwen2.5-7B", "Mistral-7B", "Falcon3-7B", "BioMistral-7B"]

# Paper policy set, in presentation order, using the curated repo naming:
# "SnapKV" is the faithful RealSnapKV implementation and "BlockSal" is the
# block-level redesign formerly called SnapKV_Modified (see Section 2.2).
POLICIES: List[str] = [
    "FullContext",
    "KiaOmni_Gaussian",
    "KiaOmni_σ8",
    "BlockSal",
    "AdaSnapKV",
    "H2O",
    "SnapKV",
]
DISPLAY: Dict[str, str] = {
    "FullContext": "FullContext",
    "KiaOmni_Gaussian": "KiaOmni-G",
    "KiaOmni_σ8": "KiaOmni-σ8",
    "BlockSal": "BlockSal",
    "AdaSnapKV": "Ada-SnapKV",
    "H2O": "H2O",
    "SnapKV": "RealSnapKV",
}

# Falcon3 / BioMistral ran the lowest budget at 96 (vs 98 for Qwen /
# Mistral); both are reported in the B≈98 column.
BUDGET_ALIAS = {96: 98}
BUDGETS = [98, 128, 256, 512]
CONTEXTS = [4096, 8192, 16384]


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    out = df[df.policy.isin(POLICIES)][
        ["model", "policy", "budget", "ctx", "task", "judge_label"]
    ].copy()
    out["budget"] = out["budget"].replace(BUDGET_ALIAS)
    out["correct"] = (out["judge_label"] == "CORRECT").astype(float)
    return out


def correct_pct(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Micro-averaged CORRECT% over the given grouping keys."""
    return df.groupby(keys + ["policy"], as_index=False)["correct"].mean()


def fmt(v: float, best: bool, oracle: bool) -> str:
    s = f"{v * 100:.1f}%"
    if oracle:
        return f"*{s}*"
    return f"**{s}**" if best else s


def render_block(piv: pd.DataFrame, row_label: str) -> List[str]:
    cols = [DISPLAY[p] + (" ✦" if p.startswith("KiaOmni") else "") for p in POLICIES]
    lines = [
        "| " + row_label + " | " + " | ".join(cols) + " |",
        "|:---|" + ":---:|" * len(POLICIES),
    ]
    for idx, row in piv.iterrows():
        evict = {p: row[p] for p in POLICIES[1:] if pd.notna(row.get(p))}
        best_v = max(evict.values()) if evict else None
        cells = []
        for p in POLICIES:
            v = row.get(p)
            if pd.isna(v):
                cells.append("—")
            else:
                cells.append(fmt(v, best=(p != "FullContext" and v == best_v),
                                 oracle=(p == "FullContext")))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    df = load()
    n_rows = len(df)
    md: List[str] = []

    # ---- Table: model x budget ------------------------------------------
    by_budget = correct_pct(df, ["model", "budget"])
    md.append("### CORRECT% by Model × Budget (pooled over all tasks and contexts)\n")
    for model in MODELS:
        sub = by_budget[by_budget.model == model].pivot(
            index="budget", columns="policy", values="correct"
        ).reindex(BUDGETS)
        sub.index = [f"**{model}** B={b}" for b in BUDGETS]
        md += render_block(sub, "Model / Budget")
        md.append("")

    # ---- Table: model x context -----------------------------------------
    by_ctx = correct_pct(df, ["model", "ctx"])
    md.append("### CORRECT% by Model × Context Length (pooled over all tasks and budgets)\n")
    for model in MODELS:
        sub = by_ctx[by_ctx.model == model].pivot(
            index="ctx", columns="policy", values="correct"
        ).reindex(CONTEXTS)
        sub.index = [f"**{model}** {c // 1024}K" for c in CONTEXTS]
        md += render_block(sub, "Model / Context")
        md.append("")

    # ---- Cross-model mean of %FC per budget ------------------------------
    md.append("### Cross-Model Mean — % of FullContext CORRECT% per Budget\n")
    rows = []
    for b in BUDGETS:
        sub = by_budget[by_budget.budget == b].pivot(
            index="model", columns="policy", values="correct"
        )
        pct_fc = sub[POLICIES[1:]].div(sub["FullContext"], axis=0).mean() * 100
        rows.append((b, pct_fc))
    header = [DISPLAY[p] + (" ✦" if p.startswith("KiaOmni") else "") for p in POLICIES[1:]]
    md.append("| Budget | " + " | ".join(header) + " |")
    md.append("|:---|" + ":---:|" * len(header))
    for b, pct_fc in rows:
        best = pct_fc.max()
        cells = [
            (f"**{pct_fc[p]:.1f}%**" if pct_fc[p] == best else f"{pct_fc[p]:.1f}%")
            for p in POLICIES[1:]
        ]
        md.append(f"| B={b} | " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"*Source: {n_rows} judged predictions across "
              f"{len(MODELS)} models × {len(BUDGETS)} budgets × "
              f"{len(CONTEXTS)} contexts. ✦ = KiaOmni variant · "
              f"**Bold** = best eviction policy per row · "
              f"*Italic* = FullContext oracle.*")

    out_md = HERE / "main_results_table.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    long = pd.concat([by_budget.assign(dim="budget"), by_ctx.assign(dim="ctx")])
    long.to_csv(HERE / "main_results_table.csv", index=False, encoding="utf-8")
    print(f"Wrote {out_md} ({len(md)} lines) from {n_rows} judged rows.")


if __name__ == "__main__":
    main()
