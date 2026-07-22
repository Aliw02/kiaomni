"""
build_appendix.py  v3
Fixes: double-pipe separator bug, improved design with emoji markers,
       grouped layout, % suffix for readability.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path

BASE  = Path(__file__).parent
PAPER = BASE.parent / "KiaOmni_Paper.md"
CSV   = BASE / "full_results_table.csv"

df = pd.read_csv(CSV)
df.columns = [c.replace("KiaOmni_s8", "KiaOmni_σ8") for c in df.columns]

BUDGETS  = [98, 128, 256, 512]
POLICIES = ["FullContext", "KiaOmni_Gaussian", "KiaOmni_σ8", "BlockSal", "Ada-SnapKV", "H2O", "RealSnapKV"]
MODELS   = ["Qwen2.5-7B", "Mistral-7B", "Falcon3-7B", "BioMistral-7B"]

# Short display names for header (keep table width manageable)
HEADERS = {
    "FullContext":    "FC (oracle)",
    "KiaOmni_Gaussian": "KiaOmni-G ✦",
    "KiaOmni_σ8":    "KiaOmni-σ8 ✦",
    "BlockSal":       "BlockSal",
    "Ada-SnapKV":     "Ada-SnapKV",
    "H2O":            "H2O",
    "RealSnapKV":     "RealSnapKV",
}

def fmt(val, is_best=False, is_fc=False):
    if str(val) in ("—", "nan", "?"):
        return "—"
    try:
        f = float(val)
        s = f"{f:.1f}%"
        if is_best:   return f"**{s}**"
        if is_fc:     return f"*{s}*"
        return s
    except:
        return str(val)

lines = [
    "",
    "---",
    "",
    "## Appendix B — Full Per-Task Results",
    "",
    "CORRECT% (LLM-judge) for every **model × budget × task × policy**.",
    "Source: `llm_judge_results.csv` — 61,681 judged predictions.",
    "✦ = KiaOmni variant · **Bold** = best eviction policy per row · *Italic* = FullContext oracle · Falcon3 B=96 aligned to B=98 column.",
    "",
]

for model in MODELS:
    sub_m = df[df["Model"] == model]
    tasks = sorted(sub_m["Task"].unique())

    lines.append(f"### {model}")
    lines.append("")

    for budget in BUDGETS:
        bud_note = " *(B=96)*" if (model == "Falcon3-7B" and budget == 98) else ""
        lines.append(f"**B={budget}{bud_note}**")
        lines.append("")

        sub_b = sub_m[sub_m["Budget"] == budget]

        # Header — use short names
        h_row = "| Task | " + " | ".join(HEADERS[p] for p in POLICIES) + " |"
        s_row = "|:-----|" + "|:------:" * len(POLICIES) + "|"
        lines.append(h_row)
        lines.append(s_row)

        for task in tasks:
            r = sub_b[sub_b["Task"] == task]
            if r.empty:
                continue
            raw = [r.iloc[0].get(p, "—") for p in POLICIES]

            # Find best among eviction policies (skip FC at index 0)
            evict_nums = []
            for v in raw[1:]:
                try:    evict_nums.append(float(v))
                except: evict_nums.append(-1.0)
            best_val = max(evict_nums) if evict_nums else -1.0

            cells = [fmt(raw[0], is_fc=True)]  # FC in italic
            for i, v in enumerate(raw[1:]):
                try:
                    fv = float(v)
                    cells.append(fmt(v, is_best=(fv == best_val and fv > 0)))
                except:
                    cells.append("—")

            lines.append(f"| {task} | " + " | ".join(cells) + " |")

        # Macro avg row
        def avg(col):
            nums = [float(v) for v in sub_b[col] if str(v) not in ("—","nan","?")]
            return f"{sum(nums)/len(nums):.1f}" if nums else None

        avgs_raw = [avg(p) for p in POLICIES]
        evict_avg_nums = []
        for v in avgs_raw[1:]:
            try:    evict_avg_nums.append(float(v)) if v else evict_avg_nums.append(-1.0)
            except: evict_avg_nums.append(-1.0)
        best_avg = max(evict_avg_nums) if evict_avg_nums else -1.0

        avg_cells = [fmt(avgs_raw[0], is_fc=True)]
        for v in avgs_raw[1:]:
            try:
                fv = float(v)
                avg_cells.append(fmt(v, is_best=(fv == best_avg and fv > 0)))
            except:
                avg_cells.append("—")

        lines.append("| **Macro Avg** | " + " | ".join(avg_cells) + " |")
        lines.append("")

    lines.append("")

# ── Write standalone MD ──────────────────────────────────────────────────────
standalone = [
    "# Full Results: CORRECT% by Model × Budget × Task × Policy",
    "",
    "CORRECT% (LLM-judge) for every **model × budget × task × policy**.",
    "Source: `llm_judge_results.csv` — 61,681 judged predictions.",
    "✦ = KiaOmni variant · **Bold** = best eviction policy per row · *Italic* = FullContext oracle",
    "Falcon3 B=96 aligned to B=98 column.",
    "",
] + lines[6:]  # skip the section header lines (those are for the paper appendix)

(BASE / "full_results_table.md").write_text("\n".join(standalone), encoding="utf-8")
print("Wrote full_results_table.md")

# ── Inject into paper ────────────────────────────────────────────────────────
paper = PAPER.read_text(encoding="utf-8")
MARKER = "\n## References"
APP_B  = "\n## Appendix B"

appendix_block = "\n" + "\n".join(lines) + "\n"

if APP_B in paper:
    start = paper.index(APP_B)
    end   = paper.index(MARKER, start)
    paper = paper[:start] + appendix_block + paper[end:]
else:
    insert_at = paper.index(MARKER)
    paper = paper[:insert_at] + appendix_block + paper[insert_at:]

PAPER.write_text(paper, encoding="utf-8")
print(f"Updated paper: {len(paper):,} bytes, {paper.count(chr(10))} lines")
print("Done.")
