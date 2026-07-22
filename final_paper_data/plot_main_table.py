"""
plot_main_table.py
Publication-quality grouped bar chart for Table 1:
  % of FullContext CORRECT% at B=512 — 4 Architectures
Output: reports/full-comparison/plots/main_table_bar.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent.parent / "reports" / "full-comparison" / "plots" / "main_table_bar.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Data from paper Table 1 (sandbox-verified) ──────────────────────────────
MODELS = ["Qwen2.5-7B", "Mistral-7B", "Falcon3-7B", "BioMistral-7B"]

DATA = {
    # policy           Qwen   Mistral  Falcon3  BioMistral
    "KiaOmni-Gaussian": [89.5,  81.2,    83.3,    98.6],
    "KiaOmni-σ8":       [87.1,  75.8,    82.5,    96.6],
    "BlockSal":         [83.6,  71.5,    77.8,    98.6],
    "Ada-SnapKV":       [76.0,  56.4,    67.5,    98.6],
    "H2O":              [66.7,  54.5,    64.3,    97.3],
    "RealSnapKV":       [63.7,  46.1,    43.7,    92.5],
}

POLICIES = list(DATA.keys())

# ── Palette — dark background, distinct vivid colors ────────────────────────
BG       = "#0f1117"
GRID_C   = "#2a2d3a"
TEXT_C   = "#e8eaf0"
ACCENT   = "#ffffff"

COLORS = {
    "KiaOmni-Gaussian": "#4fc3f7",   # cyan-blue  (primary KiaOmni)
    "KiaOmni-σ8":       "#29b6f6",   # slightly darker cyan
    "BlockSal":         "#ab47bc",   # purple
    "Ada-SnapKV":       "#66bb6a",   # green
    "H2O":              "#ffa726",   # amber
    "RealSnapKV":       "#ef5350",   # red
}

n_models  = len(MODELS)
n_policies = len(POLICIES)
width  = 0.12
gap    = 0.05
group_w = n_policies * width + gap
x = np.arange(n_models) * (group_w + 0.18)

fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
ax.set_facecolor(BG)

for i, policy in enumerate(POLICIES):
    vals   = DATA[policy]
    offset = (i - n_policies / 2 + 0.5) * (width + 0.005)
    bars   = ax.bar(x + offset, vals, width=width,
                    color=COLORS[policy], alpha=0.92,
                    edgecolor=BG, linewidth=0.6,
                    zorder=3)
    # value labels on top of each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.0f}",
                ha="center", va="bottom",
                fontsize=6.5, color=TEXT_C, fontweight="bold")

# ── FullContext 100% reference line ─────────────────────────────────────────
ax.axhline(100, color="#ffffff", linewidth=1.2, linestyle="--", alpha=0.35, zorder=2, label="FullContext (oracle)")

# ── Axes styling ─────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=12, color=TEXT_C, fontweight="bold")
ax.set_ylim(35, 107)
ax.set_ylabel("CORRECT% as % of FullContext (B=512)", fontsize=11, color=TEXT_C, labelpad=10)
ax.set_xlabel("Architecture", fontsize=11, color=TEXT_C, labelpad=8)

ax.tick_params(colors=TEXT_C, which="both", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID_C)
ax.spines["bottom"].set_color(GRID_C)
ax.yaxis.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# ── Title ────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Table 1 — KiaOmni vs Baselines: % of FullContext CORRECT% at B=512\n"
    "4 Architectures · 8 LongBench Tasks · 61,681 LLM-Judged Samples",
    fontsize=13, color=ACCENT, fontweight="bold", y=0.98, ha="center"
)

# ── Legend ───────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[p], label=p) for p in POLICIES]
patches.append(mpatches.Patch(color="#ffffff", alpha=0.35, label="FullContext 100% (oracle)"))
ax.legend(
    handles=patches,
    loc="lower right",
    fontsize=9,
    framealpha=0.15,
    facecolor=BG,
    edgecolor=GRID_C,
    labelcolor=TEXT_C,
    ncol=2,
)

# ── Cross-model mean annotations ─────────────────────────────────────────────
means = {p: np.mean(DATA[p]) for p in POLICIES}
mean_text = "  Cross-model mean:  " + "   ".join(
    f"{p.split('-')[0] if '-' in p else p[:6]}: {means[p]:.1f}%"
    for p in POLICIES
)
fig.text(0.5, 0.01, mean_text, ha="center", va="bottom",
         fontsize=8, color="#9e9e9e", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {OUT}")
