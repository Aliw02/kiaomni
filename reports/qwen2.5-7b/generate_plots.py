"""Generate comparison plots for KiaOmni Qwen2.5-7B report."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

DST = Path(r"D:\MyFolder\ProgrammingWith-Python\Ai\A+\reports\qwen2.5-7b")
PLOTS = DST / "plots"

POLICY_COLORS = {
    "FullContext": "#2c3e50",
    "KiaOmni_Gaussian": "#27ae60",
    "KiaOmni_σ8": "#2980b9",
    "BlockSal": "#e67e22",
    "AdaSnapKV": "#8e44ad",
    "H2O": "#c0392b",
    "SnapKV": "#7f8c8d",
}

POLICY_LABELS = {
    "FullContext": "FullContext",
    "KiaOmni_Gaussian": "KiaOmni\nGaussian",
    "KiaOmni_σ8": "KiaOmni\nσ8",
    "BlockSal": "BlockSal\n(ours)",
    "AdaSnapKV": "AdaSnapKV",
    "H2O": "H2O",
    "SnapKV": "SnapKV",
}

DATA = DST / "data"

def read_scores():
    scores = {}
    with open(DATA / "final_scores.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["policy"]] = {k: float(v) for k, v in row.items() if k != "policy"}
    return scores

def plot_overall_comparison(scores):
    policies = list(POLICY_COLORS.keys())
    overalls = [scores[p]["overall"] for p in policies]
    fullcontext = scores["FullContext"]["overall"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(policies)), overalls, color=[POLICY_COLORS[p] for p in policies], edgecolor="white", linewidth=0.5)

    ax.axhline(y=fullcontext, color=POLICY_COLORS["FullContext"], linestyle="--", linewidth=1, alpha=0.7)
    ax.text(-0.3, fullcontext + 0.03, f"FullContext\n{fullcontext:.3f}", fontsize=8, color=POLICY_COLORS["FullContext"], va="bottom")

    for i, (bar, score) in enumerate(zip(bars, overalls)):
        pct = (score / fullcontext) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{score:.3f}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=POLICY_COLORS[policies[i]])

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=8)
    ax.set_ylabel("Overall Score (avg across 11 tasks)", fontsize=10)
    ax.set_title("KiaOmni on Qwen2.5-7B — Overall Comparison (whitelist policies)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(overalls) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for label in ax.get_yticklabels():
        label.set_fontsize(8)

    fig.tight_layout()
    fig.savefig(PLOTS / "overall_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved overall_comparison.png")

def plot_vt_comparison(scores):
    policies = list(POLICY_COLORS.keys())
    vt_scores = [scores[p]["vt"] for p in policies]
    fullcontext_vt = scores["FullContext"]["vt"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(policies)), vt_scores, color=[POLICY_COLORS[p] for p in policies], edgecolor="white", linewidth=0.5)

    ax.axhline(y=fullcontext_vt, color=POLICY_COLORS["FullContext"], linestyle="--", linewidth=1, alpha=0.7)
    ax.text(-0.3, fullcontext_vt + 0.05, f"FullContext\n{fullcontext_vt:.2f}", fontsize=8, color=POLICY_COLORS["FullContext"], va="bottom")

    for i, (bar, score) in enumerate(zip(bars, vt_scores)):
        comparison = "vs FullContext"
        color = POLICY_COLORS[policies[i]]
        above = score >= fullcontext_vt
        label = f"{score:.2f}\n({'BEATS' if above else ''})" if abs(score - fullcontext_vt) > 0.01 else f"{score:.2f}"
        va = "bottom" if above else "top"
        y_pos = bar.get_height() + 0.05 if above else bar.get_height() - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label, ha="center", va=va, fontsize=7, fontweight="bold", color=color)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=8)
    ax.set_ylabel("Variable Tracing (VT) Score", fontsize=10)
    ax.set_title("KiaOmni on Qwen2.5-7B — Variable Tracing Task (whitelist policies)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vt_scores) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for label in ax.get_yticklabels():
        label.set_fontsize(8)

    fig.tight_layout()
    fig.savefig(PLOTS / "vt_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved vt_comparison.png")

def plot_speed_vram(scores):
    speed_data = {
        "FullContext": {"tps": 11.1, "vram": 7154},
        "KiaOmni_Gaussian": {"tps": 15.4, "vram": 6036},
        "KiaOmni_σ8": {"tps": 15.5, "vram": 6036},
        "BlockSal": {"tps": 15.4, "vram": 6036},
        "AdaSnapKV": {"tps": 15.2, "vram": 6051},
        "H2O": {"tps": 15.4, "vram": 6036},
        "SnapKV": {"tps": 15.7, "vram": 6016},
    }
    policies = list(POLICY_COLORS.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TPS
    tps_vals = [speed_data[p]["tps"] for p in policies]
    bars1 = ax1.bar(range(len(policies)), tps_vals, color=[POLICY_COLORS[p] for p in policies], edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(policies)))
    ax1.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=7)
    ax1.set_ylabel("Tokens / Second", fontsize=10)
    ax1.set_title("Generation Speed", fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, val in zip(bars1, tps_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # VRAM
    vram_vals = [speed_data[p]["vram"] for p in policies]
    bars2 = ax2.bar(range(len(policies)), vram_vals, color=[POLICY_COLORS[p] for p in policies], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(policies)))
    ax2.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=7)
    ax2.set_ylabel("VRAM (MB)", fontsize=10)
    ax2.set_title("Peak VRAM Usage", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    base_vram = speed_data["FullContext"]["vram"]
    for bar, (p, val) in zip(bars2, [(p, speed_data[p]["vram"]) for p in policies]):
        saved = (1 - val / base_vram) * 100
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f"{val:.0f}\n(-{saved:.1f}%)",
                 ha="center", va="bottom", fontsize=6)

    fig.suptitle("Speed & VRAM — KiaOmni on Qwen2.5-7B", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / "speed_vram.png", dpi=200)
    plt.close(fig)
    print("Saved speed_vram.png")

def plot_per_budget(scores):
    budget_data = {
        "FullContext": [4.69, 4.69, 4.69, 4.69],
        "KiaOmni_Gaussian": [3.53, 3.97, 4.47, 4.75],
        "KiaOmni_σ8": [3.42, 4.00, 4.47, 4.63],
        "BlockSal": [3.44, 3.85, 4.36, 4.65],
        "AdaSnapKV": [2.62, 2.83, 3.23, 3.69],
        "H2O": [2.50, 2.64, 3.08, 3.55],
        "SnapKV": [2.07, 2.14, 2.79, 3.26],
    }
    budgets = ["B98", "B128", "B256", "B512"]
    x = np.arange(len(budgets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in list(POLICY_COLORS.keys()):
        scores_list = budget_data[policy]
        ax.plot(x, scores_list, marker="o", label=policy, color=POLICY_COLORS[policy], linewidth=2, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(budgets, fontsize=9)
    ax.set_xlabel("Budget (cache size in tokens)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("KiaOmni on Qwen2.5-7B — Score by Budget", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(1.5, 5.5)
    ax.grid(axis="y", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontsize(8)

    fig.tight_layout()
    fig.savefig(PLOTS / "per_budget_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved per_budget_comparison.png")

scores = read_scores()
plot_overall_comparison(scores)
plot_vt_comparison(scores)
plot_speed_vram(scores)
plot_per_budget(scores)
print("All plots generated.")
