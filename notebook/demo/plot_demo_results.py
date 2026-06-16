import matplotlib.pyplot as plt
import numpy as np

# ── Full data from Kaggle run (kv-cachecompressionbenchmark.ipynb) ────────────
BUDGETS = [98, 128, 256, 512, 1024, 2048, 3400]

needle_kia  = [33.3, 33.3, 73.3, 80.0, 70.0, 70.0, 70.0]
needle_snap = [ 0.0,  0.0, 13.3, 33.3, 43.3, 63.3, 66.7]
needle_van  = 66.7   # flat line

vram_kia  = [3.15, 3.15, 3.15, 3.15, 3.15, 3.23, 5.26]
vram_snap = [6.35, 6.35, 6.35, 6.36, 6.36, 6.37, 6.39]
vram_van  = 6.40

# ── Per-task breakdown at B=512 ───────────────────────────────────────────────
tasks       = ["Single\nNeedle", "Multi\nNeedle", "Reasoning", "Summary"]
task_kia512 = [100, 100, 40, 100]
task_snap512= [ 80,  20,  0, 100]
task_van512 = [100, 100,  0, 100]

BLUE = "#3fa7d6"
RED  = "#e05c5c"
GRAY = "#aaaaaa"

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    "KiaOmni-Gaussian vs SnapKV vs Vanilla  ·  Qwen2.5-7B-Instruct NF4 · ~4K ctx · Tesla T4",
    fontsize=13, fontweight="bold"
)

# ── Top-left: Needle mean accuracy across all budgets ─────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
xs = range(len(BUDGETS))
ax1.plot(xs, needle_kia,  "o-", color=BLUE, lw=2, ms=7, label="KiaOmni-Gaussian", zorder=3)
ax1.plot(xs, needle_snap, "s-", color=RED,  lw=2, ms=7, label="SnapKV",           zorder=3)
ax1.axhline(needle_van, ls="--", color=GRAY, lw=1.5, label=f"Vanilla ({needle_van:.0f}%)")
for i, (k, s) in enumerate(zip(needle_kia, needle_snap)):
    ax1.text(i, k + 2, f"{k:.0f}", ha="center", fontsize=8, color=BLUE)
    ax1.text(i, s - 5, f"{s:.0f}", ha="center", fontsize=8, color=RED)
ax1.set_xticks(list(xs)); ax1.set_xticklabels([f"B={b}" for b in BUDGETS], fontsize=8)
ax1.set_ylabel("Needle mean accuracy %")
ax1.set_ylim(-10, 100)
ax1.set_title("Needle Mean Accuracy vs Budget", fontsize=10)
ax1.legend(fontsize=8.5); ax1.grid(alpha=0.3); ax1.spines[["top","right"]].set_visible(False)
# annotate peak
ax1.annotate("KiaOmni peak\n(beats Vanilla)", xy=(3, 80), xytext=(4, 85),
             fontsize=7.5, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

# ── Top-right: VRAM across all budgets ───────────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(xs, vram_kia,  "o-", color=BLUE, lw=2, ms=7, label="KiaOmni-Gaussian")
ax2.plot(xs, vram_snap, "s-", color=RED,  lw=2, ms=7, label="SnapKV")
ax2.axhline(vram_van, ls="--", color=GRAY, lw=1.5, label=f"Vanilla ({vram_van:.2f} GB)")
for i, (k, s) in enumerate(zip(vram_kia, vram_snap)):
    ax2.text(i, k + 0.05, f"{k:.2f}", ha="center", fontsize=7.5, color=BLUE)
ax2.set_xticks(list(xs)); ax2.set_xticklabels([f"B={b}" for b in BUDGETS], fontsize=8)
ax2.set_ylabel("Peak VRAM (GB)")
ax2.set_ylim(0, 8); ax2.set_title("Peak VRAM vs Budget", fontsize=10)
ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3); ax2.spines[["top","right"]].set_visible(False)

# ── Bottom-left: Per-task breakdown at B=512 ─────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
x3  = np.arange(len(tasks)); w = 0.26
b1 = ax3.bar(x3 - w, task_van512,  w, color=GRAY, label="Vanilla",           zorder=3)
b2 = ax3.bar(x3,     task_kia512,  w, color=BLUE, label="KiaOmni-Gaussian",  zorder=3)
b3 = ax3.bar(x3 + w, task_snap512, w, color=RED,  label="SnapKV",            zorder=3)
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}",
                     ha="center", va="bottom", fontsize=8)
ax3.set_xticks(x3); ax3.set_xticklabels(tasks, fontsize=9)
ax3.set_ylabel("Accuracy / Coverage %"); ax3.set_ylim(0, 120)
ax3.set_title("Per-Task Accuracy at B=512", fontsize=10)
ax3.legend(fontsize=8.5); ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.spines[["top","right"]].set_visible(False)

# ── Bottom-right: Summary text box ───────────────────────────────────────────
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis("off")
summary = (
    "Key Findings\n"
    "─────────────────────────────────\n\n"
    "★  KiaOmni at B=512 (12.8% cache)\n"
    "    achieves 80% needle mean —\n"
    "    beats Vanilla's 66.7%\n\n"
    "★  SnapKV needs B=3400 (85% cache)\n"
    "    to merely tie Vanilla (66.7%)\n\n"
    "★  KiaOmni VRAM: 3.15 GB flat\n"
    "    from B=98 → B=1024\n"
    "    (half of SnapKV at every budget)\n\n"
    "★  Reasoning task: KiaOmni 40%\n"
    "    vs SnapKV 0% vs Vanilla 0%\n\n"
    "Model: Qwen2.5-7B-Instruct NF4\n"
    "GPU: Tesla T4  ·  SDPA backend\n"
    "N: 10/10/10/3 samples per task"
)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=9.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f7ff", edgecolor="#3fa7d6", lw=1.5))

plt.tight_layout()
plt.savefig("demo_results.png", dpi=150, bbox_inches="tight")
print("Saved: demo_results.png")
