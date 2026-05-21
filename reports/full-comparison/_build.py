"""Build L8 Master Comparison: master table, heatmap, provenance, README."""
import csv
import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.dirname(BASE)
ROOT = os.path.dirname(REPORTS)
DATA_IN = os.path.join(BASE, "data")
PLOTS_IN = os.path.join(BASE, "plots")
MAIN_RESULTS = os.path.join(ROOT, "main-results", "full-comparison")
SRC_QWEN = os.path.join(REPORTS, "qwen2.5-7b", "data", "final_scores.csv")
SRC_MISTRAL = os.path.join(REPORTS, "mistral-7b", "data", "final_scores.csv")
SRC_FALCON = os.path.join(REPORTS, "cross-model", "data", "falcon3_final_scores.csv")
SRC_BIOMISTRAL = os.path.join(REPORTS, "cross-model", "data", "biomistral_final_scores.csv")

POLICY_MAP = {"KiaOmni_s8": "KiaOmni_σ8", "SnapKV_Modified": "BlockSal", "Ada-SnapKV": "AdaSnapKV", "RealSnapKV": "SnapKV"}
WHITELIST = {"FullContext", "H2O", "SnapKV", "BlockSal", "AdaSnapKV", "KiaOmni_σ8", "KiaOmni_Gaussian"}

def norm_policy(name):
    name = name.strip()
    return POLICY_MAP.get(name, name)

def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# ---- Read all source CSVs ----
qwen_rows = read_csv(SRC_QWEN)
mistral_rows = read_csv(SRC_MISTRAL)
falcon_rows = read_csv(SRC_FALCON)
biomistral_rows = read_csv(SRC_BIOMISTRAL)

def build_map(rows, key_col, val_col):
    m = {}
    for r in rows:
        p = norm_policy(r[key_col])
        if p in WHITELIST:
            m[p] = float(r[val_col])
    return m

qwen = build_map(qwen_rows, "policy", "overall")
mistral = build_map(mistral_rows, "policy", "macro_f1_B256")
falcon = build_map(falcon_rows, "Policy", "Mean")
biomistral = build_map(biomistral_rows, "Policy", "Mean")

# ---- Build master table ----
all_policies = [p for p in WHITELIST if p in qwen and p in mistral and p in falcon and p in biomistral]
# sort by mean descending
table = []
for p in all_policies:
    q = qwen[p]; m = mistral[p]; f = falcon[p]; b = biomistral[p]
    avg = (q + m + f + b) / 4.0
    table.append((p, q, m, f, b, avg))
table.sort(key=lambda x: x[5], reverse=True)

# ---- Write master_table.csv ----
csv_path = os.path.join(DATA_IN, "master_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Policy", "Qwen_Overall", "Mistral_macro_f1", "Falcon3_Mean", "BioMistral_Mean", "Mean"])
    for row in table:
        w.writerow(row)

# ---- Generate heatmap ----
headers = ["Qwen\nOverall", "Mistral\nF1", "Falcon3\nMean", "BioMistral\nMean"]
policies = [r[0] for r in table]
data = np.array([[r[1], r[2], r[3], r[4]] for r in table])

fig, ax = plt.subplots(figsize=(7, 4.5))
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=data.max())
im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(headers)))
ax.set_xticklabels(headers, fontsize=9)
ax.set_yticks(range(len(policies)))
ax.set_yticklabels(policies, fontsize=9)
ax.set_title("KiaOmni — Cross-Model Comparison", fontsize=12, fontweight="bold")

for i in range(len(policies)):
    for j in range(len(headers)):
        val = data[i, j]
        color = "white" if norm(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

fig.tight_layout()
heatmap_path = os.path.join(PLOTS_IN, "master_heatmap.png")
fig.savefig(heatmap_path, dpi=200)
plt.close(fig)
print(f"Heatmap saved: {heatmap_path}")

# ---- Provenance ----
provenance = {
    "description": "L8 Master Comparison — aggregated cross-model comparison table",
    "policies_whitelist": sorted(WHITELIST),
    "disclosure": "SnapKV = faithful arXiv:2404.14469 implementation. BlockSal = our novel block-level baseline (paper §4).",
    "sources": {
        "qwen2.5-7b": {"file": str(SRC_QWEN), "column": "overall"},
        "mistral-7b": {"file": str(SRC_MISTRAL), "column": "macro_f1_B256"},
        "falcon3-7b": {"file": str(SRC_FALCON), "column": "Mean"},
        "biomistral-7b": {"file": str(SRC_BIOMISTRAL), "column": "Mean"},
    },
    "entries": [
        {
            "policy": p,
            "Qwen_Overall": {"value": q, "provenance": "L1 qwen2.5-7b final_scores.csv -> overall"},
            "Mistral_macro_f1": {"value": m, "provenance": "L2 mistral-7b final_scores.csv -> macro_f1_B256"},
            "Falcon3_Mean": {"value": f, "provenance": "L4 falcon3_final_scores.csv -> Mean"},
            "BioMistral_Mean": {"value": b, "provenance": "L4 biomistral_final_scores.csv -> Mean"},
            "Mean": {"value": avg, "provenance": "arithmetic mean of the 4 model columns"},
        }
        for p, q, m, f, b, avg in table
    ]
}
prov_path = os.path.join(BASE, "provenance.json")
with open(prov_path, "w", encoding="utf-8") as f:
    json.dump(provenance, f, indent=2)
print(f"Provenance saved: {prov_path}")

# ---- Build README.md ----
winner = table[0][0]
winner_mean = table[0][5]

lines = []
lines.append("# L8 Master Comparison\n")
lines.append("## TL;DR")
lines.append(f"**{winner}** wins the mean column across all 4 models (x̄ = {winner_mean:.3f}).\n")
lines.append("## Methodology")
lines.append("Aggregated from:")
lines.append("- **L1 — Qwen2.5-7B**: LongBench 11-task suite, `overall` score")
lines.append("- **L2 — Mistral-7B**: RULER macro F1 at B=256")
lines.append("- **L4 — Falcon3-7B**: Mean across budgets 96/128/256/512")
lines.append("- **L4 — BioMistral-7B**: Mean across budgets 96/128/256/512")
lines.append("The **Mean** column is the arithmetic average of the 4 model columns.\n")
lines.append("## Master Comparison Table")
lines.append("| Policy | Qwen Overall | Mistral F1 | Falcon3 Mean | BioMistral Mean | Mean |")
lines.append("|--------|-------------|-----------|-------------|----------------|------|")
for r in table:
    p, q, m, f, b, avg = r
    lines.append(f"| {p} | {q:.3f} | {m:.3f} | {f:.3f} | {b:.3f} | {avg:.3f} |")

lines.append("")
lines.append("## Master Heatmap")
lines.append('![Master Heatmap](plots/master_heatmap.png)')
lines.append("*Figure: Cross-model comparison heatmap. Rows sorted by descending mean. Color intensity reflects relative score.*\n")
lines.append("## Caveats")
lines.append("- **SnapKV** = faithful arXiv:2404.14469 implementation. **BlockSal** = our novel block-level baseline (paper §4).")
lines.append("- **Amber** is excluded from this comparison; results reported separately in the Amber section.")
lines.append("- Each model uses a different metric suite: LongBench overall (Qwen), RULER macro F1 (Mistral), passkey accuracy mean across budgets (Falcon3, BioMistral). Direct cross-model comparability is limited — the **Mean** column should be treated as an aggregated indicator, not a rigorous apples-to-apples benchmark.\n")
lines.append("## Reproduce")
lines.append("```bash")
lines.append("python reports/full-comparison/_build.py")
lines.append("```\n")
lines.append("## Full Data")
lines.append("See [`data/master_table.csv`](data/master_table.csv) for the canonical CSV.\n")

# Inline a text version of the full table too
lines.append("### All Values (by model)")
lines.append("| Policy | Qwen Overall | Mistral F1 | Falcon3 Mean | BioMistral Mean |")
lines.append("|--------|-------------|-----------|-------------|----------------|")
for r in table:
    p, q, m, f, b, _ = r
    lines.append(f"| {p} | {q:.3f} | {m:.3f} | {f:.3f} | {b:.3f} |")

readme_path = os.path.join(BASE, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"README saved: {readme_path}")

# ---- Copy source CSVs to main-results/full-comparison ----
for src, name in [
    (SRC_QWEN, "qwen2.5-7b_final_scores.csv"),
    (SRC_MISTRAL, "mistral-7b_final_scores.csv"),
    (SRC_FALCON, "falcon3_final_scores.csv"),
    (SRC_BIOMISTRAL, "biomistral_final_scores.csv"),
]:
    dst = os.path.join(MAIN_RESULTS, name)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")

# Print summary
print("\n=== MASTER TABLE ===")
print(f"| {'Policy':22s} | {'Qwen':6s} | {'Mistral':6s} | {'Falcon3':6s} | {'BioMistral':6s} | {'Mean':6s} |")
print("|" + "-"*24 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*8 + "|")
for r in table:
    print(f"| {r[0]:22s} | {r[1]:6.3f} | {r[2]:6.3f} | {r[3]:6.3f} | {r[4]:6.3f} | {r[5]:6.3f} |")
print(f"\nWinner: {winner} (Mean = {winner_mean:.3f})")
