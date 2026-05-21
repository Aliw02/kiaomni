import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

OUT = os.path.dirname(__file__)

# ── Passkey Accuracy Data ──
with open(os.path.join(OUT, '..', 'data', 'passkey_results.csv')) as f:
    lines = [l.strip() for l in f if l.strip()]

header = lines[0].split(',')
rows = [l.split(',') for l in lines[1:]]

policies = ['FullContext', 'BlockSal', 'KiaOmni_Gaussian', 'KiaOmni_\\u03c38', 'H2O', 'SnapKV']
palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8D6B94']

depths = ['010', '025', '050', '075', '090']
depth_labels = ['10%', '25%', '50%', '75%', '90%']
budgets = ['98', '128', '256', '512']

# Compute mean accuracy across all contexts for each policy × depth × budget
data = {}
for r in rows:
    pol_raw = r[0]
    bud = r[1]
    vals = [float(x) for x in r[2:]]
    key = (pol_raw, bud)
    if key not in data:
        data[key] = []
    data[key].extend(vals)

# Map display names
name_map = {
    'FullContext': 'FullContext',
    'BlockSal': 'BlockSal',
    'KiaOmni_Gaussian': 'KiaOmni_Gaussian',
    'KiaOmni_\\u03c38': 'KiaOmni_σ8',
    'H2O': 'H2O',
    'SnapKV': 'SnapKV',
}

fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
fig.suptitle('Passkey Retrieval Accuracy by Budget (mean across all contexts)', fontsize=13, fontweight='bold')

for bi, bud in enumerate(budgets):
    ax = axes[bi]
    x = np.arange(len(depths))
    width = 0.12
    for pi, pol in enumerate(policies):
        pol_key = pol.replace('\\u03c38', 'σ8') if 'σ8' in pol else pol
        # Find matching rows
        vals = []
        for r in rows:
            if r[0] == pol_key and r[1] == bud:
                vals = [float(x) for x in r[2:]]
                break
        # Group by depth (every 3 context lengths per depth)
        depth_means = []
        for di in range(len(depths)):
            start = di * 3
            depth_vals = vals[start:start+3]
            depth_means.append(np.mean(depth_vals) if depth_vals else 0)
        ax.bar(x + pi * width, depth_means, width, label=name_map.get(pol, pol), color=palette[pi])
    ax.set_title(f'B = {bud}', fontsize=11)
    ax.set_xticks(x + width * (len(policies)-1) / 2)
    ax.set_xticklabels(depth_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.98, color='gray', linestyle='--', linewidth=0.5)
    if bi == 0:
        ax.set_ylabel('Accuracy')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.08), fontsize=9)
plt.tight_layout(rect=[0, 0.08, 1, 0.93])
plt.savefig(os.path.join(OUT, 'passkey_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("passkey_accuracy.png saved")

# ── PPL Bar Chart ──
with open(os.path.join(OUT, '..', 'data', 'ppl_results.csv')) as f:
    lines = [l.strip() for l in f if l.strip()]
header = lines[0].split(',')
rows = [l.split(',') for l in lines[1:]]

ppl_data = {}
for r in rows:
    pol, bud, ppl = r[0], r[1], float(r[2])
    if pol not in ppl_data:
        ppl_data[pol] = {}
    ppl_data[pol][bud] = ppl

ppl_policies = ['FullContext', 'KiaOmni_Gaussian', 'KiaOmni_σ8', 'BlockSal', 'SnapKV', 'H2O']
ppl_palette = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72', '#8D6B94', '#6A994E']

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.13

for pi, pol in enumerate(ppl_policies):
    pol_key = pol.replace('KiaOmni_σ8', 'KiaOmni_σ8')
    bud_vals = []
    for bud in ['128', '256', '512']:
        v = ppl_data.get(pol, {}).get(bud, None)
        bud_vals.append(v)
    ax.bar(x + pi * width, bud_vals, width, label=pol, color=ppl_palette[pi])

ax.set_title('Perplexity (WikiText-2) by Policy and Budget', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * (len(ppl_policies)-1) / 2)
ax.set_xticklabels(['B=128', 'B=256', 'B=512'], fontsize=10)
ax.set_ylabel('PPL ↓')
ax.legend(loc='upper right', fontsize=8)
ax.axhline(y=7.46, color='gray', linestyle='--', linewidth=0.8, label='_FC ref')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ppl_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("ppl_comparison.png saved")
