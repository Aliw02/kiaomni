import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Load the merged results
with open(r'D:\MyFolder\ProgrammingWith-Python\Ai\A+\notebook\kv_cache_benchmark\039_swap_experiment\merged_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

budgets = [98, 128, 256, 512, 1024, 2048]
protected = 96
num_kv_heads = 32
k_per_head = np.array([round(max(0, (b - protected)) / num_kv_heads, 1) for b in budgets])

# FAIR comparison: each policy with its own native signal + selector
sn = np.array([data['niah_single'][str(b)]['SnapKV_NaturalSignal']['em']['mean'] for b in budgets])
kn = np.array([data['niah_single'][str(b)]['KiaOmni_NaturalSignal']['em']['mean'] for b in budgets])

fig, ax = plt.subplots(figsize=(9, 5.5))

# PRIMARY curves — native signal+selector for each
ax.plot(k_per_head, sn, 'o-', linewidth=3, markersize=11, color='#c0392b',
        label='SnapKV (natural signal + selector)', zorder=5)
ax.plot(k_per_head, kn, '^-', linewidth=3, markersize=11, color='#27ae60',
        label='KiaOmni (natural signal + selector)', zorder=5)

# Phase transition regions
ax.axvspan(0, 5, alpha=0.12, color='red')
ax.axvspan(5, 16, alpha=0.08, color='orange')
ax.axvspan(16, 65, alpha=0.06, color='green')

ax.set_xlabel(r'k per head (tokens) = (budget - 96) / 32 heads', fontsize=12, fontweight='bold')
ax.set_ylabel('Exact Match (EM) - niah_single', fontsize=12)
ax.set_title('Phase Transition: KiaOmni vs SnapKV\nCTX=4096, TinyLlama-1.1B-Chat, n=15/cell', fontsize=13, fontweight='bold')
ax.legend(fontsize=10.5, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(-0.02, 0.55)
ax.set_xticks(k_per_head)

# Budget annotations below x-axis
for k, b in zip(k_per_head, budgets):
    ax.annotate(f'B={b}', xy=(k, -0.03), ha='center', fontsize=9, color='gray', fontweight='bold')

# Key annotations
ax.annotate('SnapKV\nflat at 0', xy=(0.1, 0.01), xytext=(2.5, 0.20),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
            fontsize=11, ha='center', color='#c0392b', fontweight='bold')
ax.annotate('KiaOmni\nrecovers first', xy=(5.0, 0.10), xytext=(8, 0.38),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
            fontsize=11, ha='center', color='#27ae60', fontweight='bold')
ax.annotate('KiaOmni wins\nat ALL budgets\nB <= 1024', xy=(13.0, 0.30), xytext=(22, 0.48),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
            fontsize=11, ha='center', color='#27ae60', fontweight='bold')

# Region labels
ax.text(2.5, 0.50, 'COLLAPSE', ha='center', fontsize=10, color='red', fontweight='bold', alpha=0.7)
ax.text(10.5, 0.50, 'TRANSITION', ha='center', fontsize=10, color='#e67e22', fontweight='bold', alpha=0.7)
ax.text(40, 0.50, 'STABLE', ha='center', fontsize=10, color='green', fontweight='bold', alpha=0.7)

# Delta annotation
ax.text(29, 0.08, 'Delta at B=1024: +0.20 EM', fontsize=10, color='#27ae60', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#27ae60', alpha=0.8))

plt.tight_layout()
outpath = r'D:\MyFolder\ProgrammingWith-Python\Ai\A+\notebook\kv_cache_benchmark\039_swap_experiment\phase_transition_actual.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath}')
print()
print('Fair comparison - each with own native setup:')
print('  KiaOmni EM:', [round(x, 3) for x in kn])
print('  SnapKV  EM:', [round(x, 3) for x in sn])
print('  Delta:     ', [round(k - s, 3) for k, s in zip(kn, sn)])
print()
print('KiaOmni Natural EM:', [round(x, 3) for x in kn])
print('SnapKV Natural EM: ', [round(x, 3) for x in sn])
print('Delta (KiaOmni - SnapKV):', [round(k - s, 3) for k, s in zip(kn, sn)])
