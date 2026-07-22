import json
import os

notebook_path = r"d:\MyFolder\ProgrammingWith-Python\Ai\A+\notebook\001_llm_inference.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Cell 4
        if "# Cell 4 — Experiment Configuration" in source:
            if "TurboCD" not in source:
                new_source = source + """
# TurboCD hyperparams
TCD_TAU_FAST = 100
TCD_TAU_SLOW = 60000
TCD_LAM = 0.15
TCD_ALPHA = 2.0
TCD_THETA = -0.5
TCD_MAX_THETA = 0.5
"""
                cell['source'] = new_source.splitlines(True)
                
        # Cell 7
        elif "# Cell 7 — Eviction Policy Scoring Functions" in source:
            if "TurboCD" not in source:
                turbocd_code = """
# 7. TurboCD — The Advanced Dynamic Ensemble
def policy_turbocd(tau_fast=TCD_TAU_FAST, tau_slow=TCD_TAU_SLOW, lam=TCD_LAM, alpha=TCD_ALPHA, theta=TCD_THETA, k=CACHE_BUDGET) -> np.ndarray:
    freq  = np.power(np.log1p(attn_received), alpha)
    decay = lam * np.exp(-dt / tau_fast) + (1 - lam) * np.exp(-dt / tau_slow)
    
    # VFCA variance penalty component
    per_layer_attn = []
    for layer_attn in _layer_attns_cpu:
        col = layer_attn.mean(axis=0).sum(axis=0)
        per_layer_attn.append(col)
    stacked = np.stack(per_layer_attn, axis=0)
    mu  = stacked.mean(axis=0) + 1e-9
    std = stacked.std(axis=0)
    cv  = std / mu
    
    raw = (freq / (cv + 1.0)) * decay
    
    # GRACERank admission component
    mu_raw = raw.mean()
    sig_raw = raw.std() + 1e-9
    z = (raw - mu_raw) / sig_raw
    admitted = np.where(z >= theta)[0]
    
    if len(admitted) <= k:
        chosen = admitted if len(admitted) > 0 else np.arange(CONTEXT_LEN)
    else:
        chosen = admitted[np.argsort(raw[admitted])[-k:]]
    return np.sort(chosen)

# 8. TurboCD x6 — Specialized for aggressive compression targets
def policy_turbocd_x6(k=CACHE_BUDGET) -> np.ndarray:
    # Optimizes dynamically for severe budget cuts
    return policy_turbocd(tau_fast=50, tau_slow=80000, lam=0.1, alpha=2.5, theta=0.0, k=k)

# 9. TurboCD MAX — Ultimate state context preservation
def policy_turbocd_max(k=CACHE_BUDGET) -> np.ndarray:
    # Rejects vast majority of tokens, preserves only critical entities 
    return policy_turbocd(tau_fast=10, tau_slow=100000, lam=0.05, alpha=3.0, theta=0.5, k=k)
"""
                insert_idx = source.find("# ── Pre-compute per-layer attention arrays for VFCA (CPU, fp32) ──")
                if insert_idx != -1:
                    new_source = source[:insert_idx] + turbocd_code + "\n" + source[insert_idx:]
                    cell['source'] = new_source.splitlines(True)

        # Cell 9
        elif "# Cell 9 — Run All Policies" in source:
            if "TurboCD" not in source:
                new_source = source.replace('("GRACERank 🆕", policy_gracerank),', '("GRACERank 🆕", policy_gracerank),\n    ("TurboCD 🚀",        policy_turbocd),\n    ("TurboCD x6 🚀",     policy_turbocd_x6),\n    ("TurboCD MAX 🚀",    policy_turbocd_max),')
                cell['source'] = new_source.splitlines(True)

        # Cell 13
        elif "# Cell 13 — Sensitivity Analysis" in source:
            if "TurboCD" not in source:
                new_source = source.replace("sensitivity = {name: [] for name in ['DualTime 🆕', 'H2O', 'Recency', 'TFCache']}", "sensitivity = {name: [] for name in ['DualTime 🆕', 'H2O', 'Recency', 'TFCache', 'TurboCD 🚀', 'TurboCD MAX 🚀']}")
                new_source = new_source.replace("'Recency':     policy_recency,", "'Recency':     policy_recency,\n    'TurboCD 🚀': policy_turbocd,\n    'TurboCD MAX 🚀': policy_turbocd_max,")
                new_source = new_source.replace("'Recency':     '#8be9fd',", "'Recency':     '#8be9fd',\n    'TurboCD 🚀': '#ff5555',\n    'TurboCD MAX 🚀': '#bd93f9',")
                new_source = new_source.replace("sens_lw = {'DualTime 🆕': 3.0, 'TFCache': 2.0, 'H2O': 2.0, 'Recency': 1.5}", "sens_lw = {'DualTime 🆕': 3.0, 'TFCache': 2.0, 'H2O': 2.0, 'Recency': 1.5, 'TurboCD 🚀': 3.5, 'TurboCD MAX 🚀': 3.5}")
                cell['source'] = new_source.splitlines(True)

    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "## 🏆 Policies Compared" in source:
            if "TurboCD" not in source:
                new_source = source.replace("| **GRACERank** | Z-score relative admission with rejection gate 🆕 |", "| **GRACERank** | Z-score relative admission with rejection gate 🆕 |\n| **TurboCD** | Hybrid dynamic ensemble with multiplicative weights 🚀 |\n| **TurboCD x6 / MAX** | High-compression optimized variants 🚀 |")
                cell['source'] = new_source.splitlines(True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✅ Successfully injected TurboCD and its variants into the notebook.")
