import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('D:/MyFolder/ProgrammingWith-Python/Ai/A+/notebook/001_llm_inference.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Keep only original 16 cells
nb['cells'] = nb['cells'][:16]

cell_A_src = (
    "# PRE-REGISTRATION PROTOCOL (Science Lock-In)\n"
    "# Run ONCE before any test. Defines ALL hyperparams and datasets.\n"
    "# Changing these after seeing results = cheating. Defense verifies hash.\n"
    "import hashlib, json as _json, datetime\n"
    "\n"
    "PROTOCOL = {\n"
    '    "version": "v1.0",\n'
    '    "date": str(datetime.date.today()),\n'
    '    "model": "Qwen/Qwen2.5-1.5B-Instruct",\n'
    '    "cache_budget": 512,\n'
    '    "context_len": 1200,\n'
    '    "eval_len": 200,\n'
    '    "n_samples_per_domain": 10,\n'
    '    "random_seed": 42,\n'
    '    "hyperparams": {\n'
    '        "TFCACHE_TAU": 800,   "TFCACHE_ALPHA": 1.5,\n'
    '        "DT_TAU_FAST": 200,   "DT_TAU_SLOW": 50000, "DT_LAM": 0.2,  "DT_ALPHA": 1.5,\n'
    '        "VFCA_TAU":    800,   "VFCA_ALPHA": 1.5,\n'
    '        "GR_TAU":      800,   "GR_ALPHA": 1.5, "GR_THETA": 0.3,\n'
    '        "TCD_TAU_FAST": 300,  "TCD_TAU_SLOW": 80000, "TCD_LAM": 0.15,\n'
    '        "TCD_ALPHA": 2.0,     "TCD_THETA": -0.5\n'
    '    },\n'
    '    "domains": {\n'
    '        "needle_retrieval": {"dataset": "abisee/cnn_dailymail", "task": "ppl+retrieval"},\n'
    '        "summarization":    {"dataset": "abisee/cnn_dailymail", "task": "ppl"},\n'
    '        "long_qa":          {"dataset": "hotpot_qa",            "task": "ppl+em"},\n'
    '        "code_completion":  {"dataset": "codeparrot/github-code","task": "ppl"}\n'
    '    },\n'
    '    "primary_metric":    "perplexity_delta_vs_baseline",\n'
    '    "secondary_metrics": ["win_rate", "mean_rank", "wilcoxon_p_vs_H2O"]\n'
    "}\n"
    "\n"
    "proto_str  = _json.dumps(PROTOCOL, sort_keys=True)\n"
    "proto_hash = hashlib.sha256(proto_str.encode()).hexdigest()[:16]\n"
    'print(f"PROTOCOL HASH: {proto_hash}")\n'
    'print("Lock this hash before tests. Hash change = cheating.")\n'
    "print(_json.dumps(PROTOCOL, indent=2))\n"
)

cell_B_src = (
    "# CELL B: Real Data Loader (no hand-crafted text, fixed seed=42)\n"
    "from datasets import load_dataset\n"
    "import numpy as np, random\n"
    "\n"
    "_RNG = np.random.default_rng(42)\n"
    "random.seed(42)\n"
    "\n"
    "def load_real_documents(domain_name, n=10):\n"
    "    if domain_name in ('summarization', 'needle_retrieval'):\n"
    "        ds   = load_dataset('abisee/cnn_dailymail', '3.0.0', split='test', streaming=True)\n"
    "        docs = [x['article'] for i, x in enumerate(ds) if i < 400]\n"
    "        idxs = _RNG.choice(len(docs), size=n, replace=False)\n"
    "        if domain_name == 'summarization':\n"
    "            return [docs[i] for i in idxs]\n"
    "        results = []\n"
    "        for i in idxs:\n"
    "            doc  = docs[i]\n"
    "            code = str(_RNG.integers(10_000_000, 99_999_999))\n"
    "            cut  = int(len(doc) * _RNG.uniform(0.05, 0.60))\n"
    "            needle_str = ' [SECRET:' + code + '] '\n"
    "            results.append({'text': doc[:cut] + needle_str + doc[cut:], 'needle': code})\n"
    "        return results\n"
    "\n"
    "    elif domain_name == 'long_qa':\n"
    "        ds    = load_dataset('hotpot_qa', 'fullwiki', split='validation', streaming=True)\n"
    "        items = [x for i, x in enumerate(ds) if i < 400]\n"
    "        idxs  = _RNG.choice(len(items), size=n, replace=False)\n"
    "        out   = []\n"
    "        for i in idxs:\n"
    "            x   = items[i]\n"
    "            ctx = ' '.join([' '.join(s) for s in x['context']['sentences']])\n"
    "            out.append({'context': ctx, 'question': x['question'], 'answer': x['answer']})\n"
    "        return out\n"
    "\n"
    "    elif domain_name == 'code_completion':\n"
    "        ds   = load_dataset('codeparrot/github-code', 'Python', split='train', streaming=True)\n"
    "        docs = [x['code'] for i, x in enumerate(ds) if i < 1000 and len(x['code']) > 2000]\n"
    "        idxs = _RNG.choice(min(len(docs), 400), size=n, replace=False)\n"
    "        return [docs[i] for i in idxs]\n"
    "\n"
    "    raise ValueError('Unknown domain: ' + domain_name)\n"
    "\n"
    "print('Data loader ready.')\n"
)

cell_C_src = (
    "# CELL C: Fair Multi-Domain Evaluation Engine\n"
    "# Rule 1: same tokenized input for ALL policies per sample\n"
    "# Rule 2: hyperparams from PROTOCOL only - no per-domain tuning\n"
    "# Rule 3: metrics computed AFTER all runs finish\n"
    "import numpy as np, torch, torch.nn.functional as F\n"
    "\n"
    "DOMAINS_TO_TEST = ['summarization', 'needle_retrieval', 'code_completion', 'long_qa']\n"
    "N_SAMPLES = 10  # raise to 30 for final Defense submission\n"
    "\n"
    "ALL_POLICIES = [\n"
    "    ('Recency',     policy_recency),\n"
    "    ('H2O',         policy_h2o),\n"
    "    ('TFCache',     policy_tfcache),\n"
    "    ('DualTime',    policy_dualtime),\n"
    "    ('VFCA',        policy_vfca),\n"
    "    ('GRACERank',   policy_gracerank),\n"
    "    ('TurboCD',     policy_turbocd),\n"
    "    ('TurboCD_x6',  policy_turbocd_x6),\n"
    "    ('TurboCD_MAX', policy_turbocd_max),\n"
    "]\n"
    "\n"
    "@torch.no_grad()\n"
    "def _ppl_only(sel_ids, eval_ids):\n"
    "    pkv  = model(input_ids=sel_ids, use_cache=True).past_key_values\n"
    "    nlls = []\n"
    "    for s in range(eval_ids.shape[1] - 1):\n"
    "        out  = model(input_ids=eval_ids[:, s:s+1], past_key_values=pkv, use_cache=True)\n"
    "        pkv  = out.past_key_values\n"
    "        nlls.append(F.cross_entropy(out.logits[:, -1, :], eval_ids[:, s+1]).item())\n"
    "    return float(np.exp(np.mean(nlls)))\n"
    "\n"
    "def run_domain(domain_name, documents):\n"
    "    pol_ppls  = {nm: [] for nm, _ in ALL_POLICIES}\n"
    "    base_ppls = []\n"
    "    for di, doc in enumerate(documents):\n"
    "        if isinstance(doc, dict):\n"
    "            text = doc.get('text') or doc.get('context', '')\n"
    "        else:\n"
    "            text = doc\n"
    "        ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)\n"
    "        if ids.shape[1] < CONTEXT_LEN + EVAL_LEN + 10:\n"
    "            continue\n"
    "        ctx_ids  = ids[:, :CONTEXT_LEN]\n"
    "        eval_ids = ids[:, CONTEXT_LEN:CONTEXT_LEN + EVAL_LEN + 1]\n"
    "        with torch.no_grad():\n"
    "            pfx = model(input_ids=ctx_ids, output_attentions=True, use_cache=False)\n"
    "        heads_sum   = torch.zeros(CONTEXT_LEN)\n"
    "        layer_attns = []\n"
    "        for la in pfx.attentions:\n"
    "            arr = la[0].float().cpu()\n"
    "            heads_sum += arr.mean(0).sum(0)\n"
    "            layer_attns.append(arr.numpy())\n"
    "        _oa = globals().get('attn_received')\n"
    "        _ol = globals().get('_layer_attns_cpu')\n"
    "        globals()['attn_received']    = heads_sum.numpy()\n"
    "        globals()['_layer_attns_cpu'] = layer_attns\n"
    "        base_ppls.append(_ppl_only(ctx_ids, eval_ids))\n"
    "        for pname, pfn in ALL_POLICIES:\n"
    "            try:\n"
    "                idx = pfn(k=CACHE_BUDGET)\n"
    "                pol_ppls[pname].append(_ppl_only(ctx_ids[:, idx], eval_ids))\n"
    "            except Exception as e:\n"
    "                print('  ERROR', pname, 'doc', di, ':', e)\n"
    "                pol_ppls[pname].append(float('nan'))\n"
    "        globals()['attn_received']    = _oa\n"
    "        globals()['_layer_attns_cpu'] = _ol\n"
    "        print(f'  [{domain_name}] {di+1}/{len(documents)} base={base_ppls[-1]:.2f}', end='\\r')\n"
    "    return pol_ppls, base_ppls\n"
    "\n"
    "all_domain_results   = {}\n"
    "all_domain_baselines = {}\n"
    "for domain in DOMAINS_TO_TEST:\n"
    "    print('\\n' + '='*60 + '\\n  ' + domain.upper() + '\\n' + '='*60)\n"
    "    docs = load_real_documents(domain, n=N_SAMPLES)\n"
    "    r, b = run_domain(domain, docs)\n"
    "    all_domain_results[domain]   = r\n"
    "    all_domain_baselines[domain] = b\n"
    "print('\\nAll domains complete.')\n"
)

cell_D_src = (
    "# CELL D: Statistical Report (Defense-ready)\n"
    "# Metrics: mean dPPL%, win-rate, mean-rank, Wilcoxon p vs H2O\n"
    "# Significant = p < 0.05 (one-sided: policy PPL < H2O PPL)\n"
    "import numpy as np\n"
    "from scipy import stats\n"
    "\n"
    "pol_names = [n for n, _ in ALL_POLICIES]\n"
    "\n"
    "for domain in DOMAINS_TO_TEST:\n"
    "    base = np.array(all_domain_baselines[domain])\n"
    "    n    = len(base)\n"
    "    print('\\n' + '='*68)\n"
    "    print('  ' + domain.upper() + f'  (n={n}, baseline mean PPL={base.mean():.2f})')\n"
    "    print('='*68)\n"
    "    print(f'  {\"Policy\":<18} {\"dPPL%\":>8} {\"Win%\":>7} {\"Rank\":>6} {\"p-vs-H2O\":>10}')\n"
    "    print('  ' + '-'*54)\n"
    "    h2o = np.array(all_domain_results[domain]['H2O'])\n"
    "    rows = []\n"
    "    for pol in pol_names:\n"
    "        ppls  = np.array(all_domain_results[domain][pol])\n"
    "        valid = ~np.isnan(ppls) & ~np.isnan(base)\n"
    "        delta = ((ppls[valid] - base[valid]) / base[valid] * 100)\n"
    "        mean_d = delta.mean() if valid.any() else float('nan')\n"
    "        win = 0\n"
    "        for i in range(n):\n"
    "            sample = {p: all_domain_results[domain][p][i] for p in pol_names\n"
    "                       if not np.isnan(all_domain_results[domain][p][i])}\n"
    "            if sample and min(sample, key=sample.get) == pol:\n"
    "                win += 1\n"
    "        rank_sum = 0\n"
    "        for i in range(n):\n"
    "            row = sorted([(p, all_domain_results[domain][p][i]) for p in pol_names\n"
    "                           if not np.isnan(all_domain_results[domain][p][i])], key=lambda x: x[1])\n"
    "            for r, (p, _) in enumerate(row, 1):\n"
    "                if p == pol:\n"
    "                    rank_sum += r\n"
    "        v2 = ~np.isnan(ppls) & ~np.isnan(h2o)\n"
    "        if pol == 'H2O' or v2.sum() < 5:\n"
    "            p_val = float('nan')\n"
    "        else:\n"
    "            try:\n"
    "                _, p_val = stats.wilcoxon(ppls[v2], h2o[v2], alternative='less')\n"
    "            except:\n"
    "                p_val = float('nan')\n"
    "        rows.append((pol, mean_d, win/n*100, rank_sum/n, p_val))\n"
    "    rows.sort(key=lambda x: x[1])\n"
    "    for pol, d, w, r, p in rows:\n"
    "        sig = ' *' if (not np.isnan(p) and p < 0.05) else '  '\n"
    "        print(f'  {pol:<18} {d:>+8.1f}% {w:>6.0f}% {r:>6.1f} {p:>10.4f}{sig}')\n"
    "    print('  * = significantly better than H2O (Wilcoxon p < 0.05)')\n"
    "\n"
    "print('\\n' + '='*68)\n"
    "print(f'  OVERALL: Mean dPPL across all {len(DOMAINS_TO_TEST)} domains')\n"
    "print('='*68)\n"
    "agg = {p: [] for p in pol_names}\n"
    "for domain in DOMAINS_TO_TEST:\n"
    "    base = np.array(all_domain_baselines[domain])\n"
    "    for pol in pol_names:\n"
    "        ppls  = np.array(all_domain_results[domain][pol])\n"
    "        valid = ~np.isnan(ppls)\n"
    "        if valid.any():\n"
    "            agg[pol].extend(((ppls[valid] - base[valid]) / base[valid] * 100).tolist())\n"
    "ranked = sorted([(p, np.mean(v)) for p, v in agg.items() if v], key=lambda x: x[1])\n"
    "for rank, (pol, d) in enumerate(ranked, 1):\n"
    "    print(f'  {rank:>2}. {pol:<18}  avg dPPL = {d:>+.1f}%')\n"
)

cells_to_add = [
    {"cell_type": "code", "metadata": {}, "outputs": [], "source": [cell_A_src]},
    {"cell_type": "code", "metadata": {}, "outputs": [], "source": [cell_B_src]},
    {"cell_type": "code", "metadata": {}, "outputs": [], "source": [cell_C_src]},
    {"cell_type": "code", "metadata": {}, "outputs": [], "source": [cell_D_src]},
]

nb['cells'].extend(cells_to_add)

with open('D:/MyFolder/ProgrammingWith-Python/Ai/A+/notebook/001_llm_inference.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
