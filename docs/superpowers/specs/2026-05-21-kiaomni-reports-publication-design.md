# Publish KiaOmni Experimental Results to the Public Repo

**Date:** 2026-05-21
**Owner:** Ali (Aliwey)
**Repo:** `Aliw02/kiaomni`
**Type:** Documentation publication (no code changes to `kiaomni/`)

---

## 1. Goal

Publish the experimental results that currently live only in
`notebook/kv_cache_benchmark/*_results/` to the public GitHub repo under a
new top-level directory `reports/`, so external readers can see the
numbers, plots, and methodology behind KiaOmni's claims.

No experiments are re-run. This is a *publication* pass, not a *generation*
pass.

---

## 2. Scope

### In scope

- 6 parallel publication lanes (see §4)
- Curation of canonical JSON/CSV from each source results directory
- Regeneration of plots from canonical data (so figures and tables agree)
- Per-lane `README.md` with method paragraph, headline table, and links
  back to the originating experiment script
- One git commit per lane, pushed independently to `main`
- Final integration commit: top-level `reports/README.md` index

### Out of scope

- Re-running any experiment
- Touching any file under `kiaomni/` (no v0.2.6 release)
- Llama-3.1 results (`034_llama_results/` — explicitly excluded by owner)
- Phi-3.5 / Phi-3 results (excluded by owner)
- Stale, debug, or superseded result files

---

## 3. Architecture

```
kiaomni/  (repo root)
└── reports/
    ├── README.md                     ← top-level index
    ├── qwen2.5-7b/                   ← L1
    │   ├── README.md
    │   ├── data/                     ← curated JSON / CSV
    │   └── plots/                    ← regenerated PNG figures
    ├── mistral-7b/                   ← L2
    │   ├── README.md
    │   ├── data/
    │   └── plots/
    ├── cross-model/                  ← L4 (Falcon3 + BioMistral + Amber)
    │   ├── README.md
    │   ├── falcon3-7b.md
    │   ├── biomistral-7b.md
    │   ├── amber-7b.md
    │   ├── data/
    │   └── plots/
    ├── benchmarks/
    │   ├── niah-heatmap/             ← L5
    │   │   ├── README.md
    │   │   ├── data/
    │   │   └── plots/
    │   └── passkey-and-ppl/          ← L6
    │       ├── README.md
    │       ├── data/
    │       └── plots/
    └── llm-judge/                    ← L7
        ├── README.md                 ← cross-model judge synthesis
        ├── data/                     ← 4 llm_judge_results.csv files
        └── plots/                    ← win-rate bars
```

Each lane is **self-contained**: a reader who lands on
`reports/mistral-7b/README.md` should see the full story for that model
without needing to jump elsewhere. The top-level `reports/README.md`
exists for cross-navigation only.

---

## 4. Lane Decomposition

Six lanes, each runnable by a single subagent in its own git worktree.

| Lane | Source on local disk | Output path | Headline content |
|---|---|---|---|
| **L1** Qwen2.5-7B full comparison | `notebook/kv_cache_benchmark/033_full_comparison_results/` | `reports/qwen2.5-7b/` | 11 tasks × 12 policies × 4 budgets; lift `final_analysis_report.md` as backbone; NIAH + VT curves |
| **L2** Mistral-7B-Instruct-v0.3 | `notebook/kv_cache_benchmark/034_mistral_results/` | `reports/mistral-7b/` | 4K/8K/16K RULER + LongBench; lift `mistral_analysis_report.md` as backbone |
| **L4** Cross-architecture | `037_falcon3_results/`, `038_biomistral_results/`, `040_amber_results/` | `reports/cross-model/` | One sub-page per model; shared cross-arch comparison plot. *Amber has no LLM-judge data — disclose explicitly.* |
| **L5** NIAH heatmaps | `035_heatmap_results/` | `reports/benchmarks/niah-heatmap/` | Per-policy heatmap PNGs + accuracy-by-depth table for FullContext / H2O / σ8 / Gaussian / Scissorhands at B∈{98,128,256,512} |
| **L6** Passkey + PPL | `034_pass_key_results/`, `035_ppl_wikitext2_results/` | `reports/benchmarks/passkey-and-ppl/` | Passkey: 100% σ8/Gaussian at all depths ≥B98; PPL: Gaussian B=512 = 27.80 (best eviction), Scissorhands = 360-411 (worst, documented anomaly) |
| **L7** LLM-judge synthesis | 4 × `llm_judge_results.csv` (Qwen / Mistral / Falcon3 / BioMistral); source scripts at `scratch/llm_judge_multi_model.py` + `scratch/llm_judge_biomistral.py` | `reports/llm-judge/` | Per-policy win-rate across 4 models; rubric documented from script docstrings; Amber explicitly absent |

Lane 3 (Llama-3.1) and the original Phi-3 sub-lane are **dropped** per owner instruction.

---

## 5. Per-Lane Subagent Contract

Each subagent receives a self-contained prompt and must perform exactly these steps in order:

1. **Inventory.** `ls` the source result directory; list every `*.json`, `*.csv`, `*.png`, `*.md` that exists. Identify the canonical files (latest timestamp, mentioned in memory or `EXPERIMENT_MAP.md`).
2. **Curate.** Copy the canonical files into `reports/<lane>/data/`. Skip files marked debug, scratch, partial, or superseded by a later run.
3. **Plot.** For each headline figure: regenerate the PNG from the curated `data/` using matplotlib. If an existing PNG is already canonical (e.g., the heatmap lane), copy it verbatim into `plots/` and note its provenance.
4. **Write `README.md`.** Required sections in this order:
   - **TL;DR** (one paragraph, ≤80 words, with the single most important number)
   - **Methodology** (one paragraph naming the experiment script, model, context lengths, budgets, metrics)
   - **Headline table** (Markdown table — policies × headline metric)
   - **Figures** (link `plots/*.png` with a one-line caption each)
   - **Caveats** (anything that would mislead a casual reader — e.g., Scissorhands PPL anomaly, RealSnapKV broken disclosure)
   - **Reproduce** (one fenced shell block: clone, install, run the originating script)
5. **Commit** within the lane's worktree:
   ```
   git add reports/<lane>/
   git commit -m "docs(reports): publish <lane> results from <source>"
   ```
6. **Push** to `origin/main`. If the push is rejected because another lane raced ahead, `git pull --rebase origin main` and retry once.
7. **Report back** to the dispatcher in ≤150 words: what was published, what was deliberately excluded, any anomalies encountered.

---

## 6. Coordination & Sequencing

- **Lane dispatch:** all 6 lanes spawn as parallel `Agent` subagents with `isolation: "worktree"`.
- **Race-condition handling:** the first lane to push creates `reports/`; subsequent lanes will pull-rebase if their push races. Worktrees isolate working-copy state, so this is safe.
- **Integration commit:** after all 6 report back, the dispatcher runs one final commit on the main checkout to add `reports/README.md` (the top-level index linking each lane).
- **Memory updates:** after the integration commit lands, save a memory note recording the publication and the canonical commit SHAs.

---

## 7. Headline Numbers (sourced from memory + analysis reports)

These are the numbers each lane is expected to surface prominently. If a
lane's curated data disagrees with these, the lane must investigate
before publishing.

| Lane | Headline |
|---|---|
| L1 Qwen | KiaOmni_Gaussian: 4.176/4.695 = **89.0%** of FullContext (#1); σ8 #3 @ 88.0%; RealSnapKV #12 @ 54.6%; VT: Gaussian *beats* FullContext (6.16 vs 4.53) |
| L2 Mistral | All KiaOmni variants = **100% niah_single** across 4K/8K/16K; VT: Gaussian ties #1 at 0.689 vs FC 0.330 (+108%) |
| L4 Falcon3 / BioMistral / Amber | Confirms cross-architecture generalization beyond Qwen and Mistral |
| L5 NIAH heatmap | Visual confirmation: σ8 and Gaussian retain needle across all depths at B≥128 |
| L6 Passkey | σ8 and Gaussian = **100% at all depths, all contexts, B≥98**; RealSnapKV = 0% at B≤256 |
| L6 PPL | Gaussian B=512 = **27.80** (FC=7.46); Scissorhands = 360-411 (worst — must disclose) |
| L7 LLM-judge | Cross-model win-rate confirmation that KiaOmni_σ8 and Gaussian are not artifacts of contains-score |

---

## 8. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Push races between parallel lanes | Worktree isolation + pull-rebase retry |
| A lane finds stale/contradictory numbers | Lane halts, reports back; dispatcher decides |
| Subagent regenerates a plot incorrectly | Plot script must read from `data/` only — never from the original results dir — so the figure is reproducible from what we publish |
| Pushing 1,800 files bloats the repo | Curation step keeps only canonical files; expected final size ≤ a few hundred files total |
| LLM-judge raw CSVs may contain judge prompts naming proprietary models | L7 must strip any third-party model identifiers from publication-bound CSVs |

---

## 9. Success Criteria

- All 6 lanes have committed + pushed
- Top-level `reports/README.md` exists with working links to every lane
- A fresh reader can click into any lane README and understand the headline result without leaving that page
- Memory file `project_kiaomni_reports_published.md` records the publication and links

---

## 10. Out of Scope (explicit)

- No Llama-3.1 lane
- No Phi-3 / Phi-3.5 lane
- No `kiaomni/` source edits (no v0.2.6)
- No experiment re-runs
- No edits to the existing analysis reports inside `notebook/kv_cache_benchmark/*_results/` — those remain the historical record; the published `reports/` are a curated subset
