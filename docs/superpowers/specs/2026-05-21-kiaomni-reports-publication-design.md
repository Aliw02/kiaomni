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

- 7 publication lanes total (6 parallel in M1a + L8 sequential in M1b — see §4)
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

### 3a. Dual-output model

Each lane writes **two copies** of its content, in two locations:

| Location | Purpose | Granularity | Tracked? |
|---|---|---|---|
| `main-results/<lane>/` | Comprehensive, paper-grade artifacts. The source-of-truth for the paper. Includes every plot, every table, every raw curated CSV. | Verbose | Local-only (gitignored, decided at M3-polish) |
| `reports/<lane>/` | Curated, public-facing GitHub version. Headline table + key plots + methodology paragraph. | Concise (~5-10 files per lane) | Tracked, committed, pushed to `origin/main` |

The curation rule: anything that would clutter a casual reader stays in `main-results/`; anything that helps a reader decide whether to use KiaOmni goes in `reports/`. When in doubt, file it under `main-results/` and link from `reports/` to the source script for full detail.

### 3b. Repo layout

```
kiaomni/  (repo root)
├── main-results/                     ← LOCAL paper workspace (likely .gitignore'd)
│   ├── qwen2.5-7b/                       full dump per lane
│   ├── mistral-7b/
│   ├── cross-model/
│   ├── benchmarks/niah-heatmap/
│   ├── benchmarks/passkey-and-ppl/
│   └── llm-judge/
└── reports/                          ← PUBLIC GitHub-visible curated version
    ├── README.md                     ← top-level index (created at M2)
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
| **L8** Master comparison | Aggregates from `033_full_comparison_results/`, `034_mistral_results/`, `037_falcon3_results/`, `038_biomistral_results/`, `040_amber_results/` (per-model headline CSVs) | `reports/full-comparison/` | **One master table** rows = all policies (FullContext, KiaOmni_σ8/Gaussian/Scissorhands/RatioAdaptive, SnapKV_Modified, RealSnapKV, H2O, StreamingLLM, …), columns = headline metric per (model × task), final "Mean" column. **One master plot** (heatmap or grouped bar — SnapKV-paper style). |

Lane 3 (Llama-3.1) and the original Phi-3 sub-lane are **dropped** per owner instruction.

---

## 4a. Anti-Cheat / Data-Integrity Mandate

This is the most important rule in this spec. Read it twice.

**Every numeric value that appears in any published `reports/<lane>/README.md` or `reports/<lane>/*.md` MUST be sourced from a specific cell in a specific file under `main-results/<lane>/`.** No exceptions.

### What this forbids

- Citing numbers "from memory" or "from the analysis report"
- Rounding to "look nice" if the source file has more precision
- Citing a different model's number by accident (cross-contamination between lanes)
- Fabricating averages — if you publish a mean, the source CSV must contain a row labeled "mean" *or* the subagent must compute the mean in a checked-in script
- Citing numbers from superseded / partial / debug runs

### What every subagent MUST produce

In addition to the README and plots, each subagent writes:

`reports/<lane>/provenance.json` — a structured manifest with one entry per published number, each entry containing:

```json
{
  "value": 4.176,
  "metric": "overall_score",
  "policy": "KiaOmni_Gaussian",
  "source_file": "main-results/qwen2.5-7b/data/final_scores.csv",
  "source_row": "KiaOmni_Gaussian",
  "source_column": "overall",
  "extraction_method": "literal_cell"  // or "computed_mean", "computed_max", etc.
}
```

The dispatcher (me, after all lanes return) MUST spot-check at least 3 random entries per lane by opening the source file and confirming the cell matches. Any mismatch halts the milestone.

### What this enables

A future reviewer — or you, six months from now — can audit any published claim by opening `provenance.json`, finding the source file, and confirming the cell. No "I think we said 89% somewhere" memory archaeology.

---

## 5. Per-Lane Subagent Contract

Each subagent receives a self-contained prompt and must perform exactly these steps in order:

1. **Inventory.** `ls` the source result directory; list every `*.json`, `*.csv`, `*.png`, `*.md` that exists. Identify the canonical files (latest timestamp, mentioned in memory or `EXPERIMENT_MAP.md`).
2. **Curate — paper grade (`main-results/<lane>/`).** Copy *every* canonical file into `main-results/<lane>/`. Preserve subdirectory structure. This is the paper's source-of-truth.
3. **Curate — repo grade (`reports/<lane>/data/`).** Copy *only the headline* JSON/CSV the public README references. Skip raw per-trial dumps, debug logs, intermediate runs.
4. **Plot.** For each headline figure: regenerate the PNG from `main-results/<lane>/` using matplotlib, write to both `main-results/<lane>/plots/` (paper) and `reports/<lane>/plots/` (repo). If a canonical PNG already exists in the source dir (e.g., the heatmap lane), copy verbatim to both locations and note provenance.
5. **Write `reports/<lane>/README.md`.** Required sections in this order:
   - **TL;DR** (one paragraph, ≤80 words, with the single most important number)
   - **Methodology** (one paragraph naming the experiment script, model, context lengths, budgets, metrics)
   - **Headline table** (Markdown table — policies × headline metric)
   - **Figures** (link `plots/*.png` with a one-line caption each)
   - **Caveats** (anything that would mislead a casual reader — e.g., Scissorhands PPL anomaly, RealSnapKV broken disclosure)
   - **Reproduce** (one fenced shell block: clone, install, run the originating script)
   - **Full data** (one-line pointer: "Comprehensive paper-grade artifacts kept locally under `main-results/<lane>/`.")
6. **Commit** within the lane's worktree (only `reports/`, never `main-results/`):
   ```
   git add reports/<lane>/
   git commit -m "docs(reports): publish <lane> results from <source>"
   ```
7. **Push** to `origin/main`. If the push is rejected because another lane raced ahead, `git pull --rebase origin main` and retry once.
8. **Report back** to the dispatcher in ≤150 words: what was published to `reports/`, what was added to `main-results/`, what was deliberately excluded, any anomalies encountered.

---

## 6. Milestones & Coordination

Execution is decomposed into **4 sequential milestones**; milestone M1 contains 6 parallel lanes dispatched via the `dispatching-parallel-agents` pattern.

### M1 — Parallel evidence dispatch (6 lanes concurrent, then L8 sequential)

**Phase M1a (parallel):** Dispatch lanes L1, L2, L4, L5, L6, L7 simultaneously with `isolation: "worktree"`. Each is an **independent problem domain** — distinct source directory, distinct output paths, independent commit. No shared state.

**Phase M1b (sequential, after M1a):** L8 (master comparison) runs alone because it *depends on* the curated data deposited by L1, L2, L4 in `main-results/`. Trying to parallel-dispatch L8 would race on data that doesn't exist yet.

| Lane | Phase | Subagent type | Worktree |
|---|---|---|---|
| L1 Qwen | M1a | `general-purpose` | yes |
| L2 Mistral | M1a | `general-purpose` | yes |
| L4 Cross-arch | M1a | `general-purpose` | yes |
| L5 NIAH heatmap | M1a | `general-purpose` | yes |
| L6 Passkey + PPL | M1a | `general-purpose` | yes |
| L7 LLM-judge | M1a | `general-purpose` | yes |
| L8 Master comparison | M1b | `general-purpose` | yes |

Race-condition handling: first lane to push creates `reports/`; subsequent lanes pull-rebase if their push races. Worktrees isolate working-copy state.

**M1 DoD:** all 6 lanes have committed + pushed to `origin/main`; each lane's `reports/<lane>/README.md` is reachable via `git show`.

### M2 — Top-level integration

Single commit on main checkout (no worktree). Adds `reports/README.md` (the index linking each of the 6 lanes) and edits the root `README.md` to add a "📊 Results" section above the fold with the one-line headline.

**M2 DoD:** clicking the repo's root `README.md` on GitHub shows the headline result + a link table to all 6 lane reports.

### M3 — Repo polish

Single commit. Verifies LICENSE, `pip install -e .` works, CHANGELOG has v0.2.5 entry, `.gitignore` excludes `main-results/` (which is local-only by definition), quickstart copy-paste runs without error.

**M3 DoD:** `pytest tests/ -v` green; quickstart from README runs and produces the documented output.

### M4 — Release v0.3.0

Bump version files, write CHANGELOG entry summarizing M1+M2 (= reports added, README integrated), `git tag v0.3.0`, push tag.

**M4 DoD:** tag visible on GitHub release page; release notes link to `reports/README.md`.

### Memory updates

After M4 lands, save memory `project_kiaomni_reports_published.md` recording the publication SHAs and tag.

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
