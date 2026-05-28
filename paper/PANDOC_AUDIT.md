# Pandoc Conversion Audit — `KiaOmni_Paper.md`

**Source file:** `D:\MyFolder\ProgrammingWith-Python\Ai\A+\KiaOmni_Paper.md` (769 lines)
**Target:** Publication-grade PDF via Pandoc + LaTeX
**Scope:** Issues that, if left unfixed, will produce broken tables, missing glyphs, code-block "math", or compile errors.

---

## 1. Math Handling Issues (fenced code blocks used as display math)

The paper uses triple-backtick fenced blocks for formulas. With `--listings`, Pandoc renders these as code listings (monospace, no math typesetting, `Σ` and `σ` glyphs lost under pdflatex). Each block below should be converted to a LaTeX display-math environment.

| Lines | Current (code fence) | Recommended Patch |
|------|----------------------|-------------------|
| 78–80 | `A[i] = mean_h( softmax(Q_h @ K_h^T)[last_query, i] )` | Replace with `$$A_i = \mathrm{mean}_h\!\left(\mathrm{softmax}(Q_h K_h^\top)_{\text{last},i}\right)$$` |
| 85–87 | `E[i] = log1p(A[i])` | `$$E_i = \log(1+A_i)$$` |
| 89–93 | `P[i] = Σ_{j=0}^{i} E[j]` and `F[i] = (P[min(i+σ, N-1)] - P[max(i-σ-1, -1)]) / (2σ+1)` | Convert to `align*` block: `$$P_i = \sum_{j=0}^{i} E_j, \qquad F_i = \frac{P_{\min(i+\sigma,N-1)} - P_{\max(i-\sigma-1,-1)}}{2\sigma+1}$$` |
| 95–99 | `keep = argsort(F, descending=True)[:B - N_SINK - RECENCY]` plus two `∪=` lines | Convert to `\begin{align*}` block or keep as pseudocode under `\begin{algorithmic}` (algorithm2e/algpseudocode). Note `∪=` and `argsort` glyphs need math mode. |
| 138–141 | `H_norm = normalized Shannon entropy of A` / `σ = σ_max × (1 - H_norm) × √(B/N)` | `$$\sigma = \sigma_{\max}\cdot(1-H_{\text{norm}})\cdot\sqrt{B/N}$$` |

**Recommendation:** strip ` ```...``` ` fences around these five blocks and replace with `$$...$$` (or `\begin{equation*}`). Leave actual code references like `extract_all_saliency` as inline backticks.

---

## 2. Table Issues

The paper contains ~20 markdown tables. Several are too wide for the default `\textwidth` and will overflow without `longtable` + `\footnotesize` or landscape rotation. **Several contain emoji** that pdflatex cannot render.

### 2a. Wide tables (>6 columns or wide text cells)

| Line | Section | Cols | Issue | Fix |
|------|---------|------|-------|-----|
| 155–164 | Table 1 (cross-arch) | 6 | borderline; pp values fit | `\footnotesize` + booktabs |
| 232–238 | Table 2 (LongBench F1) | 7 | 7 cols + `✦` emoji | Reduce cell content, replace `✦` |
| 253–266 | Table 3 (Hallucination) | 5 | Last row contains `←` arrow + `+3.1pp` annotation + dagger; row labels long | Move annotation to note, replace `←` with `$\leftarrow$` |
| 286–293 | Table 4 (niah_multikey) | 5 | OK with footnotesize | – |
| 297–304 | Table 5 (VT) | 6 | Contains `✦` and `+44%` percentage signs | Replace `✦` with `*` or `\textsuperscript{$\ast$}` |
| 314–321 | Cross-context Qwen 16K | 5 | OK | – |
| 331–337 | Table 8a | 4 + `×` symbol | `×` in "Speedup" column | Replace `×` with `$\times$` |
| 341–348 | Table 8b | 4 + `×` | Same; also `~31×` repeated | Replace |
| 362–375 | Table 9 (Falcon3) | 6 | wide; many rows | `longtable`, `\footnotesize` |
| 395–403 | Table 10 (PPL) | 6 + `✅`/`❌` emoji in "Trend" col | Heavy emoji use, last col is text | Replace ✅→"yes" / ❌→"no" or use unicode-safe engine |
| 425–438 | Table 11 (BioMistral) | 6 | wide | longtable |
| 442–449 | Table 12 (BioMistral halluc) | 5 | OK | – |
| 521–530 | Table 6 (Phi-3) | 7 | 7 cols, long policy names | `\footnotesize` + `longtable`; consider landscape |
| 542–546 | Compression Benefit | 5 | OK | – |
| 597–612 | Appendix A inventory | 3 | Last col very long ("Key Result") | `\raggedright` p-column or shrink |
| 753–761 | Appendix D.9 stats | 5 | Contains `✅` and `⚠️` in last col | Replace emoji |

### 2b. Emoji-bearing tables (pdflatex will fail or print `?`)

- Line 234: `✦` (Table 2)
- Lines 299, 300, 301, 302, 306: `✦` (Table 5)
- Lines 398–403: `✅`, `❌` (Table 10 Trend column — 6 occurrences)
- Lines 755–760: `✅` (Table D.9 Significant? column — 6 occurrences)
- Line 761: `⚠️` (Table D.9, multi-codepoint emoji — extra risk)
- Line 763: `✅` in body text

**Fix:** Either (a) compile with `xelatex` + `fontspec` + emoji-capable font (e.g. `Symbola` or `NotoColorEmoji`), or (b) global replacement in a preprocessing step: `✅`→`yes`, `❌`→`no`, `✦`→`*`, `⚠️`→`!`, `←`→`$\leftarrow$`, `†`→`\dag`.

---

## 3. Non-ASCII / Unicode Issues

pdflatex chokes on these without `xelatex`/`lualatex` or explicit `\usepackage{newunicodechar}` mappings. Representative occurrences (NOT exhaustive — pattern-replace globally):

| Char | Meaning | Example line(s) | LaTeX replacement |
|------|---------|-----------------|-------------------|
| `σ` | sigma | 11, 23, 25, 26, 27, 92, 102, 109–117, 132, 137, 140, 471–481 (very heavy use) | `$\sigma$` (inline) / `\sigma` (in math) |
| `×` | times | 11, 332–348 (Speedup col), 151, 419 | `$\times$` |
| `≥`, `≤` | inequalities | 103, 117, 415, 536, 743, 747 | `$\geq$`, `$\leq$` |
| `≈` | approx | 350, 705 | `$\approx$` |
| `→` | arrow | 49, 56, 350, 411, 532 | `$\to$` or `$\rightarrow$` |
| `←` | left arrow | 265 (Table 3) | `$\leftarrow$` |
| `μ` | mu | (none found — clean) | – |
| `²`, `³` | superscripts | (none found) | – |
| `°` | degree | (none) | – |
| `±` | plus/minus | 166, 178 | `$\pm$` |
| `Σ` | summation | 91 | `$\sum$` |
| `√` | sqrt | 140 | `$\sqrt{\,}$` |
| `∪` | union | 98, 99 | `$\cup$` |
| `∈` | element of | 29, 351, 470, 517 | `$\in$` |
| `∪=` | augmented union | 98, 99 | `\mathrel{{\cup}{=}}` |
| `≡` | equivalent | 109, 113, 513, 624 | `$\equiv$` |
| `⁻⁶`, `⁻¹⁵`, `⁻⁴⁰`, `⁻⁴` | Unicode superscript digits | 11, 184, 656, 663, 755–759 | `$10^{-6}$`, `$10^{-15}$`, etc. |
| `≪` | much less than | 39 | `$\ll$` |
| `α` | alpha (sig level) | 31, 184, 266, 278, 658, 663, 730, 755–760 | `$\alpha$` |
| `ℝ` | reals | 76 | `$\mathbb{R}$` |
| `Δ` | delta | 253, 487, 56 | `$\Delta$` |
| `†` | dagger | 266, 268 | `\dag` |
| `pp` | percentage points | many | leave as text |
| em-dash `—` | many lines | OK with `--from markdown+smart` | leave |
| en-dash `–` | 124, 128, 597, 601 | OK with smart | leave |
| `≥` greek/punctuation | already covered | – | – |

**Bulk strategy:** Run a single `sed` preprocessing pass that maps the above unicode glyphs to inline math. Alternative: use `xelatex` engine and add `\usepackage{unicode-math}` + `\setmathfont{Latin Modern Math}` — this lets you keep the Unicode source untouched. **Recommended: switch to `xelatex`** (handles σ, ≥, ×, ∪, arrows, ℝ natively; only emoji still need attention).

---

## 4. Heading / Structure Issues

- **Heading levels are consistent overall** — paper uses `#` (title), `##` (numbered sections 1–9 + Abstract + Appendices), `###` (subsections), `####` (table captions). No level-skipping detected.
- **Table caption convention is unusual:** `#### Table N: ...` (lines 153, 202, 230, 251, 284, 296, 360, 393, 423, 440, 519, 753). Pandoc will render these as `\paragraph{}` (level 4). Recommend converting to true Pandoc table captions: `Table: caption text` placed under the table, so they get `\caption{}` and `\ref{}` support.
- **Section 2.2 is duplicated:** Line 41 `### 2.2 Implementation Notes for Baselines` and line 58 `### 2.2 Existing Methods`. This is a logical-numbering bug, not a Pandoc bug, but pdflatex `\subsection{2.2 Existing Methods}` will produce two TOC entries labeled "2.2". **Renumber one of them** (e.g. 2.2 → Implementation Notes, 2.3 → Existing Methods, 2.4 → Subword Gap).
- **Section 5.0** (line 149) uses a `.0` suffix. Pandoc handles this fine, but conventional papers start at 5.1. Optional rename to "5.1 Unified Cross-Model …" and shift downstream.
- **Section 5.1b, 5.2b**: lowercase suffix sections (lines 198, 242). Pandoc renders these literally — fine, but reviewers may flag.
- **Headings with special chars:** Line 638 `### D.1 "Your SnapKV baseline is not correctly implemented"` and D.2–D.8 — contain smart double quotes inside `\subsubsection{}`. Pandoc with `+smart` converts these to `\enquote{}` only if `csquotes` is loaded. **Add `\usepackage{csquotes}` to the header**, or smart-quotes may break the PDF outline.

---

## 5. Cross-Reference / Link Issues

Cross-references are plain prose, not `\ref`. Representative samples (do not rewrite all — out of scope for first compile):

- Line 31: "is reported as a confirmed result in §5.2b" — `§5.2b` is literal text.
- Line 56: "See Appendix D.10" — literal.
- Line 117: "(see §6.4)" — literal.
- Line 156 footnote: "see `GROUND_TRUTH.md` §1" — external file ref, fine as-is.
- Line 415: "(§5.7)" — literal.

For a first compile, leave these as text. For final submission, replace with `\Cref{sec:hallucination}` and add `\label{sec:hallucination}` under each heading. The `§` symbol itself is Unicode — replace with `\S` or rely on xelatex.

External markdown link present at line 56: `[FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)` and `[NVIDIA/kvpress](...)`. Pandoc converts to `\href{}{}` — requires `\usepackage{hyperref}` (Pandoc loads by default).

---

## 6. Quote Characters

Smart quotes are used (e.g. line 54 `"SnapKV_Modified"`, line 240 `"compression benefit"`, lines 638, 649, 669, 683, 698, 711, 725, 736 — Appendix D objection headers, and §6.5 line 540 — several `"..."` pairs).

Pandoc's default `--from markdown+smart` will convert these to `\enquote{}` (csquotes) or to `` `` ... '' ``. Behavior is correct **provided `csquotes` is loaded**. No action required if the recommended Pandoc invocation (§8) is used.

---

## 7. Bibliography

**There is NO bibliography section.** The paper has a `## 7. Related Work` section (lines 552–558) with inline author-year citations of the form `H2O (Zhang et al., 2023)`, `SnapKV (Li et al., 2024)`, `ScissorHands (Liu et al., 2023)`, `PyramidKV (Cai et al., 2024)`, `Michel et al., 2019`, `Child et al., 2019`, `Zheng et al. (2023)`, plus arXiv IDs scattered in body text (e.g. `arXiv:2404.14469` line 47, `arXiv:2311.16867` line 356).

**Gap:** No `references.bib`, no `## References` section, no `\printbibliography` hook. Pandoc will not generate a References list. **For publication you must:**

1. Create `paper/references.bib` with BibTeX entries for the 7 cited works.
2. Convert inline citations to `[@zhang2023h2o]` style (Pandoc-citeproc syntax).
3. Add `# References` heading at end of paper.
4. Pass `--citeproc --bibliography=paper/references.bib` to Pandoc.

This is the single biggest publication-readiness gap.

---

## 8. Recommended Pandoc Invocation (First Compile)

Given the heavy Unicode (σ everywhere), emoji in tables, fenced math, and absence of bibliography, **use xelatex** (not pdflatex) for the first compile. This eliminates 90% of the Unicode patches above without source edits.

```bash
pandoc KiaOmni_Paper.md \
  -o paper/KiaOmni_Paper.pdf \
  --from=markdown+smart+pipe_tables+backtick_code_blocks \
  --pdf-engine=xelatex \
  --listings \
  --number-sections \
  --toc --toc-depth=3 \
  -V geometry:margin=1in \
  -V mainfont="Latin Modern Roman" \
  -V monofont="DejaVu Sans Mono" \
  -V mathfont="Latin Modern Math" \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V documentclass=article \
  -V papersize=a4 \
  -V fontsize=10pt \
  -H paper/header.tex
```

Where `paper/header.tex` should contain:

```latex
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{csquotes}
\usepackage{unicode-math}
\usepackage{microtype}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
% emoji fallback (only if you do not preprocess emoji out)
\newunicodechar{✅}{\textsc{yes}}
\newunicodechar{❌}{\textsc{no}}
\newunicodechar{✦}{$\ast$}
\newunicodechar{⚠}{!}
\newunicodechar{†}{\textsuperscript{\dag}}
\newunicodechar{←}{$\leftarrow$}
% force wide tables to shrink
\AtBeginEnvironment{longtable}{\footnotesize}
```

### Compile order (recommended workflow)

1. **First compile** with the command above — expect: tables OK, math fenced blocks still rendered as listings (acceptable for draft), no References.
2. **Second pass** (publication): apply Section 1 patches (math fences → `$$`), Section 2b patches (emoji → text), Section 7 (add bibliography + citeproc).
3. **Renumber duplicate §2.2** before final submission.

### Critical-path fixes (must do before first useful PDF)

1. **Switch engine to xelatex** (kills the σ / × / ≥ / ∪ / ℝ problem instantly).
2. **Add `header.tex`** with `longtable`, `booktabs`, `csquotes`, `newunicodechar` mappings for emoji.
3. **Fix duplicate section 2.2** (rename one heading) — otherwise the TOC has two "2.2" entries.

---

## Summary

| Category | Count |
|---|---|
| Fenced-code math blocks needing `$$` conversion | 5 |
| Tables ≥6 cols or wide content | 6 |
| Tables containing emoji | 5 |
| Distinct Unicode glyph classes needing replacement (if not using xelatex) | ~20 |
| Heading structure bugs (duplicate §2.2) | 1 |
| Bibliography missing | 1 (critical) |
| Smart-quote handling | OK with `+smart` + `csquotes` |

**Top 3 critical fixes:**
1. Switch pdf-engine from `pdflatex` to `xelatex` (one flag, fixes all Unicode).
2. Add `header.tex` with `longtable`, `booktabs`, `csquotes`, and emoji `\newunicodechar` mappings.
3. Add a real References section + `references.bib` + `--citeproc`. Without this the paper is not submittable.
