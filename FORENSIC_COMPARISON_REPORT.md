# Forensic Comparison Report
## KiaOmni_Paper.md (OLD · Draft v1.0/v1.1 · 2026-04-27 / 2026-05-09 · 824 lines) ↔ KiaOmni_Revised_Paper.md (NEW · Revised Draft · 2026-06-07 · 565 lines)

**File-size delta:** 67,920 → 37,650 bytes (−45% shorter, despite covering new ground).

**OLD path:** `D:\MyFolder\ProgrammingWith-Python\Ai\A+\KiaOmni_Paper.md`
**NEW path:** `D:\MyFolder\ProgrammingWith-Python\Ai\A+\new_draft\KiaOmni_Revised_Paper.md`

---

## 🔴 THE LOST VAULT
*(Crucial info, data, or arguments from the OLD paper that are completely missing in the NEW paper)*

### 1. The Entire Reviewer-Defense Apparatus (Appendix D, OLD §§664–789)
This is by far the **largest single structural deletion** — 8 pre-emptive reviewer defenses totaling ~125 lines of dense argumentation:
- **D.1** "Your SnapKV baseline is not correctly implemented" — full defense that full-sequence attention is a *stronger* oracle for SnapKV, making the reported gap a *conservative* lower bound
- **D.2** "The hallucination result is not statistically significant" — full closure argument with Z=2.91, p=0.0036 walkthrough
- **D.3** "NIAH is a synthetic task" — defense citing VT, LongBench, and the cross-benchmark compression benefit
- **D.4** "The compression benefit is cherry-picked" — defense using the 7/10 policies > FC observation
- **D.5** "You didn't compare against PyramidKV, AdaKV, or Quest" — explanation of prefill-vs-decode orthogonality
- **D.6** "Your architecture taxonomy predicted σ=0 for Phi-3 but σ=8 won" — philosophical defense that falsification is scientific strength
- **D.7** "N=15 is too small for the bf16 ablation" — Cohen's h ≈ 0.87 power analysis
- **D.8** "The 'no-harm guarantee' is violated on Phi-3 at low budgets" — scope-condition defense

**Verdict:** Losing these was a **serious mistake** for an ACL/NeurIPS submission, where reviewer rebuttal text is the single most-used feature by meta-reviewers. The new paper trades defensive argumentation for assertiveness — risky.

### 2. The KiaOmni_Adaptive Policy with H_norm Formula (OLD §4.1, lines 145–155)
The OLD paper's **adaptive σ** formula is gone:
```
H_norm = -Σ p_i log p_i / log N    where p_i = A_i/ΣA_j
σ = σ_max · (1 - H_norm) · √(B/N)
```
Empirical results: +3.5pp on Qwen, +1.6pp on Mistral at B=40. The **H_norm** variable, the **σ_max** parameter, the Gini-coefficient interpretation, and the Adaptive variant are **completely erased** from the new paper's main text. The variant still appears in Table 1 (line 213 of new) but is no longer explained, derived, or motivated. **Mistake** — Adaptive is now a black box in the cross-architecture comparison.

### 3. The 4-Profile Architecture Taxonomy (OLD §4, lines 131–142)
| Profile | Models | OLD Recommendation |
|---|---|---|
| Hyper-concentrated | Qwen2.5-7B | KiaOmniAdaptive, σ_max=160 |
| Intermediate-peaked | Mistral-7B, **SmolLM-1.7B** | KiaOmniAdaptive, σ_max=64–96 |
| Bimodal-switched | Phi-3-mini | KiaOmni_σ8 (predicted neutral) |
| Flat/diffuse | **TinyLlama-1.1B** | σ=0 (no smoothing) |

The OLD paper mentioned **5 architectures** (Qwen, Mistral, SmolLM, Phi-3, TinyLlama) and Experiments 015v3, D-066 to D-073. The new paper has **only 4** (Qwen, Mistral, Falcon3, BioMistral) and has **downgraded** the taxonomy from a "precise theory" to "an empirical regularity" (§8 of new, line 444). **SmolLM-1.7B and TinyLlama-1.1B have disappeared from the architecture story entirely** — the small-model regime (where σ=0 was predicted to win) is now invisible.

### 4. The Hard Constraint `BLOCK_SIZE ≤ B÷4` (OLD §3.1, line 113)
The hard feasibility constraint in the OLD paper is **gone** from the new paper. The OLD paper declared `BLOCK_SIZE=16` as a fixed parameter with `BLOCK_SIZE ≤ B÷4`. The new paper keeps `SINK=16, RECENCY=32, σ=8 (boxcar) or σ=4 (Gaussian)` but **drops BLOCK_SIZE entirely** — the block granularity of eviction is no longer exposed as a hyperparameter.

### 5. The Discards Appendix (OLD Appendix C, lines 648–654)
Three explicit "discarded claims" are gone from the new paper:
1. Unification theorem (σ=0 ≡ KiaBeast, σ=8 ≡ KiaCachePlusR2) — falsified by D-063 (Jaccard 0.534 / 0.794)
2. log1p as "noise neutralizer" — demoted to implementation detail
3. Phi-3 σ=0 prediction — taxonomy reframed

**Loss is honest** but the new paper's narrative now **silently buries** the KiaBeast / KiaCachePlusR2 lineage — the OLD paper was transparent that those were the algorithm's predecessors, the new paper pretends KiaOmni emerged in a vacuum.

### 6. The Scissorhands PPL Pathology (OLD §5.7 + §8, lines 411–439)
A striking finding in the OLD paper:
- KiaOmni_Scissorhands PPL = **302–411** on WikiText-2 (vs KiaOmni_Gaussian 27.8, FullContext 7.46)
- Identified as the **retrieval/fluency tradeoff**: Scissorhands is best for retrieval (NIAH-multikey), worst for fluency
- Tradeoff was an explicit limitation (#7 in OLD §8)

The new paper's §12 Limitations #2 mentions PPL=27.8 vs FullContext 7.46 in passing but **drops the entire Scissorhands PPL pathology table** (OLD Table 10) and the retrieval/fluency tradeoff framing. The new paper mentions "PPL of KiaOmni-Scissorhands is bad" only obliquely via "3.73×" comparison. **Mistake** — this is a concrete failure mode that reviewers will discover via the existing codebase and complain about.

### 7. The Reviewer Note Inside the Abstract/Intro (OLD line 37)
> *"The hallucination experiment was scaled to N=360 per policy (Experiment 033, 8 LongBench tasks). KiaOmni_σ8 hallucination rate: 45.0% vs FullContext 55.8%; two-proportion Z-test Z=2.91, p=0.0036 (two-sided; one-sided p=0.0018). The prior N=50 result (p=0.45) was underpowered and is superseded."*

This **epistemic transparency** about the underpowered predecessor result is gone. The new paper mentions "The prior underpowered experiment (N=50, p=0.45) is superseded" in §6.5 footnote 3, but the **prominent placement** and **contextual weight** is lost.

### 8. The φ-3 "Quantum Entanglement" Attention-Sink Pathology Detail (OLD §6.4, line 560)
OLD: *"SnapKV_Original, SnapKV_Grouped, and H2O hallucinate 'Quantum Entanglement' as the passphrase across all budgets — a Phi-3 attention-sink pathology where these policies over-concentrate retention on uninformative tokens, fully evicting the needle."*

The new §7.4 compresses this entire finding to a 5-line paragraph with **no mention of the Quantum Entanglement example**, **no ranking table** (OLD Table 6 had explicit per-policy per-task numbers), and **no failure analysis**. The Phi-3 section is the **most degraded** of all cross-architecture sections. **Mistake** — the OLD Phi-3 evaluation was the most thorough analysis in the paper.

### 9. Section 5.4 "Cross-Context Scaling" (OLD lines 326–341) with 16K Detailed Table
OLD §5.4 had a **per-budget table** for KiaOmni_Gaussian, KiaOmni_σ8, SnapKV_Modified, H2O, SnapKV_Original, FullContext at B ∈ {96, 128, 256, 512}. The new paper's §6.2 (RULER NIAH) is **shrunken** and only has a 2-column table (Qwen vs Mistral, just B=64/B=96). The 16K context data is **gone**.

### 10. The "Discovery Note" Inside the Background (OLD §2.2, line 62)
OLD: *"All evaluations prior to Experiment 033-RealSnap used a simplified SnapKV baseline (snapkv_keep) implementing block-mean page eviction — missing the observation window, voting sum, and per-head union. This was identified through a line-by-line audit against the official repository…"*

This is an **important scientific-reproducibility disclosure** — admitting that early results used a faulty baseline. The new paper **retains only the kernel of this** (line 59 of new) but loses the "Discovery Note" framing that made it a strength-of-the-paper demonstration of self-correction.

### 11. The "0.59 TPS" Throughput Floor (OLD Abstract + §5.5)
OLD: *"FullContext: 0.59 TPS"* as the dramatic 32K context number.
NEW: *"throughput (which drops as low as 0.50 TPS)"* — actually **strengthens the metric** (lower floor = stronger case for eviction) and adds a **range** ("0.50 – 5.68"). Net **improvement** in rhetorical impact.

### 12. SnapKV_Original vs RealSnapKV Naming (OLD line 338, 537)
OLD used **"SnapKV_Original"** in some tables to distinguish the truly faithful implementation. NEW consolidates to **"RealSnapKV (faithful)"**. Minor renaming, but it means a reader looking for "SnapKV_Original" in the new paper won't find it — the new paper has renamed it for clarity, which is **arguably an improvement**.

### 13. Cross-architecture details: Ada-SnapKV Anomaly, BioMistral F1 Breakdown, Mistral NIAH scoring
- OLD §5.6 had a note on **Ada-SnapKV degradation at B=512 on Falcon3** (0.219 < 0.223) — gone from new
- OLD §5.8 had a **NOISE column** in BioMistral hallucination table — new collapses to CORRECT/HALLUCINATED/REFUSED/NOISE only at B=96
- OLD §5.1 had detailed Mistral-7B-Instruct-v0.3 NIAH scores (81.7%) preserved in new

---

## 🟢 THE NEW ADDITIONS
*(Everything entirely new in the REVISED paper)*

### 1. Title Change: "Boxcar" → "Gaussian" as Primary Identity
- **OLD title:** *"KiaOmni: O(N) Boxcar Smoothing for Budget-Exact KV-Cache Eviction in Large Language Models"*
- **NEW title:** *"KiaOmni-Gaussian: Smoothed Saliency Selection for Long-Context KV-Cache Eviction"*

The **brand pivot** is fundamental — the new paper markets the **Gaussian kernel** (σ=4) as the default, demoting the boxcar (σ=8) to "dependency-free production fallback." This is a **strategic narrative shift** that aligns with the cross-model mean result (KiaOmni_Gaussian wins the §6.1 table).

### 2. Signal-Swap Causal Analysis — New §4 (NEW lines 133–167, 35 lines)
**The single most important new section.** Experiment 039 is a 4-condition causal experiment:
| Condition | Saliency Signal | Selection Mechanism |
|---|---|---|
| KiaOmni_natural | Mean attention | KiaOmni_σ8 (boxcar+top-K) |
| KiaOmni_swapped | SnapKV voting | KiaOmni_σ8 (boxcar+top-K) |
| SnapKV_natural | SnapKV voting | RealSnapKV (per-head+union) |
| SnapKV_swapped | Mean attention | RealSnapKV (per-head+union) |

Headline result: **SnapKV_swapped reaches 1.000 NIAH-single B=256** — perfect retrieval — when given KiaOmni's mean-attention signal. The OLD paper never had this experiment. **Major addition** that turns the paper from "we beat SnapKV" into "we proved our saliency signal is the *causal* driver."

### 3. Zero-Training Value Proposition — New §10 (NEW lines 466–479, 14 lines)
A new section framing KiaOmni as a product:
| Aspect | KiaOmni | Training-Aware Methods |
|---|---|---|
| Setup cost | One function call | $1,000–$50,000 per model |
| Data required | None | Training corpus + validation set |
| Cross-model transfer | Drop-in | Re-train per architecture |

This is **purely a marketing/business framing** — does not exist anywhere in the OLD paper. It reframes the lack of training (which the OLD paper treated neutrally) as the **key selling point**. **Risk**: this is the kind of section reviewers may flag as non-rigorous for a research venue.

### 4. Statistical Summary Table with Multiple-Comparison Correction — New §11 (NEW lines 484–495, 12 lines)
A consolidated 7-row table of all primary claims with **N, test, p-value, and significance flag**:
- Adds a **Wilcoxon signed-rank test (p≤0.01)** that did not appear in the OLD paper
- Adds a **"Signal-swap causal destruction" p<10⁻¹⁰** row (linked to the new Experiment 039)
- Adds an explicit **Benjamini-Hochberg FDR correction** for Table 1's 16 comparisons at q<0.05

The OLD paper had a similar table (D.9, lines 778–789) but it was **buried in Appendix D** and lacked the Wilcoxon and FDR elements. The new paper **promotes** this to §11 main-text status — **clear improvement**.

### 5. Explicit Gaussian Kernel Formula — New §3.1 Step 2 (NEW lines 98–101)
OLD had only the boxcar prefix-sum formula. NEW adds:
$$F = E * G_\sigma, \quad G_\sigma(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)$$

This is the **mathematical upgrade** that justifies the new title. The OLD paper's §3.1 mentioned "or Gaussian" parenthetically but never wrote down the formula. The new paper treats Gaussian as a co-equal peer of boxcar. **Net improvement** for completeness.

### 6. Practical Deployment Code Snippet — New §3.4 (NEW lines 122–129, 8 lines)
```python
from kiaomni import apply_kiaomni
apply_kiaomni(model, policy="kiaomni_gaussian", budget=256)
```
Plus an `ArchitectureProbe` description: *"walks the module tree at apply-time, classifies QKV layout (separate / fused-concat / fused-interleaved), detects positional encoding (RoPE / ALiBi / learned)."*

This is **code-level content from the published `kiaomni/` package** (per CHANGELOG.md v0.2.5) that does not exist in the OLD paper. The OLD paper was purely algorithm-focused. **Improvement** for reproducibility.

### 7. Prefill-Phase Scope Discussion — New §2.3 (NEW lines 62–63, 2 lines)
A short but important methodological disclosure: *"KiaOmni operates in the prefill phase: saliency is computed once from the last query's attention distribution before any decode step."* This justifies the prefill-only scope and explicitly excludes PyramidKV/AdaKV/CAKE from comparison as orthogonal.

The OLD paper had a 3-line version of this in §2.3 and a more elaborate version in Appendix B (D.5). NEW consolidates into a clean §2.3 with **one** clean disclaimer.

### 8. Full Model-Suite Table — New §5.1 (NEW lines 173–181, 9 lines)
| Model | Architecture | Heads | Context |
|---|---|---|---|
| Qwen2.5-7B-Instruct | GQA (28 layers) | 28 Q / 4 KV | 32K |
| Mistral-7B-Instruct-v0.3 | Sliding-window MHA | 32 | 32K |
| Falcon3-7B-Instruct | GQA (28 layers) | 12 Q / 4 KV | 32K |
| BioMistral-7B-DARE | MHA (biomedical fine-tune) | 32 | 8K |

OLD §5.6 had some of this in prose. NEW formalizes as a table at §5.1. **Improvement** for scanability.

### 9. Reference: Megiddo & Modha ARC Paper — NEW Reference #12
- NEW adds: *"ARC: A Self-Tuning, Low Overhead Replacement Cache. FAST 2003"* (the Megiddo ARC paper).
- OLD does not cite it.

This is a **literature gap** the OLD paper had — H2O and SnapKV were cited but their ancestor in cache-replacement theory (ARC) was not. **Improvement** for completeness of related work.

### 10. Reference: Qwen2.5 Technical Report
- NEW adds: *"Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115, 2025."* (Reference #8)
- OLD cites Qwen2.5 implicitly in §5.6 prose but not in the References section.

### 11. Reference: FlexGen
- NEW adds: *"Sheng et al. FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU. arXiv:2303.06865"* (Reference #14).
- OLD does not cite FlexGen.

### 12. Updated Experimental Provenance Footer — NEW Line 564
*"Experiments 001–039 · Platforms: Kaggle T4×2, Modal A10G/A100/L4, Lightning AI L4"*
vs OLD: *"Experiments 001–038"* — adds **Experiment 039 (signal-swap)**.

### 13. The "throughput can drop as low as 0.50 TPS" claim — NEW Abstract
NEW: *"restoring baseline throughput (which drops as low as 0.50 TPS)"*
OLD: *"0.59 TPS"*

The new paper **lowers the worst-case number** (0.50 vs 0.59), making the eviction case stronger. Whether this is a new measurement or a re-characterization is unclear.

### 14. The "3.2× – 31×" Speedup Range — NEW §6.9 (lines 383–385)
OLD had a single ~31× number at 32K. NEW has a **range** "3.2× – 31×" to capture both 4K and 32K contexts. **Improvement** in honesty/transparency.

### 15. The "51% VRAM reduction" — NEW Abstract
OLD: *"2× VRAM reduction"* (i.e., 50%)
NEW: *"51% VRAM reduction"*

Mathematically the same (5.57 GB / 11.27 GB = 0.494). NEW is just more precise. **Cosmetic** improvement.

### 16. The "Cross-Model Replication" with Llama-3.1 — NEW §9 (line 459)
NEW: *"Cross-model replication (Qwen, Mistral, Llama-3.1, Phi-3) rules out the artifact hypothesis."*
OLD §6.5: 3 models (Qwen, Mistral, Phi-3) only. NEW adds **Llama-3.1** to the cross-model replication list, citing Experiment 035.

### 17. The "Drop Earlier Taxonomy Claims" — NEW §8 (line 444)
The OLD paper treated attention entropy as a predictive theory. NEW §8 reframes it: *"We drop earlier claims of 'architecture taxonomy' as a precise theory and reframe it as an empirical regularity."* This is **epistemic honesty** — admitting a former claim is no longer being defended as a precise theory.

---

## 🔄 THE METAMORPHOSIS
*(How the story, framing, and metrics shifted between the two versions)*

### M1. From "Boxcar-Only with Gaussian Parenthetical" → "Bilateral Kernel Family with Gaussian Default"
- **OLD:** σ=8 boxcar is *the* algorithm; Gaussian mentioned parenthetically.
- **NEW:** σ=4 Gaussian is *the* default; σ=8 boxcar is the *fallback*.

**Implication:** The cross-model Table 1 winner (KiaOmni_Gaussian @ 88.2%) becomes the headline, not a side note. The OLD paper buried the Gaussian story behind a "where each is preferred per architecture" hedge.

### M2. From "Defensive" Tone → "Assertive" Tone
The OLD paper reads like a **paper under siege**:
- Every major claim is defended in Appendix D
- The intro has a "Reviewer Note" warning about an underpowered predecessor
- The architecture taxonomy is presented and then immediately hedged
- Scissorhands PPL is openly discussed as catastrophic (302–411)
- Discards are listed in Appendix C

The NEW paper reads like a **paper ready to ship**:
- Appendix D is **completely removed**
- The signal-swap causal experiment is the new defense
- Architecture taxonomy is **rebranded as empirical regularity** rather than defended
- PPL pathology is **compressed to one sentence** in Limitations
- Discards are **not listed** anywhere
- The "Zero-Training Value Proposition" is a **product pitch**

**Tone shift:** from *defensive scientist* to *confident product-builder*. Reviewers may interpret this as either **maturity** (the authors have processed the feedback) or **overconfidence** (the defensible guardrails are gone).

### M3. From "Hybrid 4-Camp Family (σ8, Gaussian, Adaptive, Scissorhands)" → "Gaussian as Hero, Others as Ablations"
- **OLD §1 abstract:** *"KiaOmni_σ8 achieves 100% needle-in-a-haystack retrieval"* — the boxcar is the headline.
- **NEW §1 abstract:** *"A signal-swap causal experiment proves that the smoothing kernel — not the selector mechanism — is the causal driver"* — the *kernel choice* is the headline, not σ=8 specifically.

**Implication:** The OLD paper had four "KiaOmni" variants that competed with each other (KiaOmni_σ8, KiaOmni_Gaussian, KiaOmni_Adaptive, KiaOmni_Scissorhands). The NEW paper **promotes Gaussian and demotes Scissorhands/Adaptive to "ablations"** explicitly (line 225): *"We present KiaOmni-Gaussian as the primary method (§3) and all other variants as ablations (§7)."*

### M4. From "Architecture Taxonomy" → "Budget-Dependent Architecture Claim"
The OLD paper's §4 is titled **"Architecture Taxonomy"** with 4 profiles and a Gini-coefficient predictor. The NEW paper's §8 is titled **"Budget-Dependent Architecture Claim"** and explicitly disowns the predictor:
- OLD line 142: *"the taxonomy predicts optimal σ from the Gini coefficient of the attention weight distribution"*
- NEW line 444: *"We drop earlier claims of 'architecture taxonomy' as a precise theory and reframe it as an empirical regularity."*

The σ_max column (160, 64–96, etc.) is **gone** from the new table. The H_norm formula is gone. **Net change:** from "we can predict σ from attention profile" → "all models improve with budget; Falcon3 needs more budget because its attention is more diffuse."

### M5. From "PPL Pathology" → "PPL Transparency Acknowledgment"
- **OLD §5.7** has a **dedicated 6-row PPL table** (FullContext 7.46, KiaOmni_Gaussian 27.8, KiaOmni_σ8 36.3, SnapKV_Modified 52.1, H2O 220.4, KiaOmni_Scissorhands 302–411, RealSnapKV 192.6).
- **NEW §12** has a **single sentence** in Limitations: *"KiaOmni at B=512 achieves PPL=27.8 vs FullContext 7.46 on WikiText-2... KiaOmni-Gaussian's 3.73× is 8× better than H2O (29.5×) and 1.4× better than BlockSal (6.99×)."*

**Risk:** The Scissorhands PPL 302–411 number is the most embarrassing metric in the entire paper. By collapsing it, the new paper avoids drawing reviewer attention to it. But the underlying data still exists in the codebase. **Net change:** defensive omission vs. full disclosure.

### M6. From "Experiment 029–038" → "Experiment 001–039"
The OLD paper's footer (line 794) lists *"Experiments 001–038"*. The NEW paper (line 564) lists *"Experiments 001–039"*. This reflects:
- **Gained:** Experiment 039 = Signal-Swap Causal Analysis
- **Lost:** No explicit experiments dropped, but Appendix A (the experiment inventory table) is **gone**

### M7. From "Appendix A/B/C/D" → "Sections 1–13"
OLD: 9 main sections + 4 appendices (A inventory, B excluded baselines, C discards, D reviewer defense) = 13 logical units.
NEW: 13 main sections (1 Intro through 13 Conclusion) + 1 References = 14 logical units, **zero appendices**.

**Trade-off:** OLD was longer with the appendices; NEW is more compact but loses: (a) experiment inventory, (b) excluded-baselines discussion, (c) discarded claims, (d) reviewer defenses. The first three are *content losses*; the fourth is a *strategic loss*.

### M8. From "Multiple Cross-Architecture Notes" → "One Consolidated Cross-Arch Section"
- OLD §5.6 (Falcon3, 3 pages) and §5.8 (BioMistral, 2 pages) — each model has its own detailed section
- NEW §6.7 (Falcon3) and §6.8 (BioMistral) — each is now ~1 page

The BioMistral section is the **most compressed**: OLD §5.8 had 12 rows in Table 12 (CORRECT/HALLUCINATED/REFUSED/NOISE at B=96), NEW §6.8 has only 3 policies at the top of the table (Ada-SnapKV, KiaOmni-Gaussian, KiaOmni-σ8) and no per-category breakdown. The "RealSnapKV 48.6% HALLUCINATED at B=96 — strongest safety finding" is preserved in the prose, but the data is gone.

### M9. From "8 LongBench tasks" → "6 LongBench tasks" (in §6.4)
- **OLD §5.2 (Experiment 031):** 6 tasks (qasper, hotpotqa, multifieldqa_en, narrativeqa, 2wikimqa, musique)
- **NEW §6.4 (Experiment 031):** Same 6 tasks

**However**, NEW §5.2 (Benchmarks) says LongBench has **8 tasks** (adds gov_report, qmsum). This 8-task set is what was used in Experiment 033 (Hallucination) and Experiment 038 (BioMistral). So the **8-task set is mentioned in setup but the F1 table uses 6 tasks** — same as OLD. **No change, but a clarity improvement**: NEW §5.2 explicitly distinguishes "6 tasks" (F1) from "8 tasks" (judge + hallucination).

### M10. From "Underspecified Gaussian" → "Mathematically Rigorous Gaussian"
- **OLD §1** (line 29): *"KiaOmni instantiates this field with a rectangular (boxcar) kernel of half-width σ, computed in O(N) via prefix sums"*
- **NEW §1** (line 27): *"KiaOmni instantiates this field with a Gaussian kernel (σ=4) or a rectangular boxcar kernel (half-width σ=8), computed in O(N) via prefix sums"*

The OLD paper treated Gaussian as an **afterthought**; the NEW paper **inverts the priority**.

### M11. From "Subword Gap as σ=0 Failure Mode" → "Subword Gap as Bipartisan Mechanism"
- **OLD §2.4** explicitly framed subword gap collapse as a *failure mode of σ=0 methods*
- **NEW §2.4** reframes it as a *mechanism that σ=8 solves*: *"σ=8 fills the intra-token gaps that σ=0 leaves open"*

**Net:** rhetorically stronger for KiaOmni (the problem exists; we solve it) but loses the *negative result* (σ=0 is bad) which was useful for the paper's contribution framing.

### M12. From "Hard Constraint BLOCK_SIZE ≤ B÷4" → "No Hard Constraint Mentioned"
- **OLD §3.1 line 113:** *"Hard constraint: BLOCK_SIZE ≤ B÷4 (ensures budget feasibility)"*
- **NEW §3.1 line 112:** *"Fixed hyperparameters: SINK=16, RECENCY=32, σ=8 (boxcar) or σ=4 (Gaussian). No per-model calibration required."*

**Risk:** A reviewer could ask "what if B=32?" — the OLD paper had a hard answer; the NEW paper does not.

### M13. From "13 References" → "14 References"
- NEW adds: Megiddo ARC, Qwen2.5 Technical Report, FlexGen
- NEW does not drop any references

The Reference #5 in NEW is **RULER** with venue "arXiv:2404.06654, 2024" (no venue specifier) vs OLD #8 with the same. **Cosmetic.**

---

## ⚠️ CONTRADICTIONS & CLASHES
*(Any place where the old paper says "X is false/bad" and the new paper says "X is true/good", or vice versa)*

### C1. SnapKV_Original: From "Failed Baseline" → "Renamed as RealSnapKV (faithful)"
- **OLD §5.6 line 396:** Table 9 lists "KiaOmni_Scissorhands 0.235" at B=512, and SnapKV_Original (called "RealSnapKV" in some places) is bottom-tier.
- **OLD line 537:** Lists "SnapKV_Original" in the bf16 ablation with judge score 0.290.
- **NEW line 217:** Renames it "RealSnapKV (faithful)" and labels it the *honest* implementation, dropping SnapKV_Original entirely.

**Interpretation:** The OLD paper sometimes used SnapKV_Original to mean "the original SnapKV repo implementation" and sometimes RealSnapKV to mean "our faithful reimplementation of arXiv:2404.14469." The new paper **standardizes on RealSnapKV (faithful)** — this is a *naming correction*, not a contradiction. But the new paper **adds a softening disclaimer** at §2.2 (line 57): *"The only deviation from the official repo is using mean-head saliency as a principled tiebreaker in the trim step rather than index-order truncation — strictly more principled, not less faithful."* The OLD paper did not make this clarification.

### C2. SnapKV_Modified/BlockSal: From "Our Own Design" → "Faithful Block-Level Variant"
- **OLD §2.2 line 60:** *"BlockSal (formerly 'SnapKV_Modified'): A novel baseline of our own design — block-level KV selection using mean saliency per block."*
- **NEW §2.2 line 59:** *"SnapKV_Modified (renamed BlockSal): A block-level design of our own — mean saliency per block of 16 tokens, observation window scaled proportionally with context length."*

Both describe it as "of our own design" but the new paper **adds the phrase "observation window scaled proportionally with context length"** — a refinement. **Net change:** clarification, not contradiction.

### C3. Architecture Taxonomy: From "Heuristic Predictor" → "Empirical Regularity"
- **OLD line 142:** *"The taxonomy is a useful heuristic for ordering candidates, not a precise predictor of absolute performance at all budget levels."*
- **NEW line 444:** *"We drop earlier claims of 'architecture taxonomy' as a precise theory and reframe it as an empirical regularity."*

**Contradiction type:** Rhetorical *regression*. OLD said "heuristic predictor" (a *useful tool*); NEW says "empirical regularity" (a *post-hoc observation*). The new paper has weakened its own scientific claim. This may be **strategic humility** in response to the Phi-3 falsification, or it may be **overshooting** in retraction.

### C4. H_norm Formula Disappearance
- **OLD §4.1 lines 148–153:** Full formula `H_norm = -Σ p_i log p_i / log N` and `σ = σ_max · (1 − H_norm) · √(B/N)` presented as a **practical deployment tool**.
- **NEW:** Formula not mentioned anywhere. H_norm variable does not appear.

**Contradiction type:** Silent retraction. The OLD paper said "this is a useful formula"; the NEW paper says nothing. Reviewers will ask "where is the adaptive σ from your earlier work?" and the new paper has no answer.

### C5. Phi-3 Failure Mode: From "Detailed Table" → "5-Line Paragraph"
- **OLD §6.4 (lines 545–562):** 6-row Phi-3 table with NIAH-Single, NIAH-Multi, VT, LongBench ROUGE-L, Overall. Includes the "Quantum Entanglement" pathology.
- **NEW §7.4 (lines 426–430):** 5-line paragraph. NIAH-Single numbers (0.972, 0.967) preserved. **No VT, no ROUGE-L, no Quantum Entanglement mention.**

**Contradiction type:** Coverage shrinkage. The OLD paper was thorough; the NEW paper is *less* thorough on the very architecture where its own theory was falsified. **This is the most reviewer-vulnerable section** of the new paper.

### C6. Scissorhands PPL: From "Catastrophic Failure" → "Compressed to 1 Sentence"
- **OLD §5.7 Table 10:** KiaOmni_Scissorhands PPL 302–411 across budgets. Discussion of retrieval/fluency tradeoff. Listed as Limitation #7.
- **NEW §12 Limitation #2:** PPL=27.8 mentioned for KiaOmni-Gaussian. **No mention of Scissorhands 302–411.** The 3.73× comparison (KiaOmni-Gaussian vs H2O) actually *hides* that Scissorhands is the worst by far.

**Contradiction type:** Omission. The OLD paper was honest about a real failure mode; the NEW paper is silent on it. Reviewers who read the codebase will discover it and may flag it as suppression.

### C7. H2O in the Falcon3 Table: From "10-Policy Table" → "5-Policy Table"
- **OLD §5.6 Table 9:** 10 policies listed (KiaOmni_Gaussian, KiaOmni_Quest, KiaOmni_σ8, SnapKV_Modified, KiaOmni_RatioAdaptive, KiaOmni_Adaptive, KiaOmni_AnchorExp, Ada-SnapKV, H2O, KiaOmni_Scissorhands, RealSnapKV). 11 rows total.
- **NEW §6.7 Table 8:** 5 policies listed (KiaOmni-Gaussian, KiaOmni-σ8, BlockSal, H2O, RealSnapKV). 5 rows.

**Contradiction type:** Coverage shrinkage. The NEW paper drops 6 policies (KiaOmni-Quest, KiaOmni-RatioAdaptive, KiaOmni-Adaptive, KiaOmni-AnchorExp, Ada-SnapKV, KiaOmni-Scissorhands) from the Falcon3 specific table. They are still present in the global Table 1 but the per-model granularity is gone. **This makes BioMistral and Falcon3 feel like an afterthought** in the new paper.

### C8. Causal Claim Strength: From "Mechanism Visualization" → "Signal-Swap Experiment"
- **OLD §6.2 (line 507):** "Panel visualizations of σ=0 vs σ=8 decisions" — *qualitative* mechanism evidence.
- **NEW §4 (lines 133–167):** "Signal-swap causal experiment" with p<10⁻¹⁰ binomial test — *quantitative* causal evidence.

**No contradiction** — strictly stronger. The signal-swap experiment is a **major upgrade** in rigor. The OLD paper's qualitative visualizations are gone (no §6.2 in new).

### C9. H_norm Predictive Power
- **OLD §4.1:** *"Trace-level evaluation showed KiaOmniAdaptive outperforms fixed σ=8 on Qwen (+3.5pp) and Mistral (+1.6pp) at budget B=40"*
- **NEW:** No such claim. The Adaptive variant appears in Table 1 (line 213) but is not motivated or explained.

**Contradiction type:** Silent removal of a positive result. The OLD paper claimed KiaOmniAdaptive was better on Qwen and Mistral; the NEW paper does not reproduce this finding and the variant has no algorithmic justification in the new text.

### C10. Mistral 65.3% Hallucination as Inconvenient Fact
- **OLD §5.2b line 267:** *"Under FullContext, Mistral hallucinates at 65.3% and Falcon3 at 72.2% — both models generate fluent confident text regardless of whether the answer is in context."*
- **NEW §6.5 line 313:** *"Mistral-7B (65.3% FC hallucination) and Falcon3-7B (72.2%) have base hallucination rates so high that policy-level comparison is uninformative."*

Both papers make the same exclusion argument, but the **NEW paper's wording is more pejorative** ("uninformative" vs. "not the dominant variable"). This is a **framing tightening** — the new paper wants to shut down questions about Mistral/Falcon3 hallucination more firmly.

### C11. "Single Layer Saliency" → "ArchitectureProbe"
- **OLD §8 Limitation #3:** *"KiaOmni uses only the last transformer layer's attention. Multi-layer saliency aggregation (KiaOmni_Scissorhands) outperforms single-layer on multi-key retrieval tasks."*
- **NEW §3.4 line 129:** *"The ArchitectureProbe walks the module tree at apply-time, classifies QKV layout (separate / fused-concat / fused-interleaved), detects positional encoding (RoPE / ALiBi / learned), and supports all HuggingFace causal LMs."*

**No direct contradiction** — but the new paper **adds an implementation claim** (ArchitectureProbe supports fused-concat/interleaved QKV) that the OLD paper did not make. This is a claim about code-level capability that may or may not be true (the actual `kiaomni/adapters/probe.py` is in the repo). **Risk:** if the code doesn't actually support fused QKV, this is a falsifiable claim.

### C12. Open-Design Loss: The "Two Recommended Defaults"
- **OLD Abstract line 15:** *"The two recommended defaults — σ=8 (boxcar, dependency-free) and Gaussian (σ=4) — require no per-model calibration"*
- **NEW Abstract line 15:** *"We recommend KiaOmni-Gaussian (σ=4) as the default, with KiaOmni-σ8 (boxcar) as the dependency-free production fallback."*

**The priority flipped.** OLD: σ=8 first. NEW: σ=4 first. The OLD paper was σ=8-centric; the NEW paper is Gaussian-centric. **This is the most significant rhetorical mutation** in the entire paper.

### C13. Reading Order: From "σ=8" → "From-Gaussian Angle"
- **OLD §1 line 31–33:** *"1. Recovers pointwise selection at σ=0. 2. Recovers block-level selection at σ = BLOCK_SIZE/2. 3. Achieves optimal retrieval at intermediate σ values."*
- **NEW §1 line 27:** *"This construction recovers pointwise selection at σ=0 and block-level selection at σ = BLOCK_SIZE/2, while achieving optimal retrieval at intermediate σ values."*

The OLD paper presented these as **three claims**. The NEW paper collapses the third claim into a single sentence and removes the emphasis on "intermediate σ". This aligns with the Gaussian-first framing — the NEW paper doesn't need to argue for "intermediate σ" because the Gaussian is at σ=4 (intermediate), not the boxcar's σ=8.

### C14. Cross-Context Scaling: From "10-Policy" to "5-Policy"
- **OLD §5.4 Table at 16K context:** 6 policies (KiaOmni_Gaussian, KiaOmni_σ8, SnapKV_Modified, H2O, SnapKV_Original, FullContext) × 4 budgets
- **NEW §6.2 Table 2 (NIAH-Single):** 4 policies (KiaOmni-σ8, SnapKV, H2O, FullContext) × 2 architectures

**Net:** The OLD paper's §5.4 had a **table for 16K context**; the NEW paper **does not have a 16K-specific table** — that data is gone. The closest analog is the new §6.2 which is at ctx=16,384 but only shows 2 budgets (B=64, B=96).

### C15. Old §2.2 "Discovery Note" Disappearance
- **OLD §2.2 line 62 (entire "Discovery Note" callout):** 9-line block admitting that early results used a *simplified* SnapKV baseline and that a line-by-line audit was needed.
- **NEW §2.2:** A 2-line "We implement baselines as faithful approximations" with no Discovery Note.

**Contradiction type:** Loss of self-corrective narrative. The OLD paper used the Discovery Note to **showcase scientific rigor** (we caught our own mistake). The NEW paper removes this, leaving the impression that no such audit ever happened.

---

## 🟡 OVERALL ASSESSMENT

| Dimension | OLD Paper | NEW Paper | Net |
|---|---|---|---|
| Length | 824 lines | 565 lines | −31% |
| Title focus | Boxcar (σ=8) | Gaussian (σ=4) | **Pivot** |
| Defensive posture | 8 reviewer defenses (Appendix D) | 0 (none) | **Risk** |
| Architecture scope | 5 models (incl. SmolLM, TinyLlama) | 4 models (Qwen/Mistral/Falcon3/BioMistral) | **−1 model** |
| Causal evidence | Qualitative visualizations | Signal-swap binomial test (p<10⁻¹⁰) | **+Major upgrade** |
| PPL disclosure | Full table (Scissorhands 302–411) | 1 sentence | **−Honesty** |
| Adaptive σ policy | Full H_norm formula | Mentioned in Table 1, not explained | **−Depth** |
| Statistical rigor | Appendix D.9 table | §11 main-text + FDR correction | **+Upgrade** |
| References | 13 | 14 (+ARC, +FlexGen, +Qwen2.5) | **+Completeness** |
| Product framing | None | "Zero-Training Value Proposition" §10 | **+New** |
| Discards | Appendix C explicit list | None | **−Transparency** |
| BlockSal coverage | Renaming story + observation-window details | Compressed to 2 lines | **−Detail** |
| Phi-3 detail | Full 6-row table + Quantum Entanglement pathology | 5-line paragraph | **−Major shrinkage** |

---

## 📋 SUMMARY OF RECOMMENDATIONS

The NEW paper makes three **strategic upgrades** that are net improvements:
1. Gaussian-as-default realignment with the cross-model winner
2. Signal-swap causal experiment (the strongest addition)
3. Statistical summary in main text (not appendix)

And three **strategic downgrades** that are net losses:
1. **All 8 reviewer defenses deleted** — vulnerable to reviewer pushback
2. **Phi-3 analysis collapsed to 5 lines** — most falsification-vulnerable section
3. **KiaOmni_Adaptive policy orphaned** (formula gone, claim unchallenged)

### Critical Recoveries Needed

If the user wants a defensible submission, the most critical recoveries are:
- **D.1–D.8 reviewer defenses** (or a condensed "Limitations & Disclosures" §12 covering all 8)
- **The KiaOmni_Adaptive H_norm formula + Phi-3 detailed table** (in §7.4 or §8)
- **Scissorhands PPL 302–411 table** (move to §6.7 or expand Limitation #2)
- **The full Discovery Note from OLD §2.2** (move to §2.2 or §3 of new)

---

*Report generated: 2026-06-07*
*Comparison scope: KiaOmni_Paper.md (OLD) vs new_draft/KiaOmni_Revised_Paper.md (NEW)*
*Methodology: line-by-line forensic diff with section-level cross-referencing*
