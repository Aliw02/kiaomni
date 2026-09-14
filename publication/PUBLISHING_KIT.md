# KiaOmni Publication Kit

Prepared for arXiv endorsement outreach and Zenodo publication.

## Canonical paper

- Title: **KiaOmni: Gaussian and Boxcar Smoothing for Long-Context KV-Cache Eviction**
- Author: **Aliwey Abood**
- Affiliation: **Independent Researcher**
- Paper file: `paper/KiaOmni_Paper.pdf`

## Recommended arXiv classification

- Primary: `cs.LG` — Machine Learning
- Possible cross-list: `cs.CL` — Computation and Language

## Short abstract for submission forms

KiaOmni is a training-free KV-cache eviction method for long-context causal language-model inference. It smooths token-importance signals with Gaussian or boxcar kernels before cache selection and is implemented as a removable Hugging Face generation patch that does not require model retraining. Across four model architectures and eight LongBench tasks, the Gaussian variant retains 88.2% of FullContext judged correctness at a 512-token cache budget in the repository's cross-model evaluation, outperforming the included H2O and RealSnapKV baselines under the same evaluation framework. Additional experiments cover Needle-in-a-Haystack retrieval, passkey retrieval, perplexity, signal-swap ablations, multiple context lengths, and aggressive cache budgets. The implementation supports common Hugging Face attention backends and provides scripts and raw result artifacts for reproduction.

## Zenodo metadata

- Resource type: Publication / Preprint
- Title: KiaOmni: Gaussian and Boxcar Smoothing for Long-Context KV-Cache Eviction
- Creator: Aliwey Abood
- Affiliation: Independent Researcher
- Publication date: use the actual Zenodo publication date
- Publisher: Zenodo
- Language: English
- Keywords: KV cache; large language models; inference efficiency; long-context inference; cache eviction; attention; memory compression; Hugging Face Transformers
- Description: use the short abstract above
- Files to upload:
  1. `paper/KiaOmni_Paper.pdf`
  2. Optional source/release archive

## DOI workflow

If the PDF does not yet contain a DOI, reserve a Zenodo DOI before publication and insert it into the final PDF if desired. After publication, use the final DOI in the repository, CV, scholarly outreach, and arXiv metadata where appropriate.

## arXiv endorsement workflow

1. Start a new arXiv submission in `cs.LG`.
2. Continue until arXiv displays the endorsement requirement and provides the endorsement request code/link.
3. Use arXiv's eligible-endorser lookup for recent papers in the same category.
4. Contact 3-5 highly relevant eligible authors in parallel rather than waiting on one person.
5. Include the endorsement code/link, paper link, and repository link.
6. Ask only for category endorsement, not a detailed review or commercial discussion.

## Distribution after DOI/arXiv

Use the DOI/arXiv page as the canonical research link. Keep scholarly outreach separate from commercial licensing outreach. Company support inboxes should not be the main discovery channel for the paper.

## Claims discipline

- Report cross-model means with the stated evaluation framework.
- State when a result is LLM-judged.
- Do not describe a non-significant directional result as statistically significant.
- Keep FullContext as an oracle/reference, not as an eviction baseline.
- Report cases where another method wins rather than cherry-picking only KiaOmni wins.
