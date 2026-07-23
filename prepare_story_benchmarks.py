import os
import json

STORY_PATH = r"d:\MyFolder\ProgrammingWith-Python\Ai\A+\ALGORITHM_STORY.md"

with open(STORY_PATH, "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
total_words = len(words)
print(f"Loaded ALGORITHM_STORY.md: {total_words:,} words, {len(text):,} chars.")

# Approximate Qwen tokens: ~1.38 tokens per word for markdown technical English
# Target words:
# 4K context  -> ~2,800 words
# 8K context  -> ~5,600 words
# 16K context -> ~11,200 words
# 32K context -> ~22,000 words (or full text)

words_4k = " ".join(words[:2800])
words_8k = " ".join(words[:5600])
words_16k = " ".join(words[:11200])
words_32k = text  # full text

out_dir = r"d:\MyFolder\ProgrammingWith-Python\Ai\A+\kiaomni_chat\static\sample_data"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "story_4k.txt"), "w", encoding="utf-8") as f:
    f.write(words_4k)

with open(os.path.join(out_dir, "story_8k.txt"), "w", encoding="utf-8") as f:
    f.write(words_8k)

with open(os.path.join(out_dir, "story_16k.txt"), "w", encoding="utf-8") as f:
    f.write(words_16k)

with open(os.path.join(out_dir, "story_32k.txt"), "w", encoding="utf-8") as f:
    f.write(words_32k)

print("Saved story_4k.txt, story_8k.txt, story_16k.txt, story_32k.txt.")

# Prepare expected answers JSON for all 4 context lengths
expected_story = {
    "4k": [
        {
            "id": "Q1",
            "question": "What priority formula pi(t) was used in the original TurboCD paper, and what were the default weights for alpha, beta, gamma?",
            "expected": "pi(t) = alpha*A_accum(t) + beta*(1/age(t)) + gamma*recency(t), defaults: alpha=0.6, beta=0.2, gamma=0.2"
        },
        {
            "id": "Q2",
            "question": "What initial constant priority score pi_init did every new token start with in Group A (A2), and why was this a cold-start problem?",
            "expected": "pi_init = 0.43 (0.6*0 + 0.2*1 + 0.2*1). New tokens started with constant 0.43 and were almost always rejected against established tokens."
        },
        {
            "id": "Q3",
            "question": "What critical simulation bias was identified in Group B (B2) regarding attn_weight and ground-truth importance?",
            "expected": "attn_weight was computed as beta(2 + importance*3, 5), giving H2O and TurboCD a hidden advantage by seeing ground-truth importance labels."
        },
        {
            "id": "Q4",
            "question": "List the 5 charges sustained by the Critic in Court Ruling Critical-07.",
            "expected": "A2 (Cold start 0.43), A3 (Gaussian threshold), B2 (Importance leakage), B4 (StreamingLLM missing), B5 (No error bars)."
        },
        {
            "id": "Q5",
            "question": "What was the initial paper score assigned by the Critic, and what did it change to after Round 2 fixes (Critical-09)?",
            "expected": "Initial score: 5/10 (reject). Round 2 fixes score: 7/10 (accept with minor revisions)."
        },
        {
            "id": "Q6",
            "question": "Summarize Part 1 and Part 2.5 of the TurboCD review in 5 bullet points.",
            "expected": "5 bullet points summarizing TurboCD metaphor, 17 charges, defense responses, court rulings, and paper score progression."
        }
    ],
    "8k": [
        {
            "id": "Q1",
            "question": "What priority formula pi(t) was used in the original TurboCD paper, and what were the default weights for alpha, beta, gamma?",
            "expected": "pi(t) = alpha*A_accum(t) + beta*(1/age(t)) + gamma*recency(t), defaults: alpha=0.6, beta=0.2, gamma=0.2"
        },
        {
            "id": "Q2",
            "question": "What critical simulation bias was identified in Group B (B2) regarding attn_weight and ground-truth importance?",
            "expected": "attn_weight was computed as beta(2 + importance*3, 5), giving H2O and TurboCD a hidden advantage by seeing ground-truth importance labels."
        },
        {
            "id": "Q3",
            "question": "Explain what D-005 (The Cold Start Problem) revealed about TurboCD's win/loss ratio across the 7 domains.",
            "expected": "TurboCD won only 2 of 7 domains (LLM_TUNE +3.4pp, CPU_SCHED +3.7pp) and lost 5 domains to H2O because the admission gate punished new tokens."
        },
        {
            "id": "Q4",
            "question": "What was the exact BM25-inspired formula for TF-Cache proposed by the authors in Defense-09, and what role did each term play?",
            "expected": "score(t) = log(1 + n(t))^alpha * 1/age(t)^(beta*0.5) * exp(-dt/tau). n=frequency (TF), age=decay (IDF), dt=forgetting."
        },
        {
            "id": "Q5",
            "question": "What optimal weights did grid search DW-1 find for TurboCD across ALL 7 domains, and what did this prove about TurboCD's priority gate?",
            "expected": "alpha=0, beta=0, gamma=1.0 (pure LRU) in ALL 7 domains. Proved TurboCD's priority gate adds no value over pure recency."
        },
        {
            "id": "Q6",
            "question": "Summarize the transition from TurboCD to TF-Cache in 5 bullet points.",
            "expected": "5 bullet points covering TurboCD flaws, cold-start problem, D-005 findings, TF-Cache BM25 formula, and DW-1 grid search."
        }
    ],
    "16k": [
        {
            "id": "Q1",
            "question": "What is the key distinction between AI/LLM workloads and Systems workloads identified in D-012 (Architectural Bifurcation)?",
            "expected": "AI/LLM workloads require Must-Admit (rejecting new tokens breaks autoregressive Markov chain). Systems workloads benefit from Admission Gates."
        },
        {
            "id": "Q2",
            "question": "What efficiency ratio eta did DualTime achieve in the NETWORK domain compared to the Belady Oracle (D-014)?",
            "expected": "eta = 0.997 (essentially optimal)."
        },
        {
            "id": "Q3",
            "question": "Explain the 6 dimensions of Workload DNA (D-016) and why ROCKETS was the hardest domain to optimize.",
            "expected": "TLI=0.038, PS=0.090, EWSR=35.3x capacity. ROCKETS was near-random with no exploitable locality or stability."
        },
        {
            "id": "Q4",
            "question": "What were the 3 algorithms designed in D-012 (VFCA, GRACERank, DualTime) and which domain did each win?",
            "expected": "DualTime (NETWORK, CPU_SCHED, DB_CACHE), GRACERank (EMBED_PRUNE, FINANCE), VFCA (penalty for bursty tokens)."
        },
        {
            "id": "Q5",
            "question": "Summarize the Phase 1 experimental findings and domain specialization in 5 bullet points.",
            "expected": "5 bullet points covering Must-Admit vs Admission Gate, Belady Oracle bounds, DualTime efficiency, Workload DNA, and domain specialization."
        }
    ],
    "32k": [
        {
            "id": "Q1",
            "question": "How did KiaCache evolve from ARC + EMA gate in D-021 to KiaCachePlusR in Critical-18 and KiaBeast in D-060?",
            "expected": "D-021 (ARC + EMA gate), Critical-18 (Critic conditions for KiaCachePlusR), D-060 (Formalized Log-Hybrid KiaCachePlusR2 as KiaBeast)."
        },
        {
            "id": "Q2",
            "question": "What were the final benchmark results of KiaOmni and Gaussian policies on RULER 032 and LongBench 031 in Kaggle evaluations (D-078)?",
            "expected": "KiaOmni and Gaussian achieved #1 rank in Variable Tracking (VT) and LongBench 031, beating FullContext@B=256 and Scissorhands #1 NIAH."
        },
        {
            "id": "Q3",
            "question": "Summarize the complete story of TurboCD to KiaOmni in 5 bullet points.",
            "expected": "5 bullet points covering TurboCD initial paper & review, simulation bug fixes & TF-Cache, KiaCachePlusR, Paged Attention validation, and KiaOmni continuous field supremacy."
        }
    ]
}

with open(os.path.join(r"d:\MyFolder\ProgrammingWith-Python\Ai\A+\kiaomni_chat\sample_data", "story_expected.json"), "w", encoding="utf-8") as f:
    json.dump(expected_story, f, indent=2)

print("Saved story_expected.json successfully.")
