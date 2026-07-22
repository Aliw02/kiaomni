"""
LLM-as-Judge for BioMistral predictions (038_biomistral_results/predictions.csv)

Four output categories:
  CORRECT      — prediction is right (exact or semantic match with ground truth)
  HALLUCINATED — non-empty, confident, but wrong answer
  REFUSED      — model explicitly says it cannot / doesn't know
  NOISE        — empty, echoes prompt instructions, generates new questions, off-topic

Strategy (cost-efficient):
  • Empty predictions       → auto NOISE   (no API call)
  • contains == 1.0         → auto CORRECT (no API call)
  • Prompt-echo / question  → auto NOISE   (regex, no API call)
  • Everything else         → LLM judge    (API call)

Saves results incrementally → fully resumable if interrupted.
Run: python llm_judge_biomistral.py
"""

import csv
import re
import time
from pathlib import Path

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = Path(
    "D:/MyFolder/ProgrammingWith-Python/Ai/A+/notebook/kv_cache_benchmark"
    "/038_biomistral_results/predictions.csv"
)
OUTPUT_PATH = Path(
    "D:/MyFolder/ProgrammingWith-Python/Ai/A+/notebook/kv_cache_benchmark"
    "/038_biomistral_results/llm_judge_results.csv"
)

# None = run all budgets (B=96, B=128, B=256, B=512)
BUDGET_FILTER: str | None = None

# Exclude tasks where FullContext itself is 100% hallucinated (annotation mismatch)
EXCLUDED_TASKS: set[str] = {"medalpaca_wiki", "pubmedqa_long"}

LIGHTNING_BASE_URL = "https://lightning.ai/api/v1/"
LIGHTNING_API_KEY  = "ae5690b5-b58f-4071-8880-69e403f5eab9"
MODEL = "anthropic/claude-haiku-4-5-20251001"   # Lightning model ID
MAX_RETRIES = 5
RETRY_DELAY = 5.0    # seconds between retries (doubles on each failure)
CALL_DELAY  = 4.0    # seconds between every API call (respects rate limit)
RATELIMIT_COOLDOWN = 30.0  # seconds to wait globally after any rate limit hit

# ── Regex auto-classifiers ────────────────────────────────────────────────────
PROMPT_ECHO_RE = re.compile(
    r"^(please\s+(do\s+not|provide|include|note|ensure|answer|give|state|list|specify)|"
    r"do\s+not\s+include|note\s+that\s+the|the\s+answer\s+should|your\s+answer\s+should)",
    re.IGNORECASE,
)
QUESTION_GEN_RE = re.compile(
    r"(\?\s+(what|who|where|when|how|which|why|please|is|are|was|were|do|does|did|can|could)\s)"
    r"|(what\s+is\s+the\s+\w+\?.*){2,}",
    re.IGNORECASE | re.DOTALL,
)

# ── Judge prompt ───────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a strict medical QA judge. You classify model predictions into exactly "
    "one of four categories. Reply with ONLY the category name — no explanation, no "
    "punctuation, just the single word."
)

JUDGE_TEMPLATE = """\
TASK: {task}
GROUND TRUTH: {ground_truth}
MODEL PREDICTION: {prediction}

Categories:
CORRECT      — prediction is right (exact match or clear semantic equivalent of ground truth)
HALLUCINATED — prediction is non-empty, sounds confident/fluent, but is factually wrong
REFUSED      — prediction explicitly signals inability to answer ("I cannot", "not mentioned", "N/A", etc.)
NOISE        — prediction is empty, repeats prompt instructions, generates new questions, or is completely off-topic

Reply with ONE WORD only: CORRECT, HALLUCINATED, REFUSED, or NOISE"""

VALID_LABELS = {"CORRECT", "HALLUCINATED", "REFUSED", "NOISE"}


def auto_classify(prediction: str, contains: float) -> str | None:
    """
    Returns a label without an API call when the answer is obvious,
    or None if LLM judgment is needed.
    """
    if contains == 1.0:
        return "CORRECT"
    pred = prediction.strip()
    if not pred:
        return "NOISE"
    if PROMPT_ECHO_RE.search(pred):
        return "NOISE"
    if QUESTION_GEN_RE.search(pred):
        return "NOISE"
    return None  # needs LLM


def judge_with_llm(client: OpenAI, row: dict) -> str:
    """Call Claude via Lightning.ai to classify a single prediction. Retries on failure."""
    prompt = JUDGE_TEMPLATE.format(
        task=row["task"],
        ground_truth=row["ground_truth"][:200],
        prediction=row["prediction"][:500],
    )
    delay = RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": [{"type": "text", "text": prompt}]},
                ],
            )
            label = completion.choices[0].message.content.strip().upper()
            if label in VALID_LABELS:
                return label
            print(f"  [warn] unexpected label {label!r}, defaulting to HALLUCINATED")
            return "HALLUCINATED"
        except Exception as exc:
            is_ratelimit = "rate limit" in str(exc).lower() or "429" in str(exc)
            if attempt < MAX_RETRIES - 1:
                wait = RATELIMIT_COOLDOWN if is_ratelimit else delay
                print(f"  [retry {attempt+1}] {'rate-limit' if is_ratelimit else 'error'} — waiting {wait:.0f}s")
                time.sleep(wait)
                delay *= 2
            else:
                print(f"  [error] giving up after {MAX_RETRIES} attempts: {exc}")
                return "ERROR"


def load_already_judged(output_path: Path) -> set[tuple]:
    """Load (source, task, ctx, trial, policy, budget) keys already saved."""
    if not output_path.exists():
        return set()
    done: set[tuple] = set()
    with open(output_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((
                row["source"], row["task"], row["ctx"],
                row["trial_or_sample"], row["policy"], row["budget"],
            ))
    return done


def row_key(row: dict) -> tuple:
    return (
        row["source"], row["task"], row["ctx"],
        row["trial_or_sample"], row["policy"], row["budget"],
    )


def main() -> None:
    client = OpenAI(
        base_url=LIGHTNING_BASE_URL,
        api_key=LIGHTNING_API_KEY,
    )

    # Load all predictions
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Apply filters
    rows = [
        r for r in all_rows
        if r["task"] not in EXCLUDED_TASKS
        and (BUDGET_FILTER is None or r["budget"] == BUDGET_FILTER)
    ]
    print(f"Loaded {len(all_rows)} total rows → {len(rows)} after filters")

    # Resume: skip already judged rows
    done_keys = load_already_judged(OUTPUT_PATH)
    print(f"Already judged: {len(done_keys)} rows — will skip these")

    # Open output file (append mode for resumability)
    output_exists = OUTPUT_PATH.exists()
    out_fields = list(all_rows[0].keys()) + ["judge_label", "judge_source"]
    out_file = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=out_fields)
    if not output_exists:
        writer.writeheader()

    # Process
    api_calls = 0
    auto_calls = 0
    errors = 0

    to_process = [r for r in rows if row_key(r) not in done_keys]
    total = len(to_process)
    print(f"Rows to judge: {total}\n")

    for i, row in enumerate(to_process, 1):
        label = auto_classify(row["prediction"], float(row["contains"]))
        source = "auto"

        if label is None:
            label = judge_with_llm(client, row)
            source = "llm"
            api_calls += 1
            if label == "ERROR":
                errors += 1
                time.sleep(RATELIMIT_COOLDOWN)  # cool down extra after any error
            else:
                time.sleep(CALL_DELAY)  # steady pace between successful calls
        else:
            auto_calls += 1

        out_row = dict(row)
        out_row["judge_label"] = label
        out_row["judge_source"] = source
        writer.writerow(out_row)
        out_file.flush()

        if i % 50 == 0 or i == total:
            print(
                f"  [{i}/{total}] api_calls={api_calls} auto={auto_calls} errors={errors}"
            )

    out_file.close()

    print(f"\nDone. Results saved to: {OUTPUT_PATH}")
    print(f"API calls made: {api_calls} | Auto-classified: {auto_calls} | Errors: {errors}")

    # ── Quick summary ──────────────────────────────────────────────────────────
    from collections import defaultdict

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    labels = ["CORRECT", "HALLUCINATED", "REFUSED", "NOISE", "ERROR"]

    # Per-budget summary: show CORRECT% and % of FullContext for each budget
    print("\n=== CORRECT% by Policy and Budget (% of FullContext in brackets) ===")
    budgets = sorted(set(r["budget"] for r in results), key=int)
    policies = sorted(set(r["policy"] for r in results))

    # Compute correct% per policy per budget
    bpol: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for r in results:
        bpol[r["budget"]][r["policy"]][r["judge_label"]] += 1

    hdr = f"{'Policy':<28} " + " ".join(f"B={b:>4}" for b in budgets)
    print(hdr)
    print("-" * len(hdr))
    for pol in policies:
        parts = []
        for b in budgets:
            counts = bpol[b].get(pol, {})
            total = sum(counts.values())
            fc_total = sum(bpol[b].get("FullContext", {}).values())
            fc_correct = bpol[b].get("FullContext", {}).get("CORRECT", 0)
            pol_correct = counts.get("CORRECT", 0)
            pct_fc = (100 * pol_correct / fc_correct) if fc_correct else 0
            pct_abs = (100 * pol_correct / total) if total else 0
            parts.append(f"{pct_abs:4.0f}%({pct_fc:3.0f}%)")
        print(f"{pol:<28} " + "  ".join(parts))

    print("\nFormat: abs_correct%(pct_of_FullContext)")

    # Full label breakdown at each budget
    for b in budgets:
        print(f"\n=== Full label breakdown — B={b} ===")
        header = f"{'Policy':<28} " + " ".join(f"{l:>14}" for l in labels) + f"  {'N':>5}"
        print(header)
        print("-" * len(header))
        for pol in policies:
            counts = bpol[b].get(pol, defaultdict(int))
            total = sum(counts.values())
            if total == 0:
                continue
            row_str = f"{pol:<28} " + " ".join(
                f"{100*counts.get(l,0)/total:>13.1f}%" for l in labels
            ) + f"  {total:>5}"
            print(row_str)


if __name__ == "__main__":
    main()
