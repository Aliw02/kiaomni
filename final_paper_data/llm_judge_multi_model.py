"""
llm_judge_multi_model.py — LLM-as-Judge for Qwen7B, Mistral, Falcon3, Amber predictions
========================================================================================

Runs Claude Haiku via Lightning.ai on four experiment prediction files:
  033_full_comparison_results/predictions.csv    → Qwen2.5-7B
  034_mistral_results/data/predictions.csv       → Mistral-7B
  037_falcon3_results/predictions.csv            → Falcon3-7B
  040_amber_results/predictions.csv              → LLM360/Amber-7B

Four output categories (identical schema to 038 BioMistral judge):
  CORRECT      — prediction is right (exact or semantic match)
  HALLUCINATED — non-empty, confident, but factually wrong
  REFUSED      — model says it cannot / doesn't know
  NOISE        — empty, echoes prompt, generates questions, off-topic

Cost strategy:
  • contains == 1.0  → auto CORRECT  (no API call)
  • empty prediction → auto NOISE    (no API call)
  • prompt-echo / question-gen regex → auto NOISE (no API call)
  • everything else  → LLM judge     (API call)

RULER tasks (niah_single, niah_multikey, vt) are skipped — they have
exact-match ground truth so contains=1.0 / contains=0.0 is definitive.
Only LongBench tasks go to the LLM judge.

Fully resumable: appends to output CSV, skips already-judged rows.

Usage:
    python llm_judge_multi_model.py                  # judge all four models
    python llm_judge_multi_model.py --model qwen     # judge only Qwen7B
    python llm_judge_multi_model.py --model mistral  # judge only Mistral
    python llm_judge_multi_model.py --model falcon   # judge only Falcon3
    python llm_judge_multi_model.py --model amber    # judge only Amber-7B
"""

import argparse
import csv
import re
import time
from pathlib import Path

from openai import OpenAI

# ── Lightning.ai config ───────────────────────────────────────────────────────
LIGHTNING_BASE_URL = "https://lightning.ai/api/v1/"
LIGHTNING_API_KEY  = "ae5690b5-b58f-4071-8880-69e403f5eab9"
MODEL              = "anthropic/claude-haiku-4-5-20251001"

MAX_RETRIES        = 5
RETRY_DELAY        = 5.0    # seconds, doubles on each failure
CALL_DELAY         = 4.0    # seconds between every successful API call
RATELIMIT_COOLDOWN = 30.0   # seconds to wait after a rate-limit hit

# ── Experiment definitions ────────────────────────────────────────────────────
BASE = Path("D:/MyFolder/ProgrammingWith-Python/Ai/A+/notebook/kv_cache_benchmark")

EXPERIMENTS: dict[str, dict] = {
    "qwen": {
        "label":       "Qwen2.5-7B",
        "predictions": BASE / "033_full_comparison_results/predictions.csv",
        "output":      BASE / "033_full_comparison_results/llm_judge_results.csv",
    },
    "mistral": {
        "label":       "Mistral-7B",
        "predictions": BASE / "034_mistral_results/data/predictions.csv",
        "output":      BASE / "034_mistral_results/data/llm_judge_results.csv",
    },
    "falcon": {
        "label":       "Falcon3-7B",
        "predictions": BASE / "037_falcon3_results/predictions.csv",
        "output":      BASE / "037_falcon3_results/llm_judge_results.csv",
    },
    "amber": {
        "label":       "Amber-7B",
        "predictions": BASE / "040_amber_results/predictions.csv",
        "output":      BASE / "040_amber_results/llm_judge_results.csv",
    },
}

# ── Tasks to judge (RULER tasks have exact-match GT — skip them) ──────────────
RULER_TASKS = {"niah_single", "niah_multikey", "vt"}

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

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a strict QA judge. You classify model predictions into exactly "
    "one of four categories. Reply with ONLY the category name — no explanation, "
    "no punctuation, just the single word."
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def auto_classify(prediction: str, contains: float) -> str | None:
    """Returns a label without an API call, or None if LLM judgment is needed."""
    if contains == 1.0:
        return "CORRECT"
    pred = prediction.strip()
    if not pred:
        return "NOISE"
    if PROMPT_ECHO_RE.search(pred):
        return "NOISE"
    if QUESTION_GEN_RE.search(pred):
        return "NOISE"
    return None


def judge_with_llm(client: OpenAI, row: dict) -> str:
    """Call Claude Haiku via Lightning.ai to classify one prediction."""
    prompt = JUDGE_TEMPLATE.format(
        task=row["task"],
        ground_truth=str(row["ground_truth"])[:300],
        prediction=str(row["prediction"])[:600],
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
            print(f"  [warn] unexpected label {label!r} → defaulting to HALLUCINATED")
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


def load_done_keys(output_path: Path) -> set[tuple]:
    """Load row keys already saved to the output CSV."""
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


def print_summary(output_path: Path, label: str) -> None:
    """Print CORRECT% per policy per budget after judging is complete."""
    from collections import defaultdict

    with open(output_path, encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    if not results:
        print("  (no results to summarise)")
        return

    labels  = ["CORRECT", "HALLUCINATED", "REFUSED", "NOISE", "ERROR"]
    budgets = sorted(set(r["budget"] for r in results), key=int)
    policies = sorted(set(r["policy"] for r in results))

    bpol: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in results:
        bpol[r["budget"]][r["policy"]][r["judge_label"]] += 1

    print(f"\n=== {label} — CORRECT% by Policy and Budget (abs%(pct-of-FC)) ===")
    hdr = f"{'Policy':<28} " + " ".join(f"B={b:>4}" for b in budgets)
    print(hdr)
    print("-" * len(hdr))
    for pol in policies:
        parts = []
        for b in budgets:
            counts  = bpol[b].get(pol, {})
            total   = sum(counts.values())
            fc_corr = bpol[b].get("FullContext", {}).get("CORRECT", 0)
            pol_corr = counts.get("CORRECT", 0)
            pct_abs = (100 * pol_corr / total) if total else 0
            pct_fc  = (100 * pol_corr / fc_corr) if fc_corr else 0
            parts.append(f"{pct_abs:4.0f}%({pct_fc:3.0f}%)")
        print(f"{pol:<28} " + "  ".join(parts))

    print("\nFormat: abs_correct%(pct_of_FullContext)")

    for b in budgets:
        print(f"\n=== {label} — Full label breakdown B={b} ===")
        header = f"{'Policy':<28} " + " ".join(f"{l:>14}" for l in labels) + f"  {'N':>5}"
        print(header)
        print("-" * len(header))
        for pol in policies:
            counts = bpol[b].get(pol, defaultdict(int))
            total  = sum(counts.values())
            if total == 0:
                continue
            row_str = (
                f"{pol:<28} "
                + " ".join(f"{100 * counts.get(l, 0) / total:>13.1f}%" for l in labels)
                + f"  {total:>5}"
            )
            print(row_str)


# ── Per-experiment runner ─────────────────────────────────────────────────────

def run_experiment(client: OpenAI, exp_key: str, exp: dict) -> None:
    pred_path   = exp["predictions"]
    output_path = exp["output"]
    label       = exp["label"]

    print(f"\n{'='*65}")
    print(f"  Judging: {label}  ({pred_path.name})")
    print(f"{'='*65}")

    if not pred_path.exists():
        print(f"  [SKIP] predictions file not found: {pred_path}")
        return

    with open(pred_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Filter: LongBench tasks only (RULER has exact-match GT)
    lb_rows = [r for r in all_rows if r["task"] not in RULER_TASKS]
    print(f"  Total rows: {len(all_rows)} | LongBench only: {len(lb_rows)}")

    done_keys = load_done_keys(output_path)
    print(f"  Already judged: {len(done_keys)} | Remaining: {len(lb_rows) - len(done_keys)}")

    to_process = [r for r in lb_rows if row_key(r) not in done_keys]
    total      = len(to_process)
    if total == 0:
        print("  Nothing to judge — all done.")
        print_summary(output_path, label)
        return

    out_fields   = list(all_rows[0].keys()) + ["judge_label", "judge_source"]
    output_exists = output_path.exists()
    out_file = open(output_path, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(out_file, fieldnames=out_fields)
    if not output_exists:
        writer.writeheader()

    api_calls = auto_calls = errors = 0

    for i, row in enumerate(to_process, 1):
        label_val = auto_classify(row["prediction"], float(row["contains"]))
        source    = "auto"

        if label_val is None:
            label_val  = judge_with_llm(client, row)
            source     = "llm"
            api_calls += 1
            if label_val == "ERROR":
                errors += 1
                time.sleep(RATELIMIT_COOLDOWN)
            else:
                time.sleep(CALL_DELAY)
        else:
            auto_calls += 1

        out_row = dict(row)
        out_row["judge_label"]  = label_val
        out_row["judge_source"] = source
        writer.writerow(out_row)
        out_file.flush()

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] api={api_calls} auto={auto_calls} errors={errors}")

    out_file.close()
    print(f"\n  Done: {output_path}")
    print(f"  API calls={api_calls} | Auto={auto_calls} | Errors={errors}")
    print_summary(output_path, label)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(EXPERIMENTS.keys()),
        help="Judge only one model (default: all three)",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=LIGHTNING_BASE_URL, api_key=LIGHTNING_API_KEY, timeout=30.0)

    targets = (
        {args.model: EXPERIMENTS[args.model]}
        if args.model
        else EXPERIMENTS
    )

    for exp_key, exp in targets.items():
        run_experiment(client, exp_key, exp)

    print("\n\nAll done.")


if __name__ == "__main__":
    main()
