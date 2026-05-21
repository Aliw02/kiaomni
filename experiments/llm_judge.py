"""
llm_judge.py — LLM-as-Judge for KV-Cache Benchmark Predictions
===============================================================
Uses Claude Haiku via Lightning.ai to classify predictions into:
  CORRECT, HALLUCINATED, REFUSED, NOISE

Auto-classifies (no API cost):
  • contains == 1.0  → CORRECT
  • empty prediction → NOISE
  • prompt-echo / question-gen → NOISE
  • RULER tasks     → skipped (exact-match ground truth)
  • BioMistral excluded tasks → skipped

Usage:
    export LIGHTNING_API_KEY="your-key-here"
    python experiments/llm_judge.py                        # all models
    python experiments/llm_judge.py --model qwen           # single model
    python experiments/llm_judge.py --model biomistral     # single model

Available models: qwen, mistral, falcon, amber, biomistral
"""

import argparse
import csv
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI


LIGHTNING_BASE_URL = "https://lightning.ai/api/v1/"
LIGHTNING_API_KEY  = os.environ.get("LIGHTNING_API_KEY", "")
MODEL              = "anthropic/claude-haiku-4-5-20251001"

MAX_RETRIES        = 5
RETRY_DELAY        = 5.0
CALL_DELAY         = 4.0
RATELIMIT_COOLDOWN = 30.0

BASE = Path("results")

EXPERIMENTS: dict[str, dict] = {
    "qwen": {
        "label":         "Qwen2.5-7B",
        "predictions":   BASE / "033_full_comparison_results/predictions.csv",
        "output":        BASE / "033_full_comparison_results/llm_judge_results.csv",
        "exclude_tasks": set(),
    },
    "mistral": {
        "label":         "Mistral-7B",
        "predictions":   BASE / "034_mistral_results/predictions.csv",
        "output":        BASE / "034_mistral_results/llm_judge_results.csv",
        "exclude_tasks": set(),
    },
    "falcon": {
        "label":         "Falcon3-7B",
        "predictions":   BASE / "037_falcon3_results/predictions.csv",
        "output":        BASE / "037_falcon3_results/llm_judge_results.csv",
        "exclude_tasks": set(),
    },
    "amber": {
        "label":         "Amber-7B",
        "predictions":   BASE / "040_amber_results/predictions.csv",
        "output":        BASE / "040_amber_results/llm_judge_results.csv",
        "exclude_tasks": set(),
    },
    "biomistral": {
        "label":         "BioMistral-7B",
        "predictions":   BASE / "038_biomistral_results/predictions.csv",
        "output":        BASE / "038_biomistral_results/llm_judge_results.csv",
        "exclude_tasks": {"medalpaca_wiki", "pubmedqa_long"},
    },
}

RULER_TASKS = {"niah_single", "niah_multikey", "vt"}

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

JUDGE_SYSTEM = (
    "You are a strict QA judge. You classify model predictions into exactly "
    "one of four categories. Reply with ONLY the category name \u2014 no explanation, "
    "no punctuation, just the single word."
)

JUDGE_TEMPLATE = """\
TASK: {task}
GROUND TRUTH: {ground_truth}
MODEL PREDICTION: {prediction}

Categories:
CORRECT      \u2014 prediction is right (exact match or clear semantic equivalent of ground truth)
HALLUCINATED \u2014 prediction is non-empty, sounds confident/fluent, but is factually wrong
REFUSED      \u2014 prediction explicitly signals inability to answer ("I cannot", "not mentioned", "N/A", etc.)
NOISE        \u2014 prediction is empty, repeats prompt instructions, generates new questions, or is completely off-topic

Reply with ONE WORD only: CORRECT, HALLUCINATED, REFUSED, or NOISE"""

VALID_LABELS = {"CORRECT", "HALLUCINATED", "REFUSED", "NOISE"}


def auto_classify(prediction: str, contains: float) -> str | None:
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
            print(f"  [warn] unexpected label {label!r} \u2192 defaulting to HALLUCINATED")
            return "HALLUCINATED"
        except Exception as exc:
            is_ratelimit = "rate limit" in str(exc).lower() or "429" in str(exc)
            if attempt < MAX_RETRIES - 1:
                wait = RATELIMIT_COOLDOWN if is_ratelimit else delay
                print(f"  [retry {attempt+1}] {'rate-limit' if is_ratelimit else 'error'} \u2014 waiting {wait:.0f}s")
                time.sleep(wait)
                delay *= 2
            else:
                print(f"  [error] giving up after {MAX_RETRIES} attempts: {exc}")
                return "ERROR"


def load_done_keys(output_path: Path) -> set[tuple]:
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

    print(f"\n=== {label} \u2014 CORRECT% by Policy and Budget (abs%(pct-of-FC)) ===")
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
        print(f"\n=== {label} \u2014 Full label breakdown B={b} ===")
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


def run_experiment(client: OpenAI, exp_key: str, exp: dict) -> None:
    pred_path    = exp["predictions"]
    output_path  = exp["output"]
    label        = exp["label"]
    exclude      = exp["exclude_tasks"]

    print(f"\n{'='*65}")
    print(f"  Judging: {label}  ({pred_path.name})")
    print(f"{'='*65}")

    if not pred_path.exists():
        print(f"  [SKIP] predictions file not found: {pred_path}")
        return

    with open(pred_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    lb_rows = [
        r for r in all_rows
        if r["task"] not in RULER_TASKS
        and r["task"] not in exclude
    ]
    print(f"  Total rows: {len(all_rows)} | Judgeable: {len(lb_rows)}")

    done_keys = load_done_keys(output_path)
    print(f"  Already judged: {len(done_keys)} | Remaining: {len(lb_rows) - len(done_keys)}")

    to_process = [r for r in lb_rows if row_key(r) not in done_keys]
    total      = len(to_process)
    if total == 0:
        print("  Nothing to judge \u2014 all done.")
        print_summary(output_path, label)
        return

    out_fields    = list(all_rows[0].keys()) + ["judge_label", "judge_source"]
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


def main() -> None:
    if not LIGHTNING_API_KEY:
        print("ERROR: LIGHTNING_API_KEY environment variable not set.")
        print("  export LIGHTNING_API_KEY='your-key-here'")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(EXPERIMENTS.keys()),
        help="Judge only one model (default: all)",
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
