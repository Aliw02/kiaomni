from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


def normalize(text: str) -> str:
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9\-]+", " ", text)
    return " ".join(text.split())


def first_answer_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def strict_score(text: str, expected_answers: list[str]) -> dict:
    first = normalize(first_answer_line(text))
    expected = [normalize(x) for x in expected_answers]
    hits = [bool(x and x in first) for x in expected]

    # Controlled benchmark questions explicitly request only the answer values.
    # Therefore credit is based on the first non-empty answer line, not on a
    # correct value appearing later inside an explanation after an incorrect
    # first answer.
    return {
        "strict_first_line": first,
        "strict_all_correct": bool(all(hits)),
        "strict_answer_recall": float(sum(hits) / max(1, len(hits))),
        "strict_hits": hits,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs_jsonl")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    src = Path(args.runs_jsonl).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(src)
    rescored = []
    changed = []

    for row in rows:
        r = dict(row)
        if r.get("status") == "ok":
            strict = strict_score(r.get("output_text", ""), list(r.get("expected_answers", [])))
            r.update(strict)
            if bool(r.get("all_correct")) != bool(r["strict_all_correct"]):
                changed.append({
                    "case_id": r.get("case_id"),
                    "task": r.get("task"),
                    "method": r.get("method"),
                    "budget_label": r.get("budget_label"),
                    "old_all_correct": r.get("all_correct"),
                    "strict_all_correct": r["strict_all_correct"],
                    "strict_first_line": r["strict_first_line"],
                    "expected_answers": r.get("expected_answers"),
                })
        rescored.append(r)

    out_jsonl = out_dir / (src.stem + "_strict.jsonl")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rescored:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    ok = [r for r in rescored if r.get("status") == "ok"]
    df = pd.DataFrame(ok)
    group_cols = ["context_tokens", "method", "budget_label"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("case_id", "count"),
            strict_accuracy=("strict_all_correct", "mean"),
            strict_answer_recall=("strict_answer_recall", "mean"),
            legacy_accuracy=("all_correct", "mean"),
            kv_reduction_ratio=("kv_reduction_ratio", "mean"),
            peak_allocated_gb=("peak_allocated_gb", "mean"),
            ttft_seconds=("ttft_seconds", "mean"),
            total_seconds=("total_seconds", "mean"),
            tokens_per_second=("tokens_per_second", "mean"),
            greedy_self_ppl=("greedy_self_ppl", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_strict.csv", index=False)

    full = {
        r["case_id"]: r for r in ok
        if r["method"] == "full_kv"
    }
    paired = []
    for r in ok:
        if r["method"] == "full_kv" or r["case_id"] not in full:
            continue
        f = full[r["case_id"]]
        fc = bool(f["strict_all_correct"])
        cc = bool(r["strict_all_correct"])
        paired.append({
            "case_id": r["case_id"],
            "context_tokens": r["context_tokens"],
            "method": r["method"],
            "budget_label": r["budget_label"],
            "budget_tokens": r["budget_tokens"],
            "full_correct": fc,
            "compressed_correct": cc,
            "full_recall": float(f["strict_answer_recall"]),
            "compressed_recall": float(r["strict_answer_recall"]),
            "recall_delta": float(r["strict_answer_recall"] - f["strict_answer_recall"]),
            "preserved": bool(fc and cc),
            "regression": bool(fc and not cc),
            "rescue": bool((not fc) and cc),
            "both_wrong": bool((not fc) and (not cc)),
        })

    if paired:
        pdf = pd.DataFrame(paired)
        pdf.to_csv(out_dir / "paired_cases_strict.csv", index=False)
        ps = (
            pdf.groupby(["context_tokens", "method", "budget_label"], dropna=False)
            .agg(
                n=("case_id", "count"),
                preserved=("preserved", "sum"),
                regressions=("regression", "sum"),
                rescues=("rescue", "sum"),
                both_wrong=("both_wrong", "sum"),
                mean_recall_delta=("recall_delta", "mean"),
            )
            .reset_index()
        )
        ps.to_csv(out_dir / "paired_summary_strict.csv", index=False)

    Path(out_dir / "strict_scoring_changes.json").write_text(
        json.dumps(changed, indent=2), encoding="utf-8"
    )
    print(f"rescored={len(ok)} changed={len(changed)}")
    for item in changed:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
