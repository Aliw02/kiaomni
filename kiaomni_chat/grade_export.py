"""Local grader for side-by-side chat exports.

Reads:
  * A saved chat JSON (from the UI's "Save chat (JSON)" button) — either
    the regular chat (chatRecord schema) or the side-by-side chat
    (compareRecord schema).
  * ``kiaomni_chat/sample_data/expected_answers.json`` — the 9 expected
    answers with their scoring rules (schema v3: ``questions`` array,
    each with ``tier``, ``type``, ``question``, ``expected_answer`` or
    ``expected_topics`` / ``expected_reasoning_chain``).

For each Q1-Q9, splits the model's response by the "Q1 ... Q9" markers
inferred from the prompt, then scores the per-question answer:
  * Tier 1 (NIAH): substring / year / number match per the ``match`` field.
  * Tier 2 (Summarization) and Tier 3 (Reasoning): flag for human review
    and dump the model's answer + the expected themes / chains side by
    side so the reviewer can score by hand.

Saves a graded JSON next to the input file.

Usage:
  python kiaomni_chat/grade_export.py kiaomni_chat/results/some_export.json
  python kiaomni_chat/grade_export.py kiaomni_chat/results/some_export.json --out graded.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PARENT = REPO.parent
EXPECTED_PATH = REPO / "sample_data" / "expected_answers.json"


def _split_answers(text: str) -> dict[str, str]:
    """Split a long model response into per-question chunks.

    Looks for "Q1", "Q2", ... markers. Falls back to single-block if no
    markers are present.
    """
    if not text:
        return {}
    pattern = re.compile(r"(?:^|\n)\s*\*?\*?Q\s*([1-9])\b", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return {"_all": text.strip()}
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[f"q{num}"] = text[start:end].strip()
    return out


def _score_tier1_single(answer: str, expected: dict) -> tuple[float, str]:
    """Score a Tier-1 NIAH single-needle question.

    Match logic by ``expected["match"]`` field:
      * "date + flight number" — accept either the date OR the flight number
      * "date" / "altitude" / "temperature" / "weight" / "number" / "year" — extract the
        numeric/date token from the expected answer and check substring
      * default — generic substring fallback
    """
    if not answer:
        return 0.0, "no answer"
    expected_ans = expected.get("expected_answer", "")
    if not expected_ans or not isinstance(expected_ans, str):
        return 0.0, "no expected answer"
    a = answer.lower()
    a_norm = _norm(answer)
    match_kind = (expected.get("match") or "").lower()

    if "flight number" in match_kind and "date" in match_kind:
        # accept EITHER the date OR the flight number
        date_m = re.search(r"[A-Z][a-z]+ \d{1,2},? \d{4}", expected_ans)
        flight_m = re.search(r"IFT-\d+", expected_ans, re.IGNORECASE)
        if date_m and _norm(date_m.group(0)) in a_norm:
            return 1.0, f"found date '{date_m.group(0)}'"
        if flight_m and flight_m.group(0).lower() in a.lower():
            return 1.0, f"found flight '{flight_m.group(0)}'"
        return 0.0, f"neither date nor flight number found — expected '{expected_ans}'"

    if "date" in match_kind:
        # Try several date formats
        for fmt in (
            r"[A-Z][a-z]+ \d{1,2},? \d{4}",          # "March 14, 2024"
            r"\d{1,2} [A-Z][a-z]+ \d{4}",             # "14 March 2024"
            r"\d{1,2}/\d{1,2}/\d{4}",                 # "3/14/2024"
        ):
            m = re.search(fmt, expected_ans)
            if m and _norm(m.group(0)) in a_norm:
                return 1.0, f"date match '{m.group(0)}'"

    if "altitude" in match_kind or "temperature" in match_kind or "weight" in match_kind or "number" in match_kind or "year" in match_kind:
        m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", expected_ans)
        if m:
            tok = m.group(1).replace(",", "")
            if tok in a.replace(",", "").replace(" ", ""):
                return 1.0, f"found numeric token '{tok}'"

    if "address" in match_kind:
        for tok in re.findall(r"[A-Za-z]+|\d{2,5}", expected_ans):
            if tok.lower() in a and len(tok) > 1:
                return 1.0, f"found address token '{tok}'"

    # Generic substring fallback
    if _norm(expected_ans) in a_norm:
        return 1.0, f"exact substring match for '{expected_ans}'"
    # Loose: any 4+ char word from expected appears in answer
    for word in re.findall(r"[A-Za-z]{4,}", expected_ans):
        if word.lower() in a:
            return 0.5, f"weak match: word '{word}' found"
    return 0.0, f"no match — expected '{expected_ans}'"


def _score_tier1_multi_needle(answer: str, expected: dict) -> tuple[float, str]:
    """Score a Tier-1 NIAH multi-needle question.

    ``expected["expected_answer"]`` is a list of dicts. Credit N/total for
    N items found in the answer. Punctuation is normalised on both sides
    so "March 14, 2024" matches "March 14 2024".
    """
    if not answer:
        return 0.0, "no answer"
    items = expected.get("expected_answer", [])
    if not isinstance(items, list):
        return 0.0, "expected_answer is not a list"
    a = _norm(answer)
    hits = 0
    hit_items: list[str] = []
    for item in items:
        text_candidates: list[str] = []
        for k in ("date", "value", "outcome", "spec"):
            if k in item:
                text_candidates.append(str(item[k]))
        if not text_candidates:
            text_candidates = [str(v) for v in item.values()]
        if any(_norm(cand) in a for cand in text_candidates):
            hits += 1
            hit_items.append(text_candidates[0])
    score = hits / len(items) if items else 0.0
    note = f"hit {hits}/{len(items)} items: {hit_items}" if hits else "no items matched"
    return score, note


def _norm(s: str) -> str:
    """Lowercase + strip commas / periods / extra spaces for fuzzy matching."""
    import re as _re
    return _re.sub(r"[,.\s]+", " ", s.lower()).strip()


def _score_tier1(question: dict, answer: str) -> tuple[float, str, str]:
    """Dispatch a Tier-1 question to the right scorer."""
    qtype = (question.get("type") or "").lower()
    if "multi-needle" in qtype:
        s, note = _score_tier1_multi_needle(answer, question)
        return s, note, "tier1_multi_needle"
    s, note = _score_tier1_single(answer, question)
    return s, note, "tier1_single"


def _index_questions(expected: dict) -> dict[str, dict]:
    """Index the ``questions`` array by q1..q9 key (derived from position)."""
    out: dict[str, dict] = {}
    for i, q in enumerate(expected.get("questions", []), start=1):
        out[f"q{i}"] = q
    return out


def _human_review_payload(question: dict) -> dict:
    """Build the expected-reasoning payload for tier 2/3 questions."""
    out: dict = {}
    for k in (
        "expected_topics",
        "expected_reasoning_chain",
        "expected_themes",
        "expected_chain",
        "expected_events",
        "expected_parallels",
    ):
        v = question.get(k)
        if v:
            out[k] = v
    return out


def grade(export: dict, expected: dict) -> dict:
    """Walk the export, score every (policy × question) pair, return a graded dict."""
    mode = export.get("mode", "chat")
    questions = _index_questions(expected)
    rows: list[dict] = []
    if mode == "side_by_side":
        for ti, turn in enumerate(export.get("turns", [])):
            for policy, resp in turn.get("responses", {}).items():
                if not isinstance(resp, dict) or resp.get("error"):
                    rows.append({
                        "turn": ti + 1,
                        "policy": policy,
                        "question_id": None,
                        "tier": None,
                        "score": 0.0,
                        "notes": resp.get("error", "no response"),
                    })
                    continue
                answers = _split_answers(resp.get("text", ""))
                stats = resp.get("stats", {})
                for qkey, q in questions.items():
                    a = answers.get(qkey, "")
                    tier = q.get("tier")
                    if tier == 1:
                        s, note, _ = _score_tier1(q, a)
                        rows.append({
                            "turn": ti + 1,
                            "policy": policy,
                            "question_id": qkey,
                            "tier": 1,
                            "question": q.get("question"),
                            "type": q.get("type"),
                            "expected": q.get("expected_answer"),
                            "actual": a,
                            "score": round(s, 3),
                            "notes": note,
                            "tokens_in": stats.get("tokens_in"),
                            "tokens_kept": stats.get("tokens_kept"),
                            "tok_per_sec": stats.get("tok_per_sec"),
                        })
                    else:
                        rows.append({
                            "turn": ti + 1,
                            "policy": policy,
                            "question_id": qkey,
                            "tier": tier,
                            "question": q.get("question"),
                            "type": q.get("type"),
                            "expected": q.get("expected_answer")
                                       or q.get("expected_topics")
                                       or q.get("expected_reasoning_chain"),
                            "actual": a,
                            "score": None,
                            "notes": "manual review required",
                            "expected_detail": _human_review_payload(q),
                            "tokens_in": stats.get("tokens_in"),
                            "tokens_kept": stats.get("tokens_kept"),
                            "tok_per_sec": stats.get("tok_per_sec"),
                        })
    else:
        # Single-stream chat — only one "policy" effectively, no per-policy split.
        for ti, m in enumerate(export.get("messages", [])):
            if m.get("role") != "assistant":
                continue
            answers = _split_answers(m.get("content", ""))
            stats = m.get("stats") or {}
            for qkey, q in questions.items():
                a = answers.get(qkey, "")
                tier = q.get("tier")
                if tier == 1:
                    s, note, _ = _score_tier1(q, a)
                    rows.append({
                        "turn": ti,
                        "policy": export.get("policy", "?"),
                        "question_id": qkey,
                        "tier": 1,
                        "question": q.get("question"),
                        "type": q.get("type"),
                        "expected": q.get("expected_answer"),
                        "actual": a,
                        "score": round(s, 3),
                        "notes": note,
                    })
                else:
                    rows.append({
                        "turn": ti,
                        "policy": export.get("policy", "?"),
                        "question_id": qkey,
                        "tier": tier,
                        "question": q.get("question"),
                        "type": q.get("type"),
                        "expected": q.get("expected_answer")
                                   or q.get("expected_topics")
                                   or q.get("expected_reasoning_chain"),
                        "actual": a,
                        "score": None,
                        "notes": "manual review required",
                        "expected_detail": _human_review_payload(q),
                    })
    return {
        "schema_version": 2,
        "graded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "export_summary": {
            "mode": mode,
            "deploy_url": export.get("deploy_url"),
            "created_at": export.get("created_at"),
            "budget": export.get("budget"),
            "max_new_tokens": export.get("max_new_tokens"),
            "turn_count": len(export.get("turns", []))
                             if mode == "side_by_side"
                             else sum(1 for m in export.get("messages", []) if m.get("role") == "user"),
        },
        "rows": rows,
        "summary_by_policy_tier1": _summary(rows),
    }


def _summary(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("tier") != 1:
            continue
        policy = r.get("policy", "?")
        if policy not in out:
            out[policy] = {"n": 0, "score_sum": 0.0}
        out[policy]["n"] += 1
        out[policy]["score_sum"] += float(r.get("score") or 0.0)
    return {k: {"n": v["n"], "mean": round(v["score_sum"] / v["n"], 3)} for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path)
    ap.add_argument("--expected", type=Path, default=EXPECTED_PATH)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.export.exists():
        print(f"export not found: {args.export}", file=sys.stderr)
        return 1
    if not args.expected.exists():
        print(f"expected_answers not found: {args.expected}", file=sys.stderr)
        return 1
    export = json.loads(args.export.read_text(encoding="utf-8"))
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    graded = grade(export, expected)
    out_path = args.out or args.export.with_name(args.export.stem + "_graded.json")
    out_path.write_text(json.dumps(graded, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[grade] saved {out_path}")
    print(f"[grade] mode={graded['export_summary']['mode']}  "
          f"turns={graded['export_summary']['turn_count']}")
    if graded["summary_by_policy_tier1"]:
        print(f"[grade] tier-1 (NIAH) mean scores by policy:")
        for p, s in graded["summary_by_policy_tier1"].items():
            print(f"  {p:24s} {s['mean']:.3f}  ({s['n']} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
