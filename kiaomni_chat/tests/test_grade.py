"""Tests for the local grader at kiaomni_chat/grade_export.py.

Locks the contract that:
  * v3 ``expected_answers.json`` schema is consumed (questions array, tier field).
  * Tier 1 single-needle questions are auto-scored (substring / numeric / date match).
  * Tier 1 multi-needle questions get partial credit.
  * Tier 2/3 questions are flagged for human review (score is None).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "kiaomni_chat"))

import grade_export  # type: ignore  # noqa: E402


SAMPLE_EXPORT = {
    "schema_version": 1,
    "created_at": "2026-07-17T00:00:00Z",
    "deploy_url": "https://example.com",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "mode": "side_by_side",
    "budget": 512,
    "max_new_tokens": 2048,
    "policies": ["fullcontext", "kiaomni_s8", "kiaomni_gaussian"],
    "turns": [
        {
            "ts": "2026-07-17T00:00:01Z",
            "user": "Answer Q1..Q9 about the article.",
            "responses": {
                "fullcontext": {
                    "text": (
                        "**Q1:** IFT-3 launched on March 14, 2024.\n\n"
                        "**Q2:** The upper stage reached 234 kilometers (peak altitude).\n\n"
                        "**Q3:** Peak temperature was approximately 1,500 °C.\n\n"
                        "**Q4:** October 13, 2024 (IFT-5).\n\n"
                        "**Q5:** IFT-3 March 14, 2024 Reached space (234 km), lost on re-entry over Indian Ocean. "
                        "IFT-4 June 6, 2024 First controlled re-entry, soft splashdown in Indian Ocean. "
                        "IFT-5 October 13, 2024 First successful Super Heavy booster catch at Starbase; ship splashed down. "
                        "IFT-6 November 19, 2024 Booster caught successfully, ship lost during coast phase due to propellant leak. "
                        "IFT-7 January 16, 2025 Both booster and ship successfully caught — first time ship was caught.\n\n"
                        "**Q6:** Total height 121 meters. Total liftoff mass approximately 5,000 metric tons. "
                        "Total liftoff thrust approximately 7,500 metric tons. "
                        "Number of first-stage engines 33 Raptor 2 engines. "
                        "Number of upper-stage engines 6 Raptor 2 engines (3 sea-level, 3 vacuum-optimized).\n\n"
                        "**Q7:** The chopstick catch is more ambitious because of positional accuracy.\n\n"
                        "**Q8:** IFT-6 booster caught, ship lost in coast phase.\n\n"
                        "**Q9:** Bullet 1. Bullet 2. Bullet 3. Bullet 4. Bullet 5."
                    ),
                    "stats": {"tokens_in": 2229, "tokens_kept": 2229, "tok_per_sec": 25.0},
                },
                "kiaomni_s8": {
                    "text": "**Q1:** March 14, 2024.\n\n**Q2:** 234 kilometers.\n\n**Q3:** 1,500 °C.\n\n"
                             "**Q4:** October 13, 2024.\n\n**Q5:** IFT-3 March 14, 2024 Reached space (234 km), lost on re-entry over Indian Ocean. "
                             "IFT-4 June 6, 2024 First controlled re-entry, soft splashdown in Indian Ocean.\n\n"
                             "**Q6:** 121 meters. 5,000 metric tons. 33 Raptor 2 engines. 6 Raptor 2 engines.\n\n"
                             "**Q7:** Catch is hard.\n\n"
                             "**Q8:** Ship lost.\n\n**Q9:** 5 bullets.",
                    "stats": {"tokens_in": 2229, "tokens_kept": 512, "tok_per_sec": 28.0},
                },
                "kiaomni_gaussian": {
                    "error": "simulated error",
                },
            },
        }
    ],
}


@pytest.fixture
def expected() -> dict:
    return json.loads((REPO / "kiaomni_chat" / "sample_data" / "expected_answers.json").read_text("utf-8"))


def test_index_questions(expected):
    idx = grade_export._index_questions(expected)
    assert len(idx) == 9
    assert "q1" in idx and "q9" in idx
    assert idx["q1"]["tier"] == 1
    assert idx["q9"]["tier"] == 2


def test_split_answers_handles_bold_markers():
    text = "**Q1:** answer 1\n\n**Q2:** answer 2\n\n**Q3:** answer 3"
    out = grade_export._split_answers(text)
    assert "q1" in out
    assert "q2" in out
    assert "q3" in out
    assert "answer 1" in out["q1"]


def test_grade_side_by_side_fullcontext_perfect_tier1(expected):
    graded = grade_export.grade(SAMPLE_EXPORT, expected)
    rows = [r for r in graded["rows"] if r["policy"] == "fullcontext" and r.get("tier") == 1]
    assert len(rows) == 6   # q1..q6 are tier 1
    # All 6 should score 1.0 for the perfect fullcontext response
    for r in rows:
        assert r["score"] == 1.0, f"{r['question_id']} got {r['score']}: {r['notes']}"


def test_grade_side_by_side_tier2_3_flagged_for_human_review(expected):
    graded = grade_export.grade(SAMPLE_EXPORT, expected)
    rows = [r for r in graded["rows"] if r["policy"] == "fullcontext" and r.get("tier") in (2, 3)]
    assert len(rows) == 3   # q7, q8, q9
    for r in rows:
        assert r["score"] is None
        assert r["notes"] == "manual review required"
        assert r.get("expected_detail")


def test_grade_handles_error_response(expected):
    graded = grade_export.grade(SAMPLE_EXPORT, expected)
    err_rows = [r for r in graded["rows"] if r["policy"] == "kiaomni_gaussian"]
    # An error response short-circuits to a single per-policy error row
    # (no per-question breakdown since there is no answer text to score).
    assert len(err_rows) == 1
    assert err_rows[0]["score"] == 0.0
    assert "simulated error" in err_rows[0]["notes"]


def test_summary_by_policy_tier1(expected):
    graded = grade_export.grade(SAMPLE_EXPORT, expected)
    summary = graded["summary_by_policy_tier1"]
    assert "fullcontext" in summary
    assert summary["fullcontext"]["mean"] == 1.0
    assert summary["fullcontext"]["n"] == 6
    # kiaomni_s8 gets full credit on q1-q4 (NIAH single), partial on q5 (2/5)
    # and partial on q6 (4/5). Mean should be high but not 1.0.
    s8 = summary["kiaomni_s8"]
    assert 0.7 < s8["mean"] < 1.0, f"unexpected kiaomni_s8 mean: {s8}"


def test_date_match_handles_us_format(expected):
    q1 = expected["questions"][0]
    s, note, _ = grade_export._score_tier1(q1, "It happened on March 14, 2024.")
    assert s == 1.0
    assert "March 14, 2024" in note


def test_numeric_match_handles_comma_thousands(expected):
    # The "altitude" question expects "234 kilometers"
    q2 = expected["questions"][1]
    s, _note, _ = grade_export._score_tier1(q2, "The peak was 234 km.")
    assert s == 1.0


def test_flight_number_match_fallback(expected):
    # q4 is "date + flight number" — accept either
    q4 = expected["questions"][3]
    s, _note, _ = grade_export._score_tier1(q4, "It happened on IFT-5.")
    assert s == 1.0


def test_expected_answers_file_is_loadable(expected) -> None:
    """The expected_answers.json file must exist, have the v3 schema,
    contain exactly 9 questions, and tier-1 questions must cover q1..q6."""
    assert expected.get("schema_version") == 3
    questions = expected.get("questions", [])
    assert len(questions) == 9, f"expected 9 questions, got {len(questions)}"
    for i, q in enumerate(questions, start=1):
        assert "tier" in q, f"Q{i} missing tier"
        assert "type" in q, f"Q{i} missing type"
        qid = q.get("id", "")
        assert f"q{i}" in qid, f"Q{i} id={qid} doesn't match position"
    # Tier-1 questions must have expected_answer
    for off in range(6):  # q1..q6 = tier 1
        q = questions[off]
        assert q["tier"] == 1, f"Q{off+1} should be tier 1, got {q['tier']}"
        assert q.get("expected_answer"), f"Q{off+1} missing expected_answer"
        # Single-needle questions need a match field; multi-needle don't
        qtype = (q.get("type") or "").lower()
        if "single" in qtype:
            assert q.get("match"), f"Q{off+1} (single needle) missing match field"


def test_multi_needle_partial_credit(expected):
    q5 = expected["questions"][4]  # the 5 IFTs
    answer = "IFT-3 March 14 2024, IFT-4 June 6 2024"
    s, note, _ = grade_export._score_tier1(q5, answer)
    assert 0.3 < s < 0.5
    assert "2/5" in note
