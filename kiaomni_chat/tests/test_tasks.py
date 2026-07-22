"""Test the deterministic demo task generators and their scoring."""
from __future__ import annotations

from kiaomni_chat.tasks import (
    CONTEXT_TOKENS, SEED, list_tasks, make_sample,
)


class _StubTok:
    """A whitespace tokenizer that approximates the demo's ``count_tokens``."""
    def __call__(self, text: str, add_special_tokens: bool = True):
        toks = text.split()
        return type("R", (), {"input_ids": list(range(len(toks)))})()


def test_list_tasks() -> None:
    assert list_tasks() == ["multi", "reason", "single", "summary"]


def test_seed_reproducible() -> None:
    tok = _StubTok()
    a = make_sample("single", 0, tok)
    b = make_sample("single", 0, tok)
    assert a.context == b.context
    assert a.question == b.question
    assert a.gold == b.gold


def test_single_grade() -> None:
    tok = _StubTok()
    s = make_sample("single", 0, tok)
    assert s.gold in s.context, "needle must be in the context"
    score, detail = s.grade(f"The code is {s.gold}.")
    assert score == 1.0
    assert detail == "found"
    score, detail = s.grade("I don't know")
    assert score == 0.0
    assert detail == "missing"


def test_multi_grade() -> None:
    tok = _StubTok()
    s = make_sample("multi", 0, tok)
    needles = [g for g in s.gold.split(", ")]
    answer = " ".join(needles).upper()
    score, detail = s.grade(answer)
    assert score == 1.0
    assert detail == "3/3 needles"
    score, detail = s.grade("just one: " + needles[0])
    assert score == 0.0


def test_reason_grade() -> None:
    tok = _StubTok()
    s = make_sample("reason", 0, tok)
    # The answer and distractor are both 5-digit numbers; the check
    # function embeds them in the context. We can look them up.
    assert s.gold in s.context
    score, _ = s.grade(f"The answer is {s.gold}.")
    assert score == 1.0


def test_summary_grade_partial() -> None:
    tok = _StubTok()
    s = make_sample("summary", 0, tok)
    # Grade is a fraction in [0, 1].
    score, detail = s.grade("blah")
    assert 0.0 <= score <= 1.0
    assert "/" in detail


def test_make_sample_unknown_raises() -> None:
    tok = _StubTok()
    import pytest
    with pytest.raises(ValueError):
        make_sample("nope", 0, tok)


def test_context_length_is_around_4k() -> None:
    """The demo's contract is ~4000 tokens of context."""
    tok = _StubTok()
    for task in list_tasks():
        s = make_sample(task, 0, tok)
        # Stub tokenizer splits on whitespace — so 4K "tokens" ≈ 4K words.
        # Real tokenizer yields fewer; we just sanity-check order of magnitude.
        assert len(s.context.split()) > 1000
