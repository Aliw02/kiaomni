"""Deterministic task generators ported from notebook/demo/kiaomni_vs_snapkv_kaggle.ipynb.

The generators here are byte-for-byte ports of the demo notebook's task builders.
They are seeded for reproducibility and designed for ~4K-token context. The
``check`` functions return ``(score, detail)`` where ``score`` is normalized to
[0, 1] for ``summary`` (8-fact coverage) and is a boolean cast to int for the
needle / reasoning tasks.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

SEED: int = 42
CONTEXT_TOKENS: int = 4000
DEPTHS: list[float] = [0.10, 0.30, 0.50, 0.70, 0.90]

_MAX_NEW: dict[str, int] = {
    "single": 24,
    "multi": 96,
    "reason": 24,
    "summary": 220,
}

_SUBJ: list[str] = [
    "The committee", "A regional survey", "The maintenance crew", "An early prototype",
    "The northern facility", "A visiting delegation", "The archive department", "Local observers",
    "The pilot program", "A follow-up study", "The logistics team", "An internal memo",
    "The harbor authority", "A quarterly audit", "The training division", "Field engineers",
]
_VERB: list[str] = [
    "reported", "confirmed", "documented", "reviewed", "scheduled", "postponed",
    "evaluated", "inspected", "catalogued", "summarized", "approved", "recorded",
]
_OBJ: list[str] = [
    "minor adjustments to the ventilation schedule", "a gradual rise in afternoon foot traffic",
    "the relocation of two storage containers", "routine calibration of the measurement rigs",
    "an updated rotation plan for night shifts", "the replacement of worn signage near gate three",
    "consistent humidity readings across all halls", "a backlog of paperwork from the previous quarter",
    "slower than expected delivery of spare parts", "general satisfaction with the new canteen layout",
    "uneven wear on the loading dock surface", "stable energy consumption throughout the week",
]


def _sentence(rng: random.Random) -> str:
    return f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)}."


def count_tokens(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def build_haystack(rng: random.Random, reserved_tokens: int, tok) -> list[str]:
    target = CONTEXT_TOKENS - reserved_tokens
    sents: list[str] = []
    step = 40
    while count_tokens(" ".join(sents), tok) < target:
        sents.extend(_sentence(rng) for _ in range(step))
    while count_tokens(" ".join(sents), tok) > target and sents:
        sents.pop()
    return sents


def _insert(sents: list[str], items: list[tuple[float, str]]) -> str:
    out = list(sents)
    for depth, s in sorted(items, key=lambda x: x[0], reverse=True):
        out.insert(int(depth * len(out)), s)
    return " ".join(out)


@dataclass
class Sample:
    task: str
    sid: int
    context: str
    question: str
    gold: str
    check: Callable[[str], tuple[float, str]]
    info: str
    max_new: int

    def grade(self, answer: str) -> tuple[float, str]:
        return self.check(answer)


# ── Task generators ────────────────────────────────────────────────────────────

def make_single(i: int, tok) -> Sample:
    rng = random.Random(SEED + 100 + i)
    code = rng.randint(100000, 999999)
    depth = DEPTHS[i % len(DEPTHS)]
    needle = f"Note for the security desk: the secret access code for vault {chr(65 + i % 26)} is {code}."
    q = "What is the secret access code mentioned in the document? Answer with the number only."
    ctx = _insert(build_haystack(rng, reserved_tokens=80, tok=tok), [(depth, needle)])
    gold = str(code)

    def check(ans: str) -> tuple[float, str]:
        ok = re.search(rf"\b{gold}\b", ans) is not None
        return (1.0 if ok else 0.0), ("found" if ok else "missing")

    return Sample(
        task="single", sid=i, context=ctx, question=q, gold=gold, check=check,
        info=f"code={gold} @ depth {int(depth*100)}%", max_new=_MAX_NEW["single"],
    )


def make_multi(i: int, tok) -> Sample:
    rng = random.Random(SEED + 200 + i)
    code = rng.randint(100000, 999999)
    locker = rng.randint(100, 999)
    name = rng.choice([
        "FALCON", "GRANITE", "MERIDIAN", "COBALT", "JUNIPER",
        "VERTEX", "HARBOR", "QUARTZ", "SABLE", "ORCHID",
    ])
    needles = [
        (0.20, f"For the record, the access code is {code}."),
        (0.50, f"Equipment update: the spare keys were moved to locker {locker}."),
        (0.80, f"Personnel note: the liaison operates under the codename {name}."),
    ]
    q = ("The document hides three secret values: an access code, a locker number, "
         "and a codename. List all three values.")
    ctx = _insert(build_haystack(rng, reserved_tokens=140, tok=tok), needles)
    gold_list = [str(code), str(locker), name]

    def check(ans: str) -> tuple[float, str]:
        up = ans.upper()
        found = sum(1 for g in gold_list if re.search(rf"\b{re.escape(g)}\b", up))
        return (1.0 if found == 3 else 0.0), f"{found}/3 needles"

    return Sample(
        task="multi", sid=i, context=ctx, question=q,
        gold=", ".join(gold_list), check=check,
        info="3 needles @ 20/50/80%", max_new=_MAX_NEW["multi"],
    )


def make_reason(i: int, tok) -> Sample:
    rng = random.Random(SEED + 300 + i)
    val = rng.randint(10000, 99999)
    dval = rng.randint(10000, 99999)
    while dval == val:
        dval = rng.randint(10000, 99999)
    A, B, C, D = (f"V{rng.randint(10,99)}{c}" for c in "ABCD")
    X, Y = (f"V{rng.randint(10,99)}{c}" for c in "XY")
    chain = [
        (0.15, f"Assignment log: variable {A} was set to {val}."),
        (0.40, f"Assignment log: variable {B} was set to the value of {A}."),
        (0.60, f"Assignment log: variable {C} was set to the value of {B}."),
        (0.85, f"Assignment log: variable {D} was set to the value of {C}."),
        (0.25, f"Assignment log: variable {X} was set to {dval}."),
        (0.70, f"Assignment log: variable {Y} was set to the value of {X}."),
    ]
    q = (f"Track the variable assignments in the document. "
         f"What is the final value of variable {D}? Answer with the number only.")
    ctx = _insert(build_haystack(rng, reserved_tokens=180, tok=tok), chain)
    gold = str(val)

    def check(ans: str) -> tuple[float, str]:
        hit = re.search(rf"\b{gold}\b", ans) is not None
        dist = re.search(rf"\b{dval}\b", ans) is not None
        ok = hit and not dist
        return (1.0 if ok else 0.0), ("correct" if ok else ("distractor value" if dist else "missing"))

    return Sample(
        task="reason", sid=i, context=ctx, question=q, gold=gold, check=check,
        info=f"4-hop chain, answer {gold}, distractor {dval}", max_new=_MAX_NEW["reason"],
    )


def make_summary(i: int, tok) -> Sample:
    rng = random.Random(SEED + 400 + i)
    company = rng.choice(["Altavera Systems", "Norwind Logistics", "Helios Materials"])
    facts = {
        "revenue":   f"{rng.randint(11, 48)}.{rng.randint(1,9)} million",
        "headcount": str(rng.randint(180, 950)),
        "churn":     f"{rng.randint(2, 9)}.{rng.randint(0,9)}%",
        "nps":       str(rng.randint(31, 78)),
        "runway":    f"{rng.randint(9, 30)} months",
        "city":      rng.choice(["Riga", "Porto", "Tallinn", "Valencia"]),
        "defects":   f"{rng.randint(1, 6)}.{rng.randint(0,9)} per thousand units",
        "launch":    rng.choice(["March", "June", "September", "November"]),
    }
    sections = [
        f"QUARTERLY REVIEW - {company}.",
        f"Finance. Total revenue for the quarter reached {facts['revenue']} dollars, "
        f"and the cash runway now stands at {facts['runway']}.",
        f"People. Headcount closed at {facts['headcount']} employees after the {facts['city']} office expansion.",
        f"Customers. Monthly churn was {facts['churn']} while the net promoter score reached {facts['nps']}.",
        f"Operations. The defect rate improved to {facts['defects']}, and the next product launch is scheduled for {facts['launch']}.",
    ]
    rng2 = random.Random(SEED + 500 + i)
    sents = build_haystack(rng2, reserved_tokens=count_tokens(" ".join(sections), tok) + 80, tok=tok)
    block = len(sents) // len(sections)
    woven: list[str] = []
    for k, sec in enumerate(sections):
        woven.append(sec)
        woven.extend(sents[k * block:(k + 1) * block])
    ctx = " ".join(woven)
    q = ("Summarize this quarterly review in 5-8 sentences. "
         "Include every concrete figure: revenue, headcount, churn, NPS, runway, "
         "expansion city, defect rate, and launch month.")
    keys = list(facts.values())

    def check(ans: str) -> tuple[float, str]:
        up = ans.upper()
        hits = sum(1 for v in keys if v.upper() in up)
        return hits / len(keys), f"{hits}/{len(keys)} key facts"

    return Sample(
        task="summary", sid=i, context=ctx, question=q,
        gold="; ".join(f"{k}={v}" for k, v in facts.items()),
        check=check, info=f"{company}, 8 planted facts", max_new=_MAX_NEW["summary"],
    )


_GENERATORS: dict[str, Callable[[int, object], Sample]] = {
    "single":  make_single,
    "multi":   make_multi,
    "reason":  make_reason,
    "summary": make_summary,
}


def make_sample(task: str, sid: int, tok) -> Sample:
    if task not in _GENERATORS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(_GENERATORS)}")
    return _GENERATORS[task](sid, tok)


def list_tasks() -> list[str]:
    return sorted(_GENERATORS)
