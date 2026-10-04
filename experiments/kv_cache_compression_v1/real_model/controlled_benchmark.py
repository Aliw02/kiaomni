from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
from typing import Iterable


FILLER_SENTENCES = [
    "Archive note {i}: routine logistics were reviewed and no exceptional action was required.",
    "Record {i}: the maintenance team completed a standard inspection and filed the ordinary report.",
    "Entry {i}: shipment counts were reconciled with the ledger and all routine values matched.",
    "Memo {i}: the project office discussed scheduling, staffing, and ordinary administrative updates.",
    "Log {i}: the monitoring service recorded normal operating conditions during the observation period.",
    "Report {i}: several departments exchanged routine status updates without changing the operational plan.",
]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    task: str
    input_ids: list[int]
    expected_answers: tuple[str, ...]
    answer_mode: str
    metadata: dict


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _filler_tokens(tokenizer, needed: int, seed: int) -> list[int]:
    if needed <= 0:
        return []
    rng = random.Random(seed)
    chunks: list[int] = []
    i = 0
    while len(chunks) < needed:
        template = FILLER_SENTENCES[rng.randrange(len(FILLER_SENTENCES))]
        chunks.extend(_encode(tokenizer, "\n" + template.format(i=i)))
        i += 1
    return chunks[:needed]


def _insert_segments(base: list[int], segments: list[tuple[float, list[int]]]) -> list[int]:
    """Insert token segments at approximate fractional depths in stable order."""
    out = list(base)
    offset = 0
    for depth, segment in sorted(segments, key=lambda x: x[0]):
        raw = int(round(depth * len(base)))
        pos = max(0, min(len(out), raw + offset))
        out[pos:pos] = segment
        offset += len(segment)
    return out


def _fit_case(
    tokenizer,
    *,
    target_tokens: int,
    prefix: str,
    facts: list[tuple[float, str]],
    question: str,
    seed: int,
) -> list[int]:
    prefix_ids = _encode(tokenizer, prefix)
    fact_segments = [(depth, _encode(tokenizer, "\n" + text + "\n")) for depth, text in facts]
    question_ids = _encode(tokenizer, "\n" + question)
    reserved = len(prefix_ids) + len(question_ids) + sum(len(seg) for _, seg in fact_segments)
    if reserved >= target_tokens:
        raise ValueError(
            f"target_tokens={target_tokens} too small for task payload ({reserved} reserved tokens)"
        )
    filler = _filler_tokens(tokenizer, target_tokens - reserved, seed)
    body = _insert_segments(filler, fact_segments)
    ids = prefix_ids + body + question_ids
    if len(ids) != target_tokens:
        # Token lists are composed directly, so this should remain exact.
        raise RuntimeError(f"expected {target_tokens} tokens, built {len(ids)}")
    return ids


def _code(rng: random.Random, digits: int = 6) -> str:
    lo = 10 ** (digits - 1)
    hi = (10**digits) - 1
    return str(rng.randint(lo, hi))


def make_case(tokenizer, task: str, target_tokens: int, seed: int) -> BenchmarkCase:
    rng = random.Random(seed)
    prefix = (
        "You are given a long archive. Read it carefully. "
        "Use only the information inside the archive.\n"
    )

    if task == "single_needle":
        answer = _code(rng)
        depth = rng.choice([0.15, 0.35, 0.55, 0.75, 0.90])
        facts = [(depth, f"IMPORTANT RECORD: The access code for ORION is {answer}.")]
        question = "Question: What is the access code for ORION? Answer with the code only.\nAnswer:"
        expected = (answer,)
        mode = "all"

    elif task == "multi_key":
        alpha, beta, gamma = _code(rng), _code(rng), _code(rng)
        facts = [
            (0.20, f"KEY RECORD: ALPHA has code {alpha}."),
            (0.50, f"KEY RECORD: BETA has code {beta}."),
            (0.80, f"KEY RECORD: GAMMA has code {gamma}."),
        ]
        question = (
            "Question: Give the codes for ALPHA, BETA, and GAMMA in that order. "
            "Answer with the three codes only.\nAnswer:"
        )
        expected = (alpha, beta, gamma)
        mode = "all"

    elif task == "multi_value":
        city = rng.choice(["NADIR", "SABLE", "EMBER", "CIRRUS"])
        room = str(rng.randint(100, 999))
        locker = str(rng.randint(1000, 9999))
        pin = _code(rng, digits=5)
        facts = [
            (0.28, f"PROFILE RECORD: Project {city} uses room {room}."),
            (0.52, f"PROFILE RECORD: Project {city} uses locker {locker}."),
            (0.76, f"PROFILE RECORD: Project {city} uses PIN {pin}."),
        ]
        question = (
            f"Question: For project {city}, return room, locker, and PIN in that order. "
            "Answer with only the three values.\nAnswer:"
        )
        expected = (room, locker, pin)
        mode = "all"

    elif task == "multi_query":
        left = _code(rng)
        right = _code(rng)
        facts = [
            (0.25, f"QUERY RECORD: The AURORA authorization number is {left}."),
            (0.73, f"QUERY RECORD: The NOVA authorization number is {right}."),
        ]
        question = (
            "Question: What are the AURORA and NOVA authorization numbers, in that order? "
            "Answer with only the two numbers.\nAnswer:"
        )
        expected = (left, right)
        mode = "all"

    elif task == "variable_tracking":
        value = "VX-" + _code(rng, digits=5)
        distractor = "DX-" + _code(rng, digits=5)
        chain = (
            f"VARIABLE TRACE: A = {value}. B copies A. C copies B. D copies C. "
            f"A separate distractor variable Z = {distractor} and never enters the A-B-C-D chain."
        )
        facts = [(0.58, chain)]
        question = "Question: What final value does D hold? Answer with the value only.\nAnswer:"
        expected = (value,)
        mode = "all"

    elif task == "summary_facts":
        values = tuple(_code(rng, digits=4) for _ in range(4))
        facts = [
            (0.18, f"QUARTERLY FACT: North division shipped {values[0]} units."),
            (0.38, f"QUARTERLY FACT: South division shipped {values[1]} units."),
            (0.62, f"QUARTERLY FACT: East division shipped {values[2]} units."),
            (0.84, f"QUARTERLY FACT: West division shipped {values[3]} units."),
        ]
        question = (
            "Question: List the North, South, East, and West shipment counts in that order. "
            "Answer with only the four numbers.\nAnswer:"
        )
        expected = values
        mode = "all"

    else:
        raise KeyError(f"unknown task {task!r}")

    ids = _fit_case(
        tokenizer,
        target_tokens=target_tokens,
        prefix=prefix,
        facts=facts,
        question=question,
        seed=seed,
    )
    digest = hashlib.sha256(bytes(str((task, target_tokens, seed, expected)), "utf-8")).hexdigest()[:12]
    return BenchmarkCase(
        case_id=f"{task}-L{target_tokens}-s{seed}-{digest}",
        task=task,
        input_ids=ids,
        expected_answers=tuple(str(x) for x in expected),
        answer_mode=mode,
        metadata={
            "target_tokens": target_tokens,
            "seed": seed,
            "fact_depths": [float(d) for d, _ in facts],
        },
    )


def normalize_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\-]+", " ", text)
    return " ".join(text.split())


def score_output(text: str, expected_answers: Iterable[str]) -> dict:
    norm = normalize_text(text)
    expected = [normalize_text(x) for x in expected_answers]
    hits = [bool(x and x in norm) for x in expected]
    recall = sum(hits) / max(1, len(hits))
    return {
        "all_correct": bool(all(hits)),
        "answer_recall": float(recall),
        "expected": expected,
        "hits": hits,
    }


DEFAULT_TASKS = (
    "single_needle",
    "multi_key",
    "multi_value",
    "multi_query",
    "variable_tracking",
    "summary_facts",
)
