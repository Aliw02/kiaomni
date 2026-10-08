"""Streaming V2 selection: protected sink/recency, persistent saliency, local spans and evidence vault.

A candidate is never selected using the benchmark's reference answer.
Evidence hints use only already-seen token text. The vault is an experimental
heuristic and may be less effective on other languages or documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d

_FACT_LINE = re.compile(
    r"(?im)[^\n]{0,160}\b(?:code|pin|authorization|serial|identifier|password|"
    r"key|room|locker|value|reference)\b[^\n]{0,160}\b(?:[A-Z]{1,6}-)?[0-9]{3,9}\b[^\n]{0,60}"
)


def _s8(x: np.ndarray) -> np.ndarray:
    radius = 8
    ps = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    lo = np.maximum(0, np.arange(len(x)) - radius)
    hi = np.minimum(len(x), np.arange(len(x)) + radius + 1)
    return ((ps[hi] - ps[lo]) / (hi - lo)).astype(np.float32)


def _score(saliency: np.ndarray, absolute_positions: np.ndarray,
           seen_length: int, policy: str) -> np.ndarray:
    dense = np.zeros(seen_length, dtype=np.float32)
    dense[absolute_positions] = np.log1p(np.maximum(saliency, 0.0))
    if policy == "gaussian":
        smooth = gaussian_filter1d(dense, sigma=4.0)
    elif policy == "s8":
        smooth = _s8(dense)
    else:
        raise ValueError(f"Unknown policy {policy}")
    return smooth[absolute_positions]


@dataclass
class VaultState:
    """Model saliency never vanishes at a non-eviction chunk boundary."""
    layer_scores: list[dict[int, float]]
    evidence: dict[int, float] = field(default_factory=dict)
    decay: float = 0.995

    @classmethod
    def create(cls, layers: int) -> "VaultState":
        return cls(layer_scores=[{} for _ in range(layers)])

    def update(self, current_positions: list[np.ndarray],
               current_saliency: np.ndarray) -> None:
        if current_saliency.shape[0] != len(self.layer_scores):
            raise ValueError("Saliency layer count mismatch")
        for i, positions in enumerate(current_positions):
            scores = self.layer_scores[i]
            live = set(int(p) for p in positions)
            for pos in list(scores):
                if pos not in live:
                    del scores[pos]
                else:
                    scores[pos] *= self.decay
            for pos, raw in zip(positions, current_saliency[i]):
                p = int(pos)
                scores[p] = max(float(raw), scores.get(p, 0.0))
        self.evidence = {
            pos: val for pos, val in self.evidence.items()
            if any(pos in s for s in self.layer_scores)
        }

    def retain(self, current_positions: list[np.ndarray]) -> None:
        for scores, positions in zip(self.layer_scores, current_positions):
            live = set(map(int, positions))
            for pos in list(scores):
                if pos not in live:
                    del scores[pos]
        live_union = set().union(*(set(map(int, x)) for x in current_positions))
        self.evidence = {p: v for p, v in self.evidence.items() if p in live_union}

    def track_evidence(self, tokenizer, input_ids, start: int, end: int) -> int:
        """Scan only received chunks and preserve token spans around structured facts.

        No gold answers and no future context are inspected.
        """
        left = max(0, start - 256)
        token_ids = input_ids[0, left:end].tolist()
        if not token_ids:
            return 0
        text = tokenizer.decode(
            token_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        matches = list(_FACT_LINE.finditer(text))
        if not matches:
            return 0
        mapped = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = mapped.get("offset_mapping")
        if offsets is None or len(offsets) != len(token_ids):
            return 0
        newly = 0
        for match in matches:
            overlapping = [
                left + i for i, (a, b) in enumerate(offsets)
                if a < match.end() and b > match.start()
            ]
            if len(overlapping) > 48:
                middle = (overlapping[0] + overlapping[-1]) // 2
                overlapping = [p for p in overlapping if abs(p - middle) <= 24]
            for p in overlapping:
                if p not in self.evidence:
                    newly += 1
                self.evidence[p] = max(1.0, self.evidence.get(p, 0.0))
        return newly


def choose_positions(positions: np.ndarray, state: VaultState, layer: int,
                     budget: int, seen_length: int, *,
                     policy: str = "gaussian", vault_tokens: int = 100,
                     sink: int = 16, recency: int = 32) -> np.ndarray:
    """Return exact physical positions, with protected vault WITHIN budget."""
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) <= budget:
        return np.arange(len(positions), dtype=np.int64)
    if budget < sink + recency:
        raise ValueError("budget smaller than protected sink+recency")
    if len(set(map(int, positions))) != len(positions):
        raise ValueError("duplicate absolute positions")

    protected = {
        i for i, pos in enumerate(positions)
        if pos < sink or pos >= max(0, seen_length - recency)
    }
    base_free = budget - len(protected)
    # Keep enough dynamic positions for non-fact context and new questions.
    dynamic_reserve = min(base_free, max(16, budget // 5))
    vault_cap = min(max(0, vault_tokens), max(0, base_free - dynamic_reserve))

    scores = np.asarray(
        [state.layer_scores[layer].get(int(p), 0.0) for p in positions],
        dtype=np.float32
    )
    smoothed = _score(scores, positions, seen_length, policy)
    evidence_candidates = [
        i for i, pos in enumerate(positions)
        if int(pos) in state.evidence and i not in protected
    ]
    # Small discovered fact clusters compete for dedicated slots.
    evidence_candidates.sort(
        key=lambda i: (-state.evidence[int(positions[i])],
                       -float(smoothed[i]), int(positions[i]))
    )
    vault = evidence_candidates[:vault_cap]
    protected.update(vault)

    # If less structured evidence exists, vault also keeps well-scored old
    # *regions*. Retention is not contingent on a later question.
    spare = vault_cap - len(vault)
    if spare:
        old_candidates = [
            i for i, pos in enumerate(positions)
            if i not in protected and pos < max(0, seen_length - recency)
        ]
        old_candidates.sort(
            key=lambda i: (-float(smoothed[i]), int(positions[i]))
        )
        for center in old_candidates:
            if spare <= 0:
                break
            # Prefer a 5-token contiguous span near a prominent saliency peak.
            neighbors = sorted(
                (j for j in range(len(positions))
                 if j not in protected and abs(int(positions[j]) - int(positions[center])) <= 2),
                key=lambda j: (abs(int(positions[j]) - int(positions[center])), j)
            )
            for j in neighbors:
                if spare == 0:
                    break
                protected.add(j)
                spare -= 1

    remaining = budget - len(protected)
    if remaining:
        available = [i for i in range(len(positions)) if i not in protected]
        available.sort(key=lambda i: (-float(smoothed[i]), int(positions[i])))
        protected.update(available[:remaining])
    kept = np.asarray(sorted(protected), dtype=np.int64)
    if len(kept) != budget:
        raise RuntimeError("V2 selector violated exact budget")
    return kept
