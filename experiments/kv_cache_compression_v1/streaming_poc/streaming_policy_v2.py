from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from streaming_kv_policy import select_streaming_entries


@dataclass
class PersistentScoreState:
    """Per-absolute-position historical saliency memory."""
    values: dict[int, float]
    decay: float = 0.995

    @classmethod
    def empty(cls, decay: float = 0.995) -> "PersistentScoreState":
        return cls(values={}, decay=float(decay))

    def update(
        self,
        absolute_positions: np.ndarray,
        current_saliency: np.ndarray,
    ) -> np.ndarray:
        absolute_positions = np.asarray(absolute_positions, dtype=np.int64)
        current_saliency = np.asarray(current_saliency, dtype=np.float32)
        if len(absolute_positions) != len(current_saliency):
            raise ValueError("position/saliency length mismatch")

        live = set(int(p) for p in absolute_positions)
        self.values = {
            int(p): float(v) * self.decay
            for p, v in self.values.items()
            if int(p) in live
        }
        for p, s in zip(absolute_positions, current_saliency):
            p = int(p)
            self.values[p] = max(self.values.get(p, 0.0), float(s))
        return np.asarray(
            [self.values[int(p)] for p in absolute_positions],
            dtype=np.float32,
        )

    def retain_only(self, absolute_positions: np.ndarray) -> None:
        live = set(int(p) for p in np.asarray(absolute_positions, dtype=np.int64))
        self.values = {
            p: v for p, v in self.values.items() if p in live
        }


def select_persistent_entries(
    state: PersistentScoreState,
    current_saliency: np.ndarray,
    absolute_positions: np.ndarray,
    budget: int,
    seen_length: int,
    *,
    n_sink: int = 4,
    recency: int = 8,
    sigma: float = 4.0,
) -> np.ndarray:
    persistent = state.update(absolute_positions, current_saliency)
    return select_streaming_entries(
        persistent,
        absolute_positions,
        budget,
        seen_length,
        n_sink=n_sink,
        recency=recency,
        sigma=sigma,
    )
