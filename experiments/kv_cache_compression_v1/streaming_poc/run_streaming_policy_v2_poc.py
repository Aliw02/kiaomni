from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def select(scores, positions, budget, seen, extra=(), n_sink=4, recency=8):
    positions = np.asarray(positions, dtype=np.int64)
    protected = set(
        np.where(
            (positions < n_sink)
            | (positions >= max(0, seen - recency))
        )[0].tolist()
    )
    protected.update(int(v) for v in extra)
    if len(protected) > budget:
        protected = set(sorted(protected)[:budget])
    free = budget - len(protected)
    candidates = np.asarray(
        [i for i in range(len(positions)) if i not in protected],
        dtype=np.int64,
    )
    if free > 0 and len(candidates):
        k = min(free, len(candidates))
        top = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        protected.update(int(v) for v in top)
    return np.asarray(sorted(protected), dtype=np.int64)


def simulate(policy, regime, seed, *, length=512, budget=64, chunk=32):
    rng = np.random.default_rng(seed)
    signals = (90, 190, 300, 410)
    positions = np.empty((0,), dtype=np.int64)
    persistent = {}
    peak = 0

    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        positions = np.concatenate(
            [positions, np.arange(start, end, dtype=np.int64)]
        )
        saliency = rng.random(len(positions)).astype(np.float32) * 0.02

        for signal in signals:
            hit = np.where(positions == signal)[0]
            if not len(hit):
                continue
            visible = (
                (regime == "early_observable" and start <= signal < end)
                or (regime == "future_only" and end == length)
            )
            if visible:
                saliency[int(hit[0])] += 3.0

        peak = max(peak, len(positions))

        live = set(int(p) for p in positions)
        persistent = {
            p: value * 0.995
            for p, value in persistent.items()
            if p in live
        }
        for p, score in zip(positions, saliency):
            p = int(p)
            persistent[p] = max(persistent.get(p, 0.0), float(score))

        if len(positions) <= budget:
            continue

        if policy == "current":
            scores = saliency
            extra = []
        elif policy == "persistent":
            scores = np.asarray(
                [persistent[int(p)] for p in positions],
                dtype=np.float32,
            )
            extra = []
        elif policy == "persistent_landmarks":
            scores = np.asarray(
                [persistent[int(p)] for p in positions],
                dtype=np.float32,
            )
            coverage = max(1, budget // 4)
            targets = np.linspace(0, end - 1, coverage, dtype=np.int64)
            extra = {
                int(np.argmin(np.abs(positions - target)))
                for target in targets
            }
        elif policy == "persistent_blocks":
            scores = np.asarray(
                [persistent[int(p)] for p in positions],
                dtype=np.float32,
            )
            coverage = max(8, budget // 4)
            block = 8
            n_blocks = max(1, coverage // block)
            starts = np.linspace(
                0,
                max(0, end - block),
                n_blocks,
                dtype=np.int64,
            )
            extra_list = []
            for bs in starts:
                extra_list.extend(
                    np.where(
                        (positions >= bs)
                        & (positions < bs + block)
                    )[0].tolist()
                )
            extra = set(extra_list[:coverage])
        else:
            raise KeyError(policy)

        keep = select(scores, positions, budget, end, extra=extra)
        positions = positions[keep]
        persistent = {
            int(p): persistent[int(p)]
            for p in positions
        }

    survived = sum(
        int(signal in set(positions.tolist()))
        for signal in signals
    ) / len(signals)
    return {
        "signal_survival": float(survived),
        "peak_entries": int(peak),
        "final_entries": int(len(positions)),
    }


def main():
    policies = (
        "current",
        "persistent",
        "persistent_landmarks",
        "persistent_blocks",
    )
    regimes = ("early_observable", "future_only")
    rows = []
    for regime in regimes:
        for policy in policies:
            trials = [
                simulate(policy, regime, seed)
                for seed in range(50)
            ]
            rows.append(
                {
                    "regime": regime,
                    "policy": policy,
                    "trials": len(trials),
                    "mean_signal_survival": float(
                        np.mean(
                            [r["signal_survival"] for r in trials]
                        )
                    ),
                    "mean_peak_entries": float(
                        np.mean([r["peak_entries"] for r in trials])
                    ),
                    "mean_final_entries": float(
                        np.mean([r["final_entries"] for r in trials])
                    ),
                }
            )

    result = {
        "poc": "streaming_policy_v2",
        "length": 512,
        "budget": 64,
        "chunk": 32,
        "retention_fraction": 0.125,
        "quality_claim_allowed": False,
        "rows": rows,
        "promotion": {
            "winner_for_real_smoke": "persistent",
            "reason": (
                "Persistent scoring preserves early-observable late-needed "
                "signals without increasing peak/final KV. Future-only "
                "importance remains fundamentally unresolved under a fixed "
                "sub-full budget."
            ),
        },
    }
    out = HERE.parent / "results" / "poc_streaming_policy_v2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
