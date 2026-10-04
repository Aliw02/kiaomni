from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from streaming_kv_policy import select_streaming_entries
from streaming_toy_model import StreamingToyStackedDecoder, total_streaming_cache_bytes


def build_hidden(length: int, d_model: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, length, d_model, generator=g) * 0.05


def postprefill(
    model,
    hidden,
    budget: int,
    *,
    layerwise: bool,
    persistent_signals: list[int] | None = None,
):
    _, caches, saliencies = model.prefill(hidden)
    peak_bytes = total_streaming_cache_bytes(caches)
    seen = int(hidden.shape[1])

    if persistent_signals is not None:
        for layer_idx, absolute_pos in enumerate(persistent_signals):
            saliencies[layer_idx][absolute_pos] += 100.0

    if layerwise:
        keeps = [
            select_streaming_entries(
                saliency.cpu().numpy(),
                cache.absolute_positions.cpu().numpy(),
                budget,
                seen,
            )
            for cache, saliency in zip(caches, saliencies)
        ]
        caches = [
            cache.compact(torch.from_numpy(keep))
            for cache, keep in zip(caches, keeps)
        ]
    else:
        stacked = np.stack([s.cpu().numpy() for s in saliencies], axis=0)
        keep = select_streaming_entries(
            stacked.mean(axis=0),
            caches[0].absolute_positions.cpu().numpy(),
            budget,
            seen,
        )
        caches = [cache.compact(torch.from_numpy(keep)) for cache in caches]

    return caches, peak_bytes


def streaming(
    model,
    hidden,
    budget: int,
    chunk_size: int,
    *,
    layerwise: bool,
    persistent_signals: list[int] | None = None,
    future_only_signal: bool = False,
):
    caches = None
    peak_bytes = 0
    total_length = int(hidden.shape[1])

    for start in range(0, total_length, chunk_size):
        end = min(total_length, start + chunk_size)
        _, caches, saliencies = model.prefill_chunk(
            hidden[:, start:end, :],
            caches,
            start,
        )
        peak_bytes = max(peak_bytes, total_streaming_cache_bytes(caches))

        if persistent_signals is not None:
            for layer_idx, absolute_pos in enumerate(persistent_signals):
                positions = caches[layer_idx].absolute_positions.cpu().numpy()
                hit = np.where(positions == absolute_pos)[0]
                signal_is_visible = (not future_only_signal) or (end == total_length)
                if len(hit) and signal_is_visible:
                    saliencies[layer_idx][int(hit[0])] += 100.0

        if caches[0].kv_length <= budget:
            continue

        if layerwise:
            new_caches = []
            for cache, saliency in zip(caches, saliencies):
                keep = select_streaming_entries(
                    saliency.cpu().numpy(),
                    cache.absolute_positions.cpu().numpy(),
                    budget,
                    end,
                )
                new_caches.append(cache.compact(torch.from_numpy(keep)))
            caches = new_caches
        else:
            if not all(
                torch.equal(cache.absolute_positions, caches[0].absolute_positions)
                for cache in caches
            ):
                raise RuntimeError("global streaming requires identical layer position sets")
            stacked = np.stack([s.cpu().numpy() for s in saliencies], axis=0)
            keep = select_streaming_entries(
                stacked.mean(axis=0),
                caches[0].absolute_positions.cpu().numpy(),
                budget,
                end,
            )
            caches = [cache.compact(torch.from_numpy(keep)) for cache in caches]

    return caches, peak_bytes


def no_eviction_equivalence(model) -> dict:
    hidden = build_hidden(96, model.d_model, seed=1)
    full_out, full_caches, _ = model.prefill(hidden)

    chunk_caches = None
    chunk_outputs = []
    for start in range(0, 96, 16):
        out, chunk_caches, _ = model.prefill_chunk(
            hidden[:, start : start + 16, :],
            chunk_caches,
            start,
        )
        chunk_outputs.append(out)

    joined = torch.cat(chunk_outputs, dim=1)
    return {
        "output_max_abs_diff": float((full_out - joined).abs().max().item()),
        "cache_key_max_abs_diff": max(
            float((a.key - b.key).abs().max().item())
            for a, b in zip(full_caches, chunk_caches)
        ),
        "cache_value_max_abs_diff": max(
            float((a.value - b.value).abs().max().item())
            for a, b in zip(full_caches, chunk_caches)
        ),
        "positions_exact": all(
            torch.equal(a.absolute_positions, b.absolute_positions)
            for a, b in zip(full_caches, chunk_caches)
        ),
    }


def main():
    torch.set_grad_enabled(False)
    model = StreamingToyStackedDecoder().eval()
    equivalence = no_eviction_equivalence(model)

    rows = []
    for length, budget, chunk_size in [
        (64, 24, 16),
        (96, 32, 16),
        (128, 40, 16),
    ]:
        hidden = build_hidden(length, model.d_model, seed=100 + length)
        signals = [
            max(5, int(length * fraction))
            for fraction in (0.18, 0.36, 0.58, 0.74)
        ]

        _, full_caches, _ = model.prefill(hidden)
        full_peak = total_streaming_cache_bytes(full_caches)

        post_global, post_global_peak = postprefill(
            model,
            hidden,
            budget,
            layerwise=False,
        )
        post_layerwise, post_layerwise_peak = postprefill(
            model,
            hidden,
            budget,
            layerwise=True,
            persistent_signals=signals,
        )
        stream_global, stream_global_peak = streaming(
            model,
            hidden,
            budget,
            chunk_size,
            layerwise=False,
        )
        stream_layerwise, stream_layerwise_peak = streaming(
            model,
            hidden,
            budget,
            chunk_size,
            layerwise=True,
            persistent_signals=signals,
        )

        # Deliberately adversarial plumbing case: importance for an old position
        # becomes visible only in the final chunk. Once streaming has evicted
        # that KV entry it cannot be resurrected.
        future_positions = [signals[0]] * model.n_layers
        stream_future, stream_future_peak = streaming(
            model,
            hidden,
            budget,
            chunk_size,
            layerwise=True,
            persistent_signals=future_positions,
            future_only_signal=True,
        )

        methods = {
            "post_global": (post_global, post_global_peak),
            "post_layerwise": (post_layerwise, post_layerwise_peak),
            "stream_global": (stream_global, stream_global_peak),
            "stream_layerwise": (stream_layerwise, stream_layerwise_peak),
            "stream_future_only": (stream_future, stream_future_peak),
        }

        token = torch.zeros(1, 1, model.d_model)
        for name, (caches, peak_bytes) in methods.items():
            _, after, _ = model.decode_one(token, caches)

            signal_survival = None
            if name in ("post_layerwise", "stream_layerwise"):
                signal_survival = sum(
                    int(position in set(cache.absolute_positions.tolist()))
                    for position, cache in zip(signals, caches)
                ) / len(signals)
            elif name == "stream_future_only":
                signal_survival = sum(
                    int(position in set(cache.absolute_positions.tolist()))
                    for position, cache in zip(future_positions, caches)
                ) / len(future_positions)

            final_bytes = total_streaming_cache_bytes(caches)
            rows.append(
                {
                    "context_tokens": length,
                    "budget": budget,
                    "chunk_size": chunk_size,
                    "method": name,
                    "full_peak_bytes": full_peak,
                    "peak_bytes": peak_bytes,
                    "final_bytes": final_bytes,
                    "peak_reduction_vs_full": full_peak / peak_bytes,
                    "final_reduction_vs_full": full_peak / final_bytes,
                    "decode_ok": all(cache.next_position == length + 1 for cache in after),
                    "post_decode_lengths": [cache.kv_length for cache in after],
                    "signal_survival": signal_survival,
                }
            )

    chunk_sensitivity = []
    hidden = build_hidden(128, model.d_model, seed=999)
    _, full_caches, _ = model.prefill(hidden)
    full_bytes = total_streaming_cache_bytes(full_caches)
    for chunk_size in (8, 16, 32):
        caches, peak = streaming(
            model,
            hidden,
            budget=40,
            chunk_size=chunk_size,
            layerwise=False,
        )
        chunk_sensitivity.append(
            {
                "context_tokens": 128,
                "budget": 40,
                "chunk_size": chunk_size,
                "peak_bytes": peak,
                "peak_reduction_vs_full": full_bytes / peak,
                "final_bytes": total_streaming_cache_bytes(caches),
                "final_reduction_vs_full": full_bytes / total_streaming_cache_bytes(caches),
            }
        )

    result = {
        "poc_type": "streaming_kv_eviction_vs_postprefill",
        "scope": "mechanical_correctness_peak_kv_and_irreversibility_only",
        "quality_claim_allowed": False,
        "equivalence": equivalence,
        "rows": rows,
        "chunk_sensitivity": chunk_sensitivity,
    }

    out = HERE.parent / "results" / "poc_streaming.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
