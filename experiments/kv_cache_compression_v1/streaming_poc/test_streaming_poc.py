from __future__ import annotations

import torch

from run_streaming_poc import build_hidden, no_eviction_equivalence, streaming
from streaming_toy_model import StreamingToyStackedDecoder, total_streaming_cache_bytes


def test_chunking_without_eviction_is_exact():
    model = StreamingToyStackedDecoder().eval()
    result = no_eviction_equivalence(model)
    assert result["output_max_abs_diff"] == 0.0
    assert result["cache_key_max_abs_diff"] == 0.0
    assert result["cache_value_max_abs_diff"] == 0.0
    assert result["positions_exact"] is True


def test_streaming_global_reduces_prefill_peak_kv():
    model = StreamingToyStackedDecoder().eval()
    hidden = build_hidden(128, model.d_model, seed=123)
    _, full_caches, _ = model.prefill(hidden)
    full_bytes = total_streaming_cache_bytes(full_caches)

    caches, peak = streaming(
        model,
        hidden,
        budget=40,
        chunk_size=16,
        layerwise=False,
    )
    assert peak < full_bytes
    assert total_streaming_cache_bytes(caches) < peak
    assert all(cache.kv_length == 40 for cache in caches)
    assert all(cache.next_position == 128 for cache in caches)


def test_streaming_layerwise_keeps_equal_memory_budget():
    model = StreamingToyStackedDecoder().eval()
    hidden = build_hidden(96, model.d_model, seed=124)
    signals = [17, 33, 49, 65]
    caches, _ = streaming(
        model,
        hidden,
        budget=32,
        chunk_size=16,
        layerwise=True,
        persistent_signals=signals,
    )
    assert all(cache.kv_length == 32 for cache in caches)
    assert all(
        position in set(cache.absolute_positions.tolist())
        for position, cache in zip(signals, caches)
    )


def test_streaming_decode_continues_with_absolute_position():
    model = StreamingToyStackedDecoder().eval()
    hidden = build_hidden(80, model.d_model, seed=125)
    caches, _ = streaming(
        model,
        hidden,
        budget=28,
        chunk_size=16,
        layerwise=False,
    )
    token = torch.zeros(1, 1, model.d_model)
    out, after, _ = model.decode_one(token, caches)
    assert out.shape == (1, 1, model.d_model)
    assert all(cache.kv_length == 29 for cache in after)
    assert all(cache.next_position == 81 for cache in after)


def test_smaller_chunk_lowers_streaming_peak_for_same_budget():
    model = StreamingToyStackedDecoder().eval()
    hidden = build_hidden(128, model.d_model, seed=126)
    _, peak8 = streaming(model, hidden, budget=40, chunk_size=8, layerwise=False)
    _, peak32 = streaming(model, hidden, budget=40, chunk_size=32, layerwise=False)
    assert peak8 < peak32
