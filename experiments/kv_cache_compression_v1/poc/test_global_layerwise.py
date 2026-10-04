from __future__ import annotations

import numpy as np
import torch

from global_mask_strategy import global_keep_indices, compact_global
from layerwise_mask_strategy import layerwise_keep_indices, compact_layerwise
from toy_stacked_kv_model import ToyStackedDecoder, total_cache_bytes


def test_global_uses_same_indices_all_layers():
    sal = [np.linspace(0, 1, 48, dtype=np.float32) for _ in range(4)]
    keep = global_keep_indices(sal, 20, n_sink=2, recency=4)
    assert len(keep) == 20
    assert np.array_equal(keep, np.sort(keep))


def test_layerwise_can_specialize_at_equal_budget():
    sal = []
    peaks = [10, 18, 26, 34]
    for p in peaks:
        s = np.zeros(48, dtype=np.float32)
        s[p] = 100.0
        sal.append(s)
    keeps = layerwise_keep_indices(sal, 16, n_sink=2, recency=2)
    assert all(len(k) == 16 for k in keeps)
    assert all(p in set(k.tolist()) for p, k in zip(peaks, keeps))
    assert len({tuple(k.tolist()) for k in keeps}) > 1


def test_equal_memory_budget_global_vs_layerwise():
    model = ToyStackedDecoder(n_layers=4, d_model=32, n_heads=4).eval()
    x = torch.randn(1, 64, 32)
    _, caches, saliencies = model.prefill(x)
    gkeep = global_keep_indices(saliencies, 24)
    lkeep = layerwise_keep_indices(saliencies, 24)
    gc = compact_global(caches, gkeep)
    lc = compact_layerwise(caches, lkeep)
    assert total_cache_bytes(gc) == total_cache_bytes(lc)
    assert all(c.kv_length == 24 for c in gc)
    assert all(c.kv_length == 24 for c in lc)


def test_decode_succeeds_for_both_strategies():
    model = ToyStackedDecoder(n_layers=4, d_model=32, n_heads=4).eval()
    x = torch.randn(1, 80, 32)
    _, caches, saliencies = model.prefill(x)
    gc = compact_global(caches, global_keep_indices(saliencies, 28))
    lc = compact_layerwise(caches, layerwise_keep_indices(saliencies, 28))
    token = torch.randn(1, 1, 32)
    gout, gnext, _ = model.decode_one(token, gc)
    lout, lnext, _ = model.decode_one(token, lc)
    assert gout.shape == lout.shape == (1, 1, 32)
    assert all(c.kv_length == 29 and c.next_position == 81 for c in gnext)
    assert all(c.kv_length == 29 and c.next_position == 81 for c in lnext)
