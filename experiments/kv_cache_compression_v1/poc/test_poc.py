from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_gaussian_policy import kv_gaussian_score, select_kv_positions
from toy_kv_model import ToyCausalAttention


def test_gaussian_shape_and_order():
    sal = np.zeros(32, dtype=np.float32)
    sal[10] = 100.0
    score = kv_gaussian_score(sal)
    assert score.shape == sal.shape
    keep = select_kv_positions(sal, budget=16, n_sink=2, recency=2)
    assert 10 in keep
    assert 0 in keep and 1 in keep and 30 in keep and 31 in keep


def test_identity_compaction_is_exact():
    model = ToyCausalAttention().eval()
    x = torch.randn(1, 20, 32)
    _, cache, _ = model.prefill(x)
    same = cache.compact(torch.arange(cache.kv_length))
    assert torch.equal(cache.key, same.key)
    assert torch.equal(cache.value, same.value)
    assert torch.equal(cache.absolute_positions, same.absolute_positions)
    assert cache.next_position == same.next_position


def test_physical_reduction_and_decode():
    model = ToyCausalAttention().eval()
    x = torch.randn(1, 40, 32)
    _, cache, sal = model.prefill(x)
    keep = select_kv_positions(sal.numpy(), budget=16, n_sink=2, recency=4)
    compact = cache.compact(torch.from_numpy(keep))
    assert compact.kv_length == 16
    assert compact.bytes < cache.bytes
    assert compact.next_position == 40
    assert compact.kv_length != compact.next_position
    y = torch.randn(1, 1, 32)
    out, after, _ = model.decode_one(y, compact)
    assert out.shape == (1, 1, 32)
    assert after.kv_length == 17
    assert after.next_position == 41
