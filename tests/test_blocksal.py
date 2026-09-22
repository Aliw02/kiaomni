import numpy as np
import pytest

from kiaomni.blocksal import select_blocksal_keep


def test_blocksal_exact_budget_and_protection():
    L = 400
    sal = np.linspace(0.0, 1.0, L, dtype=np.float32)
    out = select_blocksal_keep(sal, budget=98, L=L, block_size=16)
    keep = set(out.keep_indices.tolist())

    assert 98 - 15 <= len(keep) <= 98
    assert set(range(16)).issubset(keep)
    assert set(range(L - 32, L)).issubset(keep)
    assert out.block_size == 16


def test_blocksal_prefers_higher_mean_blocks():
    L = 128
    sal = np.zeros(L, dtype=np.float32)
    sal[32:40] = 10.0
    sal[40:48] = 1.0

    out = select_blocksal_keep(
        sal,
        budget=64,
        L=L,
        block_size=16,
        n_sink=16,
        recency=32,
    )
    keep = set(out.keep_indices.tolist())

    assert set(range(32, 48)).issubset(keep)


def test_blocksal_full_context_returns_all_positions():
    sal = np.arange(64, dtype=np.float32)
    out = select_blocksal_keep(sal, budget=64, L=64)
    np.testing.assert_array_equal(out.keep_indices, np.arange(64))


def test_blocksal_rejects_budget_below_protected_count():
    sal = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        select_blocksal_keep(sal, budget=40, L=100, n_sink=16, recency=32)
