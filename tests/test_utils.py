"""Pure-numpy tests for kiaomni.utils — no torch / no model required."""
from __future__ import annotations

import numpy as np
import pytest

from kiaomni.utils import boxcar, select_keep


def test_boxcar_sigma_zero_returns_input_as_float32():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = boxcar(x, sigma=0)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x.astype(np.float32))


def test_boxcar_sigma_one_averages_neighbors():
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0], dtype=np.float32)
    out = boxcar(x, sigma=1)
    # index 2 averages over [1,2,3] = (0+9+0)/3 = 3.0
    assert out[2] == pytest.approx(3.0)
    # index 0 averages over [0,1] = (0+0)/2 = 0
    assert out[0] == pytest.approx(0.0)


def test_boxcar_uniform_input_uniform_output():
    x = np.full(100, 5.0, dtype=np.float32)
    out = boxcar(x, sigma=8)
    np.testing.assert_allclose(out, 5.0, atol=1e-6)


def test_select_keep_protects_sink_and_recency():
    L = 100
    score = np.zeros(L, dtype=np.float32)
    keep = select_keep(score, budget=50, L=L, n_sink=16, recency=32)
    # First 16 must all be present.
    assert set(range(16)).issubset(keep.tolist())
    # Last 32 must all be present.
    assert set(range(L - 32, L)).issubset(keep.tolist())


def test_select_keep_respects_budget():
    L = 200
    score = np.random.RandomState(0).rand(L).astype(np.float32)
    keep = select_keep(score, budget=64, L=L)
    assert len(keep) == 64


def test_select_keep_picks_top_scores_in_free_region():
    L = 100
    score = np.zeros(L, dtype=np.float32)
    score[50] = 99.0
    keep = select_keep(score, budget=16 + 32 + 1, L=L, n_sink=16, recency=32)
    assert 50 in keep.tolist()


def test_select_keep_budget_smaller_than_protection_returns_protection():
    L = 50
    score = np.zeros(L, dtype=np.float32)
    keep = select_keep(score, budget=4, L=L, n_sink=16, recency=32)
    # Budget is less than protection → just return all protected indices.
    expected = sorted(set(range(16)) | set(range(L - 32, L)))
    assert keep.tolist() == expected
