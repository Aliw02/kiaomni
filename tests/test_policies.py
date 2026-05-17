"""Tests for the policy registry."""
from __future__ import annotations

import numpy as np
import pytest

from kiaomni.policies import POLICY_REGISTRY, get_policy, register_policy


def test_builtin_policies_present():
    assert "kiaomni_s8" in POLICY_REGISTRY
    assert "kiaomni_gaussian" in POLICY_REGISTRY


def test_get_policy_unknown_raises():
    with pytest.raises(KeyError):
        get_policy("definitely_does_not_exist")


def test_register_policy_adds_and_runs():
    register_policy("square", lambda s: s ** 2)
    fn = get_policy("square")
    out = fn(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([1.0, 4.0, 9.0], dtype=np.float32))


def test_register_policy_rejects_non_callable():
    with pytest.raises(TypeError):
        register_policy("bogus", "not a function")  # type: ignore[arg-type]


def test_kiaomni_s8_returns_same_length():
    sal = np.random.RandomState(0).rand(128).astype(np.float32)
    out = get_policy("kiaomni_s8")(sal)
    assert out.shape == sal.shape
    assert out.dtype == np.float32
