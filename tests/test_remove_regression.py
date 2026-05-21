"""Regression tests for the v0.2.4 `remove_kiaomni` over-unwinding bug.

History: v0.2.2 added a `__wrapped__` chain walker to `remove_kiaomni` to
handle stacked patches. That walker followed `functools.wraps` pointers
through `@torch.no_grad()` — the decorator HuggingFace `generate` is
wrapped with — and landed on the **raw unbound** `generate(self, ...)`
function. Reassigning that as an instance attribute made every subsequent
`model.generate(ids, ...)` call evaluate as `generate(self=ids)` — the
Tensor became `self`, crashing on the first attribute access.

These tests pin the v0.2.4 behavior: `remove_kiaomni` MUST leave
`model.generate` resolving to a properly bound method that, when called
with a Tensor, treats the Tensor as the input — never as `self`.
"""
from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")


class _FakeGenerationMixin:
    """Mimics HF GenerationMixin: a `generate` method decorated such that
    `__wrapped__` points to a different (unbound) callable. This reproduces
    the over-unwind trap without downloading a real HF model."""

    def __init__(self) -> None:
        self.name = "fake"

    @staticmethod
    def _inner_generate(self, x):
        # If `self` is anything other than the model instance, this fails.
        return f"{self.name}::{x}"

    # Wrap so that `generate.__wrapped__` points to `_inner_generate`,
    # exactly like `@torch.no_grad()` does to HF generate.
    @functools.wraps(_inner_generate)
    def generate(self, x):
        return _FakeGenerationMixin._inner_generate(self, x)


def test_remove_kiaomni_does_not_unwrap_past_bound_method():
    """After remove_kiaomni, model.generate must still bind self correctly.

    Reproduces the exact failure mode from the Kaggle Qwen-7B trace where
    `@torch.no_grad()`'s `__wrapped__` pointer caused remove_kiaomni to
    install the raw unbound function as `model.generate`.
    """
    from kiaomni.monkey_patch import remove_kiaomni

    model = _FakeGenerationMixin()
    # Sanity: baseline works.
    assert model.generate("hello") == "fake::hello"

    # Simulate a prior apply_kiaomni having left state on the model.
    # remove_kiaomni must restore generate to a working bound state.
    object.__setattr__(model, "_kia_arch_info", object())  # marker
    remove_kiaomni(model)

    # The critical assertion: generate still works as a bound method.
    # If remove_kiaomni had reassigned the unbound `_inner_generate`,
    # this call would raise AttributeError (Tensor-as-self analogue) or
    # produce wrong output.
    result = model.generate("world")
    assert result == "fake::world", (
        f"remove_kiaomni corrupted generate binding: got {result!r}"
    )

    # _kia_arch_info must be cleaned up too.
    assert not hasattr(model, "_kia_arch_info")


def test_remove_kiaomni_is_safe_on_unpatched_model():
    """Calling remove_kiaomni on a fresh model must be a no-op, not a
    booby trap. Users invoke it defensively in test loops."""
    from kiaomni.monkey_patch import remove_kiaomni

    model = _FakeGenerationMixin()
    remove_kiaomni(model)
    remove_kiaomni(model)  # idempotent
    assert model.generate("x") == "fake::x"


def test_remove_kiaomni_handles_tensor_first_arg():
    """The original Kaggle crash: passing a Tensor to generate after
    remove_kiaomni made the Tensor become self. Verify that doesn't
    happen — the Tensor must reach generate as the input argument."""
    from kiaomni.monkey_patch import remove_kiaomni

    model = _FakeGenerationMixin()
    object.__setattr__(model, "_kia_arch_info", object())
    remove_kiaomni(model)

    t = torch.tensor([1, 2, 3])
    # If remove_kiaomni broke binding, this would either crash with
    # AttributeError on `self.name` (Tensor has no .name) or
    # silently produce "tensor(...)::tensor(...)".
    result = model.generate(t)
    assert result.startswith("fake::"), (
        f"Tensor leaked into self position: {result!r}"
    )
