"""Pytest fixtures for the kiaomni_chat test suite.

We do **not** install the kiaomni package in the local sanity-check
environment — that is reserved for the Modal image. Locally we:
  * compile-check every module,
  * test the kiaomni-free components (tasks, schemas, sessions, telemetry),
  * mock the kiaomni import for engine-level structural tests.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))


# ── Mock kiaomni before any engine import so app/engine modules can load ──
class _MockKiaomni:
    """Minimal stub matching the kiaomni public surface we touch."""

    def apply_kiaomni(self, model, policy: str = "kiaomni_s8", budget: int = 256, **kw):
        self._patched = True
        return types.SimpleNamespace(
            policy=policy, budget=budget,
            confidence="high", qkv_pattern="separate",
        )

    def remove_kiaomni(self, model) -> None:
        self._patched = False

    def get_policy(self, name: str):
        return lambda x: x

    def register_policy(self, name, fn):
        return None

    POLICY_REGISTRY = {"kiaomni_s8": lambda x: x, "kiaomni_gaussian": lambda x: x}

    class _Probe:
        @staticmethod
        def probe(m): return None
    ArchitectureProbe = _Probe

    @property
    def __version__(self): return "0.3.0-mock"


def _install_kiaomni_mock() -> None:
    if "kiaomni" in sys.modules:
        return
    mod = types.ModuleType("kiaomni")
    mod.__version__ = "0.3.0-mock"
    mod.apply_kiaomni = _MockKiaomni().apply_kiaomni
    mod.remove_kiaomni = _MockKiaomni().remove_kiaomni
    mod.POLICY_REGISTRY = _MockKiaomni.POLICY_REGISTRY
    mod.get_policy = _MockKiaomni.get_policy
    mod.register_policy = _MockKiaomni.register_policy
    mod.ArchitectureProbe = _MockKiaomni.ArchitectureProbe
    mod.load_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mocked"))
    sys.modules["kiaomni"] = mod
    sys.modules["kiaomni.adapters"] = types.ModuleType("kiaomni.adapters")
    sys.modules["kiaomni.adapters.__init__"] = types.ModuleType("kiaomni.adapters.__init__")
    sys.modules["kiaomni.adapters.probe"] = types.ModuleType("kiaomni.adapters.probe")
    sys.modules["kiaomni.adapters.saliency"] = types.ModuleType("kiaomni.adapters.saliency")
    sys.modules["kiaomni.loading"] = types.ModuleType("kiaomni.loading")
    sys.modules["kiaomni.monkey_patch"] = types.ModuleType("kiaomni.monkey_patch")
    sys.modules["kiaomni.policies"] = types.ModuleType("kiaomni.policies")
    sys.modules["kiaomni.utils"] = types.ModuleType("kiaomni.utils")
    # Lazy imports used by _select_keep_with_chat_template_mask
    sys.modules["kiaomni.adapters.probe"].ArchitectureProbe = _MockKiaomni.ArchitectureProbe
    sys.modules["kiaomni.adapters.saliency"].SaliencyAdapter = lambda *a: type("SA", (), {"extract": lambda *a: np.zeros((1, 128), dtype=np.float32)})()
    sys.modules["kiaomni.policies"].get_policy = _MockKiaomni.get_policy
    sys.modules["kiaomni.utils"].select_keep = lambda *a, **kw: np.arange(min(a[1], a[2]), dtype=np.int64)


@pytest.fixture(scope="session", autouse=True)
def _mock_kiaomni() -> None:
    _install_kiaomni_mock()
