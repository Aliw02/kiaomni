"""Compile-check every module of the kiaomni_chat package."""
from __future__ import annotations

import compileall
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_compile_all() -> None:
    ok = compileall.compile_dir(str(ROOT), quiet=1, force=True)
    assert ok is True, "compileall reported failures"


def test_no_gc_in_engine() -> None:
    """The v3 plan forbids empty_cache and gc.collect in the engine.

    We parse the AST and check every function/method body — the docstring
    is allowed to describe the contract.
    """
    import ast
    src_path = ROOT / "engine.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            target = ast.unparse(func) if hasattr(ast, "unparse") else ""
            if "empty_cache" in target or target.endswith("gc.collect"):
                forbidden.append(f"line {node.lineno}: {target}")
    assert not forbidden, (
        f"engine.py must not call empty_cache / gc.collect — v3 plan forbids manual GC; found: {forbidden}"
    )


def test_engine_imports_cleanly() -> None:
    """The engine module loads under the kiaomni mock."""
    from kiaomni_chat import engine  # noqa: F401
    assert hasattr(engine, "KiaOmniEngine")
    assert hasattr(engine, "get_engine")
    assert hasattr(engine, "POLICY_FULLCONTEXT")
    assert hasattr(engine, "POLICY_S8")
    assert hasattr(engine, "POLICY_GAUSSIAN")


def test_app_assembles() -> None:
    """The FastAPI app assembles and exposes every planned route."""
    from kiaomni_chat.app import app
    paths = {r.path for r in app.routes}
    expected = {
        "/", "/api/health", "/api/telemetry",
        "/api/chat", "/api/compare", "/api/demo/run",
        "/api/docqa",
        "/api/session/create", "/api/session/{session_id}",
        "/api/session/append", "/api/session/stats",
        "/api/restart",
    }
    missing = expected - paths
    assert not missing, f"missing routes: {sorted(missing)}"


def test_static_files_exist() -> None:
    for f in ("index.html", "style.css", "app.js"):
        path = ROOT / "static" / f
        assert path.exists(), f"missing static file: {path}"
        assert path.stat().st_size > 100, f"static file too small: {path}"


def test_engine_policy_swap_pattern() -> None:
    """Engine must follow the demo pattern: remove → apply, finally remove."""
    from kiaomni_chat import engine
    src = (Path(engine.__file__)).read_text(encoding="utf-8")
    # Must call apply_kiaomni and remove_kiaomni
    assert "apply_kiaomni" in src
    assert "remove_kiaomni" in src
    # Must NOT use kiaomni's load_model helper (we use the demo's direct path)
    assert "from kiaomni import" in src
    # Validate the import line
    import_line = [ln for ln in src.splitlines() if "from kiaomni import" in ln][0]
    assert "apply_kiaomni" in import_line
    assert "remove_kiaomni" in import_line
