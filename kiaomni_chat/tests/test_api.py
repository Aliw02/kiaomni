"""End-to-end FastAPI test using TestClient (no model load)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from kiaomni_chat.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert "model" in body
    assert "gpu" in body
    assert "kiaomni_version" in body
    assert body["kiaomni_version"]  # truthy (varies by env: "0.3.0" in prod, "0.3.0-mock" in test)
    assert isinstance(body["kiaomni_version"], str)


def test_root_serves_index_html() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "KiaOmni Chat" in r.text


def test_static_style_css() -> None:
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]
    assert "--kia:" in r.text


def test_static_app_js() -> None:
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "renderPanel" in r.text


def test_telemetry_endpoint_shape() -> None:
    r = client.get("/api/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert "snapshots" in body
    assert "requests" in body
    assert "oom_count" in body
    assert "uptime_s" in body


def test_session_create_get_clear() -> None:
    r = client.post("/api/session/create", json={"system_prompt": "You are a test bot."})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    r = client.get(f"/api/session/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are a test bot."
    r = client.delete(f"/api/session/{sid}")
    assert r.status_code == 200
    assert r.json()["cleared"] is True


def test_session_stats() -> None:
    r = client.get("/api/session/stats")
    assert r.status_code == 200
    body = r.json()
    assert "active_sessions" in body
    assert "max_messages" in body


def test_chat_returns_not_ready_stream() -> None:
    """The engine is not loaded — the chat endpoint should yield an SSE error."""
    r = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "policy": "kiaomni_gaussian",
            "budget": 512,
            "max_new_tokens": 16,
        },
    )
    # Either 200 (streaming) or 503 — depends on whether engine is ready
    assert r.status_code in (200, 503)


def test_chat_request_validation() -> None:
    """Invalid budgets / policies must be rejected."""
    r = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "policy": "not_a_real_policy",
            "budget": 512,
            "max_new_tokens": 16,
        },
    )
    assert r.status_code == 422


def test_compare_request_validation() -> None:
    r = client.post(
        "/api/compare",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "budget": 512,
            "max_new_tokens": 16,
        },
    )
    # Engine not loaded → 503 is acceptable
    assert r.status_code in (200, 503)


def test_demo_request_validation() -> None:
    r = client.post(
        "/api/demo/run",
        json={"task": "invalid_task", "policy": "kiaomni_gaussian", "budget": 512, "n_samples": 1},
    )
    assert r.status_code == 422
