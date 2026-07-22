"""Test the in-memory session store."""
from __future__ import annotations

import time

from kiaomni_chat.schemas import ChatMessage
from kiaomni_chat.sessions import SessionStore


def test_create_get_clear() -> None:
    s = SessionStore()
    snap = s.create(system_prompt="You are a helpful assistant.")
    assert snap.session_id
    assert len(snap.messages) == 1
    assert snap.messages[0].role == "system"

    got = s.get(snap.session_id)
    assert got is not None
    assert got.messages[0].content == "You are a helpful assistant."

    assert s.clear(snap.session_id) is True
    assert s.get(snap.session_id) is None


def test_append_keeps_recent_only() -> None:
    s = SessionStore()
    snap = s.create()
    sid = snap.session_id
    for i in range(50):
        s.append(sid, "user", f"msg {i}")
    got = s.get(sid)
    assert got is not None
    assert len(got.messages) == 30  # _MAX_MESSAGES cap
    # Oldest messages are dropped
    assert got.messages[0].content == "msg 20"


def test_replace_overwrites() -> None:
    s = SessionStore()
    snap = s.create()
    sid = snap.session_id
    new_msgs = [ChatMessage(role="user", content="hi")]
    s.replace(sid, new_msgs)
    got = s.get(sid)
    assert got is not None
    assert got.messages == new_msgs


def test_stats() -> None:
    s = SessionStore()
    s.create()
    s.create()
    stats = s.stats()
    assert stats["active_sessions"] == 2
    assert stats["created_total"] == 2
    assert stats["max_messages"] == 30
