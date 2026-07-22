"""In-memory multi-turn session store.

No cleanup of memory on session expiry is performed — this is the v3
production stress test, and the session store is intentionally a leaky
abstraction whose lifetime is bounded only by the container.
"""
from __future__ import annotations

import time
import uuid
from threading import RLock
from typing import Any

from .schemas import ChatMessage, SessionResponse

_TTL_SECONDS: float = 30 * 60  # 30 min since last access
_MAX_MESSAGES: int = 30


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self._created_total: int = 0

    def create(self, system_prompt: str | None = None) -> SessionResponse:
        with self._lock:
            sid = uuid.uuid4().hex
            now = time.time()
            msgs: list[ChatMessage] = []
            if system_prompt:
                msgs.append(ChatMessage(role="system", content=system_prompt))
            self._sessions[sid] = {
                "messages": msgs,
                "created_at": now,
                "last_access": now,
            }
            self._created_total += 1
            return self._snapshot(sid)

    def get(self, session_id: str) -> SessionResponse | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry["last_access"] = time.time()
            return self._snapshot(session_id)

    def append(self, session_id: str, role: str, content: str) -> SessionResponse | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry["messages"].append(ChatMessage(role=role, content=content))
            if len(entry["messages"]) > _MAX_MESSAGES:
                entry["messages"] = entry["messages"][-_MAX_MESSAGES:]
            entry["last_access"] = time.time()
            return self._snapshot(session_id)

    def replace(self, session_id: str, messages: list[ChatMessage]) -> SessionResponse | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry["messages"] = list(messages)[-_MAX_MESSAGES:]
            entry["last_access"] = time.time()
            return self._snapshot(session_id)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _snapshot(self, sid: str) -> SessionResponse:
        e = self._sessions[sid]
        return SessionResponse(
            session_id=sid,
            messages=list(e["messages"]),
            tokens=0,  # populated by the engine with real token counts
            created_at=e["created_at"],
            last_access=e["last_access"],
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "created_total": self._created_total,
                "ttl_seconds": _TTL_SECONDS,
                "max_messages": _MAX_MESSAGES,
            }


_singleton: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _singleton
    if _singleton is None:
        _singleton = SessionStore()
    return _singleton
