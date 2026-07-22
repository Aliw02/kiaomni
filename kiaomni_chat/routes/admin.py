"""Admin endpoints — container restart escape hatch."""
from __future__ import annotations

import os
import signal
import sys

from fastapi import APIRouter

router = APIRouter()


@router.post("/api/restart")
def restart() -> dict:
    """Graceful container restart. The current process exits; Modal respawns.

    This is the only sanctioned way to clear memory during the no-GC stress
    test, representing "an operator restarting a production service".
    """
    import threading
    def _kill() -> None:
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return {"restarting": True, "pid": os.getpid()}
