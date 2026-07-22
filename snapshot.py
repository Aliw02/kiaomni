"""Snapshot script — captures the full kiaomni_chat project state to a
single JSON file in the repo for restoration and later reference.

Captures:
  * All source files (text content as utf-8)
  * Live deployment metadata (Modal app, image hash, URL)
  * Live telemetry from the deployed instance (timing, tokens, tok/s,
    VRAM, real outputs)

Run:
  python snapshot.py                       # save to kiaomni_chat_snapshot.json
  python snapshot.py --out my_backup.json  # custom output path
  python restore.py kiaomni_chat_snapshot.json   # restore files
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
PACKAGE_DIRS = [REPO / "kiaomni_chat"]
TOP_LEVEL_FILES = [REPO / "modal_app.py"]
DEPLOY_URL = "https://aliweyabood--kiaomni-chat-serve.modal.run"

# File extensions we snapshot as TEXT (utf-8) — every other file is base64.
TEXT_EXTS = {".py", ".html", ".css", ".js", ".md", ".sh", ".txt", ".json", ".toml", ".yml", ".yaml"}
SKIP_EXTS = {".pyc", ".pyd", ".so", ".dll", ".dylib"}
SKIP_DIR_NAMES = {"__pycache__", ".git", "node_modules", ".pytest_cache", ".venv", "venv"}
SKIP_FILES = {".DS_Store"}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _walk_files() -> list[Path]:
    out: list[Path] = []
    for root in PACKAGE_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.name in SKIP_FILES:
                continue
            if path.suffix.lower() in SKIP_EXTS:
                continue
            out.append(path)
    for f in TOP_LEVEL_FILES:
        if f.exists():
            out.append(f)
    return sorted(out, key=lambda p: str(p))


def _file_record(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(REPO))
    stat = path.stat()
    is_text = path.suffix.lower() in TEXT_EXTS
    rec: dict[str, Any] = {
        "path": rel,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "encoding": "utf-8" if is_text else "base64",
    }
    try:
        if is_text:
            rec["content"] = path.read_text(encoding="utf-8", errors="replace")
        else:
            rec["content_b64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"read failed: {exc!r}"
    return rec


def _modal_apps() -> list:
    try:
        out = subprocess.check_output(
            ["modal", "app", "list"], text=True, timeout=30,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"modal app list failed: {exc!r}"}]
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        if "kiaomni-chat" in line:
            rows.append({"raw": line.strip()})
    return rows


def _live_telemetry() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{DEPLOY_URL}/api/health", timeout=15) as r:
            health = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        health = {"error": f"health fetch failed: {exc!r}"}
    try:
        with urllib.request.urlopen(f"{DEPLOY_URL}/api/telemetry", timeout=15) as r:
            telemetry = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        telemetry = {"error": f"telemetry fetch failed: {exc!r}"}
    return {"health": health, "telemetry": telemetry}


def build_snapshot() -> dict[str, Any]:
    files = [_file_record(p) for p in _walk_files()]
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "deploy_url": DEPLOY_URL,
        "modal_apps": _modal_apps(),
        "live": _live_telemetry(),
        "files": files,
        "summary": {
            "total_files": len(files),
            "total_bytes": sum(f["size"] for f in files),
        },
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "kiaomni_chat_snapshot.json"))
    args = ap.parse_args()
    snap = build_snapshot()
    out = Path(args.out)
    out.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[snapshot] wrote {out} — {snap['summary']['total_files']} files, "
          f"{snap['summary']['total_bytes']:,} bytes")
    if isinstance(snap["live"].get("health"), dict) and snap["live"]["health"].get("ready"):
        t = snap["live"]["telemetry"]
        print(f"[snapshot] live: model ready, vram={snap['live']['health'].get('vram_allocated_mb', 0):.0f}MB, "
              f"requests={t.get('stats', {}).get('total_requests', 0)}, "
              f"oom={t.get('oom_count', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
