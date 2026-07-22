"""Restore script — reads a snapshot JSON and writes files back to disk.

Usage:
  python restore.py kiaomni_chat_snapshot.json            # restore all files
  python restore.py kiaomni_chat_snapshot.json --dry-run  # show what would be written
  python restore.py kiaomni_chat_snapshot.json --only engine.py tasks.py
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def restore(snap: dict, only: set[str] | None, dry_run: bool) -> tuple[int, int, list[str]]:
    written = 0
    skipped = 0
    errors: list[str] = []
    files = snap.get("files", [])
    for f in files:
        path = REPO / f["path"]
        if only and f["path"] not in only and Path(f["path"]).name not in only:
            skipped += 1
            continue
        try:
            if dry_run:
                print(f"[dry-run] would write {path} ({f['size']} bytes)")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if f.get("encoding") == "utf-8":
                content = f.get("content", "")
                path.write_text(content, encoding="utf-8")
            elif f.get("encoding") == "base64":
                content = base64.b64decode(f.get("content_b64", ""))
                path.write_bytes(content)
            else:
                errors.append(f"unknown encoding for {f['path']}: {f.get('encoding')}")
                continue
            written += 1
            print(f"[restore] wrote {path} ({f['size']} bytes)")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{f['path']}: {exc!r}")
    return written, skipped, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="restore only these file paths or basenames")
    args = ap.parse_args()
    if not args.snapshot.exists():
        print(f"file not found: {args.snapshot}", file=sys.stderr)
        return 1
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    only = set(args.only or [])
    written, skipped, errors = restore(snap, only, args.dry_run)
    print(f"\n[restore] written={written} skipped={skipped} errors={len(errors)}")
    for e in errors:
        print(f"  ! {e}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
