"""Run a real chat query against the deployed app, capture every metric
and the full response, then save to results_<timestamp>.json.

Demonstrates: time, tokens in, tokens kept, compression, prefill ms,
decode ms, tok/s, VRAM, real output text.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://aliweyabood--kiaomni-chat-serve.modal.run"


def _post_json(path: str, payload: dict, timeout: int = 300) -> dict:
    """POST a JSON body, expect a JSON response. For SSE, use _post_sse."""
    req = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_sse(path: str, payload: dict, timeout: int = 600) -> dict:
    """POST a JSON body, parse the SSE response stream into a dict."""
    req = urllib.request.Request(
        f"{URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "accept": "text/event-stream"},
        method="POST",
    )
    tokens: list[str] = []
    status: dict | None = None
    stats: dict | None = None
    error: dict | None = None
    buf = b""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        while True:
            chunk = r.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                for line in frame.split(b"\n"):
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        ev = json.loads(line[6:].decode("utf-8"))
                    except Exception:
                        continue
                    t = ev.get("type")
                    if t == "token":
                        tokens.append(ev.get("text", ""))
                    elif t == "status" and ev.get("phase") == "prefill":
                        status = ev
                    elif t == "stats":
                        stats = ev.get("stats")
                    elif t == "error":
                        error = ev
    return {
        "text": "".join(tokens),
        "prefill_status": status,
        "stats": stats,
        "error": error,
    }


def run_chat() -> dict:
    """Single chat call — captures wall time, token metrics, and output text."""
    payload = {
        "messages": [
            {"role": "user", "content":
             "In exactly 4 sentences, summarize what KV-cache eviction does and "
             "why it matters for long-context inference. Then list 3 key "
             "trade-offs of compressing the cache vs keeping it full."}
        ],
        "policy": "kiaomni_gaussian",
        "budget": 512,
        "max_new_tokens": 256,
    }
    t0 = time.perf_counter()
    result = _post_sse("/api/chat", payload, timeout=600)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "endpoint": "POST /api/chat (SSE)",
        "policy": payload["policy"],
        "budget": payload["budget"],
        "max_new_tokens": payload["max_new_tokens"],
        "wall_ms": wall_ms,
        "request_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "response": result,
    }


def run_compare() -> dict:
    """Side-by-side: 3 policies, same prompt, return per-policy text + stats."""
    payload = {
        "messages": [
            {"role": "user", "content":
             "List 5 well-known open-source LLM families and one distinguishing "
             "feature of each. Be concise — 1 line per family."}
        ],
        "budget": 512,
        "max_new_tokens": 200,
        "policies": ["fullcontext", "kiaomni_s8", "kiaomni_gaussian"],
    }
    t0 = time.perf_counter()
    result = _post_json("/api/compare", payload, timeout=900)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "endpoint": "POST /api/compare",
        "budget": payload["budget"],
        "max_new_tokens": payload["max_new_tokens"],
        "wall_ms": wall_ms,
        "request_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "policies_requested": payload["policies"],
        "results": result.get("results", []),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")

    print("[live] POST /api/chat (kiaomni_gaussian @ B=512)...")
    chat = run_chat()
    print(f"  wall={chat['wall_ms']:.0f}ms  status={chat['response'].get('status', '?')}")

    print("[live] POST /api/compare (3 policies @ B=512)...")
    cmp = run_compare()
    print(f"  wall={cmp['wall_ms']:.0f}ms  policies={len(cmp['results'])}")

    # Also grab current telemetry from the live instance
    with urllib.request.urlopen(f"{URL}/api/telemetry", timeout=15) as r:
        telemetry = json.loads(r.read().decode("utf-8"))
    with urllib.request.urlopen(f"{URL}/api/health", timeout=15) as r:
        health = json.loads(r.read().decode("utf-8"))

    # Flatten a friendly "metrics" view
    def _flatten_compare(c):
        rows = []
        for r in c["results"]:
            if "stats" not in r:
                rows.append({"policy": r["policy"], "error": r.get("error")})
                continue
            s = r["stats"]
            rows.append({
                "policy": r["policy"],
                "tokens_in":        s["tokens_in"],
                "tokens_kept":      s["tokens_kept"],
                "compression_pct":  round(s["tokens_kept"] / max(1, s["tokens_in"]) * 100, 1),
                "prefill_ms":       round(s["prefill_ms"], 1),
                "decode_ms":        round(s["decode_ms"], 1),
                "wall_ms_per_call": round(s["prefill_ms"] + s["decode_ms"], 1),
                "tok_per_sec":      round(s["tok_per_sec"], 2),
                "vram_gb":          round(s["vram_reserved_mb"] / 1024, 2),
            })
        return rows

    summary = {
        "schema_version": 1,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "deploy_url": URL,
        "health": health,
        "chat": {
            "request": {"policy": chat["policy"], "budget": chat["budget"]},
            "wall_ms": round(chat["wall_ms"], 1),
            "stats": chat["response"].get("stats"),
            "text": chat["response"].get("text", ""),
            "error": chat["response"].get("error"),
        },
        "compare": {
            "request": {"budget": cmp["budget"]},
            "wall_ms": round(cmp["wall_ms"], 1),
            "rows": _flatten_compare(cmp),
            "texts": {r["policy"]: r.get("text", "") for r in cmp["results"]},
        },
        "telemetry": {
            "oom_count": telemetry.get("oom_count"),
            "total_requests": telemetry.get("stats", {}).get("total_requests"),
            "uptime_s": telemetry.get("uptime_s"),
        },
    }

    out_path = out_dir / f"live_run_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[live] saved {out_path}")
    print(f"\n--- chat wall_ms: {summary['chat']['wall_ms']:.0f} ---")
    if summary["chat"]["text"]:
        print(f"--- chat text: {summary['chat']['text'][:200]}... ---")
    print("\n--- compare rows ---")
    for row in summary["compare"]["rows"]:
        print(f"  {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
