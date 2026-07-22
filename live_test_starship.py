"""Live test of the deployed KiaOmni chat with the Starship article.

Sends the 2229-token article + 4 short NIAH questions to /api/compare/turn
and prints the per-policy stats. Verifies the vram_max_allocated_mb peak
number vs the vram_reserved_mb pool number.
"""
from __future__ import annotations
import json
import time
import urllib.request
from pathlib import Path

URL = "https://aliweyabood--kiaomni-chat-serve.modal.run/api/compare/turn"
ARTICLE = Path("kiaomni_chat/static/sample_data/article.txt").read_text(encoding="utf-8")

# Wait for the container to be ready
for i in range(20):
    try:
        health = json.loads(urllib.request.urlopen("https://aliweyabood--kiaomni-chat-serve.modal.run/api/health", timeout=10).read())
        if health.get("ready"):
            print(f"[ready] vram_alloc={health.get('vram_allocated_mb', 0):.0f} MB  vram_reserved={health.get('vram_reserved_mb', 0):.0f} MB")
            break
    except Exception:
        pass
    print(f"[wait] #{i+1} not ready yet")
    time.sleep(10)
else:
    raise SystemExit("container never became ready")

questions = (
    "Use the document above to answer each of the following 4 questions. "
    "For factual questions, give the exact date or number. Number your answers Q1, Q2, Q3, Q4 in order.\n\n"
    "(Q1) On what date did IFT-3 launch?\n"
    "(Q2) What altitude did the Starship upper stage reach during IFT-3?\n"
    "(Q3) What peak temperature did the heatshield experience during IFT-4's re-entry?\n"
    "(Q4) On what date was the first successful booster catch achieved?"
)

body = {
    "history": [
        {"role": "system", "content": "You are a careful assistant. Use the document below to answer the user's questions."},
        {"role": "user", "content": f"Document:\n\n{ARTICLE}\n\n{questions}"},
    ],
    "budget": 512,
    "max_new_tokens": 512,
    "policies": ["fullcontext", "kiaomni_s8", "kiaomni_gaussian"],
}

print(f"[send] {len(ARTICLE)} chars article + 4 questions, budget=512, max_new=512")
t0 = time.perf_counter()
req = urllib.request.Request(
    URL, data=json.dumps(body).encode("utf-8"),
    headers={"content-type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req, timeout=300) as resp:
    payload = json.loads(resp.read())
wall = (time.perf_counter() - t0) * 1000.0
print(f"[recv] total wall {wall:.0f}ms across {len(payload['results'])} policies\n")

for r in payload["results"]:
    s = r["stats"]
    peak = s["vram_max_allocated_mb"] / 1024
    pool = s["vram_reserved_mb"] / 1024
    delta_pool = pool - peak
    pct = s["tokens_kept"] / max(1, s["tokens_in"]) * 100
    print(f"=== {r['policy']} ===")
    print(f"  in {s['tokens_in']:>4} → kept {s['tokens_kept']:>4}  ({pct:.1f}%)")
    print(f"  prefill {s['prefill_ms']/1000:.2f}s  decode {s['decode_ms']/1000:.2f}s  {s['tok_per_sec']:.1f} tok/s")
    print(f"  peak  = {peak:5.2f} GB  (true VRAM spike during this call)")
    print(f"  pool  = {pool:5.2f} GB  (CUDA allocator pool, never shrinks)")
    print(f"  pool - peak = {delta_pool:+.2f} GB  (allocator fragmentation, kept from previous calls)")
    print(f"  text: {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")
    print()
