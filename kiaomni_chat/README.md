# KiaOmni Chat

Production stress-test web chat for the [`kiaomni`](https://pypi.org/project/kiaomni/)
KV-cache eviction library. Runs the real `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4)
on a single Modal A100-40GB, served via FastAPI + vanilla HTML/JS.

## v3 plan invariants

- **No manual garbage collection.** `torch.cuda.empty_cache()` and
  `gc.collect()` are forbidden in `engine.py`. VRAM is allowed to accumulate
  naturally across requests; the PyTorch caching allocator decides what
  happens.
- **A100-40GB.** $30 budget caps burn at $20; the remaining $10 is the
  contingency.
- **All 5 scenarios** are real, live, on the deployed model:
  Chat · Demo Tasks (NIAH) · Document Q&A · Side-by-Side Compare · Multi-Turn.

## Architecture

```
kiaomni_chat/
├── modal_app.py        # Modal entry point — A100, ASGI deploy, test funcs
├── engine.py           # KiaOmniEngine (no GC, streaming, telemetry)
├── tasks.py            # 4 demo task generators (port from notebook/demo)
├── schemas.py          # Pydantic models
├── sessions.py         # In-memory multi-turn session store
├── telemetry.py        # VRAM ring buffer
├── routes/             # FastAPI routers
│   ├── health.py
│   ├── telemetry.py
│   ├── chat.py         # /api/chat (SSE streaming)
│   ├── compare.py      # /api/compare (3-policy sequential)
│   ├── demo.py         # /api/demo/run (4 NIAH tasks)
│   ├── docqa.py
│   ├── session.py
│   └── admin.py        # /api/restart escape hatch
├── static/             # HTML + CSS + JS (no framework, no CDN)
└── tests/              # py_compile, structure, schemas, sessions, telemetry
```

## Deploy

```bash
# 1. Create the HF secret (one-time)
modal secret create kiaomni-hf HF_TOKEN=hf_xxxxxxxxxxxx

# 2. Deploy (first run builds the image — ~3 min)
./deploy.sh deploy

# 3. Run acceptance tests
./deploy.sh demo-test    # 4 tasks × kiaomni_gaussian @ B=512
./deploy.sh q1q9-test    # 9 Starship questions × kiaomni_gaussian @ B=512
./deploy.sh vram-test    # 100 sequential requests, no GC

# 4. Open the URL printed by `deploy`
```

## Local sanity check (no kiaomni install)

```bash
pip install fastapi pydantic pytest
pytest kiaomni_chat/tests/ -v
```

The tests use a `conftest.py` mock for the `kiaomni` import so they run
without a GPU or the actual package.

## Frontend design

The UI follows `notebook/demo/final_report.html`'s color tokens (KiaOmni
blue `#3fa7d6`, pass/warn/fail badges). Dark theme, OKLCH throughout,
monospace for token counts, no external CDNs. Five tabs for the five
scenarios; a collapsible telemetry drawer plots VRAM over time and shows
per-request stats.

## kiaomni usage pattern

The engine follows the demo notebook's pattern exactly:

```python
from kiaomni import apply_kiaomni, remove_kiaomni

# Apply
apply_kiaomni(model, policy="kiaomni_gaussian", budget=512)  # n_sink=16, recency=32

# Run
out = model.generate(input_ids, max_new_tokens=256, do_sample=False)

# Always remove
remove_kiaomni(model)
```

The engine wraps this in a singleton, a policy-swap helper, and a streaming
generator that runs `generate()` in a background thread while consuming the
`TextIteratorStreamer` on the main thread.

The kiaomni package is installed in the Modal image via the same
`pip install --no-deps git+https://github.com/Aliw02/kiaomni.git` command
the demo notebook uses, with the `[gaussian]` extra to pull scipy.

## Why VRAM plateaus

See `engine.py` and the v3 plan. Summary: prompt-side eviction bounds every
single `generate()` call's KV cache to `O(layers × kv_heads × head_dim ×
(budget + max_new_tokens))`. PyTorch's caching allocator reuses the freed
blocks on the next call, so steady-state `memory_reserved` is the peak of
one bounded call, regardless of how many requests run.
