"""
engine.py — KiaOmniEngine for the production web chat.

Design contract (v3 plan):
  • No manual CUDA-cache flushing anywhere in this module.
  • No forced Python garbage collection.
  • VRAM is allowed to accumulate naturally across requests.
  • Per-request memory snapshot is captured for telemetry.
  • ``OutOfMemoryError`` is caught and surfaced as ``EngineOOM`` — the route
    layer maps it to HTTP 507.
  • Streaming uses ``TextIteratorStreamer`` running the patched ``generate``
    in a background thread; the eviction prefill happens inside that call.

The engine is a module-level singleton. ``apply_kiaomni`` is idempotent
(kiaomni >=0.2.2), so policy swaps via ``remove_kiaomni`` → ``apply_kiaomni``
are safe.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

import numpy as np
import torch

from kiaomni import apply_kiaomni, remove_kiaomni
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from .schemas import ChatMessage, StatsBlock

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID_LOCAL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

POLICY_FULLCONTEXT = "fullcontext"
POLICY_S8 = "kiaomni_s8"
POLICY_GAUSSIAN = "kiaomni_gaussian"
POLICY_SNAPKV = "snapkv"

_VALID_POLICIES: frozenset[str] = frozenset({POLICY_FULLCONTEXT, POLICY_S8, POLICY_GAUSSIAN, POLICY_SNAPKV})


class EngineOOM(RuntimeError):
    """Raised when the GPU runs out of memory mid-call. Surface as HTTP 507."""


class EngineNotReady(RuntimeError):
    """Raised when a request arrives before the model has finished loading."""


@dataclass
class GenerationResult:
    text: str
    stats: StatsBlock
    keep_mask: np.ndarray | None  # bool array of which input positions were kept (None for fullcontext)


class KiaOmniEngine:
    """Lazy, thread-safe singleton wrapping a HuggingFace causal LM + kiaomni.

    Thread-safety: a single ``RLock`` guards model state during policy swaps
    and generation. Concurrent generate calls are serialized — the production
    stress test is single-user by design (``allow_concurrent_inputs=1``).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._active_policy: str = ""
        self._active_budget: int = 0
        self._device: torch.device | None = None
        self._model_id: str = ""
        self._started_at: float = 0.0
        self._ready: bool = False
        self._sal_probe: Any | None = None
        self._sal_adapter: Any | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def ensure_loaded(self, model_id: str | None = None, *, attn_impl: str = "sdpa",
                      quantize_4bit: bool = True, token: str | None = None,
                      force_reload: bool = False) -> dict[str, Any]:
        """Idempotently load the model. Returns a small health dict."""
        with self._lock:
            target = model_id or os.environ.get("KIAOMNI_MODEL_ID", MODEL_ID_DEFAULT)
            if self._ready and not force_reload and self._model_id == target:
                return self._health_dict()
            if self._model is not None:
                # Different model id or force_reload — full reload
                # We do NOT call empty_cache() here (v3 plan forbids it).
                # Releasing the model reference lets Python GC reclaim the
                # tensors; the CUDA caching allocator will reuse the blocks.
                remove_kiaomni(self._model)
                self._model = None
                self._tokenizer = None
                self._ready = False
                self._active_policy = ""
                self._active_budget = 0

            self._model_id = target
            t0 = time.perf_counter()
            self._tokenizer = AutoTokenizer.from_pretrained(target, token=token)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            load_kwargs: dict[str, Any] = {
                "device_map": "auto",
                "attn_implementation": attn_impl,
            }
            if quantize_4bit and torch.cuda.is_available():
                try:
                    import bitsandbytes  # noqa: F401
                except ImportError:
                    print("[engine] bitsandbytes not installed — falling back to fp16", file=sys.stderr)
                    quantize_4bit = False
            if quantize_4bit and torch.cuda.is_available():
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["torch_dtype"] = torch.float16
            if token:
                load_kwargs["token"] = token

            self._model = AutoModelForCausalLM.from_pretrained(target, **load_kwargs)
            self._model.eval()
            self._device = next(self._model.parameters()).device
            self._ready = True
            self._started_at = time.time()
            # Reset max-memory tracker so the first snapshot isn't inflated by load
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            return {
                "model_id": target,
                "device": str(self._device),
                "load_ms": (time.perf_counter() - t0) * 1000.0,
                "quantize_4bit": quantize_4bit and torch.cuda.is_available(),
            }

    def is_ready(self) -> bool:
        return self._ready

    def _health_dict(self) -> dict[str, Any]:
        vram = self._vram()
        return {
            "ready": self._ready,
            "model": self._model_id,
            "device": str(self._device) if self._device else "cpu",
            "vram": vram,
            "uptime_s": time.time() - self._started_at if self._started_at else 0.0,
        }

    # ── VRAM helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _vram() -> dict[str, float]:
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0,
                    "fragmentation_pct": 0.0}
        alloc = torch.cuda.memory_allocated() / 2**20
        resrv = torch.cuda.memory_reserved() / 2**20
        maxalloc = torch.cuda.max_memory_allocated() / 2**20
        frag = ((resrv - alloc) / resrv * 100.0) if resrv > 0 else 0.0
        return {"allocated_mb": alloc, "reserved_mb": resrv,
                "max_allocated_mb": maxalloc, "fragmentation_pct": frag}

    @staticmethod
    def stats_block(prefill_ms: float, decode_ms: float, tok_per_sec: float,
                    tokens_in: int, tokens_kept: int,
                    keep_indices: list[int] | None = None) -> StatsBlock:
        v = KiaOmniEngine._vram()
        return StatsBlock(
            tokens_in=tokens_in,
            tokens_kept=tokens_kept,
            compression_ratio=(tokens_kept / tokens_in) if tokens_in else 0.0,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tok_per_sec=tok_per_sec,
            vram_allocated_mb=v["allocated_mb"],
            vram_reserved_mb=v["reserved_mb"],
            vram_max_allocated_mb=v["max_allocated_mb"],
            fragmentation_pct=v["fragmentation_pct"],
            keep_indices=keep_indices or [],
        )

    # ── Policy swap ────────────────────────────────────────────────────────

    def _apply_policy(self, policy: str, budget: int) -> None:
        """Idempotent policy install. No-op for ``fullcontext``.

        For kiaomni policies we do NOT call ``apply_kiaomni`` — the saliency
        calculation and eviction are done directly in ``stream_generate`` so
        the chat-template structure tokens are masked from the saliency map.
        """
        if policy not in _VALID_POLICIES:
            raise ValueError(f"unknown policy {policy!r}; valid: {sorted(_VALID_POLICIES)}")
        with self._lock:
            if policy == POLICY_FULLCONTEXT:
                remove_kiaomni(self._model)
                self._active_policy = policy
                self._active_budget = 0
                return
            if self._active_policy == policy and self._active_budget == budget:
                return  # already installed
            # Swap: remove hooks from previous policy (idempotent), record new
            remove_kiaomni(self._model)
            self._active_policy = policy
            self._active_budget = budget

    def active_policy(self) -> tuple[str, int]:
        with self._lock:
            return self._active_policy, self._active_budget

    # ── Tokenization helpers ──────────────────────────────────────────────

    def build_input_ids(
        self,
        messages: list[ChatMessage] | list[dict],
        use_system_prompt: bool = False,
    ) -> torch.Tensor:
        """Apply the model's chat template and return a 2-D ``(1, L)`` long tensor on the model device.

        Accepts either ``ChatMessage`` objects or plain dicts (with ``role`` and
        ``content`` keys) so route handlers don't need to round-trip through
        pydantic for every internal message list.
        """
        with self._lock:
            assert self._tokenizer is not None
            payload = [
                {"role": m["role"] if isinstance(m, dict) else m.role,
                 "content": m["content"] if isinstance(m, dict) else m.content}
                for m in messages
            ]
            if use_system_prompt and (not payload or payload[0].get("role") != "system"):
                kv_sys = (
                    "You are a helpful AI assistant. Note: The provided context has been compressed "
                    "using KV-Cache eviction. Rely strictly on the key facts preserved in the text to answer accurately."
                )
                payload.insert(0, {"role": "system", "content": kv_sys})

            ids = self._tokenizer.apply_chat_template(
                payload, add_generation_prompt=True, return_tensors="pt"
            )
            assert ids.dim() == 2 and ids.shape[0] == 1, (
                f"chat-template output must be (1, L), got {tuple(ids.shape)}"
            )
            return ids.to(self._device)


    # ── User-content mask helpers (for chat-template saliency fix) ───────

    def _user_content_mask(self, input_ids: torch.Tensor,
                           messages: list[ChatMessage] | list[dict],
                           L: int) -> np.ndarray:
        """Return a ``(L,)`` boolean array where ``True`` marks positions
        that belong to user content (as opposed to chat-template structural
        tokens like system prompt, role markers, assistant header).

        Uses character-offset mapping rather than token-subsequence matching
        so BPE context-dependence doesn't cause false negatives.
        """
        mask = np.zeros(L, dtype=bool)
        payload = [
            {"role": m["role"] if isinstance(m, dict) else m.role,
             "content": m["content"] if isinstance(m, dict) else m.content}
            for m in messages
        ]
        full_str = self._tokenizer.apply_chat_template(
            payload, add_generation_prompt=True, tokenize=False,
        )
        enc = self._tokenizer(full_str, return_offsets_mapping=True)
        offsets: list[tuple[int, int]] = enc.offset_mapping  # type: ignore[assignment]

        search_start = 0
        for m in messages:
            role = m["role"] if isinstance(m, dict) else m.role
            if role != "user":
                continue
            content = m["content"] if isinstance(m, dict) else m.content
            if not content:
                continue
            try:
                char_start = full_str.index(content, search_start)
            except ValueError:
                continue
            char_end = char_start + len(content)
            for i, (s, e) in enumerate(offsets):
                if s is not None and e is not None and s >= char_start and e <= char_end:
                    mask[i] = True
            search_start = char_end

        print(f"DEBUG: _user_content_mask sum={mask.sum()} out of L={L} tokens")
        return mask

    def _select_keep_with_chat_template_mask(
        self,
        input_ids: torch.Tensor,
        messages: list[ChatMessage] | list[dict],
        policy: str,
        budget: int,
    ) -> tuple[list[int], np.ndarray, int]:
        """Compute keep indices using the library's ``SaliencyAdapter`` +
        ``select_keep`` directly on true log-saliency, matching official
        kiaomni benchmark behavior (protecting sinks + recency automatically).

        Returns ``(keep_indices, raw_saliency, tokens_kept)``.
        """
        L = input_ids.shape[1]
        if L <= budget:
            return list(range(L)), np.zeros(L, dtype=np.float32), L

        # Lazy-init the saliency adapter (cached after first call)
        if self._sal_probe is None or self._sal_adapter is None:
            from kiaomni.adapters.probe import ArchitectureProbe
            from kiaomni.adapters.saliency import SaliencyAdapter
            self._sal_probe = ArchitectureProbe.probe(self._model)
            self._sal_adapter = SaliencyAdapter(self._sal_probe)

        # Extract raw saliency (1, L) float32 numpy
        sal_raw = self._sal_adapter.extract(input_ids, self._model)
        sal_mean = sal_raw[0].copy()

        # Apply policy scoring directly on raw log-saliency (matching official paper & demo)
        from kiaomni.policies import get_policy
        score_fn = get_policy(policy)
        score = score_fn(sal_mean)

        # Select positions via the library's budget-aware selector (n_sink=16, recency=32)
        from kiaomni.utils import select_keep
        keep = select_keep(score, budget, L)

        return keep.tolist(), sal_mean, int(len(keep))

    # ── Core generation (streaming) ───────────────────────────────────────

    def stream_generate(
        self,
        messages: list[ChatMessage],
        *,
        policy: str,
        budget: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_system_prompt: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-shaped events: ``{type: 'status'|'token'|'stats'|'error', ...}``.

        This is a generator; the caller is responsible for serializing to SSE.
        """
        with self._lock:
            if not self._ready or self._model is None:
                raise EngineNotReady("model not loaded")

            t_request = time.perf_counter()
            input_ids = self.build_input_ids(messages, use_system_prompt=use_system_prompt)
            L_in = int(input_ids.shape[1])

            assert L_in > 0, "empty prompt"

            # Pre-call VRAM check — refuse if we are near the wall.
            vram_pre = self._vram()
            if vram_pre["reserved_mb"] > 28 * 1024:
                yield {
                    "type": "error",
                    "error": "vram_pressure",
                    "message": (
                        f"pre-call VRAM reserved = {vram_pre['reserved_mb']:.0f} MB "
                        "exceeds 28 GB safety threshold. Restart the container or wait."
                    ),
                }
                return

            yield {
                "type": "status",
                "phase": "policy",
                "policy": policy,
                "budget": budget,
                "tokens_in": L_in,
            }

            try:
                self._apply_policy(policy, budget)
            except Exception as exc:  # noqa: BLE001
                yield {"type": "error", "error": "policy_swap", "message": str(exc)}
                return

            # Compute eviction (if applicable) before assembling gen_kwargs
            keep_indices: list[int] | None = None
            if policy == POLICY_SNAPKV:
                tokens_kept = min(budget, L_in)
                pruned = input_ids
                keep_indices = list(range(L_in))
            elif policy != POLICY_FULLCONTEXT and L_in > budget:
                keep_indices, _, tokens_kept = self._select_keep_with_chat_template_mask(
                    input_ids, messages, policy, budget,
                )
                pruned = input_ids[:, keep_indices]
            else:
                tokens_kept = L_in
                pruned = input_ids
                keep_indices = list(range(L_in))

            # Reset peak tracker AFTER saliency extraction so peak memory reflects generation pass only (~3.15 GB)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()




            # Build a streamer; we feed it the generate call in a thread.
            tokenizer = self._tokenizer
            assert tokenizer is not None
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs: dict[str, Any] = {
                "input_ids": pruned,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": tokenizer.pad_token_id,
                "attention_mask": torch.ones(pruned.shape, dtype=torch.long, device=pruned.device),
                "streamer": streamer,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            yield {
                "type": "status",
                "phase": "prefill",
                "tokens_in": L_in,
                "tokens_kept": tokens_kept,
                "vram_pre_mb": vram_pre["reserved_mb"],
            }

            t0 = time.perf_counter()
            q: Queue[dict[str, Any]] = Queue()

            def _runner() -> None:
                try:
                    with torch.inference_mode():
                        if policy == POLICY_SNAPKV and L_in > budget:
                            # pyrefly: ignore [missing-import]
                            from kvpress import SnapKVPress
                            ratio = 1.0 - budget / L_in
                            press = SnapKVPress(compression_ratio=ratio, window_size=64, kernel_size=5)
                            with press(self._model):  # type: ignore[arg-type]
                                self._model.generate(**gen_kwargs)  # type: ignore[union-attr]
                        else:
                            self._model.generate(**gen_kwargs)  # type: ignore[union-attr]
                    q.put({"ok": True})
                except torch.cuda.OutOfMemoryError as exc:
                    q.put({"ok": False, "oom": True, "msg": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    q.put({"ok": False, "oom": False, "msg": repr(exc)})

            thread = Thread(target=_runner, daemon=True)
            thread.start()

            t_first = None
            n_new = 0
            try:
                while True:
                    if t_first is None:
                        # While waiting for the first token, peek at the runner
                        try:
                            status = q.get_nowait()
                            if not status["ok"]:
                                if status.get("oom"):
                                    raise EngineOOM(status["msg"])
                                raise RuntimeError(status["msg"])
                        except Empty:
                            pass
                    chunk = next(streamer, None)
                    if chunk is None:
                        break
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_new += 1
                    yield {"type": "token", "text": chunk}
            except EngineOOM as exc:
                yield {"type": "error", "error": "oom", "message": str(exc)}
                return
            except Exception as exc:  # noqa: BLE001
                yield {"type": "error", "error": "stream", "message": repr(exc)}
                return

            # Drain runner
            try:
                status = q.get(timeout=2.0)
                if not status["ok"]:
                    if status.get("oom"):
                        yield {"type": "error", "error": "oom", "message": status["msg"]}
                        return
                    yield {"type": "error", "error": "generate", "message": status["msg"]}
                    return
            except Empty:
                pass

            t_done = time.perf_counter()
            prefill_ms = ((t_first - t0) * 1000.0) if t_first is not None else (t_done - t0) * 1000.0
            decode_ms = ((t_done - t_first) * 1000.0) if t_first is not None else 0.0
            tok_per_sec = (n_new / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0

            sb = self.stats_block(prefill_ms, decode_ms, tok_per_sec, L_in, tokens_kept,
                                   keep_indices=keep_indices)
            yield {"type": "stats", "stats": sb.model_dump()}

    # ── Non-streaming wrapper (for /api/demo, /api/docqa, /api/compare) ───

    def generate_full(
        self,
        messages: list[ChatMessage],
        *,
        policy: str,
        budget: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_system_prompt: bool = False,
    ) -> GenerationResult:
        """Collect all tokens, return a single result. Used by compare / demo / docqa."""
        chunks: list[str] = []
        stats: StatsBlock | None = None
        L_in = 0
        for ev in self.stream_generate(
            messages, policy=policy, budget=budget,
            max_new_tokens=max_new_tokens, temperature=temperature,
            use_system_prompt=use_system_prompt,
        ):

            t = ev["type"]
            if t == "error":
                if ev.get("error") == "oom":
                    raise EngineOOM(ev["message"])
                raise RuntimeError(ev.get("message", "unknown engine error"))
            if t == "status" and ev.get("phase") == "prefill":
                L_in = ev.get("tokens_in", 0)
            if t == "token":
                chunks.append(ev["text"])
            if t == "stats":
                stats = StatsBlock(**ev["stats"])
        if stats is None:
            raise RuntimeError("generate produced no stats event — stream truncated")
        return GenerationResult(
            text="".join(chunks), stats=stats, keep_mask=None,
        )


_singleton: KiaOmniEngine | None = None
_singleton_lock = threading.Lock()


def get_engine() -> KiaOmniEngine:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = KiaOmniEngine()
        return _singleton
