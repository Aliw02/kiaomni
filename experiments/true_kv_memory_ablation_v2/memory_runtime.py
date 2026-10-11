"""No-eviction chunked prefill and deferred Q/K saliency for True-KV ablation.

The entire input is processed before any KV entries are evicted. This is NOT
streaming eviction; only the temporary prefill computation is chunked.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import time

import numpy as np
import torch

from kiaomni.repro.qwen25_kv_runtime import (
    cache_layers, cuda_sync, prefill as legacy_prefill,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class QueryWindowCapture:
    """Copy only final W pre-RoPE q_proj outputs per layer to CPU."""

    def __init__(self, model, window: int = 32):
        self.layers = list(model.model.layers)
        self.window = int(window)
        self.handles = []
        self.queries = {}

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            def capture(_module, _inputs, output, idx=i):
                self.queries[idx] = output[:, -self.window:, :].detach().to(
                    device="cpu", copy=True
                )
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(capture))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def prefill_path(model, input_ids: torch.Tensor, *,
                 chunk_size: int | None, original_saliency: bool,
                 deferred_query: bool, window: int = 32) -> dict:
    """Return full physical cache, optionally saliency or saved last Q."""
    if original_saliency and (chunk_size is not None or deferred_query):
        raise ValueError("Original hook saliency is only supported on one-shot")
    if deferred_query and chunk_size is not None and chunk_size < window:
        raise ValueError("Last chunk must contain at least the query window")
    length = int(input_ids.shape[-1])

    if chunk_size is None:
        capture = QueryWindowCapture(model, window) if deferred_query else None
        with capture if capture is not None else nullcontext():
            result = legacy_prefill(
                model, input_ids, collect_saliency=original_saliency,
                window_size=window
            )
        result["queries"] = capture.queries if capture is not None else None
        return result

    cache = None
    final_logits = None
    final_queries = None
    started = time.perf_counter()
    for start in range(0, length, int(chunk_size)):
        end = min(start + int(chunk_size), length)
        chunk = input_ids[:, start:end]
        chunk_length = end - start
        history = 0 if cache is None else int(cache_layers(cache)[0].keys.shape[-2])
        for layer in cache_layers(cache) if cache is not None else []:
            if int(layer.keys.shape[-2]) != history:
                raise RuntimeError("Cache length diverged across layers")
        positions = torch.arange(start, end, device=input_ids.device).unsqueeze(0)
        mask = torch.ones(
            (1, 1, chunk_length, history + chunk_length),
            dtype=torch.bool, device=input_ids.device
        )
        mask[..., history:] = torch.tril(
            torch.ones((chunk_length, chunk_length), dtype=torch.bool,
                       device=input_ids.device)
        )
        is_last = end == length
        capture = QueryWindowCapture(model, window) if (is_last and deferred_query) else None
        with capture if capture is not None else nullcontext():
            outputs = model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
                position_ids=positions,
                cache_position=positions[0],
                attention_mask=mask,
            )
        cache = outputs.past_key_values
        final_logits = outputs.logits[:, -1, :].detach()
        if capture is not None:
            final_queries = capture.queries
        del outputs, chunk, mask
    cuda_sync()
    if cache is None or final_logits is None:
        raise RuntimeError("Empty chunked prefill")
    if any(int(layer.keys.shape[-2]) != length for layer in cache_layers(cache)):
        raise RuntimeError("No-eviction prefill failed to retain the full KV cache")
    return {
        "cache": cache,
        "logits": final_logits,
        "queries": final_queries,
        "layer_saliencies": None,
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def deferred_saliency(model, cache, q_by_layer: dict[int, torch.Tensor],
                      total_length: int) -> np.ndarray:
    """Reproduce the legacy W x L saliency using cached RoPE K after prefill.

    Original saliency intentionally does not apply a causal mask to its last-W
    probes; matching that legacy contract is necessary for a clean ablation.
    Scores are computed one layer at a time, after prefill has finished.
    """
    cfg = model.config
    nh = int(cfg.num_attention_heads)
    nkv = int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // nh)
    if nh % nkv:
        raise ValueError("Invalid grouped-query head layout")
    layers = cache_layers(cache)
    if len(q_by_layer) != len(layers):
        raise RuntimeError("Missing captured Q windows")
    results = []

    for i, layer in enumerate(layers):
        keys = layer.keys.detach()  # Native Qwen cache contains RoPE-rotated K.
        qflat = q_by_layer[i].to(device=keys.device)
        bsz, w, dim = qflat.shape
        if bsz != 1 or dim != nh * hd or keys.shape[-2] != total_length:
            raise RuntimeError("Invalid deferred saliency dimensions")
        q = qflat.view(1, w, nh, hd).transpose(1, 2)
        position_ids = torch.arange(
            total_length - w, total_length, device=q.device,
            dtype=torch.long
        ).unsqueeze(0)
        cos, sin = model.model.rotary_emb(q, position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        grouped = q.reshape(1, nkv, nh // nkv, w, hd).to(torch.float32)
        key32 = keys.to(torch.float32)
        scores = torch.einsum("bkgwd,bkld->bkgwl", grouped, key32) * hd ** -0.5
        probs = torch.softmax(scores, dim=-1)
        sal = probs.max(dim=3).values.mean(dim=(1, 2))
        results.append(sal[0].to("cpu", dtype=torch.float32).numpy().copy())
        del qflat, q, cos, sin, grouped, key32, scores, probs, sal

    return np.stack(results).astype(np.float32)


class MemoryPhases:
    """Report real PyTorch allocator peaks separately for each execution stage."""

    def __init__(self):
        self.phases = {}
        self.active = None
        self.started = None

    def start(self, name: str):
        cuda_sync()
        self.active = name
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def finish(self):
        cuda_sync()
        name = self.active
        if name is None:
            raise RuntimeError("No active memory phase")
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free, total = torch.cuda.mem_get_info()
        self.phases[name] = {
            "seconds": time.perf_counter() - self.started,
            "allocated_after_gib": allocated / (1024 ** 3),
            "reserved_after_gib": reserved / (1024 ** 3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
            "driver_free_gib": free / (1024 ** 3),
            "driver_total_gib": total / (1024 ** 3),
        }
        self.active = None

    def overall(self) -> dict:
        return {
            "peak_allocated_gib": max(
                (x["peak_allocated_gib"] for x in self.phases.values()), default=0.0
            ),
            "peak_reserved_gib": max(
                (x["peak_reserved_gib"] for x in self.phases.values()), default=0.0
            ),
        }
