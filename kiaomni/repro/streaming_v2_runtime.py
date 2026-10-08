"""Experimental V2 streaming prefill.

Separate implementation; V1, post-prefill True KV, and prompt selection
are untouched. Uses model-derived persistent saliency plus a constrained
non-oracle evidence vault. No hybrid prompt compression.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from .qwen25_kv_runtime import cache_bytes, cuda_sync
from .streaming_qwen25_runtime import (
    StreamingLayerwiseQKCollector, build_streaming_chunk_mask,
    _assert_equal_physical_lengths, _cache_lengths, _compact_layerwise
)
from .streaming_v2_policy import VaultState, choose_positions


@torch.inference_mode()
def streaming_v2_prefill(model, tokenizer, input_ids: torch.Tensor, *,
                         budget: int, chunk_size: int = 128,
                         trigger_multiplier: float = 2.0,
                         policy: str = "gaussian",
                         vault_tokens: int = 100,
                         use_vault: bool = True,
                         memory_guard_mb: int = 900,
                         sink: int = 16, recency: int = 32,
                         window_size: int = 32) -> dict:
    """Online KV compaction with guarded trigger, protected spans, and scores.

    Strict limits: storage budget is exact after every eviction. Peak live KV
    can exceed budget by up to the trigger and chunk size; GPU peak can also
    include non-KV tensors. This is a prototype, not an OOM guarantee.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected input_ids [1, L]")
    if budget < sink + recency:
        raise ValueError("Budget below protected sink+recency")
    if chunk_size <= 0 or trigger_multiplier < 1:
        raise ValueError("Invalid chunk size or trigger multiplier")
    if policy not in {"gaussian", "s8"}:
        raise ValueError("Unsupported V2 saliency policy")
    if memory_guard_mb < 0:
        raise ValueError("memory_guard_mb cannot be negative")

    total_length = int(input_ids.shape[1])
    n_layers = len(model.model.layers)
    vault = VaultState.create(n_layers)
    absolute_positions = [np.empty(0, dtype=np.int64) for _ in range(n_layers)]
    cache = None
    last_logits = None
    peak_kv_bytes = 0
    evictions = []
    saliency_seconds = 0.0
    selection_seconds = 0.0
    compaction_seconds = 0.0
    forward_seconds = 0.0
    t0 = time.perf_counter()

    threshold_tokens = max(int(budget), int(round(budget * trigger_multiplier)))

    for start in range(0, total_length, chunk_size):
        end = min(total_length, start + chunk_size)
        chunk = input_ids[:, start:end]
        history = 0 if cache is None else _assert_equal_physical_lengths(cache)
        position_ids = torch.arange(
            start, end, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        mask = build_streaming_chunk_mask(
            history, end - start, device=input_ids.device
        )
        collector = StreamingLayerwiseQKCollector(
            model, cache, position_ids, window_size=window_size
        )
        cuda_sync()
        tick = time.perf_counter()
        with collector.capture():
            outputs = model(
                input_ids=chunk, past_key_values=cache, use_cache=True,
                return_dict=True, logits_to_keep=1,
                position_ids=position_ids, cache_position=position_ids[0],
                attention_mask=mask,
            )
        cuda_sync()
        forward_seconds += time.perf_counter() - tick
        cache = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()

        tick = time.perf_counter()
        saliency = collector.stacked()
        newest = np.arange(start, end, dtype=np.int64)
        absolute_positions = [
            np.concatenate((old, newest)) for old in absolute_positions
        ]
        # This update happens after EVERY chunk, including all chunks where
        # the eviction threshold has not been reached.
        vault.update(absolute_positions, saliency)
        new_evidence = 0
        if use_vault:
            new_evidence = vault.track_evidence(
                tokenizer, input_ids, start, end
            )
        saliency_seconds += time.perf_counter() - tick

        physical = _assert_equal_physical_lengths(cache)
        peak_kv_bytes = max(peak_kv_bytes, cache_bytes(cache))
        near_oom = False
        if torch.cuda.is_available() and memory_guard_mb:
            free_bytes, _ = torch.cuda.mem_get_info(input_ids.device)
            near_oom = free_bytes < memory_guard_mb * 1024 * 1024

        should_evict = physical > budget and (
            physical > threshold_tokens or end == total_length or near_oom
        )
        if not should_evict:
            continue

        tick = time.perf_counter()
        keeps = [
            choose_positions(
                absolute_positions[i], vault, i, budget, end,
                policy=policy,
                vault_tokens=vault_tokens if use_vault else 0,
                sink=sink, recency=recency,
            )
            for i in range(n_layers)
        ]
        selection_seconds += time.perf_counter() - tick
        before_evidence = sum(
            p in vault.evidence for p in absolute_positions[0]
        )
        keep_pos = set(map(int, absolute_positions[0][keeps[0]]))
        retained_evidence = sum(p in keep_pos for p in vault.evidence)

        cuda_sync()
        tick = time.perf_counter()
        _compact_layerwise(cache, absolute_positions, keeps)
        vault.retain(absolute_positions)
        cuda_sync()
        compaction_seconds += time.perf_counter() - tick
        lengths = _cache_lengths(cache)
        if not lengths or any(x != budget for x in lengths):
            raise RuntimeError(f"V2 budget invariant failed: {lengths}")

        evictions.append({
            "seen_tokens": end, "cache_before": physical,
            "cache_after": budget, "new_evidence_positions": new_evidence,
            "candidate_evidence": int(before_evidence),
            "evidence_kept_in_layer0": int(retained_evidence),
            "memory_guard_trigger": bool(near_oom),
        })

    cuda_sync()
    total_seconds = time.perf_counter() - t0
    if cache is None or last_logits is None:
        raise RuntimeError("No KV cache or logits")
    if total_length > budget and any(x != budget for x in _cache_lengths(cache)):
        raise RuntimeError("Final V2 cache did not obey budget")
    return {
        "logits": last_logits, "cache": cache,
        "absolute_positions": absolute_positions,
        "peak_kv_bytes": int(peak_kv_bytes),
        "final_kv_bytes": int(cache_bytes(cache)),
        "eviction_events": len(evictions),
        "eviction_log": evictions,
        "selected_evidence_positions": int(len(vault.evidence)),
        "selection_seconds": selection_seconds,
        "saliency_seconds": saliency_seconds,
        "compaction_seconds": compaction_seconds,
        "model_forward_seconds": forward_seconds,
        "total_seconds": total_seconds,
        "policy": policy, "vault_tokens": int(vault_tokens if use_vault else 0),
        "trigger_tokens": threshold_tokens,
    }
