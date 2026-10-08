from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Optional

import numpy as np
import torch

from .qwen25_kv_runtime import cache_bytes, cache_layers, cuda_sync
from .streaming_kv_policy_qwen25 import select_streaming_positions


def _rotate_half_qwen(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half_qwen(x) * sin)


def build_streaming_chunk_mask(
    historical_kv_length: int,
    chunk_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Boolean SDPA mask: all retained history + causal prefix inside chunk."""
    historical_kv_length = int(historical_kv_length)
    chunk_length = int(chunk_length)
    mask = torch.ones(
        (1, 1, chunk_length, historical_kv_length + chunk_length),
        dtype=torch.bool,
        device=device,
    )
    mask[:, :, :, historical_kv_length:] = torch.tril(
        torch.ones(
            (chunk_length, chunk_length),
            dtype=torch.bool,
            device=device,
        )
    )
    return mask


class StreamingLayerwiseQKCollector:
    """Per-layer W x current-cache saliency for streaming Qwen2.5 prefill.

    Hooks see Q/K projection outputs for only the current chunk. Historical K
    states are read from the DynamicCache before the current layer appends its
    new keys. Historical keys are already RoPE-rotated; current Q/K receive
    RoPE using the current absolute position_ids.
    """

    def __init__(
        self,
        model,
        past_key_values,
        position_ids: torch.Tensor,
        *,
        window_size: int = 32,
    ):
        self.model = model
        self.cache = past_key_values
        self.position_ids = position_ids
        self.window_size = int(window_size)
        self.layers = list(model.model.layers)
        self.nh = int(model.config.num_attention_heads)
        self.nkv = int(model.config.num_key_value_heads)
        self.hd = int(
            getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // self.nh,
            )
        )
        if self.nh % self.nkv != 0:
            raise ValueError(
                f"num_attention_heads={self.nh} not divisible by num_key_value_heads={self.nkv}"
            )
        self.groups = self.nh // self.nkv
        self._hooks = []
        self._q_store: dict[int, torch.Tensor] = {}
        self._saliency: list[Optional[np.ndarray]] = [None] * len(self.layers)

    def _historical_keys(self, layer_idx: int) -> Optional[torch.Tensor]:
        if self.cache is None:
            return None
        layers = getattr(self.cache, "layers", None)
        if layers is None or layer_idx >= len(layers):
            return None
        keys = getattr(layers[layer_idx], "keys", None)
        if keys is None or keys.numel() == 0:
            return None
        return keys.detach()

    def _commit(
        self,
        layer_idx: int,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
    ) -> None:
        bsz, chunk_len, _ = k_flat.shape
        if bsz != 1:
            raise ValueError("streaming V1 requires batch_size=1")

        q = q_flat.view(bsz, q_flat.shape[1], self.nh, self.hd).transpose(1, 2)
        k_new = k_flat.view(bsz, chunk_len, self.nkv, self.hd).transpose(1, 2)

        cos, sin = self.model.model.rotary_emb(q, self.position_ids)
        q = _apply_rope_single(q, cos[:, -q.shape[-2] :, :], sin[:, -q.shape[-2] :, :])
        k_new = _apply_rope_single(k_new, cos, sin)

        historical = self._historical_keys(layer_idx)
        if historical is None:
            k_all = k_new
            historical_len = 0
        else:
            historical_len = int(historical.shape[-2])
            k_all = torch.cat([historical, k_new], dim=-2)

        w = min(self.window_size, q.shape[-2])
        q = q[:, :, -w:, :]
        q_grouped = q.view(bsz, self.nkv, self.groups, w, self.hd).to(torch.float32)
        k_float = k_all.to(torch.float32)

        scores = torch.einsum(
            "bkgwd,bkld->bkgwl",
            q_grouped,
            k_float,
        ) * (self.hd ** -0.5)

        # Match the real chunk causal contract. Historical retained keys are
        # all older than the current chunk. For current-chunk keys, a query can
        # only see positions up to its own local index.
        query_start = chunk_len - w
        query_local = torch.arange(
            query_start,
            chunk_len,
            device=scores.device,
            dtype=torch.long,
        )
        current_key_local = torch.arange(
            chunk_len,
            device=scores.device,
            dtype=torch.long,
        )
        current_allowed = (
            current_key_local.unsqueeze(0)
            <= query_local.unsqueeze(1)
        )
        allowed = torch.ones(
            (w, historical_len + chunk_len),
            dtype=torch.bool,
            device=scores.device,
        )
        allowed[:, historical_len:] = current_allowed
        scores = scores.masked_fill(
            ~allowed.view(1, 1, 1, w, historical_len + chunk_len),
            float("-inf"),
        )

        probs = torch.softmax(scores, dim=-1)
        saliency = probs.max(dim=3).values.mean(dim=(1, 2))
        self._saliency[layer_idx] = (
            saliency[0].detach().cpu().numpy().astype(np.float32)
        )

        del q, k_new, k_all, q_grouped, k_float, scores, probs, saliency

    def _register(self) -> None:
        self._saliency = [None] * len(self.layers)
        self._q_store.clear()
        self._hooks = []

        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn

            def q_hook(_module, _inputs, output, idx=layer_idx):
                self._q_store[idx] = output[
                    :, -self.window_size :, :
                ].detach()

            def k_hook(_module, _inputs, output, idx=layer_idx):
                q = self._q_store.pop(idx, None)
                if q is None:
                    raise RuntimeError(
                        f"K hook fired before Q hook at layer {idx}"
                    )
                self._commit(idx, q, output.detach())

            self._hooks.append(attn.q_proj.register_forward_hook(q_hook))
            self._hooks.append(attn.k_proj.register_forward_hook(k_hook))

    def _remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._q_store.clear()

    @contextmanager
    def capture(self):
        self._register()
        try:
            yield self
        finally:
            self._remove()

    def stacked(self) -> np.ndarray:
        missing = [
            idx
            for idx, value in enumerate(self._saliency)
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"missing streaming saliency for layers {missing}"
            )
        return np.stack(
            [value for value in self._saliency if value is not None],
            axis=0,
        ).astype(np.float32)


def _cache_lengths(cache) -> list[int]:
    return [
        int(layer.keys.shape[-2])
        for layer in cache_layers(cache)
        if layer.keys is not None
    ]


def _assert_equal_physical_lengths(cache) -> int:
    lengths = _cache_lengths(cache)
    if not lengths:
        return 0
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"streaming V1 requires equal physical cache lengths across layers, got {lengths}"
        )
    return lengths[0]


def _compact_global(
    cache,
    absolute_positions: list[np.ndarray],
    keep: np.ndarray,
) -> None:
    idx_np = np.asarray(keep, dtype=np.int64)
    for layer_idx, layer in enumerate(cache_layers(cache)):
        idx = torch.as_tensor(
            idx_np,
            dtype=torch.long,
            device=layer.keys.device,
        )
        layer.keys = layer.keys.index_select(-2, idx).contiguous()
        layer.values = layer.values.index_select(-2, idx).contiguous()
        absolute_positions[layer_idx] = absolute_positions[
            layer_idx
        ][idx_np]


def _compact_layerwise(
    cache,
    absolute_positions: list[np.ndarray],
    keeps: list[np.ndarray],
) -> None:
    if len(keeps) != len(cache_layers(cache)):
        raise ValueError("streaming layer count mismatch")
    for layer_idx, (layer, keep) in enumerate(
        zip(cache_layers(cache), keeps)
    ):
        idx_np = np.asarray(keep, dtype=np.int64)
        idx = torch.as_tensor(
            idx_np,
            dtype=torch.long,
            device=layer.keys.device,
        )
        layer.keys = layer.keys.index_select(-2, idx).contiguous()
        layer.values = layer.values.index_select(-2, idx).contiguous()
        absolute_positions[layer_idx] = absolute_positions[
            layer_idx
        ][idx_np]


@torch.inference_mode()
def streaming_prefill(
    model,
    input_ids: torch.Tensor,
    *,
    budget: int,
    chunk_size: int,
    variant: str,
    window_size: int = 32,
    n_sink: int = 16,
    recency: int = 32,
    sigma: float = 4.0,
    eviction_trigger_tokens: int | None = None,
) -> dict:
    """Run chunked Qwen2.5 prefill with optional deferred eviction.

    With an explicit trigger larger than budget, retain up to that many
    positions before eviction, then compact to budget. Final chunk always
    compacts to budget. The trigger is a memory/quality trade-off and cannot
    guarantee OOM avoidance.
    """
    allowed_variants = (
        "global",
        "layerwise",
        "persistent_global",
        "persistent_layerwise",
    )
    if variant not in allowed_variants:
        raise ValueError(f"variant must be one of {allowed_variants}")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("streaming V1 requires input_ids shape [1,L]")
    if budget < n_sink + recency:
        raise ValueError(
            f"budget={budget} below protected minimum {n_sink + recency}"
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    trigger = budget if eviction_trigger_tokens is None else int(eviction_trigger_tokens)
    if trigger < budget:
        raise ValueError("eviction_trigger_tokens cannot be smaller than budget")
    total_length = int(input_ids.shape[1])
    cache = None
    absolute_positions = [
        np.empty((0,), dtype=np.int64)
        for _ in range(len(model.model.layers))
    ]
    persistent_scores = [
        {}
        for _ in range(len(model.model.layers))
    ]
    persistent_decay = 0.995

    peak_kv_bytes = 0
    eviction_events = 0
    model_forward_seconds = 0.0
    selection_seconds = 0.0
    compaction_seconds = 0.0
    last_logits = None

    cuda_sync()
    total_started = time.perf_counter()

    for start in range(0, total_length, int(chunk_size)):
        end = min(total_length, start + int(chunk_size))
        chunk = input_ids[:, start:end]
        chunk_len = int(chunk.shape[1])

        if cache is None:
            historical_len = 0
        else:
            historical_len = _assert_equal_physical_lengths(cache)

        position_ids = torch.arange(
            start,
            end,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        cache_position = position_ids[0]
        attention_mask = build_streaming_chunk_mask(
            historical_len,
            chunk_len,
            device=input_ids.device,
        )

        collector = StreamingLayerwiseQKCollector(
            model,
            cache,
            position_ids,
            window_size=window_size,
        )

        cuda_sync()
        forward_started = time.perf_counter()
        with collector.capture():
            outputs = model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        cuda_sync()
        model_forward_seconds += time.perf_counter() - forward_started

        cache = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :].detach()
        layer_saliencies = collector.stacked()

        new_positions = np.arange(
            start,
            end,
            dtype=np.int64,
        )
        absolute_positions = [
            np.concatenate([old, new_positions])
            for old in absolute_positions
        ]

        current_bytes = cache_bytes(cache)
        peak_kv_bytes = max(peak_kv_bytes, current_bytes)

        physical_len = _assert_equal_physical_lengths(cache)
        if physical_len <= budget or (physical_len <= trigger and end < total_length):
            continue

        selection_started = time.perf_counter()

        use_persistent = variant.startswith("persistent_")
        layerwise_mode = variant.endswith("layerwise")

        if use_persistent:
            effective_layer_saliencies = []
            for layer_idx in range(len(absolute_positions)):
                positions = absolute_positions[layer_idx]
                current = layer_saliencies[layer_idx]
                live = set(int(p) for p in positions)
                persistent_scores[layer_idx] = {
                    int(p): float(v) * persistent_decay
                    for p, v in persistent_scores[layer_idx].items()
                    if int(p) in live
                }
                for p, score in zip(positions, current):
                    p = int(p)
                    persistent_scores[layer_idx][p] = max(
                        persistent_scores[layer_idx].get(p, 0.0),
                        float(score),
                    )
                effective_layer_saliencies.append(
                    np.asarray(
                        [
                            persistent_scores[layer_idx][int(p)]
                            for p in positions
                        ],
                        dtype=np.float32,
                    )
                )
            effective_layer_saliencies = np.stack(
                effective_layer_saliencies,
                axis=0,
            )
        else:
            effective_layer_saliencies = layer_saliencies

        if not layerwise_mode:
            if not all(
                np.array_equal(
                    absolute_positions[layer_idx],
                    absolute_positions[0],
                )
                for layer_idx in range(len(absolute_positions))
            ):
                raise RuntimeError(
                    "global streaming absolute-position sets diverged"
                )
            keep = select_streaming_positions(
                effective_layer_saliencies.mean(axis=0),
                absolute_positions[0],
                budget,
                end,
                n_sink=n_sink,
                recency=recency,
                sigma=sigma,
            )
        else:
            keep = [
                select_streaming_positions(
                    effective_layer_saliencies[layer_idx],
                    absolute_positions[layer_idx],
                    budget,
                    end,
                    n_sink=n_sink,
                    recency=recency,
                    sigma=sigma,
                )
                for layer_idx in range(len(absolute_positions))
            ]
        selection_seconds += time.perf_counter() - selection_started

        cuda_sync()
        compact_started = time.perf_counter()
        if not layerwise_mode:
            _compact_global(cache, absolute_positions, keep)
        else:
            _compact_layerwise(cache, absolute_positions, keep)

        if use_persistent:
            for layer_idx, positions in enumerate(absolute_positions):
                live = set(int(p) for p in positions)
                persistent_scores[layer_idx] = {
                    p: v
                    for p, v in persistent_scores[layer_idx].items()
                    if p in live
                }
        cuda_sync()
        compaction_seconds += time.perf_counter() - compact_started
        eviction_events += 1

        lengths = _cache_lengths(cache)
        if any(length != budget for length in lengths):
            raise RuntimeError(
                f"streaming compaction did not enforce budget={budget}: {lengths}"
            )

    cuda_sync()
    total_seconds = time.perf_counter() - total_started

    if cache is None or last_logits is None:
        raise RuntimeError("streaming prefill produced no cache/logits")

    final_bytes = cache_bytes(cache)
    final_lengths = _cache_lengths(cache)
    if total_length > budget and any(
        length != budget for length in final_lengths
    ):
        raise RuntimeError(
            f"final streaming cache lengths do not match budget: {final_lengths}"
        )

    return {
        "logits": last_logits,
        "cache": cache,
        "absolute_positions": absolute_positions,
        "peak_kv_bytes": int(peak_kv_bytes),
        "final_kv_bytes": int(final_bytes),
        "final_cache_lengths": final_lengths,
        "eviction_events": int(eviction_events),
        "model_forward_seconds": float(model_forward_seconds),
        "selection_seconds": float(selection_seconds),
        "compaction_seconds": float(compaction_seconds),
        "total_seconds": float(total_seconds),
        "chunk_size": int(chunk_size),
        "budget": int(budget),
        "variant": variant,
        "eviction_trigger_tokens": int(trigger),
    }


@torch.inference_mode()
def real_no_eviction_equivalence(
    model,
    input_ids: torch.Tensor,
    *,
    chunk_size: int = 128,
    window_size: int = 32,
) -> dict:
    """Compare one-shot vs chunked prefill with eviction disabled.

    Numeric equality is reported rather than assumed because SDPA kernel shape
    changes can produce small floating-point differences on GPU.
    """
    from .qwen25_kv_runtime import prefill

    one_shot = prefill(
        model,
        input_ids,
        collect_saliency=False,
        window_size=window_size,
    )
    chunked = streaming_prefill(
        model,
        input_ids,
        budget=int(input_ids.shape[1]),
        chunk_size=chunk_size,
        variant="global",
        window_size=window_size,
    )

    one_layers = cache_layers(one_shot["cache"])
    chunk_layers = cache_layers(chunked["cache"])
    if len(one_layers) != len(chunk_layers):
        raise RuntimeError("one-shot/chunked layer count mismatch")

    key_max = 0.0
    value_max = 0.0
    for one_layer, chunk_layer in zip(one_layers, chunk_layers):
        key_max = max(
            key_max,
            float(
                (
                    one_layer.keys.to(torch.float32)
                    - chunk_layer.keys.to(torch.float32)
                )
                .abs()
                .max()
                .item()
            ),
        )
        value_max = max(
            value_max,
            float(
                (
                    one_layer.values.to(torch.float32)
                    - chunk_layer.values.to(torch.float32)
                )
                .abs()
                .max()
                .item()
            ),
        )

    logits_a = one_shot["logits"].to(torch.float32)
    logits_b = chunked["logits"].to(torch.float32)
    logits_max = float((logits_a - logits_b).abs().max().item())
    same_argmax = bool(
        torch.equal(
            torch.argmax(logits_a, dim=-1),
            torch.argmax(logits_b, dim=-1),
        )
    )

    return {
        "context_tokens": int(input_ids.shape[1]),
        "chunk_size": int(chunk_size),
        "same_cache_lengths": (
            [int(layer.keys.shape[-2]) for layer in one_layers]
            == [int(layer.keys.shape[-2]) for layer in chunk_layers]
        ),
        "key_max_abs_diff": key_max,
        "value_max_abs_diff": value_max,
        "logits_max_abs_diff": logits_max,
        "same_next_token_argmax": same_argmax,
    }
