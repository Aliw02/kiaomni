from __future__ import annotations

from contextlib import contextmanager
import gc
import math
import time
from typing import Optional

import numpy as np
import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        cuda_sync()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> dict:
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "peak_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024**3)),
    }


def _rotate_half_qwen(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_qwen_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half_qwen(q) * sin), (k * cos) + (_rotate_half_qwen(k) * sin)


class LayerwiseQKSaliencyCollector:
    """Collect per-layer last-window QK saliency without full LxL attentions.

    This collector is intentionally Qwen2/Qwen2.5-specific for the first
    real-model gate. It reads q_proj/k_proj outputs, applies the model's RoPE,
    computes only W x L scores (default W=32), and stores one [L] saliency
    vector per layer on CPU.
    """

    def __init__(self, model, window_size: int = 32):
        self.model = model
        self.window_size = int(window_size)
        self.layers = list(model.model.layers)
        self.nh = int(model.config.num_attention_heads)
        self.nkv = int(model.config.num_key_value_heads)
        self.hd = int(getattr(model.config, "head_dim", model.config.hidden_size // self.nh))
        if self.nh % self.nkv != 0:
            raise ValueError(f"num_attention_heads={self.nh} is not divisible by num_key_value_heads={self.nkv}")
        self.groups = self.nh // self.nkv
        self.position_ids: Optional[torch.Tensor] = None
        self._hooks = []
        self._q_store: dict[int, torch.Tensor] = {}
        self._saliency: list[Optional[np.ndarray]] = [None] * len(self.layers)

    def set_positions(self, position_ids: torch.Tensor) -> None:
        if position_ids.ndim != 2 or position_ids.shape[0] != 1:
            raise ValueError(f"expected position_ids [1,L], got {tuple(position_ids.shape)}")
        self.position_ids = position_ids

    def _commit(self, layer_idx: int, q_flat: torch.Tensor, k_flat: torch.Tensor) -> None:
        if self.position_ids is None:
            raise RuntimeError("position_ids were not set before saliency capture")

        bsz, q_len, _ = q_flat.shape
        _, full_len, _ = k_flat.shape
        if bsz != 1:
            raise ValueError("real-model V1 currently requires batch_size=1")
        w = min(self.window_size, q_len)

        q = q_flat[:, -w:, :].view(bsz, w, self.nh, self.hd).transpose(1, 2)
        k = k_flat.view(bsz, full_len, self.nkv, self.hd).transpose(1, 2)

        # Use the exact Qwen rotary module so selection reflects the actual
        # pre-attention Q/K geometry rather than raw projections.
        with torch.no_grad():
            cos_all, sin_all = self.model.model.rotary_emb(q, self.position_ids)
            cos_q = cos_all[:, -w:, :]
            sin_q = sin_all[:, -w:, :]
            q, _ = _apply_qwen_rope(q, q[:, : self.nkv], cos_q, sin_q)
            dummy_q = k[:, : min(k.shape[1], self.nh), :, :]
            # Apply RoPE to K with a shape-compatible dummy Q. Only K output is used.
            _, k = _apply_qwen_rope(dummy_q, k, cos_all, sin_all)

            qg = q.view(bsz, self.nkv, self.groups, w, self.hd).to(torch.float32)
            kf = k.to(torch.float32)
            scores = torch.einsum("bkgwd,bkld->bkgwl", qg, kf) * (self.hd ** -0.5)
            probs = torch.softmax(scores, dim=-1)
            sal = probs.max(dim=3).values.mean(dim=(1, 2))
            self._saliency[layer_idx] = sal[0].detach().cpu().numpy().astype(np.float32)

        del q, k, qg, kf, scores, probs, sal, cos_all, sin_all, cos_q, sin_q, dummy_q

    def _register(self) -> None:
        self._saliency = [None] * len(self.layers)
        self._q_store.clear()
        self._hooks = []

        for idx, layer in enumerate(self.layers):
            attn = layer.self_attn

            def q_hook(_module, _inputs, output, layer_idx=idx):
                # Keep only the observation window, not the full Q tensor.
                self._q_store[layer_idx] = output[:, -self.window_size :, :].detach()

            def k_hook(_module, _inputs, output, layer_idx=idx):
                q = self._q_store.pop(layer_idx, None)
                if q is None:
                    raise RuntimeError(f"K hook fired before Q hook at layer {layer_idx}")
                self._commit(layer_idx, q, output.detach())

            self._hooks.append(attn.q_proj.register_forward_hook(q_hook))
            self._hooks.append(attn.k_proj.register_forward_hook(k_hook))

    def _remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._q_store.clear()

    @contextmanager
    def capture(self, position_ids: torch.Tensor):
        self.set_positions(position_ids)
        self._register()
        try:
            yield self
        finally:
            self._remove()

    def stacked(self) -> np.ndarray:
        if any(x is None for x in self._saliency):
            missing = [i for i, x in enumerate(self._saliency) if x is None]
            raise RuntimeError(f"missing layer saliency for layers {missing}")
        return np.stack([x for x in self._saliency if x is not None], axis=0).astype(np.float32)


def _forward_kwargs(input_ids: torch.Tensor, start_position: int = 0) -> dict:
    length = int(input_ids.shape[1])
    device = input_ids.device
    cache_position = torch.arange(start_position, start_position + length, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    return {
        "position_ids": position_ids,
        "cache_position": cache_position,
        "attention_mask": None,
    }


@torch.inference_mode()
def prefill(
    model,
    input_ids: torch.Tensor,
    *,
    collect_saliency: bool,
    window_size: int = 32,
):
    kwargs = _forward_kwargs(input_ids, start_position=0)
    collector = LayerwiseQKSaliencyCollector(model, window_size=window_size) if collect_saliency else None

    cuda_sync()
    started = time.perf_counter()
    if collector is None:
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            **kwargs,
        )
        layer_saliencies = None
    else:
        with collector.capture(kwargs["position_ids"]):
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
                **kwargs,
            )
        layer_saliencies = collector.stacked()
    cuda_sync()
    elapsed = time.perf_counter() - started

    return {
        "logits": outputs.logits[:, -1, :].detach(),
        "cache": outputs.past_key_values,
        "layer_saliencies": layer_saliencies,
        "elapsed_seconds": float(elapsed),
    }


@torch.inference_mode()
def saliency_only(
    model,
    input_ids: torch.Tensor,
    *,
    window_size: int = 32,
):
    kwargs = _forward_kwargs(input_ids, start_position=0)
    collector = LayerwiseQKSaliencyCollector(model, window_size=window_size)
    cuda_sync()
    started = time.perf_counter()
    with collector.capture(kwargs["position_ids"]):
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            **kwargs,
        )
    cuda_sync()
    elapsed = time.perf_counter() - started
    del outputs
    return {
        "layer_saliencies": collector.stacked(),
        "elapsed_seconds": float(elapsed),
    }


def cache_layers(cache):
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(f"expected Transformers Cache with .layers, got {type(cache).__name__}")
    return layers


def cache_bytes(cache) -> int:
    total = 0
    for layer in cache_layers(cache):
        if layer.keys is None or layer.values is None:
            continue
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return int(total)


def cache_lengths(cache) -> list[int]:
    return [int(layer.keys.shape[-2]) for layer in cache_layers(cache) if layer.keys is not None]


def assert_dynamic_full_attention_cache(cache) -> None:
    for idx, layer in enumerate(cache_layers(cache)):
        if getattr(layer, "is_sliding", False):
            raise RuntimeError(
                f"layer {idx} uses a sliding cache; V1 true-KV compaction requires full DynamicLayer caches"
            )
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"layer {idx} cache is uninitialized")
        if layer.keys.ndim != 4 or layer.values.ndim != 4:
            raise RuntimeError(f"layer {idx} expected 4-D K/V tensors")


def compact_cache_global(cache, keep: np.ndarray) -> dict:
    assert_dynamic_full_attention_cache(cache)
    before = cache_bytes(cache)
    expected = int(len(keep))
    for layer in cache_layers(cache):
        idx = torch.as_tensor(keep, dtype=torch.long, device=layer.keys.device)
        layer.keys = layer.keys.index_select(-2, idx).contiguous()
        layer.values = layer.values.index_select(-2, idx).contiguous()
    after = cache_bytes(cache)
    lengths = cache_lengths(cache)
    if any(v != expected for v in lengths):
        raise RuntimeError(f"global compaction length mismatch: {lengths}")
    return {"before_bytes": before, "after_bytes": after, "lengths": lengths}


def compact_cache_layerwise(cache, keeps: list[np.ndarray]) -> dict:
    assert_dynamic_full_attention_cache(cache)
    layers = cache_layers(cache)
    if len(layers) != len(keeps):
        raise ValueError(f"layer count mismatch: cache={len(layers)} masks={len(keeps)}")
    before = cache_bytes(cache)
    expected = None
    for layer, keep in zip(layers, keeps):
        if expected is None:
            expected = len(keep)
        elif len(keep) != expected:
            raise ValueError("V1 requires the same retained-position budget in every layer")
        idx = torch.as_tensor(keep, dtype=torch.long, device=layer.keys.device)
        layer.keys = layer.keys.index_select(-2, idx).contiguous()
        layer.values = layer.values.index_select(-2, idx).contiguous()
    after = cache_bytes(cache)
    lengths = cache_lengths(cache)
    if any(v != expected for v in lengths):
        raise RuntimeError(f"layerwise compaction length mismatch: {lengths}")
    return {"before_bytes": before, "after_bytes": after, "lengths": lengths}


@torch.inference_mode()
def greedy_decode(
    model,
    tokenizer,
    *,
    initial_logits: torch.Tensor,
    cache,
    start_position: int,
    max_new_tokens: int,
) -> dict:
    if initial_logits.ndim != 2 or initial_logits.shape[0] != 1:
        raise ValueError(f"expected initial logits [1,V], got {tuple(initial_logits.shape)}")

    generated: list[int] = []
    nlls: list[float] = []
    logits = initial_logits
    eos_id = tokenizer.eos_token_id

    cuda_sync()
    started = time.perf_counter()

    for step in range(int(max_new_tokens)):
        log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
        next_id = torch.argmax(logits, dim=-1)
        token_id = int(next_id.item())
        generated.append(token_id)
        nlls.append(float(-log_probs[0, token_id].item()))

        if eos_id is not None and token_id == int(eos_id):
            break
        if step + 1 >= int(max_new_tokens):
            break

        absolute_position = int(start_position + step)
        token = next_id.view(1, 1)
        outputs = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            position_ids=torch.tensor([[absolute_position]], device=token.device, dtype=torch.long),
            cache_position=torch.tensor([absolute_position], device=token.device, dtype=torch.long),
            attention_mask=None,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :].detach()

    cuda_sync()
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    ppl = float(math.exp(min(20.0, sum(nlls) / max(1, len(nlls)))))
    return {
        "generated_ids": generated,
        "text": text,
        "generated_tokens": len(generated),
        "decode_seconds": float(elapsed),
        "tokens_per_second": float(len(generated) / elapsed) if elapsed > 0 else None,
        "greedy_self_ppl": ppl,
        "cache": cache,
    }


def validate_qwen25_model(model) -> dict:
    cfg = model.config
    model_type = str(getattr(cfg, "model_type", ""))
    if model_type != "qwen2":
        raise RuntimeError(f"V1 harness expects Qwen2/Qwen2.5 model_type='qwen2', got {model_type!r}")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("unexpected Qwen model layout")
    layer_types = list(getattr(cfg, "layer_types", ["full_attention"] * len(model.model.layers)))
    if any(x != "full_attention" for x in layer_types):
        raise RuntimeError(f"V1 does not support sliding/hybrid layer types: {sorted(set(layer_types))}")
    return {
        "model_type": model_type,
        "num_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        "attn_implementation": str(getattr(cfg, "_attn_implementation", "unknown")),
        "layer_types": sorted(set(layer_types)),
    }
