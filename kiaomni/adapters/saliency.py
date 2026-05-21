"""
kiaomni/adapters/saliency.py
==============================
Generic, batch-aware saliency extraction across all probed QKV patterns.

Returns a saliency tensor of shape ``(B, L)`` where ``B`` is the input
batch size and ``L`` is the sequence length, regardless of the underlying
model's QKV layout.

Strategies
----------
1. ``hook-separate``          — distinct q_proj / k_proj modules
2. ``hook-fused-concat``      — single linear with out = 3 * hidden
3. ``hook-fused-interleaved`` — single linear with head-stride layout
4. ``fallback-attentions``    — output_attentions=True (slowest, always works)

The right strategy is chosen automatically from the ``ProbeResult``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .probe import ProbeResult, QKVPattern

logger = logging.getLogger(__name__)


class SaliencyAdapter:
    """Dispatches to the correct extraction strategy for a given probe result."""

    def __init__(self, probe: ProbeResult) -> None:
        self.probe = probe
        self._strategy = self._choose_strategy(probe)
        # Upfront safety: fused-interleaved layouts assume nh == nkv (one
        # K/V head per Q head). If the model actually uses GQA / MQA, the
        # reshape `view(B, L, nh, 3, hd)` is invalid — auto-route to the
        # always-correct output_attentions fallback instead of crashing
        # later inside a forward hook.
        if (
            self._strategy == "hook-fused-interleaved"
            and probe.num_attention_heads != probe.num_key_value_heads
        ):
            logger.warning(
                "fused-interleaved + GQA detected (nh=%d, nkv=%d) — "
                "routing to fallback-attentions strategy",
                probe.num_attention_heads,
                probe.num_key_value_heads,
            )
            self._strategy = "fallback-attentions"
        logger.debug("SaliencyAdapter using strategy=%s", self._strategy)

    # --- public ---------------------------------------------------------

    def extract(self, ids: torch.Tensor, model: nn.Module) -> np.ndarray:
        """Return per-token saliency of shape ``(B, L)`` as float32 numpy."""
        if ids.dim() != 2:
            raise ValueError(f"ids must be 2-D (B, L), got shape {tuple(ids.shape)}")

        if self._strategy == "fallback-attentions":
            return self._extract_via_attentions(ids, model)
        return self._extract_via_hooks(ids, model)

    # --- strategy selection --------------------------------------------

    @staticmethod
    def _choose_strategy(probe: ProbeResult) -> str:
        if probe.confidence == "low" or probe.qkv_pattern == "unknown":
            return "fallback-attentions"
        if probe.qkv_pattern == "separate":
            return "hook-separate"
        if probe.qkv_pattern == "fused_concat":
            return "hook-fused-concat"
        if probe.qkv_pattern == "fused_interleaved":
            return "hook-fused-interleaved"
        return "fallback-attentions"

    # --- shared helpers -------------------------------------------------

    def _get_layers(self, model: nn.Module) -> nn.ModuleList:
        node: Any = model
        for part in self.probe.layer_container_path.split("."):
            node = getattr(node, part)
        return node

    def _get_attn(self, layer: nn.Module) -> nn.Module:
        return getattr(layer, self.probe.attn_module_name)

    def _reshape_qk(
        self, q_flat: torch.Tensor, k_flat: torch.Tensor, B: int, L: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape (B, L, nh*hd) → (B, nh, L, hd), replicating K for GQA."""
        nh = self.probe.num_attention_heads
        nkv = self.probe.num_key_value_heads
        hd = self.probe.head_dim
        q = q_flat.view(B, L, nh, hd).transpose(1, 2)
        k = k_flat.view(B, L, nkv, hd).transpose(1, 2)
        if nkv != nh:
            k = k.repeat_interleave(nh // nkv, dim=1)
        return q, k

    @staticmethod
    def _last_query_softmax(q: torch.Tensor, k: torch.Tensor, hd: int) -> torch.Tensor:
        """Softmax(QK^T/sqrt(hd)) for last query position → shape (B, nh, L)."""
        sc = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) * (hd ** -0.5)
        return torch.softmax(sc, dim=-1)[:, :, 0, :]

    # --- strategy: hook-based ------------------------------------------

    def _extract_via_hooks(self, ids: torch.Tensor, model: nn.Module) -> np.ndarray:
        layers = self._get_layers(model)
        n_layers = len(layers)
        B, L = ids.shape
        hd = self.probe.head_dim

        per_layer: List[Optional[np.ndarray]] = [None] * n_layers
        hooks: list = []

        for l_idx, layer in enumerate(layers):
            attn = self._get_attn(layer)
            hooks.extend(self._register_layer_hooks(attn, l_idx, per_layer, B, L, hd))

        try:
            with torch.no_grad():
                model(ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()

        fallback = np.zeros((B, L), dtype=np.float32)
        stacked = np.stack([(x if x is not None else fallback) for x in per_layer])
        return stacked.mean(0).astype(np.float32)  # (B, L)

    def _register_layer_hooks(
        self,
        attn: nn.Module,
        l_idx: int,
        per_layer: List[Optional[np.ndarray]],
        B: int,
        L: int,
        hd: int,
    ) -> list:
        store: Dict[str, torch.Tensor] = {}

        def _commit(q_flat: torch.Tensor, k_flat: torch.Tensor) -> None:
            q, k = self._reshape_qk(q_flat, k_flat, B, L)
            sal_h = self._last_query_softmax(q, k, hd)         # (B, nh, L)
            per_layer[l_idx] = sal_h.mean(1).cpu().numpy().astype(np.float32)
            del q, k, sal_h

        if self._strategy == "hook-separate":
            q_mod = getattr(attn, self.probe.q_module_name)    # type: ignore[arg-type]
            k_mod = getattr(attn, self.probe.k_module_name)    # type: ignore[arg-type]

            def _q_hook(_m, _i, out):
                # Pull to CPU + float32 immediately. Under 4-bit NF4 with
                # bnb_4bit_compute_dtype=bfloat16, the projection outputs
                # are bf16 — the downstream softmax(QK^T/√d) accumulates
                # error and produces NaN/Inf on long sequences. CPU+fp32
                # gives the same numerical refuge as 039_swap_experiment.
                store["q"] = out.detach().cpu().to(torch.float32)

            def _k_hook(_m, _i, out, _li=l_idx):
                # Hook-ordering safety: HF standard attention always fires Q
                # before K (init order), but custom forwards can violate that.
                # Skip cleanly instead of crashing if Q hasn't fired.
                if "q" not in store:
                    logger.warning(
                        "K fired before Q on layer %d — skipping saliency "
                        "contribution from this layer", _li,
                    )
                    return
                k_tensor = out.detach().cpu().to(torch.float32)
                _commit(store["q"], k_tensor)
                del store["q"]   # precise cleanup (no leak on partial state)

            return [
                q_mod.register_forward_hook(_q_hook),
                k_mod.register_forward_hook(_k_hook),
            ]

        # Fused patterns share a single projection.
        fused_mod = getattr(attn, self.probe.fused_module_name)  # type: ignore[arg-type]
        nh = self.probe.num_attention_heads
        nkv = self.probe.num_key_value_heads

        def _fused_hook(_m, _i, out):
            # Mirror the separate-hook path: pull to CPU + fp32 immediately.
            # Under 4-bit NF4 + bf16 compute, fused QKV outputs are bf16 and
            # the downstream softmax(QK^T/√d) accumulates error → NaN/Inf on
            # long sequences. CPU+fp32 isolation is the same refuge that
            # 039_swap_experiment.py relies on.
            o = out.detach().cpu().to(torch.float32)
            if self._strategy == "hook-fused-concat":
                # Layout: [Q | K | V] concatenated along last dim.
                q_width = nh * hd
                k_width = nkv * hd
                q_flat = o[..., :q_width]
                k_flat = o[..., q_width : q_width + k_width]
            else:  # hook-fused-interleaved (nh == nkv, guarded at __init__)
                three = o.view(B, L, nh, 3, hd)
                q_flat = three[:, :, :, 0, :].reshape(B, L, nh * hd)
                k_flat = three[:, :, :, 1, :].reshape(B, L, nh * hd)
            _commit(q_flat, k_flat)

        return [fused_mod.register_forward_hook(_fused_hook)]

    # --- strategy: output_attentions fallback --------------------------

    def _extract_via_attentions(self, ids: torch.Tensor, model: nn.Module) -> np.ndarray:
        B, L = ids.shape
        with torch.no_grad():
            out = model(
                ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        attns = getattr(out, "attentions", None)
        if not attns:
            raise RuntimeError(
                "Model did not return attentions even though output_attentions=True. "
                "This model is incompatible with KiaOmni."
            )

        # attns: tuple of (B, nh, L, L). Take last-query row, avg over heads & layers.
        per_layer: list = []
        for a in attns:
            last_row = a[:, :, -1, :].to(torch.float32)         # (B, nh, L)
            per_layer.append(last_row.mean(1).cpu().numpy())    # (B, L)
        return np.stack(per_layer).mean(0).astype(np.float32)


__all__ = ["SaliencyAdapter"]
