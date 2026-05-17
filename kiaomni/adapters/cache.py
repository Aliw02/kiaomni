"""
kiaomni/adapters/cache.py
===========================
Per-batch-row KV-cache eviction for any HuggingFace causal LM that uses
``DynamicCache`` (transformers >= 4.50, the only supported line as of
May 2026).

The adapter:
1.  Runs one prefill with ``use_cache=True`` and gathers the cache.
2.  Indexes each layer's K and V along the sequence dim using a
    per-batch-row ``keep_indices`` tensor (rows may differ in length →
    we pad to the max kept count, the model's attention mask handles
    the unused tail).
3.  Returns a fresh ``DynamicCache`` ready to feed into
    ``model.generate(past_key_values=...)``.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import DynamicCache

from .probe import ProbeResult

logger = logging.getLogger(__name__)


class CacheAdapter:
    """Gather and rebuild a DynamicCache with per-row eviction."""

    def __init__(self, probe: ProbeResult) -> None:
        self.probe = probe

    # --- gather ---------------------------------------------------------

    def gather(self, ids: torch.Tensor, model: nn.Module) -> DynamicCache:
        """Run a prefill and return the resulting DynamicCache."""
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(ids.to(device), use_cache=True)
        pkv = out.past_key_values
        if not isinstance(pkv, DynamicCache):
            # Older API: legacy tuple — promote to DynamicCache.
            pkv = DynamicCache.from_legacy_cache(pkv)
        del out
        return pkv

    # --- evict ----------------------------------------------------------

    def evict(
        self,
        raw: DynamicCache,
        keep_per_row: List[np.ndarray],
        model: nn.Module,
    ) -> DynamicCache:
        """
        Build a new DynamicCache that retains only ``keep_per_row[b]``
        sequence positions for each batch row ``b``.

        Padding strategy
        ----------------
        DynamicCache requires a rectangular (B, nkv, K, hd) tensor per
        layer, but different batch rows may want to keep different
        numbers of positions. We pad each row up to ``max_keep =
        max(len(k) for k in keep_per_row)`` by **duplicating that row's
        last kept index**. This is safe because:

        * ``select_keep`` always protects the recency window, so the
          last kept index points to a real, semantically valid position
          (the final prompt token).
        * Re-gathering the same K/V slot multiple times produces
          identical attention weights — the duplicate positions are a
          no-op for the model.
        * Using a real index avoids inventing a zero/garbage row that
          could be attended to.
        """
        device = next(model.parameters()).device
        n_layers = self.probe.num_layers
        B = len(keep_per_row)
        max_keep = max(len(k) for k in keep_per_row)

        # Per-row index tensors, padded with the last kept index (safe filler).
        idx_rows = torch.zeros((B, max_keep), dtype=torch.long, device=device)
        for b, keep in enumerate(keep_per_row):
            n = len(keep)
            idx_rows[b, :n] = torch.from_numpy(keep).to(device)
            if n < max_keep:
                idx_rows[b, n:] = int(keep[-1])

        evicted = DynamicCache()
        for layer_idx in range(n_layers):
            k_layer = self._layer_k(raw, layer_idx)               # (B, nkv, L, hd)
            v_layer = self._layer_v(raw, layer_idx)
            k_new = self._gather_rows(k_layer, idx_rows)          # (B, nkv, max_keep, hd)
            v_new = self._gather_rows(v_layer, idx_rows)
            evicted.update(k_new, v_new, layer_idx)

        if hasattr(evicted, "_seen_tokens"):
            evicted._seen_tokens = max_keep
        elif hasattr(evicted, "seen_tokens"):
            evicted.seen_tokens = max_keep

        del raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return evicted

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _layer_k(cache: DynamicCache, i: int) -> torch.Tensor:
        if hasattr(cache, "key_cache"):
            return cache.key_cache[i]
        return cache[i][0]

    @staticmethod
    def _layer_v(cache: DynamicCache, i: int) -> torch.Tensor:
        if hasattr(cache, "value_cache"):
            return cache.value_cache[i]
        return cache[i][1]

    @staticmethod
    def _gather_rows(t: torch.Tensor, idx_rows: torch.Tensor) -> torch.Tensor:
        """
        ``t`` shape: (B, nkv, L, hd).  ``idx_rows`` shape: (B, K).
        Output shape: (B, nkv, K, hd) where row b is gathered along L
        using idx_rows[b].
        """
        B, nkv, L, hd = t.shape
        K = idx_rows.shape[1]
        idx = idx_rows.view(B, 1, K, 1).expand(B, nkv, K, hd)
        return torch.gather(t, dim=2, index=idx)


__all__ = ["CacheAdapter"]
