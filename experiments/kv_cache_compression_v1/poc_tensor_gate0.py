"""CPU-safe tensor Gate-0 POC.

This validates physical KV compaction without requiring Transformers or a model download.
It does NOT establish Transformer decode correctness or scientific quality.
"""

from __future__ import annotations

import json
import torch

from kv_gaussian_policy import kv_gaussian_smooth, kv_norm_proxy_score, select_kv_positions
from kv_tensor_cache import cache_nbytes, compact_legacy_kv


def main() -> None:
    torch.manual_seed(7)
    batch, heads, length, head_dim, layers = 1, 4, 128, 16, 6
    cache = tuple(
        (
            torch.randn(batch, heads, length, head_dim),
            torch.randn(batch, heads, length, head_dim),
        )
        for _ in range(layers)
    )

    raw = kv_norm_proxy_score(*cache[0])
    scores = kv_gaussian_smooth(raw, sigma=3.0)
    keep = select_kv_positions(scores, budget=32, n_sink=4, recency=8)

    before = cache_nbytes(cache)
    compacted = compact_legacy_kv(cache, keep)
    after = cache_nbytes(compacted)

    assert keep.numel() == 32
    assert all(t.shape[-2] == 32 for pair in compacted for t in pair)
    assert torch.equal(compacted[0][0], cache[0][0].index_select(-2, keep))
    assert before == 4 * after

    # Tensor-level decode growth check: a new token appends after physical compaction.
    grown = compacted
    for _ in range(5):
        grown = tuple(
            (
                torch.cat([key, torch.randn(batch, heads, 1, head_dim)], dim=-2),
                torch.cat([value, torch.randn(batch, heads, 1, head_dim)], dim=-2),
            )
            for key, value in grown
        )
    assert all(t.shape[-2] == 37 for pair in grown for t in pair)

    result = {
        "gate": "tensor_gate_0",
        "layers": layers,
        "tokens_before": length,
        "tokens_after": int(keep.numel()),
        "bytes_before": before,
        "bytes_after": after,
        "physical_reduction_pct": 100.0 * (1.0 - after / before),
        "compression_ratio": before / after,
        "append_growth_ok": True,
        "scientific_quality_claim": False,
        "transformer_decode_claim": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
