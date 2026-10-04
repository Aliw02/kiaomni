from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch

from global_mask_strategy import global_keep_indices, compact_global
from layerwise_mask_strategy import layerwise_keep_indices, compact_layerwise
from toy_stacked_kv_model import ToyStackedDecoder, total_cache_bytes

CASES = [
    {"id": "gl-01", "length": 64, "budget": 24, "peaks": [13, 21, 29, 37]},
    {"id": "gl-02", "length": 96, "budget": 32, "peaks": [17, 33, 49, 65]},
    {"id": "gl-03", "length": 128, "budget": 40, "peaks": [23, 47, 71, 95]},
]


def build_hidden(length: int, d_model: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, length, d_model, generator=g) * 0.05


def inject_layer_specific_policy_signal(saliencies, peaks):
    out = []
    for sal, peak in zip(saliencies, peaks):
        s = sal.detach().cpu().numpy().astype(np.float32).copy()
        s[peak] += 100.0
        out.append(s)
    return out


def run_case(model, case, seed):
    x = build_hidden(case["length"], model.d_model, seed)
    _, full_caches, model_saliencies = model.prefill(x)

    # Plumbing-only augmentation: each layer receives a different known-important
    # position so specialization can be tested without claiming model quality.
    saliencies = inject_layer_specific_policy_signal(model_saliencies, case["peaks"])

    t0 = time.perf_counter()
    gkeep = global_keep_indices(saliencies, case["budget"])
    global_select_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    lkeep = layerwise_keep_indices(saliencies, case["budget"])
    layerwise_select_ms = (time.perf_counter() - t0) * 1000.0

    global_caches = compact_global(full_caches, gkeep)
    layerwise_caches = compact_layerwise(full_caches, lkeep)

    next_hidden = torch.zeros(1, 1, model.d_model)
    full_out, _, _ = model.decode_one(next_hidden, full_caches)
    global_out, global_after, _ = model.decode_one(next_hidden, global_caches)
    layerwise_out, layerwise_after, _ = model.decode_one(next_hidden, layerwise_caches)

    full_bytes = total_cache_bytes(full_caches)
    global_bytes = total_cache_bytes(global_caches)
    layerwise_bytes = total_cache_bytes(layerwise_caches)

    global_peak_survival = [int(p in set(gkeep.tolist())) for p in case["peaks"]]
    layerwise_peak_survival = [int(p in set(k.tolist())) for p, k in zip(case["peaks"], lkeep)]

    return {
        "id": case["id"],
        "layers": model.n_layers,
        "input_tokens": case["length"],
        "budget_per_layer": case["budget"],
        "full_kv_bytes": full_bytes,
        "global_kv_bytes": global_bytes,
        "layerwise_kv_bytes": layerwise_bytes,
        "equal_compressed_memory": global_bytes == layerwise_bytes,
        "global_reduction_ratio": full_bytes / global_bytes,
        "layerwise_reduction_ratio": full_bytes / layerwise_bytes,
        "global_selection_ms": global_select_ms,
        "layerwise_selection_ms": layerwise_select_ms,
        "global_unique_masks": 1,
        "layerwise_unique_masks": len({tuple(k.tolist()) for k in lkeep}),
        "global_layer_specific_peak_survival": global_peak_survival,
        "layerwise_layer_specific_peak_survival": layerwise_peak_survival,
        "global_layer_specific_peak_survival_rate": sum(global_peak_survival) / len(global_peak_survival),
        "layerwise_layer_specific_peak_survival_rate": sum(layerwise_peak_survival) / len(layerwise_peak_survival),
        "global_decode_ok": all(c.kv_length == case["budget"] + 1 and c.next_position == case["length"] + 1 for c in global_after),
        "layerwise_decode_ok": all(c.kv_length == case["budget"] + 1 and c.next_position == case["length"] + 1 for c in layerwise_after),
        "full_vs_global_output_l2": float(torch.linalg.vector_norm(full_out - global_out).item()),
        "full_vs_layerwise_output_l2": float(torch.linalg.vector_norm(full_out - layerwise_out).item()),
    }


def main():
    torch.set_grad_enabled(False)
    model = ToyStackedDecoder(n_layers=4, d_model=32, n_heads=4, seed=100).eval()
    rows = [run_case(model, c, 200 + i) for i, c in enumerate(CASES)]
    result = {
        "poc_type": "global_vs_layerwise_true_kv_compaction",
        "scope": "plumbing_and_budget_fairness_only",
        "quality_claim_allowed": False,
        "model": "4_layer_custom_pytorch_decoder_with_rope",
        "cases": len(rows),
        "all_equal_memory": all(r["equal_compressed_memory"] for r in rows),
        "all_global_decode_ok": all(r["global_decode_ok"] for r in rows),
        "all_layerwise_decode_ok": all(r["layerwise_decode_ok"] for r in rows),
        "mean_global_peak_survival": float(np.mean([r["global_layer_specific_peak_survival_rate"] for r in rows])),
        "mean_layerwise_peak_survival": float(np.mean([r["layerwise_layer_specific_peak_survival_rate"] for r in rows])),
        "rows": rows,
    }
    out = Path(__file__).resolve().parents[1] / "results" / "poc_global_layerwise.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
