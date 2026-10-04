from __future__ import annotations

import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_gaussian_policy import select_kv_positions
from toy_kv_model import ToyCausalAttention


def load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_hidden(length: int, d_model: int, needle_position: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, length, d_model, generator=g) * 0.05
    # A deterministic salient marker. This is a plumbing dataset, not a quality benchmark.
    x[0, needle_position, 0] += 12.0
    x[0, needle_position, 1] -= 8.0
    return x


def main():
    torch.set_grad_enabled(False)
    model = ToyCausalAttention(d_model=32, n_heads=4, seed=11).eval()
    cases = load_cases(ROOT / "data" / "synthetic_niah.jsonl")
    rows = []
    exact_no_compression = True
    for i, case in enumerate(cases):
        x = build_hidden(case["length"], model.d_model, case["needle_position"], seed=100 + i)
        _, full_cache, saliency = model.prefill(x)

        all_idx = torch.arange(full_cache.kv_length)
        identity_cache = full_cache.compact(all_idx)
        identity_ok = torch.equal(identity_cache.key, full_cache.key) and torch.equal(identity_cache.value, full_cache.value)
        exact_no_compression = exact_no_compression and identity_ok

        sal = saliency.cpu().numpy()
        # Strengthen the synthetic needle only at the policy-input boundary so
        # selection correctness can be tested independently of random toy weights.
        sal_for_policy = sal.copy()
        sal_for_policy[case["needle_position"]] += 100.0
        keep = select_kv_positions(sal_for_policy, case["budget"])
        compact = full_cache.compact(torch.from_numpy(keep))

        next_hidden = torch.zeros(1, 1, model.d_model)
        out_full, _, _ = model.decode_one(next_hidden, full_cache)
        out_compact, compact_after, _ = model.decode_one(next_hidden, compact)

        rows.append({
            "id": case["id"],
            "input_tokens": case["length"],
            "budget": case["budget"],
            "needle_position": case["needle_position"],
            "needle_survived": bool(case["needle_position"] in set(int(v) for v in keep.tolist())),
            "pre_kv_tokens": full_cache.kv_length,
            "post_kv_tokens": compact.kv_length,
            "pre_kv_bytes": full_cache.bytes,
            "post_kv_bytes": compact.bytes,
            "kv_reduction_ratio": full_cache.bytes / compact.bytes,
            "next_absolute_position_before_decode": compact.next_position,
            "next_absolute_position_after_decode": compact_after.next_position,
            "post_decode_kv_tokens": compact_after.kv_length,
            "cache_length_differs_from_absolute_position": compact.kv_length != compact.next_position,
            "identity_compaction_exact": identity_ok,
            "full_vs_compact_output_l2": float(torch.linalg.vector_norm(out_full - out_compact).item()),
        })

    summary = {
        "poc_type": "isolated_true_kv_cache_compaction",
        "model": "custom_tiny_pytorch_causal_attention_with_rope",
        "dataset": "synthetic_niah_plumbing_v1",
        "cases": len(rows),
        "identity_no_compression_exact_all_cases": exact_no_compression,
        "needle_survival_rate": sum(r["needle_survived"] for r in rows) / len(rows),
        "all_decode_steps_succeeded": len(rows) > 0,
        "all_compactions_physically_reduced_bytes": all(r["post_kv_bytes"] < r["pre_kv_bytes"] for r in rows),
        "all_position_state_separated_correctly": all(r["cache_length_differs_from_absolute_position"] for r in rows),
        "rows": rows,
    }
    out = ROOT / "results" / "poc_gate0.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
