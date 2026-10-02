from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_summaries(artifacts: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for track, artifact in artifacts:
        for row in artifact.get("rows", []):
            if row.get("source") != "longbench_v2":
                continue
            result = row["result"]
            grouped[(track, result["method"])].append(row)

    output = []
    for (track, method), rows in sorted(grouped.items()):
        vals = [r["result"] for r in rows]
        routing = [v["routing"] for v in vals if v.get("routing") is not None]
        def mean(key: str) -> float | None:
            xs = [float(v[key]) for v in vals if v.get(key) is not None]
            return sum(xs) / len(xs) if xs else None
        def rmean(key: str) -> float | None:
            xs = [float(v[key]) for v in routing if v.get(key) is not None]
            return sum(xs) / len(xs) if xs else None
        output.append({
            "track": track,
            "method": method,
            "n": len(vals),
            "correct": sum(bool(v["correct"]) for v in vals),
            "accuracy": mean("correct"),
            "mean_input_tokens": mean("input_tokens"),
            "mean_kept_tokens": mean("kept_tokens"),
            "mean_retention_pct": (100.0 * mean("actual_retention_ratio") if mean("actual_retention_ratio") is not None else None),
            "mean_compression_ratio": mean("compression_ratio"),
            "mean_gold_answer_ppl": mean("gold_answer_ppl"),
            "mean_output_tokens_per_second": mean("output_tokens_per_second"),
            "max_generation_peak_vram_gb": max(float(v["generation_peak_allocated_vram_gb"]) for v in vals),
            "max_pipeline_peak_vram_gb": max(float(v["pipeline_peak_allocated_vram_gb"]) for v in vals),
            "mean_inference_path_elapsed_seconds": mean("inference_path_elapsed_seconds"),
            "routing_top1": rmean("top1_expert_agreement"),
            "routing_top8_jaccard": rmean("top8_set_jaccard"),
            "routing_weight_cosine": rmean("dispatch_weight_cosine"),
            "routing_expert_load_jsd": rmean("expert_load_jsd"),
            "routing_worst_layer_top1": rmean("worst_layer_top1_expert_agreement"),
            "routing_worst_layer_top8": rmean("worst_layer_top8_set_jaccard"),
        })
    return output


def paired_summaries(artifacts: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for track, artifact in artifacts:
        by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in artifact.get("rows", []):
            if row.get("source") != "longbench_v2":
                continue
            by_case[row["case_id"]][row["result"]["method"]] = row["result"]
        methods = sorted({m for d in by_case.values() for m in d if m != "full_context"})
        for method in methods:
            cc = cw = wc = ww = 0
            for items in by_case.values():
                if "full_context" not in items or method not in items:
                    continue
                fc = bool(items["full_context"]["correct"])
                kc = bool(items[method]["correct"])
                cc += int(fc and kc)
                cw += int(fc and not kc)
                wc += int((not fc) and kc)
                ww += int((not fc) and (not kc))
            fc_correct = cc + cw
            fc_wrong = wc + ww
            out.append({
                "track": track,
                "method": method,
                "n": cc + cw + wc + ww,
                "preserved": cc,
                "regressions": cw,
                "rescues": wc,
                "both_wrong": ww,
                "preservation_rate": cc / fc_correct if fc_correct else None,
                "regression_rate": cw / fc_correct if fc_correct else None,
                "rescue_rate": wc / fc_wrong if fc_wrong else None,
                "net_rescues_minus_regressions": wc - cw,
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(outdir: Path, summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plots: pip install matplotlib") from exc

    compressed = [r for r in summary if r["method"] != "full_context" and r["mean_compression_ratio"]]
    by_method = {(r["track"], r["method"]): r for r in compressed}

    def ordered(rows):
        return sorted(rows, key=lambda r: float(r["mean_compression_ratio"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    for track in sorted({r["track"] for r in compressed}):
        rs = ordered([r for r in compressed if r["track"] == track])
        ax.plot([r["mean_compression_ratio"] for r in rs], [100*r["accuracy"] for r in rs], marker="o", label=track)
    ax.set_xscale("log")
    ax.set_xlabel("Mean compression ratio (x)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("LongBench-v2 quality vs compression")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "quality_vs_compression.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for track in sorted({r["track"] for r in compressed}):
        rs = ordered([r for r in compressed if r["track"] == track and r["routing_top8_jaccard"] is not None])
        if rs:
            ax.plot([r["mean_compression_ratio"] for r in rs], [100*r["routing_top8_jaccard"] for r in rs], marker="o", label=track)
    ax.set_xscale("log")
    ax.set_xlabel("Mean compression ratio (x)")
    ax.set_ylabel("Top-8 routing Jaccard (%)")
    ax.set_title("Actual MoE routing stability vs compression")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "routing_vs_compression.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for track in sorted({r["track"] for r in compressed}):
        rs = ordered([r for r in compressed if r["track"] == track])
        ax.plot([r["mean_compression_ratio"] for r in rs], [r["max_generation_peak_vram_gb"] for r in rs], marker="o", label=track)
    ax.set_xscale("log")
    ax.set_xlabel("Mean compression ratio (x)")
    ax.set_ylabel("Generation peak VRAM (GiB)")
    ax.set_title("Generation memory vs compression")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "memory_vs_compression.png", dpi=220)
    plt.close(fig)

    pmap = {(r["track"], r["method"]): r for r in paired}
    fig, ax = plt.subplots(figsize=(8, 5))
    for track in sorted({r["track"] for r in compressed}):
        rs = ordered([r for r in compressed if r["track"] == track and (r["track"], r["method"]) in pmap])
        xs=[]; ys=[]
        for r in rs:
            p=pmap[(r["track"],r["method"])]
            if p["preservation_rate"] is not None:
                xs.append(r["mean_compression_ratio"]); ys.append(100*p["preservation_rate"])
        if xs:
            ax.plot(xs, ys, marker="o", label=track)
    ax.set_xscale("log")
    ax.set_xlabel("Mean compression ratio (x)")
    ax.set_ylabel("FullContext-success preservation (%)")
    ax.set_title("Capability preservation vs compression")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "preservation_vs_compression.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", required=True, help="Frozen Phase-03 fixed-budget final.json")
    p.add_argument("--percentage", required=True, help="New percentage replay final.json")
    p.add_argument("--outdir", default="frontier_report")
    args = p.parse_args()

    fixed = load_json(args.fixed)
    percentage = load_json(args.percentage)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts = [("fixed", fixed), ("percentage", percentage)]
    summary = method_summaries(artifacts)
    paired = paired_summaries(artifacts)

    write_csv(outdir / "frontier_summary.csv", summary)
    write_csv(outdir / "frontier_pairwise.csv", paired)
    (outdir / "frontier_plot_data.json").write_text(
        json.dumps({"summary": summary, "paired": paired}, indent=2),
        encoding="utf-8",
    )
    make_plots(outdir, summary, paired)
    print(json.dumps({
        "status": "PASS",
        "outdir": str(outdir),
        "files": [
            "frontier_summary.csv",
            "frontier_pairwise.csv",
            "frontier_plot_data.json",
            "quality_vs_compression.png",
            "routing_vs_compression.png",
            "memory_vs_compression.png",
            "preservation_vs_compression.png",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
