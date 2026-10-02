from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEPTH_BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def global_summary(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifact.get("rows", []):
        grouped[row["result"]["method"]].append(row)

    out = []
    for method, rows in sorted(grouped.items()):
        vals = [x["result"] for x in rows]
        routing = [x["routing"] for x in vals if x.get("routing") is not None]
        saliency = [x["saliency"] for x in vals if x.get("saliency") is not None]

        def mean(key: str) -> float | None:
            xs = [float(x[key]) for x in vals if x.get(key) is not None]
            return sum(xs) / len(xs) if xs else None

        def rmean(key: str) -> float | None:
            xs = [float(x[key]) for x in routing if x.get(key) is not None]
            return sum(xs) / len(xs) if xs else None

        out.append({
            "method": method,
            "n": len(vals),
            "ruler_score_pct": mean("ruler_string_match_pct"),
            "all_correct_rate": mean("correct"),
            "required_answer_token_recall": mean("required_answer_token_recall"),
            "complete_required_answer_rate": mean("complete_required_answer_rate"),
            "all_required_answers_complete_rate": mean("all_required_answers_complete"),
            "mean_input_tokens": mean("input_tokens"),
            "mean_kept_tokens": mean("kept_tokens"),
            "mean_retention_pct": mean("actual_retention_pct"),
            "mean_compression_ratio": mean("compression_ratio"),
            "mean_gold_answer_ppl": mean("gold_answer_ppl"),
            "mean_output_tokens_per_second": mean("output_tokens_per_second"),
            "mean_time_to_first_token_seconds": mean("time_to_first_token_seconds"),
            "mean_decode_after_first_token_seconds": mean("decode_after_first_token_seconds"),
            "mean_inference_path_elapsed_seconds": mean("inference_path_elapsed_seconds"),
            "max_generation_peak_vram_gb": max(float(x["generation_peak_allocated_vram_gb"]) for x in vals),
            "max_saliency_peak_vram_gb": (
                max(float(x["saliency_peak_allocated_vram_gb"]) for x in saliency)
                if saliency else None
            ),
            "max_pipeline_peak_vram_gb": max(float(x["pipeline_peak_allocated_vram_gb"]) for x in vals),
            "routing_top1": rmean("top1_expert_agreement"),
            "routing_top8_jaccard": rmean("top8_set_jaccard"),
            "routing_weight_cosine": rmean("dispatch_weight_cosine"),
            "routing_expert_load_jsd": rmean("expert_load_jsd"),
            "routing_worst_layer_top1": rmean("worst_layer_top1_expert_agreement"),
            "routing_worst_layer_top8": rmean("worst_layer_top8_set_jaccard"),
        })
    return out


def case_table(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in artifact.get("rows", []):
        r = item["result"]
        rows.append({
            "case_id": item["case_id"],
            "task": item["task"],
            "target_context_length": item["target_context_length"],
            "depth_bin": item["depth_bin"],
            "official_depth_pct": item["official_depth_pct"],
            "method": r["method"],
            "ruler_score_pct": r["ruler_string_match_pct"],
            "correct": r["correct"],
            "required_answer_token_recall": r["required_answer_token_recall"],
            "complete_required_answer_rate": r["complete_required_answer_rate"],
            "input_tokens": r["input_tokens"],
            "kept_tokens": r["kept_tokens"],
            "retention_pct": r["actual_retention_pct"],
            "compression_ratio": r["compression_ratio"],
            "gold_answer_ppl": r["gold_answer_ppl"],
            "time_to_first_token_seconds": r.get("time_to_first_token_seconds"),
            "generation_peak_vram_gb": r["generation_peak_allocated_vram_gb"],
            "pipeline_peak_vram_gb": r["pipeline_peak_allocated_vram_gb"],
            "routing_top8_jaccard": (
                r["routing"]["top8_set_jaccard"] if r.get("routing") is not None else None
            ),
        })
    return rows


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def make_plots(
    outdir: Path,
    artifact: dict[str, Any],
    summary: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required: pip install matplotlib numpy") from exc

    files: list[str] = []
    compressed = [x for x in summary if x["method"] != "full_context"]

    def family(method: str) -> str:
        if method.startswith("kiaomni_s8"):
            return "kiaomni_s8"
        if method.startswith("kiaomni_gaussian"):
            return "kiaomni_gaussian"
        return method

    def plot_curve(metric: str, ylabel: str, title: str, filename: str, scale100: bool = False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for fam in sorted({family(x["method"]) for x in compressed}):
            rows = [x for x in compressed if family(x["method"]) == fam and x.get(metric) is not None]
            rows.sort(key=lambda x: float(x["mean_compression_ratio"]))
            if rows:
                ys = [float(x[metric]) * (100.0 if scale100 else 1.0) for x in rows]
                ax.plot(
                    [x["mean_compression_ratio"] for x in rows],
                    ys,
                    marker="o",
                    label=fam,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Mean compression ratio (x)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=220)
        plt.close(fig)
        files.append(filename)

    plot_curve(
        "ruler_score_pct",
        "RULER string-match score (%)",
        "RULER NIAH quality vs compression",
        "ruler_quality_vs_compression.png",
    )
    plot_curve(
        "required_answer_token_recall",
        "Required-answer token recall (%)",
        "Evidence survival vs compression",
        "ruler_needle_survival_vs_compression.png",
        scale100=True,
    )
    plot_curve(
        "routing_top8_jaccard",
        "Top-8 routing Jaccard (%)",
        "Actual MoE routing stability vs compression",
        "ruler_routing_vs_compression.png",
        scale100=True,
    )
    plot_curve(
        "mean_time_to_first_token_seconds",
        "Time to first token (s)",
        "RULER TTFT vs compression",
        "ruler_ttft_vs_compression.png",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted({x["method"] for x in cases if x["method"] != "full_context"}):
        rows = [x for x in cases if x["method"] == method]
        ax.scatter(
            [100.0 * float(x["required_answer_token_recall"]) for x in rows],
            [float(x["ruler_score_pct"]) for x in rows],
            label=method,
            alpha=0.65,
        )
    ax.set_xlabel("Required-answer token recall (%)")
    ax.set_ylabel("RULER string-match score (%)")
    ax.set_title("Evidence survival vs answer quality")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    scatter_name = "ruler_survival_vs_quality_scatter.png"
    fig.savefig(outdir / scatter_name, dpi=220)
    plt.close(fig)
    files.append(scatter_name)

    tasks = sorted({x["task"] for x in cases})
    lengths = sorted({int(x["target_context_length"]) for x in cases})
    methods = sorted({x["method"] for x in cases})
    for method in methods:
        for length in lengths:
            subset = [
                x for x in cases
                if x["method"] == method and int(x["target_context_length"]) == length
            ]
            if not subset:
                continue
            matrix = np.full((len(tasks), len(DEPTH_BINS)), np.nan, dtype=float)
            for ti, task in enumerate(tasks):
                for di, depth_bin in enumerate(DEPTH_BINS):
                    vals = [
                        float(x["ruler_score_pct"])
                        for x in subset
                        if x["task"] == task and x["depth_bin"] == depth_bin
                    ]
                    if vals:
                        matrix[ti, di] = sum(vals) / len(vals)

            fig, ax = plt.subplots(figsize=(8, 4.8))
            image = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(len(DEPTH_BINS)), DEPTH_BINS)
            ax.set_yticks(range(len(tasks)), tasks)
            ax.set_xlabel("Target-answer depth bin (%)")
            ax.set_ylabel("RULER task")
            ax.set_title(f"RULER score by depth — {method} — {length}")
            for ti in range(len(tasks)):
                for di in range(len(DEPTH_BINS)):
                    if not np.isnan(matrix[ti, di]):
                        ax.text(di, ti, f"{matrix[ti, di]:.0f}", ha="center", va="center")
            fig.colorbar(image, ax=ax, label="String-match score (%)")
            fig.tight_layout()
            name = f"heatmap_score_{_safe_name(method)}_{length}.png"
            fig.savefig(outdir / name, dpi=220)
            plt.close(fig)
            files.append(name)

    return files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ruler", required=True, help="RULER final.json")
    p.add_argument("--outdir", default="ruler_report")
    args = p.parse_args()

    artifact = load_json(args.ruler)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = global_summary(artifact)
    cases = case_table(artifact)
    pairwise = artifact.get("pairwise", [])
    depth = artifact.get("depth_summary", [])

    write_csv(outdir / "ruler_global_summary.csv", summary)
    write_csv(outdir / "ruler_cases.csv", cases)
    write_csv(outdir / "ruler_pairwise.csv", pairwise)
    write_csv(outdir / "ruler_depth_summary.csv", depth)
    (outdir / "ruler_plot_data.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "cases": cases,
                "pairwise": pairwise,
                "depth_summary": depth,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plots = make_plots(outdir, artifact, summary, cases)
    print(json.dumps({
        "status": "PASS",
        "outdir": str(outdir),
        "core_files": [
            "ruler_global_summary.csv",
            "ruler_cases.csv",
            "ruler_pairwise.csv",
            "ruler_depth_summary.csv",
            "ruler_plot_data.json",
        ],
        "plots": plots,
    }, indent=2))


if __name__ == "__main__":
    main()
