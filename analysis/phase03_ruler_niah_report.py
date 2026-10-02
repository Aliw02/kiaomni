from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def flatten_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in artifact.get("rows", []):
        r = row["result"]
        routing = r.get("routing") or {}
        out.append({
            "case_id": row["case_id"],
            "task": row["task"],
            "target_context_length": row["target_context_length"],
            "depth_bin": row["depth_bin"],
            "official_depth_pct": row["official_depth_pct"],
            "method": r["method"],
            "ruler_score_pct": r["ruler_string_match_pct"],
            "all_correct": int(bool(r["correct"])),
            "required_answer_token_recall_pct": 100.0 * r["required_answer_token_recall"],
            "complete_required_answer_rate_pct": 100.0 * r["complete_required_answer_rate"],
            "actual_retention_pct": r["actual_retention_pct"],
            "compression_ratio": r["compression_ratio"],
            "gold_answer_ppl": r["gold_answer_ppl"],
            "time_to_first_token_seconds": r.get("time_to_first_token_seconds"),
            "output_tokens_per_second": r.get("output_tokens_per_second"),
            "generation_peak_vram_gb": r.get("generation_peak_allocated_vram_gb"),
            "pipeline_peak_vram_gb": r.get("pipeline_peak_allocated_vram_gb"),
            "routing_top1": routing.get("top1_expert_agreement"),
            "routing_top8_jaccard": routing.get("top8_set_jaccard"),
            "routing_worst_layer_top8": routing.get("worst_layer_top8_set_jaccard"),
        })
    return out


def method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["task"], int(row["target_context_length"]), row["method"]),
            [],
        ).append(row)
    out = []
    for (task, length, method), items in sorted(groups.items()):
        def mean(key: str):
            xs = [float(x[key]) for x in items if x.get(key) is not None]
            return sum(xs) / len(xs) if xs else None
        out.append({
            "task": task,
            "target_context_length": length,
            "method": method,
            "n": len(items),
            "mean_ruler_score_pct": mean("ruler_score_pct"),
            "all_correct_rate_pct": 100.0 * mean("all_correct"),
            "mean_required_answer_token_recall_pct": mean("required_answer_token_recall_pct"),
            "mean_complete_required_answer_rate_pct": mean("complete_required_answer_rate_pct"),
            "mean_actual_retention_pct": mean("actual_retention_pct"),
            "mean_compression_ratio": mean("compression_ratio"),
            "mean_gold_answer_ppl": mean("gold_answer_ppl"),
            "mean_ttft_seconds": mean("time_to_first_token_seconds"),
            "mean_output_tokens_per_second": mean("output_tokens_per_second"),
            "max_generation_peak_vram_gb": max(
                float(x["generation_peak_vram_gb"])
                for x in items if x.get("generation_peak_vram_gb") is not None
            ),
            "max_pipeline_peak_vram_gb": max(
                float(x["pipeline_peak_vram_gb"])
                for x in items if x.get("pipeline_peak_vram_gb") is not None
            ),
            "mean_routing_top1": mean("routing_top1"),
            "mean_routing_top8_jaccard": mean("routing_top8_jaccard"),
            "mean_routing_worst_layer_top8": mean("routing_worst_layer_top8"),
        })
    return out


def make_plots(outdir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], heatmap_method: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    methods = sorted({x["method"] for x in summary if x["method"] != "full_context"})

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in methods:
        xs=[]; ys=[]
        for row in summary:
            if row["method"] == method and row["mean_compression_ratio"] is not None:
                xs.append(float(row["mean_compression_ratio"]))
                ys.append(float(row["mean_ruler_score_pct"]))
        if xs:
            pairs=sorted(zip(xs,ys))
            ax.plot([x for x,_ in pairs],[y for _,y in pairs],marker="o",label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio (x)")
    ax.set_ylabel("RULER string-match score (%)")
    ax.set_title("Controlled long-context quality vs compression")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir/"ruler_score_vs_compression.png",dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in methods:
        xs=[]; ys=[]
        for row in summary:
            if row["method"] == method and row["mean_compression_ratio"] is not None:
                xs.append(float(row["mean_compression_ratio"]))
                ys.append(float(row["mean_required_answer_token_recall_pct"]))
        if xs:
            pairs=sorted(zip(xs,ys))
            ax.plot([x for x,_ in pairs],[y for _,y in pairs],marker="o",label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio (x)")
    ax.set_ylabel("Required-answer token recall (%)")
    ax.set_title("Needle survival vs compression")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir/"needle_survival_vs_compression.png",dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    selected=[x for x in rows if x["method"]==heatmap_method]
    lengths=sorted({int(x["target_context_length"]) for x in selected})
    bins=["0-20","20-40","40-60","60-80","80-100"]
    matrix=np.full((len(lengths),len(bins)),np.nan)
    for i,length in enumerate(lengths):
        for j,depth_bin in enumerate(bins):
            vals=[
                float(x["ruler_score_pct"])
                for x in selected
                if int(x["target_context_length"])==length and x["depth_bin"]==depth_bin
            ]
            if vals:
                matrix[i,j]=float(np.mean(vals))
    im=ax.imshow(matrix,aspect="auto",vmin=0,vmax=100)
    ax.set_xticks(range(len(bins)),bins)
    ax.set_yticks(range(len(lengths)),[f"{x//1024}K" for x in lengths])
    ax.set_xlabel("Answer depth bin (%)")
    ax.set_ylabel("Target context length")
    ax.set_title(f"RULER depth robustness: {heatmap_method}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i,j]):
                ax.text(j,i,f"{matrix[i,j]:.0f}",ha="center",va="center")
    fig.colorbar(im,ax=ax,label="RULER score (%)")
    fig.tight_layout()
    fig.savefig(outdir/"ruler_depth_heatmap.png",dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method in methods:
        vals=[x for x in rows if x["method"]==method]
        if vals:
            ax.scatter(
                [x["required_answer_token_recall_pct"] for x in vals],
                [x["ruler_score_pct"] for x in vals],
                label=method,
                alpha=0.7,
            )
    ax.set_xlabel("Required-answer token recall (%)")
    ax.set_ylabel("RULER string-match score (%)")
    ax.set_title("Selection survival vs answer quality")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir/"needle_survival_vs_answer_score.png",dpi=220)
    plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--ruler",required=True)
    p.add_argument("--outdir",default="ruler_report")
    p.add_argument("--heatmap-method",default="kiaomni_s8_r0.125")
    args=p.parse_args()

    artifact=load_json(args.ruler)
    outdir=Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    rows=flatten_rows(artifact)
    summary=method_summary(rows)
    write_csv(outdir/"ruler_case_metrics.csv",rows)
    write_csv(outdir/"ruler_method_summary.csv",summary)
    write_csv(outdir/"ruler_depth_summary.csv",artifact.get("depth_summary",[]))
    (outdir/"ruler_plot_data.json").write_text(
        json.dumps({
            "summary":summary,
            "depth_summary":artifact.get("depth_summary",[]),
            "pairwise":artifact.get("pairwise",[]),
        },indent=2),
        encoding="utf-8",
    )
    make_plots(outdir,rows,summary,args.heatmap_method)
    print(json.dumps({
        "status":"PASS",
        "outdir":str(outdir),
        "heatmap_method":args.heatmap_method,
        "files":[
            "ruler_case_metrics.csv",
            "ruler_method_summary.csv",
            "ruler_depth_summary.csv",
            "ruler_plot_data.json",
            "ruler_score_vs_compression.png",
            "needle_survival_vs_compression.png",
            "ruler_depth_heatmap.png",
            "needle_survival_vs_answer_score.png",
        ],
    },indent=2))


if __name__=="__main__":
    main()
