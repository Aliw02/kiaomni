from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter
from kiaomni.policies import get_policy

BASE_PATH = Path(__file__).with_name("qwen3_30b_percentage_replay.py")
_spec = importlib.util.spec_from_file_location("kiaomni_percentage_core", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot import shared percentage core from {BASE_PATH}")
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core
_spec.loader.exec_module(core)

MODEL_ID = core.MODEL_ID
MODEL_REVISION = core.MODEL_REVISION
POLICIES = core.POLICIES
PERCENTAGE_BUDGETS = core.PERCENTAGE_BUDGETS
N_SINK = core.N_SINK
RECENCY = core.RECENCY
SEED = 42

RULER_REPO = "NVIDIA/RULER"
RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3")
RULER_LENGTHS = (8192, 16384)
MAX_NEW_TOKENS = 128

STAGES = {
    "preflight": {
        "max_wall_seconds": 15 * 60,
        "tasks": ("niah_single_1",),
        "lengths": (8192,),
        "samples_per_task": 1,
        "ratios": (0.25,),
    },
    "final": {
        "max_wall_seconds": 110 * 60,
        "tasks": RULER_TASKS,
        "lengths": RULER_LENGTHS,
        "samples_per_task": 4,
        "ratios": PERCENTAGE_BUDGETS,
    },
}


@dataclass
class RulerCase:
    case_id: str
    task: str
    seq_length: int
    prompt: str
    outputs: list[str]
    answer_prefix: str
    official_length: int
    official_token_position_answer: int | None
    raw_index: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KiaOmni official RULER NIAH percentage frontier")
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--ruler-data-root", required=True)
    p.add_argument("--ruler-manifest", required=True)
    p.add_argument("--repo-revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    return p.parse_args()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cases(root: Path, cfg: dict[str, Any]) -> list[RulerCase]:
    cases: list[RulerCase] = []
    for seq_length in cfg["lengths"]:
        for task in cfg["tasks"]:
            path = root / str(seq_length) / task / "validation.jsonl"
            if not path.exists():
                raise RuntimeError(f"Missing official RULER data: {path}")
            rows = load_jsonl(path)
            if len(rows) < cfg["samples_per_task"]:
                raise RuntimeError(
                    f"RULER {task}@{seq_length} has {len(rows)} rows, need {cfg['samples_per_task']}"
                )
            for i, row in enumerate(rows[: cfg["samples_per_task"]]):
                outputs = [str(x) for x in row.get("outputs", [])]
                if not outputs:
                    raise RuntimeError(f"RULER row has no outputs: {task}@{seq_length}#{i}")
                answer_prefix = str(row.get("answer_prefix", ""))
                prompt = str(row["input"]) + answer_prefix
                cases.append(RulerCase(
                    case_id=f"ruler-{seq_length}-{task}-{i}",
                    task=task,
                    seq_length=int(seq_length),
                    prompt=prompt,
                    outputs=outputs,
                    answer_prefix=answer_prefix,
                    official_length=int(row.get("length_w_model_temp", row.get("length", 0))),
                    official_token_position_answer=(
                        int(row["token_position_answer"])
                        if row.get("token_position_answer") is not None else None
                    ),
                    raw_index=int(row.get("index", i)),
                ))
    return cases


def render_prompt(tokenizer, prompt: str) -> str:
    return core.render_chat(tokenizer, prompt)


def find_subsequence_positions(sequence: list[int], needle: list[int]) -> list[list[int]]:
    if not needle or len(needle) > len(sequence):
        return []
    hits = []
    end = len(sequence) - len(needle) + 1
    for i in range(end):
        if sequence[i:i + len(needle)] == needle:
            hits.append(list(range(i, i + len(needle))))
    return hits


def reference_spans(tokenizer, ids: torch.Tensor, outputs: list[str]) -> list[list[int]]:
    seq = [int(x) for x in ids[0].detach().cpu().tolist()]
    spans = []
    for ref in outputs:
        # Byte/BPE tokenizers can encode a value differently when it is preceded by
        # whitespace in the source sentence. Try both exact and leading-space forms.
        candidates = []
        for text in (ref, " " + ref):
            token_ids = [int(x) for x in tokenizer(text, add_special_tokens=False).input_ids]
            if token_ids and token_ids not in candidates:
                candidates.append(token_ids)
        hits: list[list[int]] = []
        for candidate in candidates:
            hits = find_subsequence_positions(seq, candidate)
            if hits:
                break
        if not hits:
            spans.append([])
            continue
        # RULER NIAH target values are generated uniquely; use the first exact target occurrence.
        spans.append(hits[0])
    return spans


def survival_metrics(spans: list[list[int]], keep: np.ndarray | None, full_len: int) -> dict[str, Any]:
    found = [s for s in spans if s]
    if not found:
        return {
            "reference_spans_found": 0,
            "reference_span_count": len(spans),
            "gold_token_recall": None,
            "complete_reference_survival": None,
            "all_references_survived": None,
            "needle_depth_pct": None,
        }
    keep_set = set(range(full_len)) if keep is None else {int(x) for x in keep.tolist()}
    total_tokens = sum(len(s) for s in found)
    kept_tokens = sum(sum(int(p in keep_set) for p in s) for s in found)
    complete = [all(p in keep_set for p in s) for s in found]
    centers = [sum(s) / len(s) for s in found]
    return {
        "reference_spans_found": len(found),
        "reference_span_count": len(spans),
        "gold_token_recall": kept_tokens / total_tokens if total_tokens else None,
        "complete_reference_survival": sum(complete) / len(complete),
        "all_references_survived": bool(all(complete) and len(found) == len(spans)),
        "needle_depth_pct": float(100.0 * np.mean(centers) / max(1, full_len - 1)),
    }


def ruler_string_match_all(prediction: str, refs: list[str]) -> tuple[float, int]:
    pred = prediction.lower()
    matched = sum(1 for ref in refs if ref.lower() in pred)
    return matched / len(refs), matched


def run_condition(
    model,
    tokenizer,
    case: RulerCase,
    ids: torch.Tensor,
    method: str,
    keep: np.ndarray | None,
    capture,
    full_routes,
    saliency_meta: dict[str, Any] | None,
    spans: list[list[int]],
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    used = ids
    if keep is not None:
        kt = torch.as_tensor(keep, dtype=torch.long, device=ids.device)
        used = ids[:, kt]

    gold_target = ", ".join(case.outputs)
    teacher, routes = core.teacher_gold_and_routing(model, tokenizer, used, gold_target, capture)
    answer, generation = core.generate_answer(model, tokenizer, used, MAX_NEW_TOKENS)
    score, matched = ruler_string_match_all(answer, case.outputs)

    routing = None
    if keep is not None:
        routing = core.compare_routes(
            full_routes,
            routes,
            keep,
            full_prompt_len=int(ids.shape[1]),
            compressed_prompt_len=int(used.shape[1]),
            num_experts=int(model.config.num_experts),
        )

    survival = survival_metrics(spans, keep, int(ids.shape[1]))
    pipeline_peak = max(
        generation["generation_peak_allocated_vram_gb"],
        teacher["routing_teacher_peak_allocated_vram_gb"],
        (saliency_meta or {}).get("saliency_peak_allocated_vram_gb", 0.0),
    )
    result = {
        "method": method,
        "input_tokens": int(ids.shape[1]),
        "kept_tokens": int(used.shape[1]),
        "actual_retention_ratio": float(used.shape[1] / ids.shape[1]),
        "actual_retention_pct": float(100.0 * used.shape[1] / ids.shape[1]),
        "compression_ratio": float(ids.shape[1] / used.shape[1]),
        "answer": answer,
        "references": list(case.outputs),
        "reference_matches": int(matched),
        "reference_count": len(case.outputs),
        "ruler_score": float(score),
        "correct_all_references": bool(score == 1.0),
        **survival,
        **generation,
        **teacher,
        "routing": routing,
        "saliency": saliency_meta,
        "pipeline_peak_allocated_vram_gb": pipeline_peak,
        "inference_path_elapsed_seconds": float(
            generation["generation_elapsed_seconds"]
            + ((saliency_meta or {}).get("saliency_elapsed_seconds", 0.0) if keep is not None else 0.0)
        ),
    }
    return result, routes


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["seq_length"], row["result"]["method"]), []).append(row)
    out = []
    for (task, seq_length, method), items in sorted(groups.items()):
        vals = [x["result"] for x in items]
        routing = [v["routing"] for v in vals if v.get("routing") is not None]
        def mean(key: str) -> float | None:
            xs = [float(v[key]) for v in vals if v.get(key) is not None]
            return float(np.mean(xs)) if xs else None
        def rmean(key: str) -> float | None:
            xs = [float(v[key]) for v in routing if v.get(key) is not None]
            return float(np.mean(xs)) if xs else None
        out.append({
            "task": task,
            "seq_length": seq_length,
            "method": method,
            "n": len(vals),
            "mean_ruler_score": mean("ruler_score"),
            "all_reference_accuracy": mean("correct_all_references"),
            "mean_gold_token_recall": mean("gold_token_recall"),
            "mean_complete_reference_survival": mean("complete_reference_survival"),
            "all_references_survived_rate": mean("all_references_survived"),
            "mean_needle_depth_pct": mean("needle_depth_pct"),
            "mean_kept_tokens": mean("kept_tokens"),
            "mean_actual_retention_pct": mean("actual_retention_pct"),
            "mean_compression_ratio": mean("compression_ratio"),
            "mean_gold_answer_ppl": mean("gold_answer_ppl"),
            "mean_output_tokens_per_second": mean("output_tokens_per_second"),
            "max_generation_peak_vram_gb": max(float(v["generation_peak_allocated_vram_gb"]) for v in vals),
            "max_pipeline_peak_vram_gb": max(float(v["pipeline_peak_allocated_vram_gb"]) for v in vals),
            "routing_mean_top1_expert_agreement": rmean("top1_expert_agreement"),
            "routing_mean_top8_set_jaccard": rmean("top8_set_jaccard"),
            "routing_mean_dispatch_weight_cosine": rmean("dispatch_weight_cosine"),
            "routing_mean_expert_load_jsd": rmean("expert_load_jsd"),
        })
    return out


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["result"]["method"]] = row["result"]
    methods = sorted({m for d in by_case.values() for m in d if m != "full_context"})
    out = []
    for method in methods:
        fc_scores=[]; k_scores=[]; survival=[]
        for d in by_case.values():
            if "full_context" not in d or method not in d:
                continue
            fc_scores.append(float(d["full_context"]["ruler_score"]))
            k_scores.append(float(d[method]["ruler_score"]))
            if d[method].get("complete_reference_survival") is not None:
                survival.append(float(d[method]["complete_reference_survival"]))
        out.append({
            "method": method,
            "n": len(fc_scores),
            "mean_full_context_score": float(np.mean(fc_scores)) if fc_scores else None,
            "mean_kia_score": float(np.mean(k_scores)) if k_scores else None,
            "mean_score_delta": float(np.mean(np.asarray(k_scores)-np.asarray(fc_scores))) if fc_scores else None,
            "mean_complete_reference_survival": float(np.mean(survival)) if survival else None,
        })
    return out


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    cfg = STAGES[args.stage]
    started = time.perf_counter()
    deadline = started + cfg["max_wall_seconds"]
    output = Path(args.output)

    manifest = json.loads(Path(args.ruler_manifest).read_text(encoding="utf-8"))
    if manifest.get("ruler_revision") != RULER_REVISION:
        raise RuntimeError("RULER revision mismatch")
    if tuple(manifest.get("tasks", [])) != RULER_TASKS:
        raise RuntimeError("RULER task manifest mismatch")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    env = core.assert_environment(model, args.min_free_gb)
    probe = ArchitectureProbe.probe(model)
    capture = core.ActualRoutingCapture(model)
    device = next(model.parameters()).device

    neutrality_text = core.render_chat(tokenizer, "Remember this value: 8127345. What value should be remembered?")
    neutrality_ids = core.encode_text(tokenizer, neutrality_text)[:, :512].to(device)
    neutrality = core.hook_neutrality_check(model, capture, neutrality_ids)
    if not neutrality["passed"]:
        raise RuntimeError(f"Routing preflight failed: {neutrality}")

    cases = load_cases(Path(args.ruler_data_root), cfg)
    policies = {name: get_policy(name) for name in POLICIES}
    adapter = SaliencyAdapter(probe, offload_to_cpu=False)
    rows: list[dict[str, Any]] = []

    artifact = {
        "schema": "KIAOMNI_QWEN3_RULER_NIAH_FRONTIER_V1",
        "stage": args.stage,
        "repo_revision": args.repo_revision,
        "runner_sha256": file_sha256(__file__),
        "model": {"repo": MODEL_ID, "revision": MODEL_REVISION},
        "ruler": {
            "repo": RULER_REPO,
            "revision": RULER_REVISION,
            "tasks": list(cfg["tasks"]),
            "lengths": list(cfg["lengths"]),
            "samples_per_task": cfg["samples_per_task"],
        },
        "percentage_budgets": list(cfg["ratios"]),
        "environment": env,
        "routing_preflight": neutrality,
        "rows": rows,
        "aggregate": [],
        "pairwise": [],
        "execution_gate": {"status": "RUNNING"},
        "wall_seconds": 0.0,
    }
    write_json(output, artifact)

    for case_i, case in enumerate(cases):
        if time.perf_counter() >= deadline:
            raise TimeoutError("RULER stage wall-time ceiling reached")
        rendered = render_prompt(tokenizer, case.prompt)
        ids = core.encode_text(tokenizer, rendered).to(device)
        spans = reference_spans(tokenizer, ids, case.outputs)
        if not any(spans):
            raise RuntimeError(f"Could not locate any official RULER target span in {case.case_id}")

        full, full_routes = run_condition(
            model, tokenizer, case, ids, "full_context", None, capture,
            full_routes=None, saliency_meta=None, spans=spans,
        )
        rows.append({
            "case_id": case.case_id,
            "task": case.task,
            "seq_length": case.seq_length,
            "result": full,
        })

        saliency, saliency_meta = core.extract_saliency(model, adapter, ids)
        for policy_name in POLICIES:
            scores = policies[policy_name](saliency)
            for ratio in cfg["ratios"]:
                keep, budget = core.select_ratio(scores, ratio, int(ids.shape[1]))
                method = f"{policy_name}_r{ratio:g}"
                result, _ = run_condition(
                    model, tokenizer, case, ids, method, keep, capture,
                    full_routes=full_routes, saliency_meta=saliency_meta, spans=spans,
                )
                rows.append({
                    "case_id": case.case_id,
                    "task": case.task,
                    "seq_length": case.seq_length,
                    "policy": policy_name,
                    "requested_retention_ratio": ratio,
                    "derived_budget": budget,
                    "keep_positions_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
                    "result": result,
                })

        artifact["rows"] = rows
        artifact["aggregate"] = aggregate(rows)
        artifact["pairwise"] = pairwise(rows)
        artifact["wall_seconds"] = time.perf_counter() - started
        artifact["execution_gate"] = {
            "status": "RUNNING",
            "completed_cases": case_i + 1,
            "total_cases": len(cases),
        }
        write_json(output, artifact)
        torch.cuda.empty_cache()

    expected = len(cases) * (1 + len(POLICIES) * len(cfg["ratios"]))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(rows)}")
    artifact["aggregate"] = aggregate(rows)
    artifact["pairwise"] = pairwise(rows)
    artifact["wall_seconds"] = time.perf_counter() - started
    artifact["execution_gate"] = {
        "status": "PASS",
        "completed_cases": len(cases),
        "expected_rows": expected,
        "actual_rows": len(rows),
        "actual_dispatch_verified": True,
    }
    write_json(output, artifact)
    print(json.dumps(artifact["execution_gate"], indent=2), flush=True)
    capture.close()


def _cli_value(flag: str) -> str | None:
    try:
        i = sys.argv.index(flag)
        return sys.argv[i + 1]
    except Exception:
        return None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        output_raw = _cli_value("--output")
        if output_raw:
            try:
                write_json(Path(output_raw), {
                    "schema": "KIAOMNI_QWEN3_RULER_NIAH_FRONTIER_ERROR_V1",
                    "stage": _cli_value("--stage"),
                    "repo_revision": _cli_value("--repo-revision"),
                    "execution_gate": {
                        "status": "ERROR",
                        "exception_type": type(exc).__name__,
                        "reason": str(exc),
                    },
                    "traceback": traceback.format_exc(),
                })
            except Exception:
                pass
        traceback.print_exc()
        raise
