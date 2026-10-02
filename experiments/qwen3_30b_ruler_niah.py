from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
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
    raise RuntimeError(f"Cannot load percentage core from {BASE_PATH}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

MODEL_ID = base.MODEL_ID
MODEL_REVISION = base.MODEL_REVISION
POLICIES = base.POLICIES
PERCENTAGE_BUDGETS = base.PERCENTAGE_BUDGETS
LEGACY_FIXED_BUDGETS = base.LEGACY_FIXED_BUDGETS
N_SINK = base.N_SINK
RECENCY = base.RECENCY
SEED = 42
MAX_NEW_TOKENS = 64

RULER_DATASET_ID = "VenusChenyy/RULER_50"
RULER_OFFICIAL_REPO = "NVIDIA/RULER"
RULER_OFFICIAL_GENERATION_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
RULER_TASKS = (
    "niah_single_2",
    "niah_multikey_1",
    "niah_multivalue",
    "niah_multiquery",
)
RULER_LENGTHS = (8192, 16384)
SAMPLES_PER_GROUP = 5

STAGES = {
    "preflight": {
        "max_wall_seconds": 15 * 60,
        "max_cases": 1,
        "ratios": (0.25,),
    },
    "final": {
        "max_wall_seconds": 100 * 60,
        "max_cases": None,
        "ratios": PERCENTAGE_BUDGETS,
    },
}


@dataclass
class RulerCase:
    case_id: str
    source: str
    task: str
    target_context_length: int
    user_text: str
    gold: list[str]
    source_line: int
    declared_length: int
    declared_length_with_template: int
    official_token_position_answer: int
    official_depth_pct: float
    meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KiaOmni Qwen3 RULER NIAH controlled retention audit")
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--ruler-root", required=True)
    p.add_argument("--ruler-index", required=True)
    p.add_argument("--repo-revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    return p.parse_args()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_chat(tokenizer, user_text: str) -> str:
    return base.render_chat(tokenizer, user_text)


def encode_text(tokenizer, text: str) -> torch.Tensor:
    return base.encode_text(tokenizer, text)


def load_cases(tokenizer, ruler_root: str, ruler_index: str) -> list[RulerCase]:
    root = Path(ruler_root)
    idx = json.loads(Path(ruler_index).read_text(encoding="utf-8"))
    if idx.get("schema") != "KIAOMNI_RULER_NIAH_INDEX_V1":
        raise RuntimeError(f"Unexpected RULER index schema: {idx.get('schema')}")
    if idx.get("official_generation_commit") != RULER_OFFICIAL_GENERATION_COMMIT:
        raise RuntimeError("RULER official generation commit drift")
    groups = idx.get("groups", [])
    expected_groups = len(RULER_TASKS) * len(RULER_LENGTHS)
    if len(groups) != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} RULER groups, found {len(groups)}")

    cases: list[RulerCase] = []
    for group in groups:
        task = str(group["task"])
        length = int(group["context_length"])
        if task not in RULER_TASKS or length not in RULER_LENGTHS:
            raise RuntimeError(f"Unexpected frozen RULER group: {task}/{length}")
        path = root / group["relative_path"]
        if not path.exists():
            raise RuntimeError(f"Missing frozen RULER source file: {path}")
        observed_sha = file_sha256(path)
        if observed_sha != group["sha256"]:
            raise RuntimeError(
                f"RULER source SHA mismatch for {task}/{length}: "
                f"{observed_sha} != {group['sha256']}"
            )
        rows = read_jsonl(path)
        for selected in group["selected"]:
            source_line = int(selected["source_line"])
            if not (0 <= source_line < len(rows)):
                raise RuntimeError(f"Invalid source line {source_line} for {task}/{length}")
            row = rows[source_line]
            gold = [str(x).strip() for x in row.get("outputs", []) if str(x).strip()]
            if not gold:
                raise RuntimeError(f"No RULER outputs for {task}/{length} line {source_line}")
            raw_input = str(row.get("input", ""))
            answer_prefix = str(row.get("answer_prefix", ""))
            user_text = raw_input + answer_prefix
            rendered = render_chat(tokenizer, user_text)
            rendered_tokens = int(encode_text(tokenizer, rendered).shape[1])
            declared_length = int(row.get("length", 0))
            declared_w = int(row.get("length_w_model_temp", declared_length))
            official_pos = int(row.get("token_position_answer", 0))
            depth = float(selected["official_depth_pct"])
            cases.append(RulerCase(
                case_id=f"ruler-{task}-{length}-line{source_line}",
                source="ruler_niah",
                task=task,
                target_context_length=length,
                user_text=user_text,
                gold=gold,
                source_line=source_line,
                declared_length=declared_length,
                declared_length_with_template=declared_w,
                official_token_position_answer=official_pos,
                official_depth_pct=depth,
                meta={
                    "mirror_repo": idx["dataset_repo"],
                    "mirror_revision": idx["dataset_revision"],
                    "official_repo": RULER_OFFICIAL_REPO,
                    "official_generation_commit": RULER_OFFICIAL_GENERATION_COMMIT,
                    "rendered_tokens": rendered_tokens,
                    "answer_prefix_present": bool(answer_prefix),
                    "depth_bin": selected["depth_bin"],
                },
            ))
    return cases


def gold_token_spans(tokenizer, rendered: str, gold: list[str]) -> list[list[int]]:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer must expose offset_mapping for RULER needle-survival audit")
    spans: list[list[int]] = []
    lower = rendered.lower()
    for answer in gold:
        needle = answer.lower()
        char_start = lower.find(needle)
        if char_start < 0:
            raise RuntimeError(f"Gold answer not found in rendered RULER prompt: {answer!r}")
        char_end = char_start + len(answer)
        token_positions = [
            i for i, (a, b) in enumerate(offsets)
            if int(b) > char_start and int(a) < char_end
        ]
        if not token_positions:
            raise RuntimeError(f"No token span for RULER gold answer: {answer!r}")
        spans.append(token_positions)
    return spans


def score_ruler(answer: str, gold: list[str]) -> dict[str, Any]:
    pred = answer.strip()
    lower = pred.lower()
    found = [g.lower() in lower for g in gold]
    fraction = float(sum(found) / len(found))
    return {
        "correct": bool(all(found)),
        "ruler_string_match_fraction": fraction,
        "ruler_string_match_pct": 100.0 * fraction,
        "gold_outputs_found": found,
        "gold_outputs_found_count": int(sum(found)),
        "gold_outputs_total": len(found),
        "prediction": pred,
    }


def survival_metrics(
    spans: list[list[int]],
    keep: np.ndarray | None,
    full_prompt_len: int,
) -> dict[str, Any]:
    if keep is None:
        keep_set = set(range(full_prompt_len))
    else:
        keep_set = {int(x) for x in keep.tolist()}
    per_answer = []
    for span in spans:
        retained = sum(int(p in keep_set) for p in span)
        recall = retained / len(span)
        per_answer.append({
            "token_start": int(min(span)),
            "token_end": int(max(span)),
            "token_count": len(span),
            "retained_tokens": retained,
            "token_recall": float(recall),
            "complete": bool(retained == len(span)),
            "depth_pct": float(100.0 * min(span) / max(1, full_prompt_len)),
        })
    complete = sum(int(x["complete"]) for x in per_answer)
    return {
        "required_answer_token_recall": float(np.mean([x["token_recall"] for x in per_answer])),
        "complete_required_answers": int(complete),
        "required_answers_total": len(per_answer),
        "complete_required_answer_rate": float(complete / len(per_answer)),
        "all_required_answers_complete": bool(complete == len(per_answer)),
        "mean_required_answer_depth_pct": float(np.mean([x["depth_pct"] for x in per_answer])),
        "min_required_answer_depth_pct": float(np.min([x["depth_pct"] for x in per_answer])),
        "max_required_answer_depth_pct": float(np.max([x["depth_pct"] for x in per_answer])),
        "per_answer": per_answer,
    }


def run_condition(
    model,
    tokenizer,
    case: RulerCase,
    ids: torch.Tensor,
    method: str,
    keep: np.ndarray | None,
    capture,
    full_routes,
    full_prompt_len: int,
    saliency_meta: dict[str, Any] | None,
    spans: list[list[int]],
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    used = ids
    if keep is not None:
        kt = torch.as_tensor(keep, dtype=torch.long, device=ids.device)
        used = ids[:, kt]

    canonical_gold = ", ".join(case.gold)
    teacher, routes = base.teacher_gold_and_routing(
        model, tokenizer, used, canonical_gold, capture
    )
    answer, generation = base.generate_answer(model, tokenizer, used, MAX_NEW_TOKENS)
    score = score_ruler(answer, case.gold)
    survival = survival_metrics(spans, keep, full_prompt_len)

    routing = None
    if keep is not None:
        if full_routes is None:
            raise RuntimeError("Missing FullContext routes for compressed RULER condition")
        routing = base.compare_routes(
            full_routes,
            routes,
            keep,
            full_prompt_len=full_prompt_len,
            compressed_prompt_len=int(used.shape[1]),
            num_experts=int(model.config.num_experts),
        )

    pipeline_peak = max(
        generation["generation_peak_allocated_vram_gb"],
        teacher["routing_teacher_peak_allocated_vram_gb"],
        (saliency_meta or {}).get("saliency_peak_allocated_vram_gb", 0.0),
    )
    saliency_seconds = (
        (saliency_meta or {}).get("saliency_elapsed_seconds", 0.0)
        if keep is not None else 0.0
    )

    result = {
        "method": method,
        "input_tokens": int(ids.shape[1]),
        "kept_tokens": int(used.shape[1]),
        "compression_ratio": float(ids.shape[1] / used.shape[1]),
        "actual_retention_ratio": float(used.shape[1] / ids.shape[1]),
        "actual_retention_pct": float(100.0 * used.shape[1] / ids.shape[1]),
        "answer": answer,
        **score,
        **survival,
        **generation,
        **teacher,
        "inference_path_elapsed_seconds": float(
            generation["generation_elapsed_seconds"] + saliency_seconds
        ),
        "measurement_pipeline_elapsed_seconds": float(
            generation["generation_elapsed_seconds"]
            + teacher["routing_teacher_elapsed_seconds"]
            + saliency_seconds
        ),
        "pipeline_peak_allocated_vram_gb": pipeline_peak,
        "saliency": saliency_meta,
        "routing": routing,
    }
    return result, routes


def print_result(case: RulerCase, result: dict[str, Any]) -> None:
    print("\n" + "=" * 88, flush=True)
    print(
        f"CASE={case.case_id} TASK={case.task} TARGET_LEN={case.target_context_length} "
        f"DEPTH={case.official_depth_pct:.2f}%",
        flush=True,
    )
    print(
        f"METHOD={result['method']} SCORE={result['ruler_string_match_pct']:.2f} "
        f"ALL_CORRECT={result['correct']}",
        flush=True,
    )
    print(
        f"TOKENS={result['kept_tokens']}/{result['input_tokens']} "
        f"RETENTION={result['actual_retention_pct']:.3f}% "
        f"COMPRESSION={result['compression_ratio']:.3f}x",
        flush=True,
    )
    print(
        f"NEEDLE_RECALL={result['required_answer_token_recall']:.6f} "
        f"COMPLETE={result['complete_required_answers']}/{result['required_answers_total']}",
        flush=True,
    )
    print("--- RAW ANSWER ---", flush=True)
    print(result["answer"], flush=True)
    print(
        f"PPL={result['gold_answer_ppl']:.6f} "
        f"TTFT={result.get('time_to_first_token_seconds')} "
        f"OUT_TOK_S={result['output_tokens_per_second']:.3f} "
        f"GEN_PEAK_VRAM_GB={result['generation_peak_allocated_vram_gb']:.3f} "
        f"PIPELINE_PEAK_VRAM_GB={result['pipeline_peak_allocated_vram_gb']:.3f}",
        flush=True,
    )
    if result["routing"] is not None:
        r = result["routing"]
        print(
            "ROUTING_ACTUAL "
            f"TOP1={r['top1_expert_agreement']:.6f} "
            f"TOP8={r['top8_set_jaccard']:.6f} "
            f"WORST_TOP8={r['worst_layer_top8_set_jaccard']:.6f} "
            f"COS={r['dispatch_weight_cosine']:.6f} "
            f"LOAD_JSD={r['expert_load_jsd']:.6f}",
            flush=True,
        )
    print("=" * 88, flush=True)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task"], int(row["target_context_length"]), row["result"]["method"])
        groups.setdefault(key, []).append(row)

    out = []
    for (task, length, method), items in sorted(groups.items()):
        vals = [x["result"] for x in items]
        routing_vals = [x["routing"] for x in vals if x.get("routing") is not None]
        saliency_vals = [x["saliency"] for x in vals if x.get("saliency") is not None]

        def mean(key: str) -> float:
            return float(np.mean([float(x[key]) for x in vals]))

        def rmean(key: str) -> float | None:
            if not routing_vals:
                return None
            return float(np.mean([float(x[key]) for x in routing_vals]))

        ci = base._bootstrap_accuracy_ci([bool(x["correct"]) for x in vals], seed=SEED)
        out.append({
            "task": task,
            "target_context_length": length,
            "method": method,
            "n": len(vals),
            "ruler_score_pct": mean("ruler_string_match_pct"),
            "all_correct_rate": mean("correct"),
            "all_correct_ci95_low": ci[0],
            "all_correct_ci95_high": ci[1],
            "mean_required_answer_token_recall": mean("required_answer_token_recall"),
            "mean_complete_required_answer_rate": mean("complete_required_answer_rate"),
            "all_required_answers_complete_rate": mean("all_required_answers_complete"),
            "mean_required_answer_depth_pct": mean("mean_required_answer_depth_pct"),
            "mean_kept_tokens": mean("kept_tokens"),
            "mean_actual_retention_pct": mean("actual_retention_pct"),
            "mean_compression_ratio": mean("compression_ratio"),
            "mean_gold_answer_ppl": mean("gold_answer_ppl"),
            "mean_output_tokens_per_second": mean("output_tokens_per_second"),
            "mean_generation_elapsed_seconds": mean("generation_elapsed_seconds"),
            "mean_time_to_first_token_seconds": mean("time_to_first_token_seconds"),
            "mean_decode_after_first_token_seconds": mean("decode_after_first_token_seconds"),
            "mean_inference_path_elapsed_seconds": mean("inference_path_elapsed_seconds"),
            "max_generation_peak_vram_gb": max(float(x["generation_peak_allocated_vram_gb"]) for x in vals),
            "max_saliency_peak_vram_gb": (
                max(float(x["saliency_peak_allocated_vram_gb"]) for x in saliency_vals)
                if saliency_vals else None
            ),
            "max_pipeline_peak_vram_gb": max(float(x["pipeline_peak_allocated_vram_gb"]) for x in vals),
            "routing_mean_top1_expert_agreement": rmean("top1_expert_agreement"),
            "routing_mean_top8_set_jaccard": rmean("top8_set_jaccard"),
            "routing_mean_dispatch_weight_cosine": rmean("dispatch_weight_cosine"),
            "routing_mean_expert_load_jsd": rmean("expert_load_jsd"),
            "routing_mean_worst_layer_top1": rmean("worst_layer_top1_expert_agreement"),
            "routing_mean_worst_layer_top8": rmean("worst_layer_top8_set_jaccard"),
        })
    return out


def pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["result"]["method"]] = row["result"]

    methods = sorted({
        row["result"]["method"]
        for row in rows
        if row["result"]["method"] != "full_context"
    })
    out = []
    for method in methods:
        cc = cw = wc = ww = 0
        score_delta = []
        needle_delta = []
        n = 0
        for items in by_case.values():
            if "full_context" not in items or method not in items:
                continue
            f = items["full_context"]
            k = items[method]
            n += 1
            fc = bool(f["correct"])
            kc = bool(k["correct"])
            cc += int(fc and kc)
            cw += int(fc and not kc)
            wc += int((not fc) and kc)
            ww += int((not fc) and (not kc))
            score_delta.append(
                float(k["ruler_string_match_pct"]) - float(f["ruler_string_match_pct"])
            )
            needle_delta.append(
                float(k["required_answer_token_recall"]) - float(f["required_answer_token_recall"])
            )
        fc_correct = cc + cw
        fc_wrong = wc + ww
        out.append({
            "method": method,
            "n": n,
            "fc_correct_to_kia_correct": cc,
            "fc_correct_to_kia_wrong": cw,
            "fc_wrong_to_kia_correct": wc,
            "fc_wrong_to_kia_wrong": ww,
            "fc_correct_preservation_rate": (cc / fc_correct if fc_correct else None),
            "regression_rate_on_fc_correct": (cw / fc_correct if fc_correct else None),
            "rescue_rate_on_fc_wrong": (wc / fc_wrong if fc_wrong else None),
            "net_rescues_minus_regressions": wc - cw,
            "mcnemar_exact_p": base._exact_mcnemar_p(cw, wc),
            "mean_ruler_score_delta_pp": float(np.mean(score_delta)) if score_delta else None,
            "mean_required_answer_token_recall_delta": (
                float(np.mean(needle_delta)) if needle_delta else None
            ),
        })
    return out


def depth_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["task"],
            int(row["target_context_length"]),
            row["result"]["method"],
            str(row["depth_bin"]),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for (task, length, method, depth_bin), items in sorted(groups.items()):
        vals = [x["result"] for x in items]
        out.append({
            "task": task,
            "target_context_length": length,
            "method": method,
            "depth_bin": depth_bin,
            "n": len(vals),
            "mean_ruler_score_pct": float(np.mean([x["ruler_string_match_pct"] for x in vals])),
            "mean_required_answer_token_recall": float(np.mean([x["required_answer_token_recall"] for x in vals])),
            "all_correct_rate": float(np.mean([x["correct"] for x in vals])),
        })
    return out


def main() -> None:
    args = parse_args()
    cfg = STAGES[args.stage]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    deadline = started + cfg["max_wall_seconds"]
    output = Path(args.output)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()

    env = base.assert_environment(model, args.min_free_gb)
    probe = ArchitectureProbe.probe(model)
    if probe.confidence != "high":
        raise RuntimeError(f"Architecture probe confidence must be high, got {probe.confidence}")

    capture = base.ActualRoutingCapture(model)
    if len(capture.layer_ids) == 0:
        raise RuntimeError("No sparse MoE routing layers discovered")

    device = next(model.parameters()).device
    neutrality_text = render_chat(
        tokenizer,
        "A short RULER routing preflight. The target number is 1234567.",
    )
    neutrality_ids = encode_text(tokenizer, neutrality_text).to(device)
    neutrality = base.hook_neutrality_check(model, capture, neutrality_ids)
    print(f"ROUTING_PREFLIGHT={json.dumps(neutrality)}", flush=True)
    if not neutrality["passed"]:
        raise RuntimeError(f"Actual routing hook neutrality failed: {neutrality}")

    cases = load_cases(tokenizer, args.ruler_root, args.ruler_index)
    max_cases = cfg["max_cases"]
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    gpu_adapter = SaliencyAdapter(probe, offload_to_cpu=False)
    policy_fns = {name: get_policy(name) for name in POLICIES}
    rows: list[dict[str, Any]] = []

    artifact: dict[str, Any] = {
        "schema": "KIAOMNI_QWEN3_30B_RULER_NIAH_V1",
        "stage": args.stage,
        "repo_revision": args.repo_revision,
        "runner_sha256": file_sha256(__file__),
        "model": {"repo": MODEL_ID, "revision": MODEL_REVISION},
        "ruler": {
            "mirror_repo": RULER_DATASET_ID,
            "official_repo": RULER_OFFICIAL_REPO,
            "official_generation_commit": RULER_OFFICIAL_GENERATION_COMMIT,
            "tasks": list(RULER_TASKS),
            "lengths": list(RULER_LENGTHS),
            "samples_per_group": SAMPLES_PER_GROUP,
        },
        "policies": list(POLICIES),
        "percentage_budgets": list(cfg["ratios"]),
        "legacy_fixed_budgets_reference": list(LEGACY_FIXED_BUDGETS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "environment": env,
        "routing_preflight": neutrality,
        "cases": [
            asdict(c) | {"user_text": "<omitted-from-artifact>"}
            for c in cases
        ],
        "rows": rows,
        "aggregate": [],
        "pairwise": [],
        "depth_summary": [],
        "execution_gate": {"status": "RUNNING"},
        "wall_seconds": 0.0,
    }
    write_json(output, artifact)

    for case_i, case in enumerate(cases):
        if time.perf_counter() >= deadline:
            raise TimeoutError("RULER stage wall-time ceiling reached before next case")

        rendered = render_chat(tokenizer, case.user_text)
        ids = encode_text(tokenizer, rendered).to(device)
        input_len = int(ids.shape[1])
        spans = gold_token_spans(tokenizer, rendered, case.gold)

        full_result, full_routes = run_condition(
            model, tokenizer, case, ids, "full_context", None, capture,
            full_routes=None, full_prompt_len=input_len, saliency_meta=None,
            spans=spans,
        )
        rows.append({
            "case_id": case.case_id,
            "source": case.source,
            "task": case.task,
            "target_context_length": case.target_context_length,
            "depth_bin": case.meta["depth_bin"],
            "official_depth_pct": case.official_depth_pct,
            "result": full_result,
        })
        print_result(case, full_result)

        saliency, saliency_meta = base.extract_saliency(model, gpu_adapter, ids)

        for policy_name in POLICIES:
            scores = policy_fns[policy_name](saliency)
            if scores.shape != saliency.shape or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid policy scores for {policy_name}")
            for ratio in cfg["ratios"]:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("RULER stage wall-time ceiling reached before next condition")
                keep, budget = base.select_ratio(scores, ratio, input_len)
                method = f"{policy_name}_r{ratio:g}"
                result, _ = run_condition(
                    model, tokenizer, case, ids, method, keep, capture,
                    full_routes=full_routes, full_prompt_len=input_len,
                    saliency_meta=saliency_meta, spans=spans,
                )
                rows.append({
                    "case_id": case.case_id,
                    "source": case.source,
                    "task": case.task,
                    "target_context_length": case.target_context_length,
                    "depth_bin": case.meta["depth_bin"],
                    "official_depth_pct": case.official_depth_pct,
                    "policy": policy_name,
                    "requested_retention_ratio": ratio,
                    "requested_retention_pct": 100.0 * ratio,
                    "derived_budget": budget,
                    "keep_positions_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
                    "result": result,
                })
                print_result(case, result)

        artifact["rows"] = rows
        artifact["aggregate"] = aggregate(rows)
        artifact["pairwise"] = pairwise_summary(rows)
        artifact["depth_summary"] = depth_summary(rows)
        artifact["wall_seconds"] = time.perf_counter() - started
        artifact["execution_gate"] = {
            "status": "RUNNING",
            "completed_cases": case_i + 1,
            "total_cases": len(cases),
        }
        write_json(output, artifact)
        gc.collect()
        torch.cuda.empty_cache()

    expected_rows = len(cases) * (1 + len(POLICIES) * len(cfg["ratios"]))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} RULER rows, got {len(rows)}")

    artifact["aggregate"] = aggregate(rows)
    artifact["pairwise"] = pairwise_summary(rows)
    artifact["depth_summary"] = depth_summary(rows)
    artifact["wall_seconds"] = time.perf_counter() - started
    artifact["execution_gate"] = {
        "status": "PASS",
        "completed_cases": len(cases),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "actual_dispatch_verified": True,
        "note": (
            "PASS means execution/measurement integrity. RULER quality is reported "
            "through official string-match-compatible scores and needle-survival metrics."
        ),
    }
    write_json(output, artifact)
    print("\nFINAL EXECUTION GATE", flush=True)
    print(json.dumps(artifact["execution_gate"], indent=2), flush=True)
    print("\nPAIRWISE SUMMARY", flush=True)
    print(json.dumps(artifact["pairwise"], indent=2), flush=True)
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
            payload = {
                "schema": "KIAOMNI_QWEN3_30B_RULER_NIAH_ERROR_V1",
                "stage": _cli_value("--stage"),
                "repo_revision": _cli_value("--repo-revision"),
                "execution_gate": {
                    "status": "ERROR",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                },
                "traceback": traceback.format_exc(),
            }
            try:
                write_json(Path(output_raw), payload)
            except Exception:
                pass
        traceback.print_exc()
        raise
