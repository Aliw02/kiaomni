from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^CASE=(\S+)\s+SOURCE=(\S+)\s+TASK=(.+)$")
METHOD_RE = re.compile(r"^METHOD=(\S+)\s+GOLD=([A-D])$")
TOKENS_RE = re.compile(r"^TOKENS=(\d+)/(\d+)\s+COMPRESSION=([0-9.]+)x$")
SCORES_RE = re.compile(
    r"^PARSED=(\S+)\s+CORRECT=(True|False)\s+PARSE=(\S+)\s+"
    r"GENERATED=(\d+)\s+HIT_LIMIT=(True|False)$"
)
METRICS_RE = re.compile(
    r"^PPL=([0-9.eE+-]+)\s+NLL=([0-9.eE+-]+)\s+OUT_TOK_S=([0-9.eE+-]+)\s+"
    r"GEN_PEAK_VRAM_GB=([0-9.eE+-]+)\s+PIPELINE_PEAK_VRAM_GB=([0-9.eE+-]+)\s+"
    r"ACTUAL_DISPATCH_VERIFIED=(True|False)$"
)
SALIENCY_RE = re.compile(
    r"^SALIENCY_SECONDS=([0-9.eE+-]+)\s+SALIENCY_PEAK_VRAM_GB=([0-9.eE+-]+)$"
)
ROUTING_RE = re.compile(
    r"^ROUTING_ACTUAL\s+TOP1=([0-9.eE+-]+)\s+TOP8_JACCARD=([0-9.eE+-]+)\s+"
    r"WEIGHT_COS=([0-9.eE+-]+)\s+ENTROPY_DELTA=([0-9.eE+-]+)\s+LOAD_JSD=([0-9.eE+-]+)$"
)


def b(v: str) -> bool:
    return v == "True"


def parse_log(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = CASE_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        case_id, source, task = m.groups()
        block: list[str] = [lines[i].strip()]
        i += 1
        while i < len(lines) and not CASE_RE.match(lines[i].strip()):
            block.append(lines[i].strip())
            i += 1

        method_m = next((METHOD_RE.match(x) for x in block if METHOD_RE.match(x)), None)
        tokens_m = next((TOKENS_RE.match(x) for x in block if TOKENS_RE.match(x)), None)
        scores_m = next((SCORES_RE.match(x) for x in block if SCORES_RE.match(x)), None)
        metrics_m = next((METRICS_RE.match(x) for x in block if METRICS_RE.match(x)), None)
        if not all((method_m, tokens_m, scores_m, metrics_m)):
            continue

        method, gold = method_m.groups()
        kept, inp, compression = tokens_m.groups()
        parsed, correct, parse_status, generated, hit_limit = scores_m.groups()
        ppl, nll, out_tps, gen_vram, pipe_vram, dispatch = metrics_m.groups()
        sal_m = next((SALIENCY_RE.match(x) for x in block if SALIENCY_RE.match(x)), None)
        route_m = next((ROUTING_RE.match(x) for x in block if ROUTING_RE.match(x)), None)

        answer = ""
        try:
            a = block.index("--- RAW ANSWER ---")
            z = block.index("--- SCORES ---")
            answer = "\n".join(block[a + 1:z]).strip()
        except ValueError:
            pass

        result: dict[str, Any] = {
            "method": method,
            "answer": answer,
            "parsed_answer": None if parsed == "None" else parsed,
            "parse_status": parse_status,
            "correct": b(correct),
            "gold": gold,
            "input_tokens": int(inp),
            "kept_tokens": int(kept),
            "actual_retention_ratio": int(kept) / int(inp),
            "compression_ratio": float(compression),
            "generated_tokens": int(generated),
            "hit_token_limit": b(hit_limit),
            "gold_answer_ppl": float(ppl),
            "gold_answer_nll": float(nll),
            "output_tokens_per_second": float(out_tps),
            "generation_peak_allocated_vram_gb": float(gen_vram),
            "pipeline_peak_allocated_vram_gb": float(pipe_vram),
            "dispatch_verified": b(dispatch),
            "time_to_first_token_seconds": None,
            "decode_after_first_token_seconds": None,
            "inference_path_elapsed_seconds": None,
            "routing": None,
            "saliency": None,
        }
        if sal_m:
            sec, peak = sal_m.groups()
            result["saliency"] = {
                "saliency_elapsed_seconds": float(sec),
                "saliency_peak_allocated_vram_gb": float(peak),
            }
        if route_m:
            top1, top8, cos, entropy, jsd = route_m.groups()
            result["routing"] = {
                "top1_expert_agreement": float(top1),
                "top8_set_jaccard": float(top8),
                "dispatch_weight_cosine": float(cos),
                "dispatch_entropy_delta": float(entropy),
                "expert_load_jsd": float(jsd),
                "worst_layer_top1_expert_agreement": None,
                "worst_layer_top8_set_jaccard": None,
            }
        rows.append({
            "case_id": case_id,
            "source": source,
            "task": task,
            "result": result,
            "recovered_from_log": True,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--tail", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    recovered = parse_log(Path(args.log))
    by_case: dict[str, list[dict[str, Any]]] = {}
    case_order: list[str] = []
    for row in recovered:
        cid = row["case_id"]
        if cid not in by_case:
            by_case[cid] = []
            case_order.append(cid)
        by_case[cid].append(row)

    complete_prefix: list[dict[str, Any]] = []
    for cid in case_order:
        rs = by_case[cid]
        if len(rs) == 7:
            complete_prefix.extend(rs)
        else:
            break

    tail = json.loads(Path(args.tail).read_text(encoding="utf-8"))
    tail_rows = list(tail.get("rows", []))
    rows = complete_prefix + tail_rows

    counts = Counter(str(r["case_id"]) for r in rows)
    if len(counts) != 27:
        raise RuntimeError(f"Expected 27 unique cases after recovery, got {len(counts)}")
    bad = {k: v for k, v in counts.items() if v != 7}
    if bad:
        raise RuntimeError(f"Each case must have exactly 7 rows; bad counts: {bad}")
    if len(rows) != 189:
        raise RuntimeError(f"Expected 189 rows, got {len(rows)}")

    payload = {
        "schema": "KIAOMNI_QWEN3_30B_PERCENTAGE_REPLAY_RECOVERED_V1",
        "stage": "final",
        "rows": rows,
        "execution_gate": {
            "status": "PASS_RECOVERED",
            "completed_cases": 27,
            "expected_rows": 189,
            "actual_rows": 189,
            "core_metrics_complete": True,
            "note": (
                "Cases recovered from the timed-out log preserve answer/correctness/PPL/"
                "throughput/VRAM/saliency and mean routing metrics printed by the frozen runner. "
                "TTFT/decode timing and worst-layer routing fields are unavailable for those "
                "log-recovered rows and are null. Tail cases come from the structured recovery artifact."
            ),
        },
        "recovery": {
            "complete_prefix_cases_from_log": len(complete_prefix) // 7,
            "tail_cases_from_structured_artifact": len(tail_rows) // 7,
            "log_path": str(args.log),
            "tail_path": str(args.tail),
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["execution_gate"], indent=2))
    print(json.dumps(payload["recovery"], indent=2))


if __name__ == "__main__":
    main()
