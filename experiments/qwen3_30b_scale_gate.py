from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter
from kiaomni.policies import get_policy
from kiaomni.utils import N_SINK_DEFAULT, RECENCY_DEFAULT, select_keep

MODEL_DEFAULT = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION_DEFAULT = "0d7cf23"
DATASET_DEFAULT = "THUDM/LongBench-v2"
DATASET_REVISION_DEFAULT = "b0db4901b856522026b7353ab541b8535ff2a4b8"
SEED_DEFAULT = 42
POLICY = "kiaomni_s8"
N_SINK = N_SINK_DEFAULT
RECENCY = RECENCY_DEFAULT
PRIMARY_RATIO = 0.125

STAGE_PLANS: dict[str, dict[str, Any]] = {
    "preflight": {
        "synthetic_target_tokens": 2048,
        "retention_ratios": [0.25],
        "synthetic_tasks": ["hard_multi"],
        "synthetic_per_task": 1,
        "real_cases": 0,
        "max_new_tokens": 24,
        "max_wall_seconds": 15 * 60,
    },
    "smoke": {
        "synthetic_target_tokens": 4096,
        "retention_ratios": [0.25, 0.125],
        "synthetic_tasks": ["single", "multi", "hard_multi", "reason"],
        "synthetic_per_task": 1,
        "real_cases": 2,
        "real_min_tokens": 8192,
        "real_max_tokens": 12288,
        "max_new_tokens": 32,
        "max_wall_seconds": 35 * 60,
    },
    "final": {
        "synthetic_target_tokens": 8192,
        "retention_ratios": [0.25, 0.125, 0.0625],
        "synthetic_tasks": ["single", "multi", "hard_multi", "reason"],
        "synthetic_per_task": 2,
        "real_cases": 6,
        "real_min_tokens": 8192,
        "real_max_tokens": 16384,
        "max_new_tokens": 32,
        "max_wall_seconds": 85 * 60,
    },
}


@dataclass
class EvalCase:
    case_id: str
    source: str
    task: str
    context: str
    question: str
    gold: list[str]
    distractors: list[str]
    meta: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Budget-capped Qwen3-30B-A3B prompt-side KiaOmni scale gate"
    )
    p.add_argument("--stage", choices=sorted(STAGE_PLANS), required=True)
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--model-revision", default=MODEL_REVISION_DEFAULT)
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--dataset-revision", default=DATASET_REVISION_DEFAULT)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--asset-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    return p.parse_args()


def repo_git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def render_user(tokenizer, context: str, question: str) -> str:
    user = f"Document:\n{context}\n\nQuestion:\n{question}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user


def encode_case(tokenizer, case: EvalCase) -> torch.Tensor:
    rendered = render_user(tokenizer, case.context, case.question)
    return tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def token_len(tokenizer, context: str, question: str) -> int:
    return int(encode_case(
        tokenizer,
        EvalCase("_len", "_", "_", context, question, [], []),
    ).shape[1])


def _filler_sentence(rng: random.Random) -> str:
    subjects = [
        "The committee", "A regional survey", "The maintenance crew",
        "An early prototype", "The northern facility", "A visiting delegation",
        "The archive department", "Local observers", "The pilot program",
        "A follow-up study", "The logistics team", "An internal memo",
    ]
    verbs = [
        "reported", "confirmed", "documented", "reviewed", "scheduled",
        "postponed", "evaluated", "inspected", "catalogued", "summarized",
    ]
    objects = [
        "minor adjustments to the ventilation schedule",
        "a gradual rise in afternoon foot traffic",
        "the relocation of two storage containers",
        "routine calibration of the measurement rigs",
        "an updated rotation plan for night shifts",
        "stable energy consumption throughout the week",
        "a backlog of paperwork from the previous quarter",
        "slower than expected delivery of spare parts",
    ]
    return f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}."


def _fit_synthetic(
    tokenizer,
    question: str,
    items: list[tuple[float, str]],
    rng: random.Random,
    target_tokens: int,
) -> str:
    fillers = [_filler_sentence(rng) for _ in range(max(900, target_tokens // 4))]

    def build(n: int) -> str:
        out = list(fillers[:n])
        for depth, sentence in sorted(items, key=lambda x: x[0], reverse=True):
            out.insert(int(depth * len(out)), sentence)
        return " ".join(out)

    lo, hi = 0, len(fillers)
    best = build(0)
    best_len = token_len(tokenizer, best, question)
    while lo <= hi:
        mid = (lo + hi) // 2
        ctx = build(mid)
        n = token_len(tokenizer, ctx, question)
        if n <= target_tokens:
            best, best_len = ctx, n
            lo = mid + 1
        else:
            hi = mid - 1
    if best_len < target_tokens - 160:
        raise RuntimeError(
            f"synthetic prompt underfilled: {best_len} < {target_tokens - 160}"
        )
    return best


def make_synthetic_case(
    tokenizer,
    task: str,
    sample_id: int,
    seed: int,
    target_tokens: int,
) -> EvalCase:
    rng = random.Random(seed + 1009 * sample_id + sum(map(ord, task)))
    if task == "single":
        code = str(rng.randint(100000, 999999))
        question = "What is the vault access code? Answer with the number only."
        items = [(0.52, f"Security register: the vault access code is {code}.")]
        return EvalCase(
            f"syn-single-{sample_id}", "synthetic", task,
            _fit_synthetic(tokenizer, question, items, rng, target_tokens),
            question, [code], []
        )

    if task == "multi":
        vals = [str(rng.randint(10000, 99999)) for _ in range(3)]
        question = (
            "Return the three project codes for ALPHA, BRAVO, and CHARLIE in that "
            "order, separated by commas."
        )
        items = [
            (0.14, f"Registry record: project ALPHA code is {vals[0]}."),
            (0.50, f"Registry record: project BRAVO code is {vals[1]}."),
            (0.86, f"Registry record: project CHARLIE code is {vals[2]}."),
        ]
        return EvalCase(
            f"syn-multi-{sample_id}", "synthetic", task,
            _fit_synthetic(tokenizer, question, items, rng, target_tokens),
            question, vals, []
        )

    if task == "hard_multi":
        vals = [str(rng.randint(10000, 99999)) for _ in range(3)]
        bad = [str(rng.randint(10000, 99999)) for _ in range(3)]
        question = (
            "Using only the AURORA records, return the codes for ALPHA, BRAVO, "
            "and CHARLIE in that order, separated by commas."
        )
        items = [
            (0.10, f"AURORA record: ALPHA code is {vals[0]}."),
            (0.28, f"BOREAL record: ALPHA code is {bad[0]}."),
            (0.46, f"AURORA record: BRAVO code is {vals[1]}."),
            (0.63, f"BOREAL record: BRAVO code is {bad[1]}."),
            (0.79, f"AURORA record: CHARLIE code is {vals[2]}."),
            (0.91, f"BOREAL record: CHARLIE code is {bad[2]}."),
        ]
        return EvalCase(
            f"syn-hard-{sample_id}", "synthetic", task,
            _fit_synthetic(tokenizer, question, items, rng, target_tokens),
            question, vals, bad
        )

    if task == "reason":
        a = rng.randint(1000, 8000)
        d = a + 31
        distractor = a + 99
        question = "What is the final value of variable D? Answer with the number only."
        items = [
            (0.15, f"Variable A is initialized to {a}."),
            (0.36, "Variable B equals variable A plus 7."),
            (0.57, "Variable C equals variable B plus 11."),
            (0.78, "Variable D equals variable C plus 13."),
            (0.88, f"Unrelated note: variable X is {distractor}."),
        ]
        return EvalCase(
            f"syn-reason-{sample_id}", "synthetic", task,
            _fit_synthetic(tokenizer, question, items, rng, target_tokens),
            question, [str(d)], [str(distractor)]
        )

    raise ValueError(f"unknown synthetic task: {task}")


def _mc_question(row: dict[str, Any]) -> str:
    return (
        f"{row['question']}\n\nChoices:\n"
        f"A: {row['choice_A']}\n"
        f"B: {row['choice_B']}\n"
        f"C: {row['choice_C']}\n"
        f"D: {row['choice_D']}\n\n"
        "Answer with one letter only: A, B, C, or D."
    )


def load_real_cases(
    tokenizer,
    dataset_dir: str,
    count: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> list[EvalCase]:
    if count <= 0:
        return []
    from datasets import load_from_disk

    ds = load_from_disk(dataset_dir)
    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for raw in ds:
        row = dict(raw)
        q = _mc_question(row)
        n = token_len(tokenizer, str(row["context"]), q)
        if min_tokens <= n <= max_tokens:
            candidates.append((str(row.get("domain", "unknown")), n, row))

    if len(candidates) < count:
        raise RuntimeError(
            f"LongBench-v2 has only {len(candidates)} naturally fitting cases in "
            f"[{min_tokens}, {max_tokens}] tokens; need {count}. No truncation allowed."
        )

    rng = random.Random(seed + 7301)
    by_domain: dict[str, list[tuple[str, int, dict[str, Any]]]] = {}
    for item in candidates:
        by_domain.setdefault(item[0], []).append(item)
    for items in by_domain.values():
        rng.shuffle(items)

    picked: list[tuple[str, int, dict[str, Any]]] = []
    domains = sorted(by_domain)
    while len(picked) < count:
        progressed = False
        for domain in domains:
            if by_domain[domain] and len(picked) < count:
                picked.append(by_domain[domain].pop())
                progressed = True
        if not progressed:
            break

    cases: list[EvalCase] = []
    for domain, n, row in picked:
        answer = str(row["answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue
        cases.append(
            EvalCase(
                case_id=f"longbenchv2-{row['_id']}",
                source="longbench_v2",
                task=str(row.get("sub_domain", domain)),
                context=str(row["context"]),
                question=_mc_question(row),
                gold=[answer],
                distractors=[],
                meta={
                    "domain": domain,
                    "sub_domain": row.get("sub_domain"),
                    "difficulty": row.get("difficulty"),
                    "length": row.get("length"),
                    "rendered_tokens": n,
                },
            )
        )
    if len(cases) < count:
        raise RuntimeError(f"Only {len(cases)} valid LongBench-v2 cases selected; need {count}")
    return cases[:count]


def score_answer(case: EvalCase, answer: str) -> dict[str, Any]:
    if case.source == "longbench_v2":
        upper = answer.strip().upper()
        match = re.search(r"\b([ABCD])\b", upper)
        parsed = match.group(1) if match else None
        return {
            "success": parsed == case.gold[0],
            "recall": 1.0 if parsed == case.gold[0] else 0.0,
            "prediction": answer.strip(),
            "parsed_choice": parsed,
            "distractor_hit": False,
        }

    text = normalize_text(answer)
    found = [normalize_text(x) in text for x in case.gold]
    distractor_hit = any(normalize_text(x) in text for x in case.distractors)
    recall = sum(found) / len(found) if found else 0.0
    return {
        "success": bool(found) and all(found) and not distractor_hit,
        "recall": recall,
        "prediction": answer.strip(),
        "parsed_choice": None,
        "distractor_hit": distractor_hit,
    }


def _cuda_peak_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _generate_ids(
    model,
    tokenizer,
    ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    _reset_peak()
    t0 = time.perf_counter()
    with torch.inference_mode():
        seq = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = seq[:, ids.shape[1]:]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return text, {
        "elapsed_s": elapsed,
        "generated_tokens": int(new_tokens.shape[1]),
        "output_tokens_per_s": (
            float(new_tokens.shape[1]) / elapsed if elapsed > 0 else None
        ),
        "peak_allocated_vram_gb": _cuda_peak_gb(),
    }


def _extract_saliency(
    model,
    adapter: SaliencyAdapter,
    ids: torch.Tensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    _reset_peak()
    t0 = time.perf_counter()
    sal = adapter.extract(ids, model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if sal.shape != tuple(ids.shape):
        raise RuntimeError(f"saliency shape mismatch: {sal.shape} vs {tuple(ids.shape)}")
    if not np.isfinite(sal).all():
        raise RuntimeError("Non-finite saliency detected")
    return sal[0], {
        "elapsed_s": elapsed,
        "peak_allocated_vram_gb": _cuda_peak_gb(),
    }


def _budget_for_ratio(length: int, ratio: float) -> int:
    budget = int(round(length * ratio))
    return max(N_SINK + RECENCY, min(length - 1, budget))


def _kia_keep(scores: np.ndarray, length: int, budget: int) -> np.ndarray:
    return np.sort(
        select_keep(
            scores,
            budget,
            length,
            n_sink=N_SINK,
            recency=RECENCY,
        )
    )


def _recency_keep(length: int, budget: int) -> np.ndarray:
    sink = np.arange(min(N_SINK, length), dtype=np.int64)
    remaining = max(0, budget - len(sink))
    tail_start = max(len(sink), length - remaining)
    tail = np.arange(tail_start, length, dtype=np.int64)
    return np.unique(np.concatenate([sink, tail]))[:budget]


def _random_keep(length: int, budget: int, seed: int) -> np.ndarray:
    protected = set(range(min(N_SINK, length)))
    protected.update(range(max(0, length - RECENCY), length))
    free = max(0, budget - len(protected))
    candidates = [i for i in range(length) if i not in protected]
    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=min(free, len(candidates)))
    return np.array(sorted(protected | set(chosen)), dtype=np.int64)


def _run_condition(
    model,
    tokenizer,
    case: EvalCase,
    ids: torch.Tensor,
    method: str,
    keep: np.ndarray | None,
    max_new_tokens: int,
    saliency_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    used = ids
    kept_tokens = int(ids.shape[1])
    if keep is not None:
        keep_t = torch.as_tensor(keep, device=ids.device, dtype=torch.long)
        used = ids[:, keep_t]
        kept_tokens = int(used.shape[1])
    answer, perf = _generate_ids(model, tokenizer, used, max_new_tokens)
    score = score_answer(case, answer)
    return {
        "method": method,
        "input_tokens": int(ids.shape[1]),
        "kept_tokens": kept_tokens,
        "retention_ratio_actual": kept_tokens / int(ids.shape[1]),
        "compression_ratio": int(ids.shape[1]) / kept_tokens,
        "answer": answer,
        "generation": perf,
        "saliency": saliency_meta,
        **score,
    }


def saliency_parity_check(
    model,
    tokenizer,
    probe,
    seed: int,
) -> dict[str, Any]:
    case = make_synthetic_case(tokenizer, "single", 991, seed, 512)
    ids = encode_case(tokenizer, case).to(model.device)
    cpu_adapter = SaliencyAdapter(probe, offload_to_cpu=True)
    gpu_adapter = SaliencyAdapter(probe, offload_to_cpu=False)
    cpu, cpu_meta = _extract_saliency(model, cpu_adapter, ids)
    gpu, gpu_meta = _extract_saliency(model, gpu_adapter, ids)
    delta = np.abs(cpu - gpu)
    if np.std(cpu) == 0 or np.std(gpu) == 0:
        corr = 1.0 if np.allclose(cpu, gpu, rtol=1e-4, atol=1e-6) else 0.0
    else:
        corr = float(np.corrcoef(cpu, gpu)[0, 1])
    k = min(128, len(cpu))
    cpu_top = set(np.argpartition(-cpu, k - 1)[:k].tolist())
    gpu_top = set(np.argpartition(-gpu, k - 1)[:k].tolist())
    jaccard = len(cpu_top & gpu_top) / len(cpu_top | gpu_top)
    passed = bool(
        np.isfinite(cpu).all()
        and np.isfinite(gpu).all()
        and corr >= 0.999
        and jaccard >= 0.98
    )
    return {
        "passed": passed,
        "pearson": corr,
        "top128_jaccard": jaccard,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "cpu": cpu_meta,
        "gpu": gpu_meta,
        "criteria": {"pearson_min": 0.999, "top128_jaccard_min": 0.98},
    }


def environment_snapshot(model, tokenizer, cfg: dict[str, Any]) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    mc = model.config
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": {
            "name": props.name,
            "total_memory_gb": props.total_memory / 2**30,
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "model_type": getattr(mc, "model_type", None),
        "num_hidden_layers": getattr(mc, "num_hidden_layers", None),
        "num_attention_heads": getattr(mc, "num_attention_heads", None),
        "num_key_value_heads": getattr(mc, "num_key_value_heads", None),
        "num_experts": getattr(mc, "num_experts", None),
        "num_experts_per_tok": getattr(mc, "num_experts_per_tok", None),
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer_class": type(tokenizer).__name__,
        "stage_config": cfg,
        "n_sink": N_SINK,
        "recency": RECENCY,
        "policy": POLICY,
    }


def check_probe(model) -> tuple[Any, dict[str, Any]]:
    probe = ArchitectureProbe.probe(model)
    return probe, {
        "confidence": probe.confidence,
        "qkv_pattern": probe.qkv_pattern,
        "num_layers": probe.num_layers,
        "num_attention_heads": probe.num_attention_heads,
        "num_key_value_heads": probe.num_key_value_heads,
        "head_dim": probe.head_dim,
        "layer_container_path": probe.layer_container_path,
        "attn_module_name": probe.attn_module_name,
        "attn_implementation": probe.attn_implementation,
        "pos_encoding": probe.pos_encoding,
        "detection_notes": list(probe.detection_notes),
    }


def assert_model_safety(model, min_free_gb: float) -> dict[str, Any]:
    if getattr(model, "is_quantized", False):
        raise RuntimeError("Quantized model detected; Phase 03 requires BF16")
    device_map = getattr(model, "hf_device_map", None) or {}
    forbidden = {
        str(v).lower()
        for v in device_map.values()
        if str(v).lower() in {"cpu", "disk", "meta"}
    }
    if forbidden:
        raise RuntimeError(
            f"CPU/disk offload is forbidden in Phase 03; found {sorted(forbidden)}"
        )
    non_cuda_param_devices = sorted({
        str(p.device) for p in model.parameters() if p.device.type != "cuda"
    })
    if non_cuda_param_devices:
        raise RuntimeError(
            "CPU/disk offload is forbidden in Phase 03; "
            f"non-CUDA parameters found on {non_cuda_param_devices}"
        )
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient post-load GPU headroom: {free_gb:.2f} GiB free; "
            f"require >= {min_free_gb:.2f} GiB"
        )
    return {
        "free_gb": free_gb,
        "total_gb": total_gb,
        "hf_device_map": {str(k): str(v) for k, v in device_map.items()},
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {
        row["case_id"]: bool(row["result"]["success"])
        for row in rows
        if row["result"]["method"] == "full_context"
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["source"], row["result"]["method"])
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (source, method), items in sorted(groups.items()):
        successes = np.array(
            [1.0 if x["result"]["success"] else 0.0 for x in items], dtype=float
        )
        conditioned = [x for x in items if full.get(x["case_id"], False)]
        out.append(
            {
                "source": source,
                "method": method,
                "n": len(items),
                "accuracy": float(successes.mean()) if len(successes) else None,
                "mean_recall": float(np.mean([x["result"]["recall"] for x in items])),
                "full_context_solved_n": len(conditioned),
                "full_context_conditioned_accuracy": (
                    float(np.mean([1.0 if x["result"]["success"] else 0.0 for x in conditioned]))
                    if conditioned else None
                ),
            }
        )
    return out


def _find_aggregate(
    aggregate_rows: list[dict[str, Any]],
    source: str,
    method: str,
) -> dict[str, Any] | None:
    return next(
        (x for x in aggregate_rows if x["source"] == source and x["method"] == method),
        None,
    )


def build_gate(
    stage: str,
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    parity: dict[str, Any] | None,
    expected_cases: int,
) -> dict[str, Any]:
    completed = len({row["case_id"] for row in rows})
    if stage == "preflight":
        passed = completed == expected_cases and parity is not None and parity["passed"]
        return {
            "status": "PASS" if passed else "FAIL",
            "criteria": {
                "completed_cases": f"{completed}/{expected_cases}",
                "gpu_cpu_saliency_parity": bool(parity and parity["passed"]),
            },
        }

    if completed != expected_cases:
        return {
            "status": "FAIL",
            "criteria": {"completed_cases": f"{completed}/{expected_cases}"},
        }

    if stage == "smoke":
        return {
            "status": "PASS",
            "criteria": {"completed_cases": f"{completed}/{expected_cases}"},
        }

    method4 = "kiaomni_r0.25"
    method8 = "kiaomni_r0.125"
    overall4_syn = _find_aggregate(aggregate_rows, "synthetic", method4)
    overall8_syn = _find_aggregate(aggregate_rows, "synthetic", method8)
    overall8_real = _find_aggregate(aggregate_rows, "longbench_v2", method8)
    random8_syn = _find_aggregate(aggregate_rows, "synthetic", "random_r0.125")
    recency8_syn = _find_aggregate(aggregate_rows, "synthetic", "recency_r0.125")

    required = [overall4_syn, overall8_syn, overall8_real, random8_syn, recency8_syn]
    if any(x is None for x in required):
        return {"status": "FAIL", "reason": "missing required aggregate rows"}

    assert overall4_syn and overall8_syn and overall8_real and random8_syn and recency8_syn
    syn_denom = int(overall8_syn["full_context_solved_n"])
    real_denom = int(overall8_real["full_context_solved_n"])
    if syn_denom < 4 or real_denom < 3:
        return {
            "status": "INCONCLUSIVE",
            "reason": "FullContext solved too few frozen cases for a quality-retention decision",
            "full_context_solved": {"synthetic": syn_denom, "real": real_denom},
        }

    checks = {
        "synthetic_4x_conditioned_ge_0.80":
            float(overall4_syn["full_context_conditioned_accuracy"]) >= 0.80,
        "synthetic_8x_conditioned_ge_0.70":
            float(overall8_syn["full_context_conditioned_accuracy"]) >= 0.70,
        "real_8x_conditioned_ge_0.60":
            float(overall8_real["full_context_conditioned_accuracy"]) >= 0.60,
        "kiaomni_8x_not_below_random_synthetic":
            float(overall8_syn["full_context_conditioned_accuracy"])
            >= float(random8_syn["full_context_conditioned_accuracy"]),
        "kiaomni_8x_not_below_recency_synthetic":
            float(overall8_syn["full_context_conditioned_accuracy"])
            >= float(recency8_syn["full_context_conditioned_accuracy"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "criteria": checks,
        "full_context_solved": {"synthetic": syn_denom, "real": real_denom},
        "note": "16x is a stress diagnostic and is intentionally not a PASS requirement.",
    }


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    cfg = dict(STAGE_PLANS[args.stage])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    deadline = started + int(cfg["max_wall_seconds"])
    output = Path(args.output)

    manifest = json.loads(Path(args.asset_manifest).read_text(encoding="utf-8"))
    if manifest.get("model_repo") != args.model:
        raise RuntimeError("asset manifest model repo mismatch")
    requested_model_rev = str(manifest.get("model_revision_requested", ""))
    resolved_model_rev = str(manifest.get("model_revision_resolved", ""))
    if (
        requested_model_rev != args.model_revision
        or not resolved_model_rev.startswith(args.model_revision)
    ):
        raise RuntimeError("asset manifest model revision mismatch")
    if manifest.get("dataset_repo") != args.dataset:
        raise RuntimeError("asset manifest dataset repo mismatch")
    if manifest.get("dataset_revision") != args.dataset_revision:
        raise RuntimeError("asset manifest dataset revision mismatch")
    resolved_dataset_rev = str(manifest.get("dataset_revision_resolved", ""))
    if resolved_dataset_rev != args.dataset_revision:
        raise RuntimeError("resolved dataset revision mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required; do not silently downgrade precision")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()

    safety = assert_model_safety(model, args.min_free_gb)
    probe, probe_record = check_probe(model)
    gpu_adapter = SaliencyAdapter(probe, offload_to_cpu=False)
    score_fn = get_policy(POLICY)

    parity = None
    if args.stage == "preflight":
        parity = saliency_parity_check(model, tokenizer, probe, args.seed)
        if not parity["passed"]:
            raise RuntimeError(f"GPU saliency parity failed: {parity}")

    cases: list[EvalCase] = []
    for task in cfg["synthetic_tasks"]:
        for i in range(int(cfg["synthetic_per_task"])):
            cases.append(
                make_synthetic_case(
                    tokenizer,
                    task,
                    i,
                    args.seed,
                    int(cfg["synthetic_target_tokens"]),
                )
            )

    if int(cfg["real_cases"]) > 0:
        cases.extend(
            load_real_cases(
                tokenizer,
                args.dataset_dir,
                int(cfg["real_cases"]),
                int(cfg["real_min_tokens"]),
                int(cfg["real_max_tokens"]),
                args.seed,
            )
        )

    expected_cases = len(cases)
    rows: list[dict[str, Any]] = []
    artifact: dict[str, Any] = {
        "schema": "KIAOMNI_QWEN3_30B_SCALE_GATE_V2",
        "claim_scope": (
            "Prompt-side KiaOmni policy scaling on Qwen3-30B-A3B-Instruct-2507. "
            "This is not evidence of real past_key_values KV-cache eviction."
        ),
        "stage": args.stage,
        "model": {"repo": args.model, "revision": args.model_revision},
        "dataset": {"repo": args.dataset, "revision": args.dataset_revision},
        "seed": args.seed,
        "repo_git_head": repo_git_head(),
        "runner_sha256": file_sha256(__file__),
        "asset_manifest": manifest,
        "environment": environment_snapshot(model, tokenizer, cfg),
        "post_load_safety": safety,
        "probe": probe_record,
        "saliency_cpu_gpu_parity": parity,
        "cases": [asdict(c) | {"context": "<omitted-from-artifact>"} for c in cases],
        "rows": rows,
        "aggregate": [],
        "gate": {"status": "RUNNING"},
        "wall_seconds": 0.0,
    }
    write_artifact(output, artifact)

    for case_index, case in enumerate(cases):
        if time.perf_counter() >= deadline:
            raise TimeoutError("experiment wall-time budget reached before next case")
        ids = encode_case(tokenizer, case).to(model.device)
        input_len = int(ids.shape[1])

        full_result = _run_condition(
            model, tokenizer, case, ids, "full_context", None,
            int(cfg["max_new_tokens"]),
        )
        rows.append({
            "case_id": case.case_id,
            "source": case.source,
            "task": case.task,
            "rendered_tokens": input_len,
            "result": full_result,
        })

        saliency, saliency_meta = _extract_saliency(model, gpu_adapter, ids)
        policy_scores = score_fn(saliency)

        for ratio in [float(x) for x in cfg["retention_ratios"]]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("experiment wall-time budget reached before next condition")
            budget = _budget_for_ratio(input_len, ratio)
            keep = _kia_keep(policy_scores, input_len, budget)
            result = _run_condition(
                model, tokenizer, case, ids, f"kiaomni_r{ratio:g}", keep,
                int(cfg["max_new_tokens"]), saliency_meta,
            )
            rows.append({
                "case_id": case.case_id,
                "source": case.source,
                "task": case.task,
                "rendered_tokens": input_len,
                "requested_retention_ratio": ratio,
                "result": result,
            })

        if args.stage == "final":
            ratio = PRIMARY_RATIO
            budget = _budget_for_ratio(input_len, ratio)
            recency = _recency_keep(input_len, budget)
            random_keep = _random_keep(
                input_len,
                budget,
                args.seed + case_index * 10007,
            )
            for method, keep in (
                ("recency_r0.125", recency),
                ("random_r0.125", random_keep),
            ):
                result = _run_condition(
                    model, tokenizer, case, ids, method, keep,
                    int(cfg["max_new_tokens"]),
                )
                rows.append({
                    "case_id": case.case_id,
                    "source": case.source,
                    "task": case.task,
                    "rendered_tokens": input_len,
                    "requested_retention_ratio": ratio,
                    "result": result,
                })

        artifact["rows"] = rows
        artifact["aggregate"] = aggregate(rows)
        artifact["wall_seconds"] = time.perf_counter() - started
        artifact["gate"] = {"status": "RUNNING", "completed_cases": case_index + 1}
        write_artifact(output, artifact)
        print(
            f"[{case_index + 1}/{expected_cases}] {case.case_id} "
            f"full={full_result['success']} input_tokens={input_len}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    aggregate_rows = aggregate(rows)
    gate = build_gate(args.stage, rows, aggregate_rows, parity, expected_cases)
    artifact["aggregate"] = aggregate_rows
    artifact["gate"] = gate
    artifact["wall_seconds"] = time.perf_counter() - started
    write_artifact(output, artifact)
    print(json.dumps({"gate": gate, "output": str(output)}, indent=2))

    if gate["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
