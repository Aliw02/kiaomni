from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from kiaomni.utils import select_keep

MODEL_DEFAULT = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION_DEFAULT = "b9b7053e66b5de60c03b1913dbc21e900ef7ded7"
DATASET_DEFAULT = "THUDM/LongBench-v2"
DATASET_REVISION_DEFAULT = "b0db4901b856522026b7353ab541b8535ff2a4b8"
SEED_DEFAULT = 42
N_SINK = 16
RECENCY = 32
POLICY = "kiaomni_s8"

STAGE_PLANS: dict[str, dict[str, Any]] = {
    "preflight": {
        "target_tokens": 8192,
        "budgets": [1024],
        "synthetic_per_task": 1,
        "real_cases": 0,
        "tasks": ["hard_multi"],
        "max_new_tokens": 24,
        "max_wall_seconds": 28 * 60,
    },
    "smoke": {
        "target_tokens": 8192,
        "budgets": [2048, 1024],
        "synthetic_per_task": 1,
        "real_cases": 2,
        "tasks": ["single", "multi", "hard_multi", "reason"],
        "max_new_tokens": 24,
        "max_wall_seconds": 43 * 60,
    },
    "final": {
        "target_tokens": 8192,
        "budgets": [2048, 1024, 512],
        "synthetic_per_task": 2,
        "real_cases": 8,
        "tasks": ["single", "multi", "hard_multi", "reason"],
        "max_new_tokens": 24,
        "max_wall_seconds": 118 * 60,
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
    choices: dict[str, str] | None = None
    meta: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Budget-capped Qwen3-30B-A3B KiaOmni MoE scaling gate"
    )
    p.add_argument("--stage", choices=sorted(STAGE_PLANS), required=True)
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--model-revision", default=MODEL_REVISION_DEFAULT)
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--dataset-revision", default=DATASET_REVISION_DEFAULT)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--budgets", default="")
    p.add_argument("--synthetic-per-task", type=int)
    p.add_argument("--real-cases", type=int)
    p.add_argument("--max-wall-seconds", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def stage_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(STAGE_PLANS[args.stage])
    if args.target_tokens is not None:
        cfg["target_tokens"] = args.target_tokens
    if args.budgets:
        cfg["budgets"] = [int(x) for x in args.budgets.split(",") if x.strip()]
    if args.synthetic_per_task is not None:
        cfg["synthetic_per_task"] = args.synthetic_per_task
    if args.real_cases is not None:
        cfg["real_cases"] = args.real_cases
    if args.max_wall_seconds is not None:
        cfg["max_wall_seconds"] = args.max_wall_seconds
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    validate_stage_config(cfg)
    return cfg


def validate_stage_config(cfg: dict[str, Any]) -> None:
    target = int(cfg["target_tokens"])
    budgets = [int(x) for x in cfg["budgets"]]
    if target < 512:
        raise ValueError("target_tokens must be >= 512")
    if not budgets:
        raise ValueError("at least one budget is required")
    if any(b >= target for b in budgets):
        raise ValueError("every compressed budget must be smaller than target_tokens")
    if any(b < N_SINK + RECENCY for b in budgets):
        raise ValueError(
            f"budgets must be >= n_sink+recency={N_SINK + RECENCY}"
        )
    if int(cfg["max_wall_seconds"]) <= 0:
        raise ValueError("max_wall_seconds must be positive")


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


def render_user(tokenizer, context: str, question: str, choices: dict[str, str] | None = None) -> str:
    if choices:
        opts = "\n".join(f"{k}. {v}" for k, v in choices.items())
        user = (
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nChoices:\n{opts}\n\n"
            "Answer with exactly one letter: A, B, C, or D."
        )
    else:
        user = f"Document:\n{context}\n\nQuestion:\n{question}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user

def token_len(tokenizer, context: str, question: str) -> int:
    rendered = render_user(tokenizer, context, question, None)
    return len(tokenizer(rendered, add_special_tokens=False).input_ids)


def encode_case(tokenizer, case: EvalCase) -> torch.Tensor:
    rendered = render_user(tokenizer, case.context, case.question, case.choices)
    return tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


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
    fillers = [_filler_sentence(rng) for _ in range(max(900, target_tokens // 5))]

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
        context = _fit_synthetic(tokenizer, question, items, rng, target_tokens)
        return EvalCase(
            case_id=f"syn-single-{sample_id}", source="synthetic", task=task,
            context=context, question=question, gold=[code], distractors=[]
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
        context = _fit_synthetic(tokenizer, question, items, rng, target_tokens)
        return EvalCase(
            case_id=f"syn-multi-{sample_id}", source="synthetic", task=task,
            context=context, question=question, gold=vals, distractors=[]
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
        context = _fit_synthetic(tokenizer, question, items, rng, target_tokens)
        return EvalCase(
            case_id=f"syn-hard-{sample_id}", source="synthetic", task=task,
            context=context, question=question, gold=vals, distractors=bad
        )

    if task == "reason":
        a = rng.randint(1000, 8000)
        b, c, d = a + 7, a + 18, a + 31
        distractor = a + 99
        question = "What is the final value of variable D? Answer with the number only."
        items = [
            (0.15, f"Variable A is initialized to {a}."),
            (0.36, "Variable B equals variable A plus 7."),
            (0.57, "Variable C equals variable B plus 11."),
            (0.78, "Variable D equals variable C plus 13."),
            (0.88, f"Unrelated note: variable X is {distractor}."),
        ]
        context = _fit_synthetic(tokenizer, question, items, rng, target_tokens)
        return EvalCase(
            case_id=f"syn-reason-{sample_id}", source="synthetic", task=task,
            context=context, question=question, gold=[str(d)],
            distractors=[str(distractor)], meta={"chain": [a, b, c, d]}
        )

    raise ValueError(f"unknown synthetic task: {task}")


def load_real_cases(
    tokenizer,
    count: int,
    target_tokens: int,
    seed: int,
    cache_dir: str | None,
    local_files_only: bool,
    dataset_repo: str,
    dataset_revision: str,
) -> list[EvalCase]:
    if count <= 0:
        return []
    if local_files_only:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(
        dataset_repo,
        split="train",
        revision=dataset_revision,
        cache_dir=os.environ.get("HF_DATASETS_CACHE") or cache_dir,
    )
    indices = list(range(len(ds)))
    random.Random(seed + target_tokens).shuffle(indices)
    min_tokens = max(4096, target_tokens // 2)
    cases: list[EvalCase] = []
    scanned = 0
    for i in indices:
        if len(cases) >= count or scanned >= 160:
            break
        scanned += 1
        row = ds[i]
        choices = {k: str(row[f"choice_{k}"]) for k in "ABCD"}
        question = str(row["question"])
        context = str(row["context"])
        rendered = render_user(tokenizer, context, question, choices)
        n = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        if not (min_tokens <= n <= target_tokens):
            continue
        cases.append(
            EvalCase(
                case_id=f"longbench-v2-{row['_id']}",
                source="longbench-v2",
                task=f"{row.get('domain', '')}/{row.get('sub_domain', '')}",
                context=context,
                question=question,
                gold=[str(row["answer"])],
                distractors=[],
                choices=choices,
                meta={
                    "difficulty": row.get("difficulty"),
                    "length": row.get("length"),
                    "rendered_tokens": n,
                },
            )
        )
    if len(cases) < count:
        raise RuntimeError(
            f"Only found {len(cases)} naturally fitting LongBench-v2 cases in "
            f"[{min_tokens}, {target_tokens}] tokens; need {count}."
        )
    return cases

def score_answer(case: EvalCase, answer: str) -> dict[str, Any]:
    text = normalize_text(answer)
    if case.source == "longbench-v2":
        m = re.search(r"\b([ABCD])\b", answer.upper())
        success = bool(m and m.group(1) == case.gold[0].upper())
        return {
            "success": success,
            "recall": 1.0 if success else 0.0,
            "prediction": answer.strip(),
            "distractor_hit": False,
        }

    gold_numbers = [re.findall(r"\d+", x) for x in case.gold]
    flat_gold = [n for group in gold_numbers for n in group]
    if flat_gold:
        pred_numbers = re.findall(r"\d+", answer)
        success = pred_numbers[:len(flat_gold)] == flat_gold
        return {
            "success": success,
            "recall": 1.0 if success else 0.0,
            "prediction": answer.strip(),
            "distractor_hit": any(d in answer for d in case.distractors),
        }

    found = [normalize_text(x) in text for x in case.gold]
    distractor_hit = any(normalize_text(x) in text for x in case.distractors)
    recall = sum(found) / len(found) if found else 0.0
    return {
        "success": bool(found) and all(found) and not distractor_hit,
        "recall": recall,
        "prediction": answer.strip(),
        "distractor_hit": distractor_hit,
    }

def _cuda_stats() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_gb": None, "reserved_gb": None, "max_allocated_gb": None}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def generate_once(
    model,
    tokenizer,
    case: EvalCase,
    max_new_tokens: int,
    budget: int | None,
) -> dict[str, Any]:
    remove_kiaomni(model)
    ids = encode_case(tokenizer, case).to(model.device)
    input_len = int(ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    method = "full_context"
    if budget is not None:
        method = f"kiaomni_{budget}"
        apply_kiaomni(
            model,
            policy=POLICY,
            budget=budget,
            n_sink=N_SINK,
            recency=RECENCY,
            verbose=False,
        )

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    seq = out.sequences if hasattr(out, "sequences") else out
    generated = seq[:, input_len:]
    answer = tokenizer.decode(generated[0], skip_special_tokens=True)
    score = score_answer(case, answer)
    compression = getattr(model, "_kia_last_compression", None)
    stats = _cuda_stats()
    remove_kiaomni(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "method": method,
        "budget": budget,
        "input_tokens": input_len,
        "generated_tokens": int(generated.shape[1]),
        "answer": answer,
        "elapsed_s": elapsed,
        "output_tokens_per_s": (
            float(generated.shape[1]) / elapsed if elapsed > 0 else None
        ),
        "compression": compression,
        "cuda": stats,
        **score,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["source"], row["result"]["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for (source, method), items in sorted(groups.items()):
        successes = [1.0 if x["result"]["success"] else 0.0 for x in items]
        recalls = [float(x["result"]["recall"]) for x in items]
        elapsed = [float(x["result"]["elapsed_s"]) for x in items]
        out.append(
            {
                "source": source,
                "method": method,
                "n": len(items),
                "accuracy": float(np.mean(successes)),
                "mean_recall": float(np.mean(recalls)),
                "mean_elapsed_s": float(np.mean(elapsed)),
            }
        )
    return out


def environment_snapshot(model, tokenizer, cfg: dict[str, Any]) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_gb": props.total_memory / 2**30,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    mc = model.config
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "model_type": getattr(mc, "model_type", None),
        "num_hidden_layers": getattr(mc, "num_hidden_layers", None),
        "num_attention_heads": getattr(mc, "num_attention_heads", None),
        "num_key_value_heads": getattr(mc, "num_key_value_heads", None),
        "num_experts": getattr(mc, "num_experts", None),
        "num_experts_per_tok": getattr(mc, "num_experts_per_tok", None),
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer_class": type(tokenizer).__name__,
        "stage_config": cfg,
    }


def check_probe(model) -> dict[str, Any]:
    probe = ArchitectureProbe.probe(model)
    return {
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


def assert_memory_headroom(min_free_gb: float = 7.0) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient post-load GPU headroom: {free_gb:.2f} GiB free; "
            f"require >= {min_free_gb:.2f} GiB. Do not offload or quantize this gate."
        )
    return {"free_gb": free_gb, "total_gb": total_gb}


def main() -> None:
    args = parse_args()
    cfg = stage_config(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    deadline = started + int(cfg["max_wall_seconds"])

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required; do not silently downgrade precision")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()

    headroom = assert_memory_headroom()
    probe = check_probe(model)

    cases: list[EvalCase] = []
    for task in cfg["tasks"]:
        for i in range(int(cfg["synthetic_per_task"])):
            cases.append(
                make_synthetic_case(
                    tokenizer, task, i, args.seed, int(cfg["target_tokens"])
                )
            )
    cases.extend(
        load_real_cases(
            tokenizer,
            int(cfg["real_cases"]),
            int(cfg["target_tokens"]),
            args.seed,
            args.cache_dir,
            args.local_files_only,
        )
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        if time.perf_counter() >= deadline:
            raise TimeoutError("experiment wall-time budget reached before next case")
        case_tokens = token_len(tokenizer, case.context, case.question)
        methods: list[int | None] = [None, *[int(x) for x in cfg["budgets"]]]
        for budget in methods:
            if time.perf_counter() >= deadline:
                raise TimeoutError("experiment wall-time budget reached before next method")
            result = generate_once(
                model,
                tokenizer,
                case,
                int(cfg["max_new_tokens"]),
                budget,
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "source": case.source,
                    "task": case.task,
                    "rendered_tokens": case_tokens,
                    "gold": case.gold,
                    "meta": case.meta,
                    "result": result,
                }
            )
            print(
                f"[{case.case_id}] {result['method']} "
                f"success={result['success']} recall={result['recall']:.3f} "
                f"elapsed={result['elapsed_s']:.2f}s"
            )

    artifact = {
        "schema": "KIAOMNI_QWEN3_30B_SCALE_GATE_V1",
        "claim_scope": (
            "Prompt-side KiaOmni policy scaling on a substantially larger MoE. "
            "This artifact is not evidence of real past_key_values KV-cache eviction."
        ),
        "stage": args.stage,
        "model": args.model,
        "seed": args.seed,
        "repo_git_head": repo_git_head(),
        "runner_sha256": file_sha256(__file__),
        "environment": environment_snapshot(model, tokenizer, cfg),
        "post_load_headroom": headroom,
        "probe": probe,
        "cases": [asdict(c) | {"context": "<omitted-from-artifact>"} for c in cases],
        "rows": rows,
        "aggregate": aggregate(rows),
        "wall_seconds": time.perf_counter() - started,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
