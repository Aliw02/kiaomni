from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiaomni import ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter
from kiaomni.policies import get_policy
from kiaomni.utils import N_SINK_DEFAULT, RECENCY_DEFAULT, select_keep

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23"
DATASET_ID = "THUDM/LongBench-v2"
DATASET_REVISION = "b0db4901b856522026b7353ab541b8535ff2a4b8"

POLICIES = ("kiaomni_s8", "kiaomni_gaussian")
FIXED_BUDGETS = (512, 256, 128, 98)
N_SINK = N_SINK_DEFAULT
RECENCY = RECENCY_DEFAULT
MAX_NEW_TOKENS = 256
SEED = 42

STAGES = {
    "preflight": {"max_wall_seconds": 15 * 60, "real_cases": 1, "reason_cases": 0},
    "final": {"max_wall_seconds": 100 * 60, "real_cases": 27, "reason_cases": 2},
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
    p = argparse.ArgumentParser(description="KiaOmni Qwen3-30B adjudication + actual routing audit")
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--adjudication-index", required=True)
    p.add_argument("--asset-manifest", required=True)
    p.add_argument("--repo-revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    return p.parse_args()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_mc_prompt(row: dict[str, Any]) -> str:
    return (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
        f"{str(row['context']).strip()}\n"
        "</text>\n\n"
        f"What is the correct answer to this question: {str(row['question']).strip()}\n"
        "Choices:\n"
        f"(A) {str(row['choice_A']).strip()}\n"
        f"(B) {str(row['choice_B']).strip()}\n"
        f"(C) {str(row['choice_C']).strip()}\n"
        f"(D) {str(row['choice_D']).strip()}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def render_chat(tokenizer, user_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def encode_text(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def build_real_case(row: dict[str, Any], rendered_tokens: int) -> EvalCase:
    gold = str(row["answer"]).strip().upper()
    if gold not in {"A", "B", "C", "D"}:
        raise RuntimeError(f"Invalid LongBench-v2 gold label: {gold!r}")
    prompt = official_mc_prompt(row)
    return EvalCase(
        case_id=f"longbenchv2-{row['_id']}",
        source="longbench_v2",
        task=str(row.get("sub_domain", row.get("domain", "unknown"))),
        context="",
        question=prompt,
        gold=[gold],
        distractors=[],
        meta={
            "domain": str(row.get("domain", "unknown")),
            "sub_domain": str(row.get("sub_domain", "unknown")),
            "difficulty": str(row.get("difficulty", "unknown")),
            "length": str(row.get("length", "unknown")),
            "rendered_tokens": int(rendered_tokens),
        },
    )


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


def make_reason_case(tokenizer, sample_id: int, seed: int, target_tokens: int = 8192) -> EvalCase:
    rng = random.Random(seed + 1009 * sample_id + sum(map(ord, "reason")))
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
    fillers = [_filler_sentence(rng) for _ in range(max(900, target_tokens // 4))]

    def build(n: int) -> str:
        out = list(fillers[:n])
        for depth, sentence in sorted(items, key=lambda x: x[0], reverse=True):
            out.insert(int(depth * len(out)), sentence)
        return " ".join(out)

    def count(ctx: str) -> int:
        rendered = render_chat(tokenizer, f"Document:\n{ctx}\n\nQuestion:\n{question}")
        return int(encode_text(tokenizer, rendered).shape[1])

    lo, hi = 0, len(fillers)
    best = build(0)
    best_len = count(best)
    while lo <= hi:
        mid = (lo + hi) // 2
        ctx = build(mid)
        n = count(ctx)
        if n <= target_tokens:
            best, best_len = ctx, n
            lo = mid + 1
        else:
            hi = mid - 1
    if best_len < target_tokens - 160:
        raise RuntimeError(f"reason synthetic underfilled: {best_len}")

    user_text = f"Document:\n{best}\n\nQuestion:\n{question}"
    return EvalCase(
        case_id=f"syn-reason-{sample_id}",
        source="synthetic",
        task="reason",
        context="",
        question=user_text,
        gold=[str(d)],
        distractors=[str(distractor)],
        meta={"rendered_tokens": best_len},
    )


def load_cases(tokenizer, dataset_dir: str, adjudication_index: str, real_count: int, reason_count: int, seed: int) -> list[EvalCase]:
    from datasets import load_from_disk

    idx = json.loads(Path(adjudication_index).read_text(encoding="utf-8"))
    rows = idx["rows"]
    if len(rows) != 27:
        raise RuntimeError(f"Adjudication index must freeze exactly 27 real cases; found {len(rows)}")
    if real_count > len(rows):
        raise RuntimeError(f"Requested {real_count} real cases but index has {len(rows)}")

    ids = {str(x["_id"]) for x in rows}
    ds = load_from_disk(dataset_dir)
    by_id = {str(raw["_id"]): dict(raw) for raw in ds if str(raw["_id"]) in ids}
    if set(by_id) != ids:
        missing = sorted(ids - set(by_id))
        raise RuntimeError(f"Adjudication index references missing rows: {missing}")

    cases: list[EvalCase] = []
    for item in rows[:real_count]:
        row = by_id[str(item["_id"])]
        prompt = render_chat(tokenizer, official_mc_prompt(row))
        actual = int(encode_text(tokenizer, prompt).shape[1])
        if actual != int(item["official_rendered_tokens"]):
            raise RuntimeError(
                f"Token-count drift for {row['_id']}: index={item['official_rendered_tokens']} runtime={actual}"
            )
        cases.append(build_real_case(row, actual))

    for i in range(reason_count):
        cases.append(make_reason_case(tokenizer, i, seed))
    return cases


def rendered_case_text(tokenizer, case: EvalCase) -> str:
    return render_chat(tokenizer, case.question)


def parse_mc_answer(text: str) -> tuple[str | None, str]:
    cleaned = text.strip()
    patterns = [
        r"(?i)the\s+correct\s+answer\s+is\s*\(?\s*([A-D])\s*\)?",
        r"(?i)final\s+answer\s*(?:is|:)\s*\(?\s*([A-D])\s*\)?",
        r"(?i)answer\s*(?:is|:)\s*\(?\s*([A-D])\s*\)?",
    ]
    for p in patterns:
        m = re.search(p, cleaned)
        if m:
            return m.group(1).upper(), "explicit"
    m = re.fullmatch(r"\s*\(?\s*([A-D])\s*\)?[\s\.!]*", cleaned, flags=re.I)
    if m:
        return m.group(1).upper(), "bare"
    return None, "unparsed"


def parse_numeric_answer(text: str) -> tuple[str | None, str]:
    patterns = [
        r"(?i)final\s+(?:value|answer)\s*(?:of\s+variable\s+D\s*)?(?:is|:)\s*(-?\d+)",
        r"(?i)answer\s*(?:is|:)\s*(-?\d+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1), "explicit"
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1], "last_number"
    return None, "unparsed"


def score_answer(case: EvalCase, answer: str) -> dict[str, Any]:
    if case.source == "longbench_v2":
        parsed, parse_status = parse_mc_answer(answer)
    else:
        parsed, parse_status = parse_numeric_answer(answer)
    correct = parsed == case.gold[0]
    return {
        "parsed_answer": parsed,
        "parse_status": parse_status,
        "correct": bool(correct),
        "prediction": answer.strip(),
    }


def reset_peak() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def peak_gb() -> float:
    return float(torch.cuda.max_memory_allocated() / 2**30)


class ActualRoutingCapture:
    """Capture routing from the exact Transformers 4.57.6 execution path.

    Qwen3MoeSparseMoeBlock computes top-k routing internally from the output
    of its real gate Linear and then calls each selected expert module. We
    capture the actual gate logits, reconstruct the exact top-k with the same
    softmax/topk/normalization operations, and independently count the token
    rows that are physically sent through every expert module. The derived
    dispatch counts must match the observed expert-call counts exactly.
    """

    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.routes: dict[int, dict[str, torch.Tensor]] = {}
        self.observed_counts: dict[int, torch.Tensor] = {}
        self.handles = []
        self.layer_ids: list[int] = []
        self.blocks: dict[int, Any] = {}
        self.dispatch_verified = True
        self.dispatch_count_mismatches: list[dict[str, Any]] = []

        for layer_idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            experts = getattr(mlp, "experts", None)
            if gate is None or experts is None or not isinstance(experts, torch.nn.ModuleList):
                continue
            if not hasattr(mlp, "top_k") or not hasattr(mlp, "num_experts"):
                continue

            self.layer_ids.append(layer_idx)
            self.blocks[layer_idx] = mlp
            self.handles.append(gate.register_forward_hook(self._gate_hook(layer_idx)))
            for expert_idx, expert in enumerate(experts):
                self.handles.append(
                    expert.register_forward_pre_hook(
                        self._expert_hook(layer_idx, expert_idx)
                    )
                )

    def _gate_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            logits = output.detach()
            block = self.blocks[layer_idx]
            probs = torch.softmax(logits, dim=1, dtype=torch.float)
            weights, indices = torch.topk(probs, int(block.top_k), dim=-1)
            if bool(block.norm_topk_prob):
                weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.to(logits.dtype)
            self.routes[layer_idx] = {
                "indices": indices.detach().to(device="cpu", dtype=torch.uint8),
                "weights": weights.detach().to(device="cpu", dtype=torch.float16),
            }
        return hook

    def _expert_hook(self, layer_idx: int, expert_idx: int):
        def hook(module, inputs):
            if not self.enabled:
                return
            if not inputs:
                raise RuntimeError(
                    f"Expert layer={layer_idx} expert={expert_idx} called without input"
                )
            current_state = inputs[0]
            rows = int(current_state.shape[0])
            if layer_idx not in self.observed_counts:
                self.observed_counts[layer_idx] = torch.zeros(
                    int(self.blocks[layer_idx].num_experts), dtype=torch.int64
                )
            self.observed_counts[layer_idx][expert_idx] += rows
        return hook

    def start(self) -> None:
        self.routes = {}
        self.observed_counts = {}
        self.dispatch_verified = True
        self.dispatch_count_mismatches = []
        self.enabled = True

    def stop(self) -> dict[int, dict[str, torch.Tensor]]:
        self.enabled = False
        if set(self.routes) != set(self.layer_ids):
            missing = sorted(set(self.layer_ids) - set(self.routes))
            raise RuntimeError(f"Missing router-logit capture for sparse layers: {missing}")

        for layer_idx in self.layer_ids:
            indices = self.routes[layer_idx]["indices"].long()
            expected = torch.bincount(
                indices.reshape(-1),
                minlength=int(self.blocks[layer_idx].num_experts),
            ).to(torch.int64)
            observed = self.observed_counts.get(
                layer_idx,
                torch.zeros_like(expected),
            )
            if not torch.equal(expected, observed):
                self.dispatch_verified = False
                diff = (expected - observed).abs()
                self.dispatch_count_mismatches.append({
                    "layer": layer_idx,
                    "mismatched_experts": int((diff != 0).sum().item()),
                    "max_abs_count_error": int(diff.max().item()),
                    "expected_assignments": int(expected.sum().item()),
                    "observed_assignments": int(observed.sum().item()),
                })

        if not self.dispatch_verified:
            raise RuntimeError(
                "Actual expert-dispatch count verification failed: "
                + json.dumps(self.dispatch_count_mismatches)
            )
        return self.routes

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _safe_prob(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _entropy(weights: torch.Tensor) -> torch.Tensor:
    p = _safe_prob(weights)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)


def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float()
    q = q.float()
    p = p / p.sum().clamp_min(1e-12)
    q = q / q.sum().clamp_min(1e-12)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum()
    kl_qm = (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def compare_routes(
    full_routes: dict[int, dict[str, torch.Tensor]],
    compressed_routes: dict[int, dict[str, torch.Tensor]],
    keep: np.ndarray,
    full_prompt_len: int,
    compressed_prompt_len: int,
    num_experts: int,
) -> dict[str, Any]:
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    per_layer = []
    for layer_idx in sorted(full_routes):
        f_idx = full_routes[layer_idx]["indices"][:full_prompt_len][keep_t].long()
        f_w = full_routes[layer_idx]["weights"][:full_prompt_len][keep_t].float()
        c_idx = compressed_routes[layer_idx]["indices"][:compressed_prompt_len].long()
        c_w = compressed_routes[layer_idx]["weights"][:compressed_prompt_len].float()
        if f_idx.shape != c_idx.shape:
            raise RuntimeError(
                f"Routing shape mismatch layer={layer_idx} full={tuple(f_idx.shape)} comp={tuple(c_idx.shape)}"
            )

        top1 = float((f_idx[:, 0] == c_idx[:, 0]).float().mean().item())
        intersection = (f_idx.unsqueeze(2) == c_idx.unsqueeze(1)).sum(dim=(1, 2)).float()
        union = 2.0 * f_idx.shape[1] - intersection
        jaccard = float((intersection / union.clamp_min(1.0)).mean().item())

        fd = torch.zeros((f_idx.shape[0], num_experts), dtype=torch.float32)
        cd = torch.zeros((c_idx.shape[0], num_experts), dtype=torch.float32)
        fd.scatter_add_(1, f_idx, f_w)
        cd.scatter_add_(1, c_idx, c_w)
        cosine = float(F.cosine_similarity(fd, cd, dim=1, eps=1e-12).mean().item())

        f_ent = _entropy(f_w)
        c_ent = _entropy(c_w)
        entropy_delta = float((c_ent - f_ent).mean().item())

        f_load = torch.bincount(f_idx.reshape(-1), minlength=num_experts).float()
        c_load = torch.bincount(c_idx.reshape(-1), minlength=num_experts).float()
        load_jsd = _jsd(f_load, c_load)

        per_layer.append({
            "layer": layer_idx,
            "top1_expert_agreement": top1,
            "topk_set_jaccard": jaccard,
            "dispatch_weight_cosine": cosine,
            "dispatch_entropy_delta": entropy_delta,
            "expert_load_jsd": load_jsd,
        })

    def mean(key: str) -> float:
        return float(np.mean([x[key] for x in per_layer]))

    return {
        "matched_tokens": int(len(keep)),
        "layers": len(per_layer),
        "top1_expert_agreement": mean("top1_expert_agreement"),
        "top8_set_jaccard": mean("topk_set_jaccard"),
        "dispatch_weight_cosine": mean("dispatch_weight_cosine"),
        "dispatch_entropy_delta": mean("dispatch_entropy_delta"),
        "expert_load_jsd": mean("expert_load_jsd"),
        "per_layer": per_layer,
    }


def teacher_gold_and_routing(
    model,
    tokenizer,
    ids: torch.Tensor,
    gold_target: str,
    capture: ActualRoutingCapture,
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    target_ids = tokenizer(gold_target, add_special_tokens=False).input_ids
    if not target_ids:
        raise RuntimeError("Gold target tokenized to zero tokens")
    target = torch.tensor(target_ids, dtype=torch.long, device=ids.device).unsqueeze(0)
    teacher_ids = torch.cat([ids, target[:, :-1]], dim=1)

    reset_peak()
    capture.start()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(
            input_ids=teacher_ids,
            attention_mask=torch.ones_like(teacher_ids),
            use_cache=False,
            logits_to_keep=len(target_ids),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    routes = capture.stop()

    logits = out.logits.float()
    if logits.shape[1] != len(target_ids):
        raise RuntimeError(f"Expected {len(target_ids)} teacher logits, got {logits.shape[1]}")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="mean",
    )
    nll = float(loss.item())
    ppl = float(math.exp(min(nll, 50.0)))
    return {
        "gold_target": gold_target,
        "gold_target_tokens": len(target_ids),
        "gold_answer_nll": nll,
        "gold_answer_ppl": ppl,
        "routing_teacher_elapsed_seconds": elapsed,
        "routing_teacher_peak_allocated_vram_gb": peak_gb(),
        "dispatch_verified": capture.dispatch_verified,
        "dispatch_count_mismatches": list(capture.dispatch_count_mismatches),
    }, routes


def generate_answer(model, tokenizer, ids: torch.Tensor, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
    reset_peak()
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
    answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    eos_ids = tokenizer.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    last_id = int(new_tokens[0, -1].item()) if new_tokens.shape[1] else None
    eos_reached = last_id in eos_set if last_id is not None else False
    generated = int(new_tokens.shape[1])
    return answer, {
        "generation_elapsed_seconds": elapsed,
        "generated_tokens": generated,
        "output_tokens_per_second": (generated / elapsed if elapsed > 0 else None),
        "generation_peak_allocated_vram_gb": peak_gb(),
        "eos_reached": eos_reached,
        "hit_token_limit": bool(generated >= max_new_tokens and not eos_reached),
    }


def extract_saliency(model, adapter: SaliencyAdapter, ids: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    reset_peak()
    t0 = time.perf_counter()
    sal = adapter.extract(ids, model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if sal.shape != tuple(ids.shape):
        raise RuntimeError(f"Saliency shape mismatch: {sal.shape} vs {tuple(ids.shape)}")
    if not np.isfinite(sal).all():
        raise RuntimeError("Non-finite saliency")
    return sal[0], {
        "saliency_elapsed_seconds": elapsed,
        "saliency_peak_allocated_vram_gb": peak_gb(),
    }


def select_budget(scores: np.ndarray, budget: int, length: int) -> np.ndarray:
    if budget not in FIXED_BUDGETS:
        raise RuntimeError(f"Non-frozen budget requested: {budget}")
    keep = select_keep(scores, budget, length, n_sink=N_SINK, recency=RECENCY)
    if len(keep) != budget:
        raise RuntimeError(f"Selector returned {len(keep)} tokens for frozen budget {budget}")
    return np.sort(keep)


def assert_environment(model, min_free_gb: float) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    if getattr(model, "is_quantized", False):
        raise RuntimeError("Quantized model is forbidden")
    non_cuda = sorted({str(p.device) for p in model.parameters() if p.device.type != "cuda"})
    if non_cuda:
        raise RuntimeError(f"Non-CUDA model parameters found: {non_cuda}")
    cfg = model.config
    expected = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "head_dim": 128,
    }
    observed = {k: getattr(cfg, k, None) for k in expected}
    if observed != expected:
        raise RuntimeError(f"Model config mismatch: expected={expected} observed={observed}")
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 2**30
    if free_gb < min_free_gb:
        raise RuntimeError(f"Post-load free VRAM {free_gb:.2f} GiB < {min_free_gb:.2f} GiB")
    return {
        "model_config": observed,
        "free_gb": free_gb,
        "total_gb": total_b / 2**30,
        "dtype": str(next(model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0),
    }


def hook_neutrality_check(model, capture: ActualRoutingCapture, ids: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        capture.enabled = False
        a = model(input_ids=ids, use_cache=False, logits_to_keep=4).logits.detach()
        capture.start()
        b = model(input_ids=ids, use_cache=False, logits_to_keep=4).logits.detach()
        routes = capture.stop()
    delta = (a.float() - b.float()).abs()
    allclose = bool(torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-5))
    return {
        "passed": bool(allclose and capture.dispatch_verified),
        "logits_exact_equal": bool(torch.equal(a, b)),
        "logits_allclose_rtol_1e-4_atol_1e-5": allclose,
        "max_abs_logit_error": float(delta.max().item()),
        "dispatch_verified": capture.dispatch_verified,
        "dispatch_count_mismatches": list(capture.dispatch_count_mismatches),
        "captured_sparse_layers": len(routes),
    }


def run_condition(
    model,
    tokenizer,
    case: EvalCase,
    ids: torch.Tensor,
    method: str,
    keep: np.ndarray | None,
    capture: ActualRoutingCapture,
    full_routes: dict[int, dict[str, torch.Tensor]] | None,
    full_prompt_len: int,
    saliency_meta: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    used = ids
    if keep is not None:
        kt = torch.as_tensor(keep, dtype=torch.long, device=ids.device)
        used = ids[:, kt]

    if case.source == "longbench_v2":
        gold_target = f"The correct answer is ({case.gold[0]})"
    else:
        gold_target = case.gold[0]

    teacher, routes = teacher_gold_and_routing(model, tokenizer, used, gold_target, capture)
    answer, generation = generate_answer(model, tokenizer, used, MAX_NEW_TOKENS)
    score = score_answer(case, answer)

    routing = None
    if keep is not None:
        assert full_routes is not None
        routing = compare_routes(
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

    result = {
        "method": method,
        "input_tokens": int(ids.shape[1]),
        "kept_tokens": int(used.shape[1]),
        "compression_ratio": float(ids.shape[1] / used.shape[1]),
        "answer": answer,
        **score,
        **generation,
        **teacher,
        "pipeline_peak_allocated_vram_gb": pipeline_peak,
        "saliency": saliency_meta,
        "routing": routing,
    }
    return result, routes


def print_result(case: EvalCase, result: dict[str, Any]) -> None:
    print("\n" + "=" * 88, flush=True)
    print(f"CASE={case.case_id} SOURCE={case.source} TASK={case.task}", flush=True)
    print(f"METHOD={result['method']} GOLD={case.gold[0]}", flush=True)
    print(
        f"TOKENS={result['kept_tokens']}/{result['input_tokens']} "
        f"COMPRESSION={result['compression_ratio']:.3f}x",
        flush=True,
    )
    print("--- RAW ANSWER ---", flush=True)
    print(result["answer"], flush=True)
    print("--- SCORES ---", flush=True)
    print(
        f"PARSED={result['parsed_answer']} CORRECT={result['correct']} "
        f"PARSE={result['parse_status']} GENERATED={result['generated_tokens']} "
        f"HIT_LIMIT={result['hit_token_limit']}",
        flush=True,
    )
    print(
        f"PPL={result['gold_answer_ppl']:.6f} "
        f"NLL={result['gold_answer_nll']:.6f} "
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
            f"TOP8_JACCARD={r['top8_set_jaccard']:.6f} "
            f"WEIGHT_COS={r['dispatch_weight_cosine']:.6f} "
            f"ENTROPY_DELTA={r['dispatch_entropy_delta']:.6f} "
            f"LOAD_JSD={r['expert_load_jsd']:.6f}",
            flush=True,
        )
    print("=" * 88, flush=True)


def pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["source"] != "longbench_v2":
            continue
        by_case.setdefault(row["case_id"], {})[row["result"]["method"]] = row["result"]

    methods = sorted({
        row["result"]["method"]
        for row in rows
        if row["source"] == "longbench_v2" and row["result"]["method"] != "full_context"
    })
    out = []
    for method in methods:
        cc = cw = wc = ww = agree = parsed = hit_limit = 0
        n = 0
        for case_id, items in by_case.items():
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
            agree += int(f["parsed_answer"] is not None and f["parsed_answer"] == k["parsed_answer"])
            parsed += int(k["parsed_answer"] is not None)
            hit_limit += int(k["hit_token_limit"])
        out.append({
            "method": method,
            "n": n,
            "fc_correct_to_kia_correct": cc,
            "fc_correct_to_kia_wrong": cw,
            "fc_wrong_to_kia_correct": wc,
            "fc_wrong_to_kia_wrong": ww,
            "parsed_answer_agreement": (agree / n if n else None),
            "parse_rate": (parsed / n if n else None),
            "token_limit_hit_rate": (hit_limit / n if n else None),
        })
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["source"], row["result"]["method"]), []).append(row)
    out = []
    for (source, method), items in sorted(groups.items()):
        vals = [x["result"] for x in items]
        def mean(key: str) -> float:
            return float(np.mean([float(x[key]) for x in vals]))
        out.append({
            "source": source,
            "method": method,
            "n": len(vals),
            "accuracy": mean("correct"),
            "parse_rate": float(np.mean([x["parsed_answer"] is not None for x in vals])),
            "mean_gold_answer_ppl": mean("gold_answer_ppl"),
            "mean_output_tokens_per_second": mean("output_tokens_per_second"),
            "mean_generation_peak_vram_gb": mean("generation_peak_allocated_vram_gb"),
            "max_pipeline_peak_vram_gb": max(float(x["pipeline_peak_allocated_vram_gb"]) for x in vals),
            "token_limit_hit_rate": mean("hit_token_limit"),
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    deadline = started + cfg["max_wall_seconds"]
    output = Path(args.output)

    manifest = json.loads(Path(args.asset_manifest).read_text(encoding="utf-8"))
    if manifest.get("model_revision_resolved", "").startswith(MODEL_REVISION) is False:
        raise RuntimeError("Frozen model revision mismatch")
    if manifest.get("dataset_revision_resolved") != DATASET_REVISION:
        raise RuntimeError("Frozen dataset revision mismatch")

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

    env = assert_environment(model, args.min_free_gb)
    probe = ArchitectureProbe.probe(model)
    if probe.confidence != "high":
        raise RuntimeError(f"Architecture probe confidence must be high, got {probe.confidence}")

    capture = ActualRoutingCapture(model)
    if len(capture.layer_ids) == 0:
        raise RuntimeError("No sparse MoE routing layers discovered")

    device = next(model.parameters()).device
    neutrality_text = render_chat(
        tokenizer,
        "Document:\nVariable A is initialized to 10. Variable B equals A plus 5.\n\n"
        "Question:\nWhat is variable B? Answer with the number only.",
    )
    neutrality_ids = encode_text(tokenizer, neutrality_text)[:, :512].to(device)
    neutrality = hook_neutrality_check(model, capture, neutrality_ids)
    print(f"ROUTING_PREFLIGHT={json.dumps(neutrality)}", flush=True)
    if not neutrality["passed"]:
        raise RuntimeError(f"Actual routing hook neutrality/dispatch verification failed: {neutrality}")

    cases = load_cases(
        tokenizer,
        args.dataset_dir,
        args.adjudication_index,
        real_count=int(cfg["real_cases"]),
        reason_count=int(cfg["reason_cases"]),
        seed=args.seed,
    )

    gpu_adapter = SaliencyAdapter(probe, offload_to_cpu=False)
    policy_fns = {name: get_policy(name) for name in POLICIES}
    rows: list[dict[str, Any]] = []

    artifact: dict[str, Any] = {
        "schema": "KIAOMNI_QWEN3_30B_ADJUDICATION_ROUTING_V1",
        "stage": args.stage,
        "repo_revision": args.repo_revision,
        "runner_sha256": file_sha256(__file__),
        "model": {"repo": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {"repo": DATASET_ID, "revision": DATASET_REVISION},
        "policies": list(POLICIES),
        "fixed_budgets": list(FIXED_BUDGETS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "environment": env,
        "probe": {
            "confidence": probe.confidence,
            "num_layers": probe.num_layers,
            "num_attention_heads": probe.num_attention_heads,
            "num_key_value_heads": probe.num_key_value_heads,
            "head_dim": probe.head_dim,
        },
        "routing_preflight": neutrality,
        "cases": [asdict(c) for c in cases],
        "rows": rows,
        "aggregate": [],
        "pairwise": [],
        "execution_gate": {"status": "RUNNING"},
        "wall_seconds": 0.0,
    }
    write_json(output, artifact)

    for case_i, case in enumerate(cases):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Stage wall-time ceiling reached before next case")

        rendered = rendered_case_text(tokenizer, case)
        ids = encode_text(tokenizer, rendered).to(device)
        input_len = int(ids.shape[1])
        if case.source == "longbench_v2" and not (8192 <= input_len <= 16384):
            raise RuntimeError(f"Frozen real case outside 8K-16K under official prompt: {case.case_id}={input_len}")

        full_result, full_routes = run_condition(
            model, tokenizer, case, ids, "full_context", None, capture,
            full_routes=None, full_prompt_len=input_len, saliency_meta=None,
        )
        rows.append({
            "case_id": case.case_id,
            "source": case.source,
            "task": case.task,
            "result": full_result,
        })
        print_result(case, full_result)

        saliency, saliency_meta = extract_saliency(model, gpu_adapter, ids)

        for policy_name in POLICIES:
            scores = policy_fns[policy_name](saliency)
            if scores.shape != saliency.shape or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid policy scores for {policy_name}")
            for budget in FIXED_BUDGETS:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Stage wall-time ceiling reached before next condition")
                keep = select_budget(scores, budget, input_len)
                method = f"{policy_name}_B{budget}"
                result, _ = run_condition(
                    model, tokenizer, case, ids, method, keep, capture,
                    full_routes=full_routes,
                    full_prompt_len=input_len,
                    saliency_meta=saliency_meta,
                )
                rows.append({
                    "case_id": case.case_id,
                    "source": case.source,
                    "task": case.task,
                    "policy": policy_name,
                    "budget": budget,
                    "keep_positions_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
                    "result": result,
                })
                print_result(case, result)

        artifact["rows"] = rows
        artifact["aggregate"] = aggregate(rows)
        artifact["pairwise"] = pairwise_summary(rows)
        artifact["wall_seconds"] = time.perf_counter() - started
        artifact["execution_gate"] = {
            "status": "RUNNING",
            "completed_cases": case_i + 1,
            "total_cases": len(cases),
        }
        write_json(output, artifact)
        gc.collect()
        torch.cuda.empty_cache()

    expected_rows = len(cases) * (1 + len(POLICIES) * len(FIXED_BUDGETS))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} condition rows, got {len(rows)}")

    artifact["aggregate"] = aggregate(rows)
    artifact["pairwise"] = pairwise_summary(rows)
    artifact["wall_seconds"] = time.perf_counter() - started
    artifact["execution_gate"] = {
        "status": "PASS",
        "completed_cases": len(cases),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "actual_dispatch_verified": True,
        "note": "PASS means execution/measurement integrity only; quality is adjudicated from the recorded metrics.",
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
                "schema": "KIAOMNI_QWEN3_30B_ADJUDICATION_ROUTING_ERROR_V1",
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
