"""Public wrapper for three distinct inference paths.

Only Qwen2/Qwen2.5 currently has an implemented post-prefill KV adapter.
Other Hugging Face models can attempt the original prompt-selection adapter.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from .capabilities import inspect_architecture
from .license import require_license

class UnsupportedArchitecture(RuntimeError):
    pass

@dataclass
class GenerationResult:
    text: str
    mode: str
    metrics: dict[str, Any]

class KiaOmniModel:
    def __init__(self, model, tokenizer, *, model_id: str = "loaded"):
        require_license()
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.capabilities = inspect_architecture(model.config)

    @classmethod
    def from_pretrained(cls, model_id: str, *, quantization: str = "4bit",
                        device: str = "cuda",
                        revision: str | None = None):
        require_license()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if device not in {"cuda", "auto"} or not torch.cuda.is_available():
            raise RuntimeError("The evaluation NF4 loader requires one CUDA GPU.")
        kwargs = {
            "torch_dtype": torch.float16, "attn_implementation": "sdpa",
            "device_map": {"": 0}, "low_cpu_mem_usage": True,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        elif quantization != "fp16":
            raise ValueError("quantization should be '4bit' or 'fp16'")
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        return cls(model, tok, model_id=model_id)

    def generate(self, prompt: str, *, mode: str = "true_kv",
                 budget: int = 256, max_new_tokens: int = 128,
                 window_size: int = 32, n_sink: int = 16,
                 recency: int = 32, sigma: float = 4.0) -> GenerationResult:
        require_license()
        import torch
        if mode not in {"full_kv", "prompt_selection", "true_kv"}:
            raise ValueError("Invalid mode: full_kv | prompt_selection | true_kv")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be nonempty text")
        if budget < 1 or max_new_tokens < 1:
            raise ValueError("budget and max_new_tokens must be positive")
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        ids = ids.to(next(self.model.parameters()).device)
        if ids.shape[0] != 1:
            raise ValueError("Only batch size one is supported")
        if self.capabilities["model_type"] == "qwen2":
            return self._qwen(ids, mode=mode, budget=budget, max_new_tokens=max_new_tokens,
                              window_size=window_size, n_sink=n_sink,
                              recency=recency, sigma=sigma)
        if mode == "true_kv":
            raise UnsupportedArchitecture(
                f"Physical true-KV unsupported for {self.capabilities['model_type']}; "
                "no silent fallback is allowed."
            )
        return self._huggingface(ids, mode=mode, budget=budget,
                                 max_new_tokens=max_new_tokens,
                                 n_sink=n_sink, recency=recency)

    def _huggingface(self, ids, *, mode, budget, max_new_tokens, n_sink, recency):
        import torch
        from kiaomni.monkey_patch import apply_kiaomni, remove_kiaomni
        started = perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if mode == "prompt_selection":
                apply_kiaomni(self.model, budget=budget, n_sink=n_sink, recency=recency)
            with torch.inference_mode():
                out = self.model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id)
            generated = out[0, ids.shape[1]:]
            metrics = {
                "protocol": "native_generate",
                "physical_kv_compression": False,
                "latency_seconds": perf_counter() - started,
                "generated_tokens": int(generated.numel()),
                "peak_allocated_gb": (
                    float(torch.cuda.max_memory_allocated() / (1024**3))
                    if torch.cuda.is_available() else None),
            }
            if mode == "prompt_selection":
                metrics["prompt_selection"] = getattr(self.model, "_kia_last_compression", None)
            return GenerationResult(
                text=self.tokenizer.decode(generated, skip_special_tokens=True),
                mode=mode, metrics=metrics)
        finally:
            if mode == "prompt_selection":
                remove_kiaomni(self.model)

    def _qwen(self, ids, *, mode, budget, max_new_tokens,
              window_size, n_sink, recency, sigma):
        import torch
        from .kv_policy_qwen25 import global_mask, layerwise_masks
        from .qwen25_kv_runtime import (
            validate_qwen25_model, cache_bytes, prefill, saliency_only,
            compact_cache_layerwise, prime_cache, greedy_decode,
            reset_peak_memory, peak_memory_gb, cuda_sync)
        validate_qwen25_model(self.model)
        length = int(ids.shape[1])
        if mode != "full_kv" and budget < min(length, n_sink + recency):
            raise ValueError("Budget smaller than protected sink and recency positions")
        bridge_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if not bridge_ids:
            raise RuntimeError("Tokenizer produced no bridge tokens")
        reset_peak_memory()
        started = perf_counter()
        saliency_seconds = 0.0
        selection_seconds = 0.0
        compaction_seconds = 0.0
        if mode == "prompt_selection" and budget < length:
            sal = saliency_only(self.model, ids, window_size=window_size)
            saliency_seconds = float(sal["elapsed_seconds"])
            t = perf_counter()
            keep = global_mask(sal["layer_saliencies"], budget,
                               n_sink=n_sink, recency=recency, sigma=sigma)
            selection_seconds = perf_counter() - t
            ids = ids.index_select(1, torch.as_tensor(keep, dtype=torch.long, device=ids.device))
        prefill_tokens = int(ids.shape[1])
        pf = prefill(self.model, ids,
                     collect_saliency=(mode == "true_kv" and budget < length),
                     window_size=window_size)
        before_bytes = cache_bytes(pf["cache"])
        after_bytes = before_bytes
        physical = False
        if mode == "true_kv" and budget < length:
            t = perf_counter()
            keep_per_layer = layerwise_masks(pf["layer_saliencies"], budget,
                                             n_sink=n_sink, recency=recency, sigma=sigma)
            selection_seconds = perf_counter() - t
            cuda_sync()
            t = perf_counter()
            reduced = compact_cache_layerwise(pf["cache"], keep_per_layer)
            cuda_sync()
            compaction_seconds = perf_counter() - t
            after_bytes = reduced["after_bytes"]
            physical = True
            if after_bytes >= before_bytes:
                raise RuntimeError("True KV claimed compression without reducing KV bytes")
        bridge = prime_cache(self.model, cache=pf["cache"], forced_token_ids=bridge_ids,
                             start_position=prefill_tokens)
        output = greedy_decode(
            self.model, self.tokenizer, initial_logits=bridge["logits"],
            cache=bridge["cache"], start_position=bridge["next_position"],
            max_new_tokens=max_new_tokens)
        cuda_sync()
        total = perf_counter() - started
        metrics = {
            "protocol": "qwen2_post_prefill_bridge",
            "original_prompt_tokens": length,
            "prefill_tokens": prefill_tokens,
            "physical_kv_compression": physical,
            "kv_before_bytes": before_bytes if mode != "prompt_selection" else None,
            "kv_after_bytes": after_bytes,
            "physical_kv_reduction_ratio": (before_bytes / after_bytes if physical else None),
            "prefill_seconds": pf["elapsed_seconds"],
            "saliency_seconds": saliency_seconds,
            "selection_seconds": selection_seconds,
            "compaction_seconds": compaction_seconds,
            "bridge_seconds": bridge["elapsed_seconds"],
            "ttft_seconds": saliency_seconds + pf["elapsed_seconds"] + selection_seconds + compaction_seconds + bridge["elapsed_seconds"],
            "total_seconds": total,
            "decode_seconds": output["decode_seconds"],
            "tokens_per_second": output["tokens_per_second"],
            "generated_tokens": output["generated_tokens"],
            **peak_memory_gb(),
        }
        return GenerationResult(text=output["text"], mode=mode, metrics=metrics)
