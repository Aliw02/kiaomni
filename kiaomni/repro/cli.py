"""Command line entry point for Kaggle and Windows evaluation."""
from __future__ import annotations
from importlib.metadata import version
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from .license import require_license
from .capabilities import inspect_architecture

def main():
    require_license()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: kiaomni-repro check-model --model-id ID | demo [benchmark options]")
        return 0
    command, args = sys.argv[1], sys.argv[2:]
    if command == "check-model":
        from transformers import AutoConfig
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-id", required=True)
        opts = parser.parse_args(args)
        cfg = AutoConfig.from_pretrained(opts.model_id, trust_remote_code=False)
        print(json.dumps({"model_id": opts.model_id, **inspect_architecture(cfg)}, indent=2))
        return 0
    if command == "streaming-gate":
        from .streaming_gate import main as gate_main
        sys.argv = [sys.argv[0], *args]
        return gate_main()
    if command == "demo":
        import torch
        from transformers import AutoConfig
        from . import benchmark
        if version("transformers") != "4.57.6":
            raise RuntimeError("Use transformers==4.57.6 for benchmark. Found: " + version("transformers"))
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
        parser.add_argument("--output-dir", default="kiaomni_results")
        opts, _ = parser.parse_known_args(args)
        cfg = AutoConfig.from_pretrained(opts.model_id, trust_remote_code=False)
        capabilities = inspect_architecture(cfg)
        if capabilities["true_kv"] == "unsupported":
            raise RuntimeError("Bundled paired demo requires Qwen2-family full-attention model. "
                               "Use the SDK for Full KV / Prompt Selection on other compatible models.")
        path = Path(opts.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "model_id": opts.model_id,
            "adapter": capabilities,
            "benchmark_branch": "exp/kiaomni-repro-sdk-v1",
        }
        (path / "run_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        sys.argv = [sys.argv[0], *args]
        benchmark.main()
        return 0
    raise SystemExit(f"Unknown command {command!r}; expected 'check-model' or 'demo'")

if __name__ == "__main__":
    raise SystemExit(main())
