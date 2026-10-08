"""Conservative capabilities; do not mistake model-family similarity for validation."""
from __future__ import annotations

def inspect_architecture(config) -> dict:
    family = str(getattr(config, "model_type", "unknown")).lower()
    layer_types = getattr(config, "layer_types", None)
    layers = list(layer_types) if layer_types is not None else []
    unsupported_layers = any(str(v) != "full_attention" for v in layers)
    sliding = bool(getattr(config, "use_sliding_window", False))
    verified_layout = family == "qwen2" and not unsupported_layers and not sliding
    return {
        "model_type": family,
        "full_kv": "candidate",
        "prompt_selection": "candidate" if family in {"qwen2","llama","mistral","gemma","phi","qwen3"} else "experimental",
        "true_kv": "candidate_needs_runtime_validation" if verified_layout else "unsupported",
        "reason": ("Qwen2 dynamic-cache layout; verify at runtime" if verified_layout
                   else "No validated true-KV adapter for this architecture"),
    }
