"""Baseline compatibility helpers used by KiaOmni experiments."""

from .mobilemoe_snapkv import (
    ADAPTER_NAME,
    ALGORITHM_MODIFIED,
    MobileMoESnapKVCompatibilityError,
    adapter_provenance,
    describe_position_embeddings,
    is_mobilemoe_module,
    make_mobilemoe_snapkv_press,
    mobilemoe_prerope_query_states,
    resolve_mobilemoe_rope,
)

__all__ = [
    "ADAPTER_NAME",
    "ALGORITHM_MODIFIED",
    "MobileMoESnapKVCompatibilityError",
    "adapter_provenance",
    "describe_position_embeddings",
    "is_mobilemoe_module",
    "make_mobilemoe_snapkv_press",
    "mobilemoe_prerope_query_states",
    "resolve_mobilemoe_rope",
]
