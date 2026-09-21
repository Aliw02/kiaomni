"""kiaomni — generic monkey-patch KV-cache eviction for any HF causal LM."""

from .adapters import (
    ArchitectureProbe,
    Confidence,
    KiaomniConfigError,
    PosEncoding,
    ProbeResult,
    QKVPattern,
)
from .loading import ATTN_FALLBACK_CHAIN, load_model
from .monkey_patch import apply_kiaomni, remove_kiaomni
from .moe_stability import (
    AdaptiveMoERouteController,
    apply_moe_route_stability,
    remove_moe_route_stability,
)
from .policies import POLICY_REGISTRY, get_policy, register_policy
from ._version import __version__

__all__ = [
    "apply_kiaomni",
    "remove_kiaomni",
    "apply_moe_route_stability",
    "remove_moe_route_stability",
    "AdaptiveMoERouteController",
    "load_model",
    "ATTN_FALLBACK_CHAIN",
    "POLICY_REGISTRY",
    "register_policy",
    "get_policy",
    "ArchitectureProbe",
    "ProbeResult",
    "QKVPattern",
    "PosEncoding",
    "Confidence",
    "KiaomniConfigError",
    "__version__",
]
