"""kiaomni — generic monkey-patch KV-cache eviction for any HF causal LM."""

from .adapters import (
    ArchitectureProbe,
    Confidence,
    KiaomniConfigError,
    PosEncoding,
    ProbeResult,
    QKVPattern,
)
from .monkey_patch import apply_kiaomni, remove_kiaomni
from .policies import POLICY_REGISTRY, get_policy, register_policy
from ._version import __version__

__all__ = [
    "apply_kiaomni",
    "remove_kiaomni",
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
