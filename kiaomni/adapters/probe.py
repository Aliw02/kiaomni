"""
kiaomni/adapters/probe.py
===========================
ArchitectureProbe — generic discovery of any HuggingFace causal LM's
attention internals, without hardcoding module paths or config names.

The probe runs once per model and caches its result on
``model._kia_arch_info`` so downstream adapters can reuse it.

Detection covers:
    * Layer container path        (model.model.layers, model.transformer.h, ...)
    * Attention module path       (self_attn, attn, attention, self_attention)
    * QKV projection pattern      (separate / fused-concat / fused-interleaved)
    * Head dimensions             (num_heads, num_kv_heads, head_dim, hidden_size)
    * Positional encoding kind    (rope / alibi / learned / none)
    * Attention implementation    (must be 'eager' — raises otherwise)

Design rules
------------
1.  Never assume a config field name — try a priority list, fall back to math.
2.  Never assume batch == 1 — record batch_dim_pos for downstream adapters.
3.  Confidence is exposed so the saliency adapter can route low-confidence
    models to the safer ``output_attentions=True`` fallback path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────────

QKVPattern  = Literal["separate", "fused_concat", "fused_interleaved", "unknown"]
PosEncoding = Literal["rope", "alibi", "learned", "none", "unknown"]
Confidence  = Literal["high", "medium", "low"]


# ── Exceptions ───────────────────────────────────────────────────────────────

class KiaomniConfigError(RuntimeError):
    """Raised when the model's runtime configuration is incompatible with KiaOmni."""


# ── Probe result ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeResult:
    """Immutable record of everything the probe discovered about a model."""

    # Module paths (dotted, relative to model root)
    layer_container_path: str
    attn_module_name: str

    # QKV pattern
    qkv_pattern: QKVPattern
    q_module_name: Optional[str]            # e.g. "q_proj" or None if fused
    k_module_name: Optional[str]
    v_module_name: Optional[str]
    fused_module_name: Optional[str]        # e.g. "c_attn" or "query_key_value"

    # Dimensions
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int

    # Batch + positional
    batch_dim_pos: int                       # always 0 for HF causal LMs
    pos_encoding: PosEncoding

    # Meta
    confidence: Confidence
    attn_implementation: str
    detection_notes: Tuple[str, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return (
            f"ProbeResult(layers='{self.layer_container_path}', "
            f"attn='{self.attn_module_name}', "
            f"qkv={self.qkv_pattern}, "
            f"n_layers={self.num_layers}, "
            f"nh={self.num_attention_heads}, nkv={self.num_key_value_heads}, "
            f"hd={self.head_dim}, pe={self.pos_encoding}, "
            f"confidence={self.confidence})"
        )


# ── Regex patterns ───────────────────────────────────────────────────────────

_ATTN_NAME_RE          = re.compile(r"^(self_)?attn(ention)?$|^self_attention$")
_LAYER_LIST_NAME_HINTS = ("layer", "block", "h")   # last segment of container path

_SEPARATE_Q_NAMES = ("q_proj", "query")
_SEPARATE_K_NAMES = ("k_proj", "key")
_SEPARATE_V_NAMES = ("v_proj", "value")

_FUSED_CONCAT_NAMES      = ("c_attn", "Wqkv", "qkv_proj", "qkv")
_FUSED_INTERLEAVED_NAMES = ("query_key_value",)


# ── Public probe ─────────────────────────────────────────────────────────────

class ArchitectureProbe:
    """Stateless discovery of an HF causal-LM's attention structure."""

    _CACHE_ATTR = "_kia_arch_info"

    # --- entry point -----------------------------------------------------

    @classmethod
    def probe(cls, model: nn.Module, *, force: bool = False) -> ProbeResult:
        """Run discovery on ``model`` and cache the result on the model."""
        cached = getattr(model, cls._CACHE_ATTR, None)
        if cached is not None and not force:
            return cached  # type: ignore[return-value]

        cls._verify_attn_implementation(model)

        notes: List[str] = []

        layer_container_path, layers, layer_conf = cls._find_layer_container(model, notes)
        attn_name, attn_conf = cls._find_attn_module_name(layers[0], notes)
        attn_module = getattr(layers[0], attn_name)

        (
            qkv_pattern,
            q_name, k_name, v_name, fused_name,
            qkv_conf,
        ) = cls._classify_qkv(attn_module, notes)

        nh, nkv, hd, hs = cls._resolve_dims(model, attn_module, qkv_pattern, fused_name, notes)
        pos_enc = cls._detect_pos_encoding(model, attn_module, notes)

        confidence = cls._aggregate_confidence(layer_conf, attn_conf, qkv_conf)
        if qkv_pattern == "unknown":
            confidence = "low"
            notes.append("qkv_pattern unresolved — fallback strategy required")

        result = ProbeResult(
            layer_container_path=layer_container_path,
            attn_module_name=attn_name,
            qkv_pattern=qkv_pattern,
            q_module_name=q_name,
            k_module_name=k_name,
            v_module_name=v_name,
            fused_module_name=fused_name,
            num_layers=len(layers),
            num_attention_heads=nh,
            num_key_value_heads=nkv,
            head_dim=hd,
            hidden_size=hs,
            batch_dim_pos=0,
            pos_encoding=pos_enc,
            confidence=confidence,
            attn_implementation=cls._read_attn_impl(model),
            detection_notes=tuple(notes),
        )

        try:
            object.__setattr__(model, cls._CACHE_ATTR, result)
        except Exception:
            logger.debug("Could not cache probe result on model instance.")

        logger.info("KiaOmni probe: %s", result)
        return result

    # --- attn-implementation gate ---------------------------------------

    @staticmethod
    def _read_attn_impl(model: nn.Module) -> str:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return "unknown"
        return (
            getattr(cfg, "_attn_implementation", None)
            or getattr(cfg, "attn_implementation", None)
            or "eager"
        )

    @classmethod
    def _verify_attn_implementation(cls, model: nn.Module) -> None:
        impl = cls._read_attn_impl(model)
        if impl in ("flash_attention_2", "flash_attention_3", "sdpa"):
            raise KiaomniConfigError(
                f"KiaOmni requires attn_implementation='eager', got '{impl}'. "
                "Forward hooks cannot observe Q/K when fused attention kernels "
                "are used. Reload the model with:\n\n"
                "    AutoModelForCausalLM.from_pretrained(..., "
                "attn_implementation='eager')\n"
            )

    # --- step 1: find layer container -----------------------------------

    @classmethod
    def _find_layer_container(
        cls, model: nn.Module, notes: List[str]
    ) -> Tuple[str, nn.ModuleList, Confidence]:
        """Locate the largest homogeneous ModuleList of transformer blocks."""
        best: Optional[Tuple[str, nn.ModuleList, int]] = None

        for path, module in model.named_modules():
            if not isinstance(module, nn.ModuleList) or len(module) == 0:
                continue
            first = module[0]
            if not isinstance(first, nn.Module):
                continue
            # Heuristic: blocks must themselves contain an attention submodule.
            has_attn = any(
                _ATTN_NAME_RE.match(child_name) is not None
                for child_name, _ in first.named_children()
            )
            if not has_attn:
                continue
            if best is None or len(module) > best[2]:
                best = (path, module, len(module))

        if best is None:
            raise KiaomniConfigError(
                "Could not locate a transformer-block ModuleList containing an "
                "attention submodule. Model architecture is unrecognised."
            )

        path, layers, _ = best
        last_segment = path.split(".")[-1] if path else path
        conf: Confidence = (
            "high" if any(h in last_segment for h in _LAYER_LIST_NAME_HINTS) else "medium"
        )
        notes.append(f"layer container at '{path}' (n={len(layers)})")
        return path, layers, conf

    # --- step 2: find attention module name -----------------------------

    @classmethod
    def _find_attn_module_name(
        cls, layer: nn.Module, notes: List[str]
    ) -> Tuple[str, Confidence]:
        for name, _ in layer.named_children():
            if _ATTN_NAME_RE.match(name):
                notes.append(f"attn submodule '{name}'")
                return name, "high"

        # Heuristic fallback: any child containing 'attn' substring.
        for name, _ in layer.named_children():
            if "attn" in name.lower() or "attention" in name.lower():
                notes.append(f"attn submodule '{name}' (heuristic)")
                return name, "medium"

        raise KiaomniConfigError(
            "No attention submodule found inside transformer block. "
            f"Children present: {[n for n, _ in layer.named_children()]}"
        )

    # --- step 3: classify QKV pattern -----------------------------------

    @classmethod
    def _classify_qkv(
        cls, attn: nn.Module, notes: List[str]
    ) -> Tuple[QKVPattern, Optional[str], Optional[str], Optional[str], Optional[str], Confidence]:

        children = dict(attn.named_children())

        q_name = next((n for n in _SEPARATE_Q_NAMES if n in children), None)
        k_name = next((n for n in _SEPARATE_K_NAMES if n in children), None)
        v_name = next((n for n in _SEPARATE_V_NAMES if n in children), None)
        if q_name and k_name and v_name:
            notes.append(f"QKV separate: {q_name}/{k_name}/{v_name}")
            return "separate", q_name, k_name, v_name, None, "high"

        for fused in _FUSED_INTERLEAVED_NAMES:
            if fused in children:
                notes.append(f"QKV fused-interleaved via '{fused}'")
                return "fused_interleaved", None, None, None, fused, "high"

        for fused in _FUSED_CONCAT_NAMES:
            if fused in children:
                notes.append(f"QKV fused-concat via '{fused}'")
                return "fused_concat", None, None, None, fused, "high"

        # Last resort: any single Linear/Conv1D child with out_features == 3*in
        for name, child in children.items():
            out_f = getattr(child, "out_features", None) or getattr(child, "nf", None)
            in_f  = getattr(child, "in_features",  None) or getattr(child, "nx", None)
            if out_f and in_f and out_f == 3 * in_f:
                notes.append(f"QKV fused-concat (heuristic) via '{name}'")
                return "fused_concat", None, None, None, name, "medium"

        notes.append("QKV pattern unresolved")
        return "unknown", None, None, None, None, "low"

    # --- step 4: dimensions ---------------------------------------------

    _NH_KEYS  = ("num_attention_heads", "n_head", "num_heads", "n_heads")
    _NKV_KEYS = ("num_key_value_heads", "num_kv_heads", "n_kv_heads")
    _HS_KEYS  = ("hidden_size", "n_embd", "d_model", "model_dim")
    _HD_KEYS  = ("head_dim", "head_size", "d_head")

    @classmethod
    def _resolve_dims(
        cls,
        model: nn.Module,
        attn: nn.Module,
        qkv_pattern: QKVPattern,
        fused_name: Optional[str],
        notes: List[str],
    ) -> Tuple[int, int, int, int]:
        cfg = getattr(model, "config", object())

        def _pick(keys: Tuple[str, ...]) -> Optional[int]:
            for k in keys:
                v = getattr(cfg, k, None)
                if isinstance(v, int) and v > 0:
                    return v
            return None

        nh = _pick(cls._NH_KEYS)
        if nh is None:
            raise KiaomniConfigError(
                f"Could not read num_attention_heads from config "
                f"(tried {cls._NH_KEYS})."
            )

        hs = _pick(cls._HS_KEYS)
        if hs is None:
            # Derive from Q-projection input width if separate pattern.
            for child in attn.children():
                in_f = getattr(child, "in_features", None) or getattr(child, "nx", None)
                if in_f:
                    hs = int(in_f)
                    notes.append(f"hidden_size derived from projection.in_features={hs}")
                    break
        if hs is None:
            raise KiaomniConfigError("Could not determine hidden_size.")

        hd = _pick(cls._HD_KEYS) or (hs // nh)
        nkv = _pick(cls._NKV_KEYS) or nh

        if hs % nh != 0:
            notes.append(f"warning: hidden_size {hs} not divisible by nh {nh}")

        return nh, nkv, hd, hs

    # --- step 5: positional encoding ------------------------------------

    @classmethod
    def _detect_pos_encoding(
        cls, model: nn.Module, attn: nn.Module, notes: List[str]
    ) -> PosEncoding:
        cfg = getattr(model, "config", object())

        # RoPE: attn module has rotary_emb attribute (or rope_theta in config)
        for name, _ in attn.named_modules():
            if "rotary" in name.lower() or "rope" in name.lower():
                notes.append("pos_encoding=rope (attn has rotary submodule)")
                return "rope"
        if getattr(cfg, "rope_theta", None) or getattr(cfg, "rotary_dim", None):
            notes.append("pos_encoding=rope (config flags)")
            return "rope"

        # ALiBi
        if getattr(cfg, "alibi", False) or getattr(cfg, "use_alibi", False):
            notes.append("pos_encoding=alibi")
            return "alibi"
        pet = getattr(cfg, "position_embedding_type", "")
        if isinstance(pet, str) and "alibi" in pet.lower():
            notes.append("pos_encoding=alibi (position_embedding_type)")
            return "alibi"

        # Learned absolute (GPT-2 wpe, BERT-style position_embeddings)
        for name, _ in model.named_modules():
            tail = name.split(".")[-1]
            if tail in ("wpe", "position_embeddings", "embed_positions"):
                notes.append(f"pos_encoding=learned ('{tail}')")
                return "learned"

        if isinstance(pet, str) and pet:
            notes.append(f"pos_encoding={pet} (raw)")
            return "learned" if "absolute" in pet.lower() else "unknown"

        notes.append("pos_encoding could not be determined")
        return "unknown"

    # --- step 6: confidence aggregation ---------------------------------

    @staticmethod
    def _aggregate_confidence(*levels: Confidence) -> Confidence:
        rank = {"high": 2, "medium": 1, "low": 0}
        worst = min(rank[l] for l in levels)
        return ("low", "medium", "high")[worst]


__all__ = [
    "ArchitectureProbe",
    "ProbeResult",
    "QKVPattern",
    "PosEncoding",
    "Confidence",
    "KiaomniConfigError",
]
