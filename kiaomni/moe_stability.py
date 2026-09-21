"""
Inference-time adaptive MoE route stabilizer.

This module revives the causal-inertia idea from the earlier EG-MoT work
as a lightweight wrapper for pretrained HuggingFace MoE models. It does
not retrain or replace experts. Instead it hooks the router gate outputs
and applies confidence-gated causal smoothing:

    s_t = alpha_t * s_(t-1) + (1 - alpha_t) * z_t

where alpha_t grows only when the current router distribution is uncertain
and the current hidden state is similar to the preceding hidden state.

The implementation is intentionally architecture-light:
- router gates are discovered by output width == num_experts;
- no expert IDs or model-family paths are hardcoded;
- alpha_max=0 is an identity path;
- state resets at a fresh prefill and persists across decode steps.

The controller also records raw-vs-stabilized routing diagnostics so a POC
can measure whether lower switching comes with quality regressions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class _RouteState:
    smoothed_logits: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None
    raw_topk: Optional[torch.Tensor] = None
    stable_topk: Optional[torch.Tensor] = None


@dataclass
class RouteMetrics:
    tokens: int = 0
    raw_top1_switches: int = 0
    stable_top1_switches: int = 0
    raw_jaccard_sum: float = 0.0
    stable_jaccard_sum: float = 0.0
    transition_pairs: int = 0
    intervention_tokens: int = 0
    alpha_sum: float = 0.0
    alpha_max_seen: float = 0.0
    uncertainty_sum: float = 0.0
    similarity_sum: float = 0.0

    def snapshot(self) -> dict:
        pairs = max(self.transition_pairs, 1)
        tokens = max(self.tokens, 1)
        return {
            "tokens_observed": self.tokens,
            "raw_top1_transition_rate": self.raw_top1_switches / pairs,
            "stable_top1_transition_rate": self.stable_top1_switches / pairs,
            "raw_topk_jaccard": self.raw_jaccard_sum / pairs,
            "stable_topk_jaccard": self.stable_jaccard_sum / pairs,
            "intervention_rate": self.intervention_tokens / tokens,
            "mean_alpha": self.alpha_sum / tokens,
            "max_alpha_seen": self.alpha_max_seen,
            "mean_uncertainty": self.uncertainty_sum / tokens,
            "mean_hidden_similarity": self.similarity_sum / tokens,
        }


class AdaptiveMoERouteController:
    """Stateful controller installed by :func:`apply_moe_route_stability`."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        alpha_max: float = 0.10,
    ) -> None:
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if not 0.0 <= alpha_max < 1.0:
            raise ValueError("alpha_max must satisfy 0 <= alpha_max < 1")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.alpha_max = float(alpha_max)
        self.states: Dict[str, _RouteState] = {}
        self.metrics = RouteMetrics()
        self.router_names: list[str] = []
        self._handles: list = []

    def reset_state(self) -> None:
        self.states.clear()

    def reset_metrics(self) -> None:
        self.metrics = RouteMetrics()

    def snapshot(self) -> dict:
        data = self.metrics.snapshot()
        data.update(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "alpha_max": self.alpha_max,
                "router_count": len(self.router_names),
                "router_names": list(self.router_names),
            }
        )
        return data

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.reset_state()

    @staticmethod
    def _to_btd(x: torch.Tensor, last_dim: int) -> tuple[torch.Tensor, bool]:
        if x.ndim == 3:
            return x, False
        if x.ndim == 2:
            return x.unsqueeze(0), True
        if x.ndim == 1 and x.numel() == last_dim:
            return x.view(1, 1, last_dim), True
        raise ValueError(f"Unsupported router tensor shape: {tuple(x.shape)}")

    @staticmethod
    def _mean_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
        # a,b: [B,K]. K is tiny, so the broadcast formulation is cheap.
        matches = a.unsqueeze(-1).eq(b.unsqueeze(-2))
        inter = matches.any(dim=-1).sum(dim=-1).float()
        union = (2 * a.shape[-1] - inter).clamp_min(1.0)
        return float((inter / union).mean().item())

    def _selection_topk(
        self,
        logits: torch.Tensor,
        expert_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = logits
        if expert_bias is not None:
            # LFM2.5 selects experts using sigmoid(router_logits)+expert_bias.
            # For architectures without this bias, ranking raw logits is
            # equivalent to ranking softmax/sigmoid probabilities.
            bias = expert_bias.detach().to(device=logits.device, dtype=torch.float32)
            scores = torch.sigmoid(logits) + bias.view(1, -1)
        return torch.topk(scores, k=self.top_k, dim=-1).indices

    def transform(
        self,
        name: str,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(logits) or logits.shape[-1] != self.num_experts:
            return logits
        if not torch.is_tensor(hidden):
            return logits

        original_shape = logits.shape
        z, squeezed = self._to_btd(logits.float(), self.num_experts)
        h, _ = self._to_btd(hidden.float(), hidden.shape[-1])

        if z.shape[:2] != h.shape[:2]:
            return logits

        state = self.states.setdefault(name, _RouteState())
        out_tokens: list[torch.Tensor] = []

        for t in range(z.shape[1]):
            raw_t = z[:, t, :]
            h_t = h[:, t, :]

            probs = torch.softmax(raw_t, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / math.log(self.num_experts)
            uncertainty = entropy.clamp(0.0, 1.0)

            if (
                state.smoothed_logits is None
                or state.hidden is None
                or state.smoothed_logits.shape[0] != raw_t.shape[0]
            ):
                similarity = torch.zeros_like(uncertainty)
                alpha = torch.zeros_like(uncertainty)
                stable_t = raw_t
            else:
                similarity = (F.cosine_similarity(h_t, state.hidden, dim=-1) + 1.0) * 0.5
                similarity = similarity.clamp(0.0, 1.0)
                alpha = self.alpha_max * uncertainty * similarity
                stable_t = (
                    alpha.unsqueeze(-1) * state.smoothed_logits
                    + (1.0 - alpha).unsqueeze(-1) * raw_t
                )

            raw_topk = self._selection_topk(raw_t, expert_bias)
            stable_topk = self._selection_topk(stable_t, expert_bias)

            if state.raw_topk is not None and state.stable_topk is not None:
                self.metrics.transition_pairs += raw_t.shape[0]
                self.metrics.raw_top1_switches += int(
                    raw_topk[:, 0].ne(state.raw_topk[:, 0]).sum().item()
                )
                self.metrics.stable_top1_switches += int(
                    stable_topk[:, 0].ne(state.stable_topk[:, 0]).sum().item()
                )
                self.metrics.raw_jaccard_sum += self._mean_jaccard(raw_topk, state.raw_topk)
                self.metrics.stable_jaccard_sum += self._mean_jaccard(
                    stable_topk, state.stable_topk
                )

            self.metrics.tokens += raw_t.shape[0]
            self.metrics.intervention_tokens += int(
                stable_topk.ne(raw_topk).any(dim=-1).sum().item()
            )
            self.metrics.alpha_sum += float(alpha.sum().item())
            self.metrics.alpha_max_seen = max(
                self.metrics.alpha_max_seen,
                float(alpha.max().item()) if alpha.numel() else 0.0,
            )
            self.metrics.uncertainty_sum += float(uncertainty.sum().item())
            self.metrics.similarity_sum += float(similarity.sum().item())

            state.smoothed_logits = stable_t.detach()
            state.hidden = h_t.detach()
            state.raw_topk = raw_topk.detach()
            state.stable_topk = stable_topk.detach()
            out_tokens.append(stable_t)

        stable = torch.stack(out_tokens, dim=1)
        if squeezed:
            stable = stable.squeeze(0)
        return stable.reshape(original_shape).to(dtype=logits.dtype)


def _config_value(config, name: str):
    value = getattr(config, name, None)
    if value is not None:
        return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return getattr(text_config, name, None)
    return None


def _discover_router_gates(model, num_experts: int) -> list[tuple[str, object, object]]:
    modules = dict(model.named_modules())
    found: list[tuple[str, object, object]] = []

    for name, module in modules.items():
        if not name:
            continue
        tail = name.rsplit(".", 1)[-1].lower()
        if tail not in {"gate", "router"}:
            continue
        out_features = getattr(module, "out_features", None)
        if out_features is not None and int(out_features) != num_experts:
            continue

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        parent_text = parent_name.lower()
        if (
            "feed_forward" not in parent_text
            and "mlp" not in parent_text
            and "moe" not in parent_text
            and not hasattr(parent, "experts")
            and not hasattr(parent, "switch_mlp")
        ):
            continue
        found.append((name, module, parent))

    return found


def _is_fresh_prefill(kwargs: dict) -> bool:
    cache_position = kwargs.get("cache_position")
    if torch.is_tensor(cache_position) and cache_position.numel() > 0:
        return int(cache_position.reshape(-1)[0].item()) == 0
    return kwargs.get("past_key_values") is None


def apply_moe_route_stability(
    model,
    *,
    alpha_max: float = 0.10,
    num_experts: Optional[int] = None,
    top_k: Optional[int] = None,
    verbose: bool = False,
) -> AdaptiveMoERouteController:
    """Install adaptive causal smoothing on discovered MoE router gates."""

    remove_moe_route_stability(model)

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model.config is required for MoE router discovery")

    resolved_experts = num_experts or _config_value(cfg, "num_experts")
    resolved_top_k = top_k or _config_value(cfg, "num_experts_per_tok")
    if not isinstance(resolved_experts, int) or resolved_experts < 2:
        raise ValueError("Could not resolve config.num_experts")
    if not isinstance(resolved_top_k, int) or resolved_top_k < 1:
        raise ValueError("Could not resolve config.num_experts_per_tok")

    controller = AdaptiveMoERouteController(
        num_experts=resolved_experts,
        top_k=resolved_top_k,
        alpha_max=alpha_max,
    )

    discovered = _discover_router_gates(model, resolved_experts)
    if not discovered:
        raise RuntimeError(
            "No MoE router gate was discovered. Expected a gate/router module "
            f"with output width {resolved_experts} under an MoE/MLP block."
        )

    for name, module, parent in discovered:
        controller.router_names.append(name)
        expert_bias = getattr(parent, "expert_bias", None)

        def _hook(_module, inputs, output, *, _name=name, _bias=expert_bias):
            if not inputs:
                return output
            return controller.transform(
                _name,
                inputs[0],
                output,
                expert_bias=_bias,
            )

        controller._handles.append(module.register_forward_hook(_hook))

    def _prefill_hook(_model, args, kwargs):
        if _is_fresh_prefill(kwargs):
            controller.reset_state()

    controller._handles.append(
        model.register_forward_pre_hook(_prefill_hook, with_kwargs=True)
    )
    model._kia_moe_route_controller = controller

    if verbose:
        print(
            f"[KiaOmni-MoE] installed adaptive route stability on "
            f"{len(discovered)} router gates; alpha_max={alpha_max}"
        )

    return controller


def remove_moe_route_stability(model) -> None:
    controller = getattr(model, "_kia_moe_route_controller", None)
    if controller is not None:
        controller.remove()
        try:
            delattr(model, "_kia_moe_route_controller")
        except AttributeError:
            pass


__all__ = [
    "AdaptiveMoERouteController",
    "RouteMetrics",
    "apply_moe_route_stability",
    "remove_moe_route_stability",
]
