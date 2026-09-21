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
    hook_calls: int = 0
    structured_outputs: int = 0
    shape_reconciliations: int = 0
    functional_router_calls: int = 0

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
            "hook_calls": self.hook_calls,
            "structured_outputs": self.structured_outputs,
            "shape_reconciliations": self.shape_reconciliations,
            "functional_router_calls": self.functional_router_calls,
        }


def _find_router_tensor(output, num_experts: int):
    """Find a tensor carrying per-expert router scores in structured outputs."""
    if torch.is_tensor(output):
        if output.ndim >= 1 and output.shape[-1] == num_experts:
            return output
        return None
    if isinstance(output, tuple):
        for item in output:
            found = _find_router_tensor(item, num_experts)
            if found is not None:
                return found
        return None
    if isinstance(output, list):
        for item in output:
            found = _find_router_tensor(item, num_experts)
            if found is not None:
                return found
        return None
    if isinstance(output, dict):
        for item in output.values():
            found = _find_router_tensor(item, num_experts)
            if found is not None:
                return found
        return None
    return None


def _replace_router_tensor(output, target: torch.Tensor, replacement: torch.Tensor):
    """Replace one tensor inside a nested router output while preserving structure."""
    if torch.is_tensor(output):
        return replacement if output is target else output
    if isinstance(output, tuple):
        values = tuple(_replace_router_tensor(x, target, replacement) for x in output)
        if hasattr(output, "_fields"):
            return type(output)(*values)
        return values
    if isinstance(output, list):
        return [_replace_router_tensor(x, target, replacement) for x in output]
    if isinstance(output, dict):
        return type(output)(
            (k, _replace_router_tensor(v, target, replacement))
            for k, v in output.items()
        )
    return output


class AdaptiveMoERouteController:
    """Stateful controller installed by :func:`apply_moe_route_stability`."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        alpha_max: float = 0.10,
        score_func: str = "softmax",
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
        self.score_func = str(score_func or "softmax").lower()
        self.states: Dict[str, _RouteState] = {}
        self.metrics = RouteMetrics()
        self.router_names: list[str] = []
        self._handles: list = []
        self._functional_linear_original = None
        self._router_weight_map: dict[int, tuple[str, Optional[torch.Tensor]]] = {}
        self._already_transformed_output_ids: set[int] = set()

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
                "score_func": self.score_func,
                "router_count": len(self.router_names),
                "router_names": list(self.router_names),
            }
        )
        return data

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._functional_linear_original is not None:
            F.linear = self._functional_linear_original
            self._functional_linear_original = None
        self._router_weight_map.clear()
        self._already_transformed_output_ids.clear()
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
            # Some custom MoE implementations flatten tokens before routing
            # while passing unflattened hidden states to surrounding modules.
            # Reconcile when both tensors represent the same token count.
            z_tokens = z.numel() // self.num_experts
            h_tokens = h.numel() // h.shape[-1]
            if z_tokens != h_tokens:
                return logits
            z = z.reshape(1, z_tokens, self.num_experts)
            h = h.reshape(1, h_tokens, h.shape[-1])
            squeezed = logits.ndim <= 2
            self.metrics.shape_reconciliations += 1

        state = self.states.setdefault(name, _RouteState())
        out_tokens: list[torch.Tensor] = []

        for t in range(z.shape[1]):
            raw_t = z[:, t, :]
            h_t = h[:, t, :]

            if self.score_func == "sigmoid":
                probs = torch.sigmoid(raw_t)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
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


def _first_config_int(config, names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        value = _config_value(config, name)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _first_attr_int(obj, names: tuple[str, ...]) -> Optional[int]:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _discover_router_gates(
    model,
    num_experts: Optional[int] = None,
) -> list[tuple[str, object, object]]:
    modules = dict(model.named_modules())
    found: list[tuple[str, object, object]] = []

    for name, module in modules.items():
        if not name:
            continue
        tail = name.rsplit(".", 1)[-1].lower()
        if tail not in {"gate", "router"}:
            continue
        out_features = getattr(module, "out_features", None)
        if out_features is not None:
            try:
                out_features = int(out_features)
            except (TypeError, ValueError):
                continue
            if out_features < 2:
                continue
            if num_experts is not None and out_features != num_experts:
                continue
        elif num_experts is not None:
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
    score_func: Optional[str] = None,
    verbose: bool = False,
) -> AdaptiveMoERouteController:
    """Install adaptive causal smoothing on discovered MoE router gates."""

    remove_moe_route_stability(model)

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model.config is required for MoE router discovery")

    resolved_experts = num_experts or _first_config_int(
        cfg,
        (
            "num_experts",
            "num_local_experts",
            "num_routed_experts",
            "n_routed_experts",
            "routed_experts",
        ),
    )
    resolved_top_k = top_k or _first_config_int(
        cfg,
        (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k",
            "topk",
            "moe_top_k",
        ),
    )
    resolved_score_func = (
        score_func
        or _config_value(cfg, "score_func")
        or _config_value(cfg, "scoring_func")
        or _config_value(cfg, "router_score_func")
        or _config_value(cfg, "router_score_function")
        or _config_value(cfg, "router_activation")
        or "softmax"
    )

    # Discover router modules even when a custom architecture uses different
    # config field names. If every candidate router has the same output width,
    # that width is the expert count.
    discovered = _discover_router_gates(model, resolved_experts)
    if not discovered and resolved_experts is not None:
        discovered = _discover_router_gates(model, None)
    if not discovered:
        raise RuntimeError(
            "No MoE router gate was discovered under an MoE/MLP block."
        )

    if resolved_experts is None:
        widths = {
            int(getattr(module, "out_features"))
            for _, module, _ in discovered
            if getattr(module, "out_features", None) is not None
        }
        if len(widths) == 1:
            resolved_experts = widths.pop()

    if resolved_top_k is None:
        parent_topks = {
            value
            for _, _, parent in discovered
            for value in [
                _first_attr_int(
                    parent,
                    (
                        "num_experts_per_tok",
                        "num_experts_per_token",
                        "top_k",
                        "topk",
                        "moe_top_k",
                    ),
                )
            ]
            if value is not None
        }
        if len(parent_topks) == 1:
            resolved_top_k = parent_topks.pop()

    if not isinstance(resolved_experts, int) or resolved_experts < 2:
        raise ValueError(
            "Could not infer MoE expert count from config or router output width."
        )
    if not isinstance(resolved_top_k, int) or not (1 <= resolved_top_k <= resolved_experts):
        raise ValueError(
            "Could not infer MoE top-k from config or router parent attributes."
        )

    # Re-filter now that expert count is known.
    discovered = _discover_router_gates(model, resolved_experts)
    if not discovered:
        raise RuntimeError(
            f"No MoE router gate with output width {resolved_experts} was discovered."
        )

    controller = AdaptiveMoERouteController(
        num_experts=resolved_experts,
        top_k=resolved_top_k,
        alpha_max=alpha_max,
        score_func=resolved_score_func,
    )

    # Some custom MoE implementations call F.linear(hidden, router.weight)
    # directly instead of router(hidden), bypassing nn.Module forward hooks.
    # Intercept only calls whose weight object belongs to a discovered router.
    for _name, _module, _parent in discovered:
        weight = getattr(_module, "weight", None)
        if torch.is_tensor(weight):
            controller._router_weight_map[id(weight)] = (
                _name,
                getattr(_parent, "expert_bias", None),
            )

    if controller._router_weight_map:
        controller._functional_linear_original = F.linear
        _orig_linear = controller._functional_linear_original

        def _linear_with_router_stability(input, weight, bias=None):
            output = _orig_linear(input, weight, bias)
            route_info = controller._router_weight_map.get(id(weight))
            if route_info is None:
                return output

            _name, _bias = route_info
            controller.metrics.hook_calls += 1
            controller.metrics.functional_router_calls += 1
            stable = controller.transform(
                _name,
                input,
                output,
                expert_bias=_bias,
            )
            controller._already_transformed_output_ids.add(id(stable))
            return stable

        F.linear = _linear_with_router_stability

    for name, module, parent in discovered:
        controller.router_names.append(name)
        expert_bias = getattr(parent, "expert_bias", None)

        def _hook(_module, inputs, output, *, _name=name, _bias=expert_bias):
            if not inputs:
                return output

            router_tensor = _find_router_tensor(output, controller.num_experts)
            if router_tensor is None:
                controller.metrics.hook_calls += 1
                return output

            # F.linear fallback may already have transformed this exact router
            # result. In that case the module hook is observation-only.
            if id(router_tensor) in controller._already_transformed_output_ids:
                controller._already_transformed_output_ids.discard(id(router_tensor))
                return output

            controller.metrics.hook_calls += 1
            if router_tensor is not output:
                controller.metrics.structured_outputs += 1

            stable = controller.transform(
                _name,
                inputs[0],
                router_tensor,
                expert_bias=_bias,
            )
            if stable is router_tensor:
                return output
            return _replace_router_tensor(output, router_tensor, stable)

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
            f"{len(discovered)} router gates; alpha_max={alpha_max}; "
            f"score_func={resolved_score_func}"
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
