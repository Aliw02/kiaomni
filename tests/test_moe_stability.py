from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiaomni.moe_stability import (
    apply_moe_route_stability,
    remove_moe_route_stability,
)


class _Config:
    num_experts = 3
    num_experts_per_tok = 1


class _SparseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(2, 3, bias=False)
        self.experts = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, x):
        return self.gate(x)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _SparseMLP()


class _TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.layers = nn.ModuleList([_Block()])

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        x = inputs_embeds
        if x is None:
            x = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return self.layers[0].mlp(x)


def test_alpha_zero_is_exact_identity():
    torch.manual_seed(1)
    model = _TinyMoE()
    x = torch.randn(1, 7, 2)

    expected = model(inputs_embeds=x).detach()
    controller = apply_moe_route_stability(model, alpha_max=0.0)
    actual = model(inputs_embeds=x).detach()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert controller.snapshot()["router_count"] == 1

    remove_moe_route_stability(model)


def test_adaptive_inertia_reduces_near_tie_switching():
    model = _TinyMoE()
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )

    # Adjacent hidden states are almost collinear, while the tiny sign change
    # makes the memoryless top-1 router alternate between expert 0 and 1.
    first = torch.tensor([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    x = torch.stack([first, torch.ones_like(first)], dim=-1).unsqueeze(0)

    controller = apply_moe_route_stability(model, alpha_max=0.80)
    _ = model(inputs_embeds=x)
    metrics = controller.snapshot()

    assert metrics["raw_top1_transition_rate"] > 0.9
    assert metrics["stable_top1_transition_rate"] < metrics["raw_top1_transition_rate"]
    assert metrics["mean_alpha"] > 0.0

    remove_moe_route_stability(model)


def test_fresh_prefill_resets_route_state():
    model = _TinyMoE()
    controller = apply_moe_route_stability(model, alpha_max=0.5)
    x = torch.tensor([[[0.1, 1.0], [-0.1, 1.0], [0.1, 1.0]]])

    first = model(inputs_embeds=x).detach()
    second = model(inputs_embeds=x).detach()

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert controller.snapshot()["tokens_observed"] == 6

    remove_moe_route_stability(model)


def test_remove_restores_unmodified_forward():
    torch.manual_seed(3)
    model = _TinyMoE()
    x = torch.randn(1, 4, 2)
    expected = model(inputs_embeds=x).detach()

    apply_moe_route_stability(model, alpha_max=0.6)
    _ = model(inputs_embeds=x)
    remove_moe_route_stability(model)

    actual = model(inputs_embeds=x).detach()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_sigmoid_router_semantics_are_detected():
    model = _TinyMoE()
    model.config.score_func = "sigmoid"
    controller = apply_moe_route_stability(model, alpha_max=0.10)

    x = torch.tensor([[[0.1, 1.0], [0.11, 1.0], [0.12, 1.0]]])
    _ = model(inputs_embeds=x)
    metrics = controller.snapshot()

    assert metrics["score_func"] == "sigmoid"
    assert 0.0 <= metrics["mean_uncertainty"] <= 1.0

    remove_moe_route_stability(model)


class _AliasConfig:
    num_routed_experts = 3
    num_experts_per_token = 1
    scoring_func = "sigmoid"


class _AliasTinyMoE(_TinyMoE):
    def __init__(self):
        super().__init__()
        self.config = _AliasConfig()


def test_custom_moe_config_aliases_are_inferred():
    model = _AliasTinyMoE()
    controller = apply_moe_route_stability(model, alpha_max=0.10)

    snapshot = controller.snapshot()
    assert snapshot["num_experts"] == 3
    assert snapshot["top_k"] == 1
    assert snapshot["score_func"] == "sigmoid"
    assert snapshot["router_count"] == 1

    remove_moe_route_stability(model)


class _FunctionalSparseMLP(_SparseMLP):
    def forward(self, x):
        # Mimics custom MoE code that bypasses router.forward() and consumes
        # the router weight through torch.nn.functional.linear directly.
        return F.linear(x, self.gate.weight)


class _FunctionalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FunctionalSparseMLP()


class _FunctionalTinyMoE(_TinyMoE):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = _Config()
        self.layers = nn.ModuleList([_FunctionalBlock()])


def test_functional_linear_router_bypass_is_intercepted():
    torch.manual_seed(11)
    model = _FunctionalTinyMoE()
    x = torch.randn(1, 9, 2)

    controller = apply_moe_route_stability(model, alpha_max=0.2)
    _ = model(inputs_embeds=x)
    metrics = controller.snapshot()

    assert metrics["tokens_observed"] > 0
    assert metrics["functional_router_calls"] > 0
    assert metrics["hook_calls"] > 0

    remove_moe_route_stability(model)


def test_shadow_counterfactual_observes_jitter_without_mutating_model():
    model = _TinyMoE()
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )

    first = torch.tensor([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    x = torch.stack([first, torch.ones_like(first)], dim=-1).unsqueeze(0)
    expected = model(inputs_embeds=x).detach()

    controller = apply_moe_route_stability(
        model,
        alpha_max=0.80,
        routing_mode="shadow_counterfactual",
    )
    actual = model(inputs_embeds=x).detach()
    metrics = controller.snapshot()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert metrics["routing_mode"] == "shadow_counterfactual"
    assert metrics["active_intervention"] is False
    assert metrics["shadow_router_calls"] > 0
    assert metrics["tokens_observed"] > 0
    assert metrics["raw_top1_transition_rate"] > 0.9
    assert metrics["stable_top1_transition_rate"] < metrics["raw_top1_transition_rate"]

    remove_moe_route_stability(model)



def test_decode_only_trace_skips_prefill_and_records_expert_sequence():
    model = _TinyMoE()
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )

    controller = apply_moe_route_stability(
        model,
        alpha_max=0.50,
        routing_mode="shadow_counterfactual",
        observe_decode_only=True,
        record_trace=True,
    )

    prefill = torch.tensor(
        [[[0.3, 1.0], [0.2, 1.0], [0.1, 1.0], [-0.1, 1.0]]]
    )
    _ = model(
        inputs_embeds=prefill,
        cache_position=torch.arange(4),
    )

    for idx, value in enumerate((0.10, -0.10, 0.11), start=4):
        step = torch.tensor([[[value, 1.0]]])
        _ = model(
            inputs_embeds=step,
            cache_position=torch.tensor([idx]),
            past_key_values=object(),
        )

    snap = controller.snapshot()
    assert snap["observe_decode_only"] is True
    assert snap["record_trace"] is True
    assert snap["tokens_observed"] == 3
    assert snap["expert_trace"]["scope"] == "decode_only"

    layer = snap["expert_trace"]["layers"]["layers.0.mlp.gate"]
    assert layer["decode_steps"] == 3
    assert len(layer["raw_top1_sequence"]) == 3
    assert len(layer["stable_top1_sequence"]) == 3
    assert len(layer["raw_topk_sequence"]) == 3
    assert layer["raw_top1_transition_rate"] is not None

    remove_moe_route_stability(model)
