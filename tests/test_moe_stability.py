from __future__ import annotations

import torch
import torch.nn as nn

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
