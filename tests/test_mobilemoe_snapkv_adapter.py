from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kiaomni.baselines.mobilemoe_snapkv import (
    ALGORITHM_MODIFIED,
    MobileMoESnapKVCompatibilityError,
    adapter_provenance,
    describe_position_embeddings,
    mobilemoe_prerope_query_states,
    resolve_mobilemoe_rope,
)


class ScaleNorm(nn.Module):
    def forward(self, x):
        return x * 2.0


class FakeMobileMoEAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="mobilemoe",
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        self.head_dim = 2
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = ScaleNorm()
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))


def test_mobilemoe_query_reconstruction_uses_native_q_norm():
    module = FakeMobileMoEAttention()
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )

    query = mobilemoe_prerope_query_states(module, hidden)

    expected = hidden.view(1, 2, 2, 2).transpose(1, 2) * 2.0
    torch.testing.assert_close(query, expected)


def test_rope_pair_is_forwarded_without_approximation():
    module = FakeMobileMoEAttention()
    hidden = torch.zeros(1, 4, 4)
    cos = torch.ones(1, 4, 2)
    sin = torch.zeros(1, 4, 2)

    kind, value = resolve_mobilemoe_rope(
        module,
        hidden,
        {"position_embeddings": (cos, sin)},
    )

    assert kind == "kwargs_cos_sin"
    assert value[0] is cos
    assert value[1] is sin


def test_unsupported_real_tensor_rope_fails_closed():
    module = FakeMobileMoEAttention()
    hidden = torch.zeros(1, 4, 4)

    with pytest.raises(MobileMoESnapKVCompatibilityError):
        resolve_mobilemoe_rope(
            module,
            hidden,
            {"position_embeddings": torch.zeros(1, 4, 2)},
        )


def test_position_embedding_diagnostics_are_structured():
    value = (torch.ones(1, 3, 2), torch.zeros(1, 3, 2))
    desc = describe_position_embeddings(value)

    assert desc["kind"] == "tuple"
    assert desc["length"] == 2
    assert desc["items"][0]["shape"] == [1, 3, 2]


def test_adapter_declares_snapkv_algorithm_unchanged():
    provenance = adapter_provenance()

    assert ALGORITHM_MODIFIED is False
    assert provenance["algorithm_modified"] is False
    assert provenance["official_snapkv_score_equation_preserved"] is True
    assert provenance["snapkv_scoring_changed"] is False
    assert provenance["snapkv_window_changed"] is False
    assert provenance["snapkv_pooling_changed"] is False
    assert provenance["snapkv_pruning_changed"] is False
