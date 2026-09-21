from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kiaomni.adapters import ArchitectureProbe
from kiaomni.adapters.saliency import SaliencyAdapter


class _Config:
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 16
    head_dim = 4
    rope_theta = 10000.0
    _attn_implementation = "eager"


class _ConvOnlyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 16, kernel_size=1)

    def forward(self, x):
        return x


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False)
        self.v_proj = nn.Linear(16, 8, bias=False)
        self.o_proj = nn.Linear(8, 16, bias=False)
        self.rotary_emb = nn.Identity()

    def forward(self, x):
        # Fire Q/K/V projections so SaliencyAdapter can observe them.
        _ = self.q_proj(x)
        _ = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(v)


class _AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, x):
        return x + self.self_attn(x)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, 16)
        self.layers = nn.ModuleList(
            [
                _ConvOnlyBlock(),
                _ConvOnlyBlock(),
                _AttentionBlock(),
                _ConvOnlyBlock(),
                _AttentionBlock(),
            ]
        )


class _HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.config = _Config()

    def forward(self, input_ids, use_cache=False, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_probe_accepts_hybrid_container_whose_first_layer_is_not_attention():
    model = _HybridModel()
    probe = ArchitectureProbe.probe(model, force=True)

    assert probe.layer_container_path == "model.layers"
    assert probe.attn_module_name == "self_attn"
    assert probe.qkv_pattern == "separate"
    assert probe.num_layers == 5


def test_saliency_skips_non_attention_hybrid_layers():
    torch.manual_seed(4)
    model = _HybridModel()
    probe = ArchitectureProbe.probe(model, force=True)
    adapter = SaliencyAdapter(probe)
    ids = torch.randint(0, 128, (1, 12))

    saliency = adapter.extract(ids, model)

    assert saliency.shape == (1, 12)
    assert saliency.dtype == np.float32
    assert np.isfinite(saliency).all()
