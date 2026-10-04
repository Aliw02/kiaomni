from __future__ import annotations

from dataclasses import dataclass
import math
import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("head_dim must be even")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
    )
    angles = positions.to(dtype=x.dtype).unsqueeze(-1) * inv_freq
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    return x * cos + rotate_half(x) * sin


@dataclass
class StreamingKVCache:
    key: torch.Tensor
    value: torch.Tensor
    absolute_positions: torch.Tensor
    next_position: int

    @property
    def kv_length(self) -> int:
        return int(self.key.shape[-2])

    @property
    def bytes(self) -> int:
        return int(
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )

    def compact(self, keep_indices: torch.Tensor) -> "StreamingKVCache":
        keep_indices = keep_indices.to(device=self.key.device, dtype=torch.long)
        return StreamingKVCache(
            key=self.key.index_select(-2, keep_indices).contiguous(),
            value=self.value.index_select(-2, keep_indices).contiguous(),
            absolute_positions=self.absolute_positions.index_select(0, keep_indices).contiguous(),
            next_position=self.next_position,
        )


class StreamingToyDecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4, seed: int = 0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.o_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 2, d_model, bias=False),
        )
        torch.random.set_rng_state(state)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def prefill(self, hidden: torch.Tensor):
        _, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device, dtype=torch.long)
        q = apply_rope(self._split(self.q_proj(hidden)), positions)
        k = apply_rope(self._split(self.k_proj(hidden)), positions)
        v = self._split(self.v_proj(hidden))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(length, length, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal.view(1, 1, length, length), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = hidden + self.o_proj(self._merge(torch.matmul(probs, v)))
        out = out + self.ffn(out)
        window = min(8, length)
        saliency = probs[:, :, -window:, :].max(dim=2).values.mean(dim=(0, 1)).detach()
        cache = StreamingKVCache(
            k.detach().clone(),
            v.detach().clone(),
            positions.detach().clone(),
            length,
        )
        return out, cache, saliency

    def prefill_chunk(
        self,
        hidden: torch.Tensor,
        cache: StreamingKVCache | None,
        start_position: int,
    ):
        _, chunk_len, _ = hidden.shape
        positions = torch.arange(
            start_position,
            start_position + chunk_len,
            device=hidden.device,
            dtype=torch.long,
        )
        q = apply_rope(self._split(self.q_proj(hidden)), positions)
        k_new = apply_rope(self._split(self.k_proj(hidden)), positions)
        v_new = self._split(self.v_proj(hidden))

        if cache is None:
            old_len = 0
            k_all = k_new
            v_all = v_new
            absolute_positions = positions
        else:
            old_len = cache.kv_length
            k_all = torch.cat([cache.key, k_new], dim=-2)
            v_all = torch.cat([cache.value, v_new], dim=-2)
            absolute_positions = torch.cat([cache.absolute_positions, positions], dim=0)

        scores = torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Every retained historical KV entry is earlier than every query in the
        # current chunk. The new chunk itself still uses normal causal order.
        allowed = torch.ones(
            chunk_len,
            old_len + chunk_len,
            device=hidden.device,
            dtype=torch.bool,
        )
        allowed[:, old_len:] = torch.tril(
            torch.ones(chunk_len, chunk_len, device=hidden.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(
            ~allowed.view(1, 1, chunk_len, old_len + chunk_len),
            float("-inf"),
        )
        probs = torch.softmax(scores, dim=-1)
        out = hidden + self.o_proj(self._merge(torch.matmul(probs, v_all)))
        out = out + self.ffn(out)

        # Streaming saliency: newest chunk queries vote over the currently
        # available historical+new KV states.
        saliency = probs.max(dim=2).values.mean(dim=(0, 1)).detach()
        new_cache = StreamingKVCache(
            k_all.detach().clone(),
            v_all.detach().clone(),
            absolute_positions.detach().clone(),
            start_position + chunk_len,
        )
        return out, new_cache, saliency

    def decode_one(self, hidden: torch.Tensor, cache: StreamingKVCache):
        position = torch.tensor(
            [cache.next_position],
            device=hidden.device,
            dtype=torch.long,
        )
        q = apply_rope(self._split(self.q_proj(hidden)), position)
        k_new = apply_rope(self._split(self.k_proj(hidden)), position)
        v_new = self._split(self.v_proj(hidden))
        k_all = torch.cat([cache.key, k_new], dim=-2)
        v_all = torch.cat([cache.value, v_new], dim=-2)
        absolute_positions = torch.cat([cache.absolute_positions, position], dim=0)
        probs = torch.softmax(
            torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(self.head_dim),
            dim=-1,
        )
        out = hidden + self.o_proj(self._merge(torch.matmul(probs, v_all)))
        out = out + self.ffn(out)
        new_cache = StreamingKVCache(
            k_all.detach(),
            v_all.detach(),
            absolute_positions.detach(),
            cache.next_position + 1,
        )
        return out, new_cache, probs.detach()


class StreamingToyStackedDecoder(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        seed: int = 100,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.layers = torch.nn.ModuleList(
            [
                StreamingToyDecoderLayer(d_model, n_heads, seed + i * 17)
                for i in range(n_layers)
            ]
        )

    def prefill(self, hidden: torch.Tensor):
        x = hidden
        caches = []
        saliencies = []
        for layer in self.layers:
            x, cache, saliency = layer.prefill(x)
            caches.append(cache)
            saliencies.append(saliency)
        return x, caches, saliencies

    def prefill_chunk(
        self,
        hidden: torch.Tensor,
        caches: list[StreamingKVCache] | None,
        start_position: int,
    ):
        x = hidden
        if caches is None:
            caches = [None] * self.n_layers
        new_caches = []
        saliencies = []
        for layer, cache in zip(self.layers, caches):
            x, new_cache, saliency = layer.prefill_chunk(x, cache, start_position)
            new_caches.append(new_cache)
            saliencies.append(saliency)
        return x, new_caches, saliencies

    def decode_one(self, hidden: torch.Tensor, caches: list[StreamingKVCache]):
        x = hidden
        new_caches = []
        probs = []
        for layer, cache in zip(self.layers, caches):
            x, new_cache, p = layer.decode_one(x, cache)
            new_caches.append(new_cache)
            probs.append(p)
        return x, new_caches, probs


def total_streaming_cache_bytes(caches: list[StreamingKVCache]) -> int:
    return sum(cache.bytes for cache in caches)
