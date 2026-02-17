from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _FlashAttentionBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = int(hidden_dim) // int(heads)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm_attn(x)
        qkv = self.qkv(h).view(h.shape[0], h.shape[1], 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        if bool(q.is_cuda):
            sdp_context = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        else:
            sdp_context = nullcontext()
        try:
            with sdp_context:
                attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "flash attention indisponivel para SequenceTower; use tower_type=mamba_like para evitar OOM em sequencias longas"
            ) from exc
        attn = attn.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[1], self.hidden_dim)
        x = residual + self.out(attn)
        x = x + self.ff(self.norm_ff(x))
        return x


class _MambaLikeBlock(nn.Module):
    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.norm_state = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.state_decay = nn.Parameter(torch.zeros(hidden_dim))
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm_state(x)
        u = self.in_proj(hidden)
        gate = torch.sigmoid(self.gate_proj(hidden))
        decay = torch.sigmoid(self.state_decay).to(dtype=x.dtype, device=x.device)
        state_f = torch.zeros((x.shape[1],), dtype=x.dtype, device=x.device)
        forward_out = torch.zeros_like(u)
        for index in range(int(x.shape[0])):
            state_f = (decay * state_f) + u[index]
            forward_out[index] = state_f * gate[index]

        state_b = torch.zeros((x.shape[1],), dtype=x.dtype, device=x.device)
        backward_out = torch.zeros_like(u)
        for index in reversed(range(int(x.shape[0]))):
            state_b = (decay * state_b) + u[index]
            backward_out[index] = state_b * gate[index]

        state_out = 0.5 * (forward_out + backward_out)
        x = x + self.out_proj(state_out)
        x = x + self.ff(self.norm_ff(x))
        return x


class SequenceTower(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        tower_type: str,
        heads: int,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.tower_type = str(tower_type).strip().lower()
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        if self.tower_type == "flash":
            self.layers = nn.ModuleList(
                [
                    _FlashAttentionBlock(hidden_dim=int(hidden_dim), heads=int(heads))
                    for _ in range(int(num_layers))
                ]
            )
        elif self.tower_type == "mamba_like":
            self.layers = nn.ModuleList([_MambaLikeBlock(hidden_dim=int(hidden_dim)) for _ in range(int(num_layers))])
        else:
            raise ValueError(f"sequence_tower invalido: {tower_type}")

    def _forward_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        if self.tower_type == "flash":
            x = self.input_proj(node_features).unsqueeze(0)
            for layer in self.layers:
                x = self._forward_layer(layer, x)
            return self.norm(x.squeeze(0))
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = self._forward_layer(layer, x)
        return self.norm(x)
