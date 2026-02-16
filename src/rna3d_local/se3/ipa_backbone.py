from __future__ import annotations

import math

import torch
from torch import nn

from .sparse_graph import build_sparse_radius_graph


def _segment_softmax(logits: torch.Tensor, src: torch.Tensor, node_count: int) -> torch.Tensor:
    heads = int(logits.shape[0])
    edge_count = int(logits.shape[1])
    src_expand = src.unsqueeze(0).expand(heads, edge_count)
    if hasattr(torch.Tensor, "scatter_reduce_"):
        max_per_node = torch.full((heads, node_count), float("-inf"), dtype=logits.dtype, device=logits.device)
        max_per_node.scatter_reduce_(1, src_expand, logits, reduce="amax", include_self=True)
        centered = logits - max_per_node.gather(1, src_expand)
        exp_vals = torch.exp(centered)
        sum_per_node = torch.zeros((heads, node_count), dtype=logits.dtype, device=logits.device)
        sum_per_node.scatter_add_(1, src_expand, exp_vals)
        denom = torch.clamp(sum_per_node.gather(1, src_expand), min=1e-9)
        return exp_vals / denom
    out = torch.zeros_like(logits)
    for head in range(heads):
        logits_head = logits[head]
        for node in range(node_count):
            mask = src == node
            if bool(mask.any()):
                out[head, mask] = torch.softmax(logits_head[mask], dim=0)
    return out

class IpaBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = int(hidden_dim) // int(heads)
        if self.head_dim <= 0:
            raise ValueError("hidden_dim/heads invalid")
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dist_bias = nn.Linear(1, heads)
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, *, src: torch.Tensor, dst: torch.Tensor, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(h.shape[0])
        q = self.query(h).view(node_count, self.heads, self.head_dim)
        k = self.key(h).view(node_count, self.heads, self.head_dim)
        v = self.value(h).view(node_count, self.heads, self.head_dim)
        q_src = q.index_select(0, src)
        k_dst = k.index_select(0, dst)
        v_dst = v.index_select(0, dst)
        logits = torch.einsum("ehd,ehd->he", q_src, k_dst) / math.sqrt(float(self.head_dim))
        bias = self.dist_bias((distances * distances).unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        weights = _segment_softmax(logits=logits - bias, src=src, node_count=node_count)
        weighted_v = v_dst * weights.transpose(0, 1).unsqueeze(-1).to(dtype=v_dst.dtype)
        h_update = torch.zeros((node_count, self.heads, self.head_dim), dtype=h.dtype, device=h.device)
        h_update.index_add_(0, src, weighted_v.to(dtype=h.dtype))
        h_update = h_update.reshape(node_count, self.hidden_dim)
        h_out = h + self.out(h_update)
        weights_mean = weights.mean(dim=0)
        displacement = torch.zeros((node_count, 3), dtype=x.dtype, device=x.device)
        displacement.index_add_(
            0,
            src,
            (x.index_select(0, dst) - x.index_select(0, src)) * weights_mean.unsqueeze(-1).to(dtype=x.dtype),
        )
        x_out = x + (self.coord_gate(h_out) * displacement)
        return h_out, x_out


class IpaBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        graph_backend: str,
        radius_angstrom: float,
        max_neighbors: int,
        graph_chunk_size: int,
        stage: str,
        location: str,
    ) -> None:
        super().__init__()
        self.stage = str(stage)
        self.location = str(location)
        self.graph_backend = str(graph_backend)
        self.radius_angstrom = float(radius_angstrom)
        self.max_neighbors = int(max_neighbors)
        self.graph_chunk_size = int(graph_chunk_size)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([IpaBlock(hidden_dim=hidden_dim, heads=heads) for _ in range(int(num_layers))])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_features)
        x = coords
        for layer in self.layers:
            graph = build_sparse_radius_graph(
                coords=x,
                radius_angstrom=self.radius_angstrom,
                max_neighbors=self.max_neighbors,
                backend=self.graph_backend,
                chunk_size=self.graph_chunk_size,
                stage=self.stage,
                location=self.location,
            )
            h, x = layer(h, x, src=graph.src, dst=graph.dst, distances=graph.distances)
        return self.norm(h), x
