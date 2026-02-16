from __future__ import annotations

import math

import torch
from torch import nn

from .geometry import build_rna_local_frames
from .sparse_graph import build_sparse_radius_graph, compute_chain_relative_features, lookup_sparse_pair_bias


def _segment_softmax(logits: torch.Tensor, src: torch.Tensor, node_count: int) -> torch.Tensor:
    logits_work = logits.to(dtype=torch.float32)
    heads = int(logits.shape[0])
    edge_count = int(logits.shape[1])
    src_expand = src.unsqueeze(0).expand(heads, edge_count)
    if hasattr(torch.Tensor, "scatter_reduce_"):
        max_per_node = torch.full((heads, node_count), float("-inf"), dtype=logits_work.dtype, device=logits.device)
        max_per_node.scatter_reduce_(1, src_expand, logits_work, reduce="amax", include_self=True)
        centered = logits_work - max_per_node.gather(1, src_expand)
        exp_vals = torch.exp(centered)
        sum_per_node = torch.zeros((heads, node_count), dtype=logits_work.dtype, device=logits.device)
        sum_per_node.scatter_add_(1, src_expand, exp_vals)
        denom = torch.clamp(sum_per_node.gather(1, src_expand), min=1e-9)
        return (exp_vals / denom).to(dtype=logits.dtype)
    out = torch.zeros_like(logits_work)
    for head in range(heads):
        logits_head = logits_work[head]
        for node in range(node_count):
            mask = src == node
            if bool(mask.any()):
                out[head, mask] = torch.softmax(logits_head[mask], dim=0)
    return out.to(dtype=logits.dtype)

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
        self.bpp_bias = nn.Linear(1, heads)
        self.msa_bias = nn.Linear(1, heads)
        self.chem_bias = nn.Linear(1, heads)
        self.rel_bias = nn.Linear(1, heads)
        self.chain_break_bias = nn.Linear(1, heads)
        self.orientation_bias = nn.Linear(3, heads)
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        *,
        src: torch.Tensor,
        dst: torch.Tensor,
        distances: torch.Tensor,
        bpp_edge_bias: torch.Tensor,
        msa_edge_bias: torch.Tensor,
        chem_edge_bias: torch.Tensor,
        relative_offset: torch.Tensor,
        chain_break_mask: torch.Tensor,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(h.shape[0])
        if frames.ndim != 3 or int(frames.shape[0]) != node_count or int(frames.shape[1]) != 3 or int(frames.shape[2]) != 3:
            raise ValueError(f"frames com shape invalido para IpaBlock: {tuple(frames.shape)}")
        q = self.query(h).view(node_count, self.heads, self.head_dim)
        k = self.key(h).view(node_count, self.heads, self.head_dim)
        v = self.value(h).view(node_count, self.heads, self.head_dim)
        q_src = q.index_select(0, src)
        k_dst = k.index_select(0, dst)
        v_dst = v.index_select(0, dst)
        edge_delta = x.index_select(0, dst) - x.index_select(0, src)
        src_frames = frames.index_select(0, src)
        edge_local_delta = torch.einsum("eij,ej->ei", src_frames.transpose(1, 2), edge_delta)
        edge_local_norm = torch.clamp(torch.linalg.norm(edge_local_delta, dim=-1, keepdim=True), min=1e-6)
        edge_local_unit = edge_local_delta / edge_local_norm
        logits = torch.einsum("ehd,ehd->he", q_src, k_dst) / math.sqrt(float(self.head_dim))
        bias = self.dist_bias((distances * distances).unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        bpp_term = self.bpp_bias(bpp_edge_bias.unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        msa_term = self.msa_bias(msa_edge_bias.unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        chem_term = self.chem_bias(chem_edge_bias.unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        rel_term = self.rel_bias(relative_offset.unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        chain_term = self.chain_break_bias(chain_break_mask.unsqueeze(-1).to(dtype=h.dtype)).transpose(0, 1)
        orientation_term = (0.1 * torch.tanh(self.orientation_bias(edge_local_unit.to(dtype=h.dtype)))).transpose(0, 1)
        weights = _segment_softmax(
            logits=logits - bias + bpp_term + msa_term + chem_term + rel_term + chain_term + orientation_term,
            src=src,
            node_count=node_count,
        )
        weighted_v = v_dst * weights.transpose(0, 1).unsqueeze(-1).to(dtype=v_dst.dtype)
        h_update = torch.zeros((node_count, self.heads, self.head_dim), dtype=h.dtype, device=h.device)
        h_update.index_add_(0, src, weighted_v.to(dtype=h.dtype))
        h_update = h_update.reshape(node_count, self.hidden_dim)
        h_out = h + self.out(h_update)
        weights_mean = weights.mean(dim=0)
        displacement_local = torch.zeros((node_count, 3), dtype=x.dtype, device=x.device)
        displacement_local.index_add_(
            0,
            src,
            edge_local_delta * weights_mean.unsqueeze(-1).to(dtype=x.dtype),
        )
        displacement_global_raw = torch.zeros((node_count, 3), dtype=x.dtype, device=x.device)
        displacement_global_raw.index_add_(
            0,
            src,
            edge_delta * weights_mean.unsqueeze(-1).to(dtype=x.dtype),
        )
        displacement_global = torch.einsum("nij,nj->ni", frames, displacement_local)
        displacement_global = 0.5 * displacement_global + 0.5 * displacement_global_raw
        x_out = x + (self.coord_gate(h_out) * displacement_global)
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

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        *,
        bpp_pair_src: torch.Tensor,
        bpp_pair_dst: torch.Tensor,
        bpp_pair_prob: torch.Tensor,
        msa_pair_src: torch.Tensor,
        msa_pair_dst: torch.Tensor,
        msa_pair_prob: torch.Tensor,
        base_features: torch.Tensor,
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        chem_exposure: torch.Tensor,
        chain_break_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_features.ndim != 2 or int(base_features.shape[0]) != int(node_features.shape[0]) or int(base_features.shape[1]) < 4:
            raise ValueError(f"base_features com shape invalido para IpaBackbone: {tuple(base_features.shape)}")
        h = self.input_proj(node_features)
        x = coords
        base_features_layer = base_features.to(device=x.device, dtype=x.dtype)
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
            bpp_edge_bias = lookup_sparse_pair_bias(
                src=graph.src,
                dst=graph.dst,
                pair_src=bpp_pair_src.to(device=graph.src.device),
                pair_dst=bpp_pair_dst.to(device=graph.src.device),
                pair_prob=bpp_pair_prob.to(device=graph.src.device),
                n_nodes=int(h.shape[0]),
            )
            msa_edge_bias = lookup_sparse_pair_bias(
                src=graph.src,
                dst=graph.dst,
                pair_src=msa_pair_src.to(device=graph.src.device),
                pair_dst=msa_pair_dst.to(device=graph.src.device),
                pair_prob=msa_pair_prob.to(device=graph.src.device),
                n_nodes=int(h.shape[0]),
            )
            chem_values = chem_exposure.to(device=graph.src.device, dtype=torch.float32)
            chem_edge_bias = 0.5 * (chem_values.index_select(0, graph.src) + chem_values.index_select(0, graph.dst))
            relative_offset, chain_break_mask = compute_chain_relative_features(
                src=graph.src,
                dst=graph.dst,
                residue_index=residue_index.to(device=graph.src.device),
                chain_index=chain_index.to(device=graph.src.device),
                chain_break_offset=int(chain_break_offset),
            )
            frames, _p_proxy, _c4_proxy, _n_proxy = build_rna_local_frames(c1_coords=x, base_features=base_features_layer)
            h, x = layer(
                h,
                x,
                src=graph.src,
                dst=graph.dst,
                distances=graph.distances,
                bpp_edge_bias=bpp_edge_bias,
                msa_edge_bias=msa_edge_bias,
                chem_edge_bias=chem_edge_bias,
                relative_offset=relative_offset,
                chain_break_mask=chain_break_mask,
                frames=frames,
            )
        return self.norm(h), x
