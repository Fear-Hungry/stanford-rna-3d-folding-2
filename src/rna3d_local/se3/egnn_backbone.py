from __future__ import annotations

import torch
from torch import nn

from .sparse_graph import build_sparse_radius_graph, compute_chain_relative_features, lookup_sparse_pair_bias

class EgnnLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear((hidden_dim * 2) + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.h_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
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
        bpp_bias: torch.Tensor,
        msa_bias: torch.Tensor,
        chem_bias: torch.Tensor,
        relative_offset: torch.Tensor,
        chain_break_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(h.shape[0])
        h_src = h.index_select(0, src)
        h_dst = h.index_select(0, dst)
        diff = x.index_select(0, src) - x.index_select(0, dst)
        dist2 = (distances * distances).unsqueeze(-1).to(dtype=h.dtype)
        m_ij = self.msg_mlp(
            torch.cat(
                [
                    h_src,
                    h_dst,
                    dist2,
                    bpp_bias.unsqueeze(-1).to(dtype=h.dtype),
                    msa_bias.unsqueeze(-1).to(dtype=h.dtype),
                    chem_bias.unsqueeze(-1).to(dtype=h.dtype),
                    relative_offset.unsqueeze(-1).to(dtype=h.dtype),
                    chain_break_mask.unsqueeze(-1).to(dtype=h.dtype),
                ],
                dim=-1,
            )
        )
        counts = torch.zeros((node_count, 1), dtype=h.dtype, device=h.device)
        counts.index_add_(0, src, torch.ones((int(src.numel()), 1), dtype=h.dtype, device=h.device))
        m_i = torch.zeros((node_count, h.shape[-1]), dtype=h.dtype, device=h.device)
        m_i.index_add_(0, src, m_ij)
        m_i = m_i / torch.clamp(counts, min=1.0)
        h_out = h + self.h_mlp(torch.cat([h, m_i], dim=-1))
        coef = self.coord_mlp(m_ij)
        delta = torch.zeros((node_count, 3), dtype=x.dtype, device=x.device)
        delta.index_add_(0, src, diff * coef.to(dtype=x.dtype))
        delta = delta / torch.clamp(counts.to(dtype=x.dtype), min=1.0)
        x_out = x + delta
        return h_out, x_out


class EgnnBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
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
        self.layers = nn.ModuleList([EgnnLayer(hidden_dim=hidden_dim) for _ in range(int(num_layers))])
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
        residue_index: torch.Tensor,
        chain_index: torch.Tensor,
        chem_exposure: torch.Tensor,
        chain_break_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            bpp_bias = lookup_sparse_pair_bias(
                src=graph.src,
                dst=graph.dst,
                pair_src=bpp_pair_src.to(device=graph.src.device),
                pair_dst=bpp_pair_dst.to(device=graph.src.device),
                pair_prob=bpp_pair_prob.to(device=graph.src.device),
                n_nodes=int(h.shape[0]),
            )
            msa_bias = lookup_sparse_pair_bias(
                src=graph.src,
                dst=graph.dst,
                pair_src=msa_pair_src.to(device=graph.src.device),
                pair_dst=msa_pair_dst.to(device=graph.src.device),
                pair_prob=msa_pair_prob.to(device=graph.src.device),
                n_nodes=int(h.shape[0]),
            )
            chem_values = chem_exposure.to(device=graph.src.device, dtype=torch.float32)
            chem_bias = 0.5 * (chem_values.index_select(0, graph.src) + chem_values.index_select(0, graph.dst))
            relative_offset, chain_break_mask = compute_chain_relative_features(
                src=graph.src,
                dst=graph.dst,
                residue_index=residue_index.to(device=graph.src.device),
                chain_index=chain_index.to(device=graph.src.device),
                chain_break_offset=int(chain_break_offset),
            )
            h, x = layer(
                h,
                x,
                src=graph.src,
                dst=graph.dst,
                distances=graph.distances,
                bpp_bias=bpp_bias,
                msa_bias=msa_bias,
                chem_bias=chem_bias,
                relative_offset=relative_offset,
                chain_break_mask=chain_break_mask,
            )
        return self.norm(h), x
