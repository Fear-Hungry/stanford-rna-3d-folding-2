from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .geometry import build_rna_local_frames
from .geometry import rotation_matrix_from_6d
from .sparse_graph import build_sparse_radius_graph, compute_chain_relative_features, lookup_sparse_pair_bias


def _scatter_add_per_src(*, values: torch.Tensor, src: torch.Tensor, out: torch.Tensor) -> None:
    heads = int(values.shape[0])
    edge_count = int(values.shape[1])
    src_expand = src.unsqueeze(0).expand(heads, edge_count)
    if hasattr(out, "scatter_add_"):
        out.scatter_add_(1, src_expand, values)
        return
    for head in range(heads):
        out[head].index_add_(0, src, values[head])


def _scatter_amax_per_src(*, values: torch.Tensor, src: torch.Tensor, out: torch.Tensor) -> None:
    heads = int(values.shape[0])
    edge_count = int(values.shape[1])
    src_expand = src.unsqueeze(0).expand(heads, edge_count)
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out.scatter_reduce_(1, src_expand, values, reduce="amax", include_self=True)
        return
    for head in range(heads):
        for edge_idx in range(edge_count):
            node_idx = int(src[edge_idx].item())
            out[head, node_idx] = torch.maximum(out[head, node_idx], values[head, edge_idx])


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
    def __init__(self, hidden_dim: int, heads: int, edge_chunk_size: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = int(hidden_dim) // int(heads)
        self.edge_chunk_size = int(edge_chunk_size)
        if self.head_dim <= 0:
            raise ValueError("hidden_dim/heads invalid")
        if self.edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size invalid")
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
        self.base_orientation_bias = nn.Linear(3, heads)
        self.base_frame_delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def _chunk_logits(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        x: torch.Tensor,
        src_chunk: torch.Tensor,
        dst_chunk: torch.Tensor,
        distances_chunk: torch.Tensor,
        bpp_edge_bias_chunk: torch.Tensor,
        msa_edge_bias_chunk: torch.Tensor,
        chem_edge_bias_chunk: torch.Tensor,
        relative_offset_chunk: torch.Tensor,
        chain_break_mask_chunk: torch.Tensor,
        frames: torch.Tensor,
        base_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_src = q.index_select(0, src_chunk).to(dtype=torch.float32)
        k_dst = k.index_select(0, dst_chunk).to(dtype=torch.float32)
        logits = torch.einsum("ehd,ehd->he", q_src, k_dst) / math.sqrt(float(self.head_dim))
        edge_delta = x.index_select(0, dst_chunk) - x.index_select(0, src_chunk)
        src_frames = frames.index_select(0, src_chunk)
        edge_local_delta = torch.einsum("eij,ej->ei", src_frames.transpose(1, 2), edge_delta)
        edge_local_norm = torch.clamp(torch.linalg.norm(edge_local_delta, dim=-1, keepdim=True), min=1e-6)
        edge_local_unit = edge_local_delta / edge_local_norm
        src_base_frames = base_frames.index_select(0, src_chunk)
        base_local_delta = torch.einsum("eij,ej->ei", src_base_frames.transpose(1, 2), edge_delta)
        base_local_norm = torch.clamp(torch.linalg.norm(base_local_delta, dim=-1, keepdim=True), min=1e-6)
        base_local_unit = base_local_delta / base_local_norm

        autocast_context = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_context:
            bias = self.dist_bias((distances_chunk.to(dtype=torch.float32) * distances_chunk.to(dtype=torch.float32)).unsqueeze(-1)).transpose(0, 1).to(dtype=torch.float32)
            bpp_term = self.bpp_bias(bpp_edge_bias_chunk.to(dtype=torch.float32).unsqueeze(-1)).transpose(0, 1).to(dtype=torch.float32)
            msa_term = self.msa_bias(msa_edge_bias_chunk.to(dtype=torch.float32).unsqueeze(-1)).transpose(0, 1).to(dtype=torch.float32)
            chem_term = self.chem_bias(chem_edge_bias_chunk.to(dtype=torch.float32).unsqueeze(-1)).transpose(0, 1).to(dtype=torch.float32)
            rel_term = self.rel_bias(relative_offset_chunk.to(dtype=torch.float32).unsqueeze(-1)).transpose(0, 1).to(dtype=torch.float32)
            chain_term = self.chain_break_bias(chain_break_mask_chunk.to(dtype=torch.float32).unsqueeze(-1)).transpose(0, 1).to(dtype=torch.float32)
            orientation_term = (0.1 * torch.tanh(self.orientation_bias(edge_local_unit.to(dtype=torch.float32)))).transpose(0, 1).to(dtype=torch.float32)
            base_orientation_term = (0.1 * torch.tanh(self.base_orientation_bias(base_local_unit.to(dtype=torch.float32)))).transpose(0, 1).to(dtype=torch.float32)
        logits = logits - bias + bpp_term + msa_term + chem_term + rel_term + chain_term + orientation_term + base_orientation_term
        return logits, edge_delta, edge_local_delta

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
        # Base frame: rotation relative to ribose frame (SO(3) per residue).
        autocast_context = (
            torch.autocast(device_type=h.device.type, enabled=False)
            if h.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_context:
            delta_rot = rotation_matrix_from_6d(self.base_frame_delta(h.to(dtype=torch.float32)).to(dtype=torch.float32)).to(dtype=frames.dtype)
            q = self.query(h.to(dtype=torch.float32)).view(node_count, self.heads, self.head_dim)
            k = self.key(h.to(dtype=torch.float32)).view(node_count, self.heads, self.head_dim)
            v = self.value(h.to(dtype=torch.float32)).view(node_count, self.heads, self.head_dim)
        base_frames = torch.einsum("nij,njk->nik", frames, delta_rot)
        edge_count = int(src.shape[0])
        chunk_size = min(int(self.edge_chunk_size), edge_count)

        max_per_node = torch.full((self.heads, node_count), float("-inf"), dtype=torch.float32, device=h.device)
        for start in range(0, edge_count, chunk_size):
            end = min(edge_count, start + chunk_size)
            src_chunk = src[start:end]
            dst_chunk = dst[start:end]
            logits_chunk, _edge_delta, _edge_local_delta = self._chunk_logits(
                q=q,
                k=k,
                x=x,
                src_chunk=src_chunk,
                dst_chunk=dst_chunk,
                distances_chunk=distances[start:end],
                bpp_edge_bias_chunk=bpp_edge_bias[start:end],
                msa_edge_bias_chunk=msa_edge_bias[start:end],
                chem_edge_bias_chunk=chem_edge_bias[start:end],
                relative_offset_chunk=relative_offset[start:end],
                chain_break_mask_chunk=chain_break_mask[start:end],
                frames=frames,
                base_frames=base_frames,
            )
            _scatter_amax_per_src(values=logits_chunk, src=src_chunk, out=max_per_node)

        sum_per_node = torch.zeros((self.heads, node_count), dtype=torch.float32, device=h.device)
        for start in range(0, edge_count, chunk_size):
            end = min(edge_count, start + chunk_size)
            src_chunk = src[start:end]
            dst_chunk = dst[start:end]
            src_expand = src_chunk.unsqueeze(0).expand(self.heads, int(src_chunk.numel()))
            logits_chunk, _edge_delta, _edge_local_delta = self._chunk_logits(
                q=q,
                k=k,
                x=x,
                src_chunk=src_chunk,
                dst_chunk=dst_chunk,
                distances_chunk=distances[start:end],
                bpp_edge_bias_chunk=bpp_edge_bias[start:end],
                msa_edge_bias_chunk=msa_edge_bias[start:end],
                chem_edge_bias_chunk=chem_edge_bias[start:end],
                relative_offset_chunk=relative_offset[start:end],
                chain_break_mask_chunk=chain_break_mask[start:end],
                frames=frames,
                base_frames=base_frames,
            )
            centered = logits_chunk - max_per_node.gather(1, src_expand)
            exp_vals = torch.exp(centered)
            _scatter_add_per_src(values=exp_vals, src=src_chunk, out=sum_per_node)

        h_update = torch.zeros((node_count, self.heads, self.head_dim), dtype=h.dtype, device=h.device)
        displacement_local = torch.zeros((node_count, 3), dtype=x.dtype, device=x.device)
        displacement_global_raw = torch.zeros((node_count, 3), dtype=x.dtype, device=x.device)
        for start in range(0, edge_count, chunk_size):
            end = min(edge_count, start + chunk_size)
            src_chunk = src[start:end]
            dst_chunk = dst[start:end]
            src_expand = src_chunk.unsqueeze(0).expand(self.heads, int(src_chunk.numel()))
            logits_chunk, edge_delta_chunk, edge_local_delta_chunk = self._chunk_logits(
                q=q,
                k=k,
                x=x,
                src_chunk=src_chunk,
                dst_chunk=dst_chunk,
                distances_chunk=distances[start:end],
                bpp_edge_bias_chunk=bpp_edge_bias[start:end],
                msa_edge_bias_chunk=msa_edge_bias[start:end],
                chem_edge_bias_chunk=chem_edge_bias[start:end],
                relative_offset_chunk=relative_offset[start:end],
                chain_break_mask_chunk=chain_break_mask[start:end],
                frames=frames,
                base_frames=base_frames,
            )
            centered = logits_chunk - max_per_node.gather(1, src_expand)
            exp_vals = torch.exp(centered)
            denom = torch.clamp(sum_per_node.gather(1, src_expand), min=1e-9)
            weights_chunk = exp_vals / denom

            v_dst_chunk = v.index_select(0, dst_chunk)
            weighted_v = v_dst_chunk * weights_chunk.transpose(0, 1).unsqueeze(-1).to(dtype=v_dst_chunk.dtype)
            h_update.index_add_(0, src_chunk, weighted_v.to(dtype=h.dtype))

            weights_mean = weights_chunk.mean(dim=0).to(dtype=x.dtype)
            displacement_local.index_add_(0, src_chunk, edge_local_delta_chunk * weights_mean.unsqueeze(-1))
            displacement_global_raw.index_add_(0, src_chunk, edge_delta_chunk * weights_mean.unsqueeze(-1))

        h_update = h_update.reshape(node_count, self.hidden_dim)
        h_out = h + self.out(h_update)
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
        ipa_edge_chunk_size: int,
        graph_pair_edges: str,
        graph_pair_min_prob: float,
        graph_pair_max_per_node: int,
        stage: str,
        location: str,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.stage = str(stage)
        self.location = str(location)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.graph_backend = str(graph_backend)
        self.radius_angstrom = float(radius_angstrom)
        self.max_neighbors = int(max_neighbors)
        self.graph_chunk_size = int(graph_chunk_size)
        self.ipa_edge_chunk_size = int(ipa_edge_chunk_size)
        self.graph_pair_edges = str(graph_pair_edges).strip().lower()
        self.graph_pair_min_prob = float(graph_pair_min_prob)
        self.graph_pair_max_per_node = int(graph_pair_max_per_node)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                IpaBlock(hidden_dim=hidden_dim, heads=heads, edge_chunk_size=self.ipa_edge_chunk_size)
                for _ in range(int(num_layers))
            ]
        )
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
        bpp_pair_src_dev = bpp_pair_src.to(device=x.device)
        bpp_pair_dst_dev = bpp_pair_dst.to(device=x.device)
        bpp_pair_prob_dev = bpp_pair_prob.to(device=x.device)
        msa_pair_src_dev = msa_pair_src.to(device=x.device)
        msa_pair_dst_dev = msa_pair_dst.to(device=x.device)
        msa_pair_prob_dev = msa_pair_prob.to(device=x.device)
        for layer in self.layers:
            pair_src = None
            pair_dst = None
            pair_prob = None
            pair_min_prob = 0.0
            pair_max_per_node = 0
            if self.graph_pair_edges == "bpp":
                pair_src = bpp_pair_src_dev
                pair_dst = bpp_pair_dst_dev
                pair_prob = bpp_pair_prob_dev
                pair_min_prob = float(self.graph_pair_min_prob)
                pair_max_per_node = int(self.graph_pair_max_per_node)
            graph = build_sparse_radius_graph(
                coords=x,
                radius_angstrom=self.radius_angstrom,
                max_neighbors=self.max_neighbors,
                backend=self.graph_backend,
                chunk_size=self.graph_chunk_size,
                pair_src=pair_src,
                pair_dst=pair_dst,
                pair_prob=pair_prob,
                pair_min_prob=pair_min_prob,
                pair_max_per_node=pair_max_per_node,
                stage=self.stage,
                location=self.location,
            )
            bpp_edge_bias = lookup_sparse_pair_bias(
                src=graph.src,
                dst=graph.dst,
                pair_src=bpp_pair_src_dev.to(device=graph.src.device),
                pair_dst=bpp_pair_dst_dev.to(device=graph.src.device),
                pair_prob=bpp_pair_prob_dev.to(device=graph.src.device),
                n_nodes=int(h.shape[0]),
            )
            msa_edge_bias = lookup_sparse_pair_bias(
                src=graph.src,
                dst=graph.dst,
                pair_src=msa_pair_src_dev.to(device=graph.src.device),
                pair_dst=msa_pair_dst_dev.to(device=graph.src.device),
                pair_prob=msa_pair_prob_dev.to(device=graph.src.device),
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
            if self.use_gradient_checkpointing and self.training:
                def _layer_forward(
                    h_in: torch.Tensor,
                    x_in: torch.Tensor,
                    _layer: IpaBlock = layer,
                    _src: torch.Tensor = graph.src,
                    _dst: torch.Tensor = graph.dst,
                    _distances: torch.Tensor = graph.distances,
                    _bpp_edge_bias: torch.Tensor = bpp_edge_bias,
                    _msa_edge_bias: torch.Tensor = msa_edge_bias,
                    _chem_edge_bias: torch.Tensor = chem_edge_bias,
                    _relative_offset: torch.Tensor = relative_offset,
                    _chain_break_mask: torch.Tensor = chain_break_mask,
                    _frames: torch.Tensor = frames,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return _layer(
                        h_in,
                        x_in,
                        src=_src,
                        dst=_dst,
                        distances=_distances,
                        bpp_edge_bias=_bpp_edge_bias,
                        msa_edge_bias=_msa_edge_bias,
                        chem_edge_bias=_chem_edge_bias,
                        relative_offset=_relative_offset,
                        chain_break_mask=_chain_break_mask,
                        frames=_frames,
                    )

                h, x = checkpoint(_layer_forward, h, x, use_reentrant=False)
            else:
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
