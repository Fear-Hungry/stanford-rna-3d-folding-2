from __future__ import annotations

from dataclasses import dataclass

import torch

from ..errors import raise_error


@dataclass(frozen=True)
class SparseRadiusGraph:
    src: torch.Tensor
    dst: torch.Tensor
    distances: torch.Tensor
    adjacency: torch.Tensor


def _enforce_min_degree(
    *,
    src: torch.Tensor,
    n_nodes: int,
    stage: str,
    location: str,
) -> None:
    degree = torch.zeros((n_nodes,), dtype=torch.int32, device=src.device)
    degree.index_add_(0, src, torch.ones_like(src, dtype=torch.int32))
    isolated = torch.nonzero(degree == 0, as_tuple=False).flatten()
    if int(isolated.numel()) > 0:
        examples = [str(int(item) + 1) for item in isolated[:8].tolist()]
        raise_error(
            stage,
            location,
            "grafo 3d esparso com residuos isolados; ajuste radius_angstrom/max_neighbors",
            impact=str(int(isolated.numel())),
            examples=examples,
        )


def _build_torch_sparse_edges(
    *,
    coords: torch.Tensor,
    radius_angstrom: float,
    max_neighbors: int,
    chunk_size: int,
    stage: str,
    location: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_nodes = int(coords.shape[0])
    node_index = torch.arange(n_nodes, device=coords.device)
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    dist_parts: list[torch.Tensor] = []
    for start in range(0, n_nodes, int(chunk_size)):
        end = min(n_nodes, start + int(chunk_size))
        local = coords[start:end]
        distances_chunk = torch.cdist(local, coords)
        for local_row in range(int(end - start)):
            src_idx = int(start + local_row)
            distances_row = distances_chunk[local_row]
            mask = (distances_row <= float(radius_angstrom)) & (node_index != src_idx)
            neighbors = torch.nonzero(mask, as_tuple=False).flatten()
            if int(neighbors.numel()) == 0:
                continue
            neighbor_dist = distances_row.index_select(0, neighbors)
            if int(neighbors.numel()) > int(max_neighbors):
                keep = torch.topk(neighbor_dist, k=int(max_neighbors), largest=False).indices
                neighbors = neighbors.index_select(0, keep)
                neighbor_dist = neighbor_dist.index_select(0, keep)
            src_parts.append(torch.full((int(neighbors.numel()),), src_idx, dtype=torch.long, device=coords.device))
            dst_parts.append(neighbors.to(dtype=torch.long, device=coords.device))
            dist_parts.append(neighbor_dist.to(dtype=torch.float32, device=coords.device))
    if not src_parts:
        raise_error(
            stage,
            location,
            "grafo 3d esparso sem arestas; ajuste radius_angstrom/max_neighbors",
            impact=str(n_nodes),
            examples=[f"radius={radius_angstrom}", f"max_neighbors={max_neighbors}"],
        )
    src = torch.cat(src_parts, dim=0)
    dst = torch.cat(dst_parts, dim=0)
    distances = torch.cat(dist_parts, dim=0)
    return src, dst, distances


def _build_torch_geometric_edges(
    *,
    coords: torch.Tensor,
    radius_angstrom: float,
    max_neighbors: int,
    stage: str,
    location: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torch_geometric.nn import radius_graph
    except Exception as exc:
        raise_error(
            stage,
            location,
            "graph_backend=torch_geometric indisponivel",
            impact="1",
            examples=[f"{type(exc).__name__}:{exc}"],
        )
    try:
        edge_index = radius_graph(coords, r=float(radius_angstrom), loop=False, max_num_neighbors=int(max_neighbors))
    except Exception as exc:
        raise_error(
            stage,
            location,
            "radius_graph indisponivel no backend torch_geometric",
            impact="1",
            examples=[f"{type(exc).__name__}:{exc}"],
        )
    if int(edge_index.numel()) == 0:
        raise_error(
            stage,
            location,
            "radius_graph retornou zero arestas",
            impact=str(int(coords.shape[0])),
            examples=[f"radius={radius_angstrom}", f"max_neighbors={max_neighbors}"],
        )
    src = edge_index[0].to(dtype=torch.long, device=coords.device)
    dst = edge_index[1].to(dtype=torch.long, device=coords.device)
    distances = torch.linalg.norm(coords.index_select(0, src) - coords.index_select(0, dst), dim=-1).to(dtype=torch.float32)
    return src, dst, distances


def build_sparse_radius_graph(
    *,
    coords: torch.Tensor,
    radius_angstrom: float,
    max_neighbors: int,
    backend: str,
    chunk_size: int,
    stage: str,
    location: str,
) -> SparseRadiusGraph:
    if coords.ndim != 2 or int(coords.shape[1]) != 3:
        raise_error(stage, location, "coords com shape invalido para radius graph", impact="1", examples=[str(tuple(coords.shape))])
    n_nodes = int(coords.shape[0])
    if n_nodes <= 1:
        raise_error(stage, location, "coords insuficientes para construir grafo", impact="1", examples=[str(n_nodes)])
    if radius_angstrom <= 0:
        raise_error(stage, location, "radius_angstrom invalido (<=0)", impact="1", examples=[str(radius_angstrom)])
    if max_neighbors <= 0:
        raise_error(stage, location, "max_neighbors invalido (<=0)", impact="1", examples=[str(max_neighbors)])
    if chunk_size <= 0:
        raise_error(stage, location, "graph_chunk_size invalido (<=0)", impact="1", examples=[str(chunk_size)])
    backend_name = str(backend).strip().lower()
    if backend_name == "torch_sparse":
        src, dst, distances = _build_torch_sparse_edges(
            coords=coords,
            radius_angstrom=radius_angstrom,
            max_neighbors=max_neighbors,
            chunk_size=chunk_size,
            stage=stage,
            location=location,
        )
    elif backend_name == "torch_geometric":
        src, dst, distances = _build_torch_geometric_edges(
            coords=coords,
            radius_angstrom=radius_angstrom,
            max_neighbors=max_neighbors,
            stage=stage,
            location=location,
        )
    else:
        raise_error(stage, location, "backend de grafo invalido", impact="1", examples=[backend_name])
    _enforce_min_degree(src=src, n_nodes=n_nodes, stage=stage, location=location)
    indices = torch.stack([src, dst], dim=0)
    adjacency = torch.sparse_coo_tensor(
        indices=indices,
        values=torch.ones((int(src.numel()),), dtype=torch.float32, device=coords.device),
        size=(n_nodes, n_nodes),
        device=coords.device,
    ).coalesce()
    return SparseRadiusGraph(src=src, dst=dst, distances=distances, adjacency=adjacency)
