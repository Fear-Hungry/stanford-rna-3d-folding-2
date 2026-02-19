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


def lookup_sparse_pair_bias(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    pair_prob: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    edge_count = int(src.numel())
    if edge_count == 0:
        return torch.zeros((0,), dtype=torch.float32, device=src.device)
    if int(pair_src.numel()) == 0:
        return torch.zeros((edge_count,), dtype=torch.float32, device=src.device)
    pair_keys = (pair_src.to(dtype=torch.long) * int(n_nodes)) + pair_dst.to(dtype=torch.long)
    order = torch.argsort(pair_keys)
    sorted_keys = pair_keys.index_select(0, order)
    sorted_vals = pair_prob.index_select(0, order).to(dtype=torch.float32)
    edge_keys = (src.to(dtype=torch.long) * int(n_nodes)) + dst.to(dtype=torch.long)
    pos = torch.searchsorted(sorted_keys, edge_keys)
    out = torch.zeros((edge_count,), dtype=torch.float32, device=src.device)
    if int(sorted_keys.numel()) == 0:
        return out
    safe_pos = torch.clamp(pos, max=int(sorted_keys.numel()) - 1)
    matches = sorted_keys.index_select(0, safe_pos) == edge_keys
    if bool(matches.any()):
        out[matches] = sorted_vals.index_select(0, safe_pos[matches])
    return out


def compute_chain_relative_features(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    chain_break_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chain_break_offset <= 0:
        raise ValueError("chain_break_offset deve ser > 0")
    src_pos = residue_index.index_select(0, src).to(dtype=torch.float32)
    dst_pos = residue_index.index_select(0, dst).to(dtype=torch.float32)
    src_chain = chain_index.index_select(0, src).to(dtype=torch.float32)
    dst_chain = chain_index.index_select(0, dst).to(dtype=torch.float32)
    adjusted_src = src_pos + (src_chain * float(chain_break_offset))
    adjusted_dst = dst_pos + (dst_chain * float(chain_break_offset))
    rel_offset = (adjusted_dst - adjusted_src) / float(chain_break_offset)
    break_mask = (src_chain != dst_chain).to(dtype=torch.float32)
    return rel_offset, break_mask


def _select_topk_pair_edges_per_src(
    *,
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    pair_prob: torch.Tensor,
    n_nodes: int,
    min_prob: float,
    max_per_node: int,
    stage: str,
    location: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_per_node <= 0 or int(pair_src.numel()) == 0:
        empty_long = torch.zeros((0,), dtype=torch.long, device=pair_src.device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=pair_src.device)
        return empty_long, empty_long, empty_float
    if pair_src.ndim != 1 or pair_dst.ndim != 1 or pair_prob.ndim != 1:
        raise_error(
            stage,
            location,
            "pares esparsos com shape invalido (esperado vetores 1D)",
            impact="1",
            examples=[f"src={tuple(pair_src.shape)}", f"dst={tuple(pair_dst.shape)}", f"prob={tuple(pair_prob.shape)}"],
        )
    if int(pair_src.numel()) != int(pair_dst.numel()) or int(pair_src.numel()) != int(pair_prob.numel()):
        raise_error(
            stage,
            location,
            "pares esparsos com comprimentos inconsistentes",
            impact="1",
            examples=[f"src={int(pair_src.numel())}", f"dst={int(pair_dst.numel())}", f"prob={int(pair_prob.numel())}"],
        )
    if not (0.0 <= float(min_prob) <= 1.0):
        raise_error(stage, location, "min_prob fora de [0,1] para pares esparsos", impact="1", examples=[str(min_prob)])
    src = pair_src.to(dtype=torch.long)
    dst = pair_dst.to(dtype=torch.long)
    prob = pair_prob.to(dtype=torch.float32)
    if int(src.numel()) == 0:
        empty_long = torch.zeros((0,), dtype=torch.long, device=src.device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=src.device)
        return empty_long, empty_long, empty_float
    if int(src.min().item()) < 0 or int(dst.min().item()) < 0:
        raise_error(
            stage,
            location,
            "pares esparsos com indices negativos",
            impact="1",
            examples=[f"src_min={int(src.min().item())}", f"dst_min={int(dst.min().item())}"],
        )
    if int(src.max().item()) >= int(n_nodes) or int(dst.max().item()) >= int(n_nodes):
        bad_mask = (src >= int(n_nodes)) | (dst >= int(n_nodes))
        bad_idx = torch.nonzero(bad_mask, as_tuple=False).flatten()[:8]
        examples = [f"{int(src[i].item())}->{int(dst[i].item())}" for i in bad_idx.tolist()]
        raise_error(
            stage,
            location,
            "pares esparsos com indices fora do intervalo do grafo",
            impact=str(int(bad_mask.sum().item())),
            examples=examples,
        )
    keep = (src != dst) & (prob >= float(min_prob))
    if not bool(keep.any()):
        empty_long = torch.zeros((0,), dtype=torch.long, device=src.device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=src.device)
        return empty_long, empty_long, empty_float
    src = src[keep]
    dst = dst[keep]
    prob = prob[keep]

    # Ordena por prob desc; depois ordena estavelmente por src asc -> (src asc, prob desc) por grupo.
    order = torch.argsort(prob, descending=True, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    prob = prob.index_select(0, order)
    order = torch.argsort(src, descending=False, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    prob = prob.index_select(0, order)

    counts = torch.bincount(src, minlength=int(n_nodes))
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(int(src.numel()), device=src.device, dtype=torch.long)
    rank = positions - starts.index_select(0, src)
    keep = rank < int(max_per_node)
    src = src[keep]
    dst = dst[keep]
    prob = prob[keep]
    return src, dst, prob


def _sort_edges_by_src_then_metric(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    metric: torch.Tensor,
    metric_desc: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(src.numel()) == 0:
        return src, dst, metric
    order = torch.argsort(metric, descending=bool(metric_desc), stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    metric = metric.index_select(0, order)
    order = torch.argsort(src, descending=False, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    metric = metric.index_select(0, order)
    return src, dst, metric


def _dedupe_directed_edges_keep_first(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    values: torch.Tensor,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(src.numel()) == 0:
        return src, dst, values
    keys = (src.to(dtype=torch.long) * int(n_nodes)) + dst.to(dtype=torch.long)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    keep_sorted = torch.ones((int(sorted_keys.numel()),), dtype=torch.bool, device=sorted_keys.device)
    keep_sorted[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep_indices = order.index_select(0, torch.nonzero(keep_sorted, as_tuple=False).flatten())
    keep_mask = torch.zeros((int(src.numel()),), dtype=torch.bool, device=src.device)
    keep_mask[keep_indices] = True
    return src[keep_mask], dst[keep_mask], values[keep_mask]


def _enforce_min_degree(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
    stage: str,
    location: str,
) -> None:
    degree = torch.zeros((n_nodes,), dtype=torch.int32, device=src.device)
    degree.index_add_(0, src, torch.ones_like(src, dtype=torch.int32))
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.int32))
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

    def _build_torch_cluster() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        try:
            from torch_cluster import radius_graph as torch_cluster_radius_graph  # type: ignore
        except Exception:
            return None
        try:
            edge_index = torch_cluster_radius_graph(coords, r=float(radius_angstrom), loop=False, max_num_neighbors=int(max_neighbors))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{stage}] [{location}] torch_cluster.radius_graph falhou; usando fallback cdist | "
                f"impacto=1 | exemplos={type(exc).__name__}:{exc}",
            )
            return None
        if int(edge_index.numel()) == 0:
            return None
        # torch_cluster returns edges in PyG convention: edge_index[0]=source (neighbor), edge_index[1]=target (center).
        # Our convention is src=receiver/center and dst=neighbor/sender to match index_add_(..., src, ...).
        src = edge_index[1].to(dtype=torch.long, device=coords.device)
        dst = edge_index[0].to(dtype=torch.long, device=coords.device)
        distances = torch.linalg.norm(coords.index_select(0, src) - coords.index_select(0, dst), dim=-1).to(dtype=torch.float32)
        return src, dst, distances

    cluster_edges = _build_torch_cluster()
    if cluster_edges is not None:
        src_cluster, dst_cluster, dist_cluster = cluster_edges
        degree = torch.zeros((n_nodes,), dtype=torch.int32, device=coords.device)
        degree.index_add_(0, src_cluster, torch.ones_like(src_cluster, dtype=torch.int32))
        degree.index_add_(0, dst_cluster, torch.ones_like(dst_cluster, dtype=torch.int32))
        if int(torch.nonzero(degree == 0, as_tuple=False).numel()) == 0:
            return src_cluster, dst_cluster, dist_cluster

    # Fallback vetorizado (GPU-friendly): construcao por blocos usando cdist.
    coords_dense = coords.to(dtype=torch.float32)
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    dist_parts: list[torch.Tensor] = []
    block = min(int(chunk_size), n_nodes)
    for start in range(0, n_nodes, block):
        end = min(start + block, n_nodes)
        query = coords_dense[start:end]
        dist_matrix = torch.cdist(query, coords_dense, p=2)
        local_len = end - start
        local_nodes = torch.arange(start, end, dtype=torch.long, device=coords.device)
        row_ids = torch.arange(local_len, dtype=torch.long, device=coords.device)
        # Remove auto-aresta.
        dist_matrix[row_ids, local_nodes] = float("inf")
        within = dist_matrix <= float(radius_angstrom)
        masked_dist = torch.where(within, dist_matrix, torch.full_like(dist_matrix, float("inf")))
        topk_dist, topk_dst = torch.topk(masked_dist, k=min(int(max_neighbors), n_nodes), largest=False, dim=1)
        keep = torch.isfinite(topk_dist)
        if not bool(keep.any()):
            continue
        src_chunk = local_nodes.unsqueeze(1).expand(-1, topk_dst.shape[1])[keep]
        dst_chunk = topk_dst[keep]
        dist_chunk = topk_dist[keep]
        # Garantia extra contra auto-aresta em casos numericos extremos.
        non_self = src_chunk != dst_chunk
        if not bool(non_self.any()):
            continue
        src_parts.append(src_chunk[non_self].to(dtype=torch.long))
        dst_parts.append(dst_chunk[non_self].to(dtype=torch.long))
        dist_parts.append(dist_chunk[non_self].to(dtype=torch.float32))

    if not src_parts:
        raise_error(
            stage,
            location,
            "grafo 3d esparso sem arestas; ajuste radius_angstrom/max_neighbors/coords",
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
    # PyG convention: edge_index[0]=source (neighbor), edge_index[1]=target (center).
    src = edge_index[1].to(dtype=torch.long, device=coords.device)
    dst = edge_index[0].to(dtype=torch.long, device=coords.device)
    distances = torch.linalg.norm(coords.index_select(0, src) - coords.index_select(0, dst), dim=-1).to(dtype=torch.float32)
    return src, dst, distances


def build_sparse_radius_graph(
    *,
    coords: torch.Tensor,
    radius_angstrom: float,
    max_neighbors: int,
    backend: str,
    chunk_size: int,
    pair_src: torch.Tensor | None = None,
    pair_dst: torch.Tensor | None = None,
    pair_prob: torch.Tensor | None = None,
    pair_min_prob: float = 0.0,
    pair_max_per_node: int = 0,
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

    # Opcional: injeta pares esparsos (ex.: BPP) como arestas topologicas, mantendo o budget total por src.
    if pair_src is not None or pair_dst is not None or pair_prob is not None:
        if pair_src is None or pair_dst is None or pair_prob is None:
            raise_error(
                stage,
                location,
                "pares esparsos incompletos (src/dst/prob)",
                impact="1",
                examples=[f"src={pair_src is not None}", f"dst={pair_dst is not None}", f"prob={pair_prob is not None}"],
            )
        pair_src_dev = pair_src.to(device=coords.device)
        pair_dst_dev = pair_dst.to(device=coords.device)
        pair_prob_dev = pair_prob.to(device=coords.device)
        pair_src_sel, pair_dst_sel, pair_prob_sel = _select_topk_pair_edges_per_src(
            pair_src=pair_src_dev,
            pair_dst=pair_dst_dev,
            pair_prob=pair_prob_dev,
            n_nodes=n_nodes,
            min_prob=float(pair_min_prob),
            max_per_node=int(pair_max_per_node),
            stage=stage,
            location=location,
        )
        if int(pair_src_sel.numel()) > 0:
            pair_dist = torch.linalg.norm(coords.index_select(0, pair_src_sel) - coords.index_select(0, pair_dst_sel), dim=-1).to(dtype=torch.float32)
            order = torch.argsort(pair_prob_sel, descending=True, stable=True)
            pair_src_sel = pair_src_sel.index_select(0, order)
            pair_dst_sel = pair_dst_sel.index_select(0, order)
            pair_dist = pair_dist.index_select(0, order)
            order = torch.argsort(pair_src_sel, descending=False, stable=True)
            pair_src_sel = pair_src_sel.index_select(0, order)
            pair_dst_sel = pair_dst_sel.index_select(0, order)
            pair_dist = pair_dist.index_select(0, order)
            src, dst, distances = _sort_edges_by_src_then_metric(src=src, dst=dst, metric=distances, metric_desc=False)
            src = torch.cat([pair_src_sel, src], dim=0)
            dst = torch.cat([pair_dst_sel, dst], dim=0)
            distances = torch.cat([pair_dist, distances], dim=0)
            order = torch.argsort(src, descending=False, stable=True)
            src = src.index_select(0, order)
            dst = dst.index_select(0, order)
            distances = distances.index_select(0, order)
            src, dst, distances = _dedupe_directed_edges_keep_first(src=src, dst=dst, values=distances, n_nodes=n_nodes)
            counts = torch.bincount(src, minlength=int(n_nodes))
            starts = torch.cumsum(counts, dim=0) - counts
            positions = torch.arange(int(src.numel()), device=src.device, dtype=torch.long)
            rank = positions - starts.index_select(0, src)
            keep = rank < int(max_neighbors)
            src = src[keep]
            dst = dst[keep]
            distances = distances[keep]

    _enforce_min_degree(src=src, dst=dst, n_nodes=n_nodes, stage=stage, location=location)
    indices = torch.stack([src, dst], dim=0)
    adjacency = torch.sparse_coo_tensor(
        indices=indices,
        values=torch.ones((int(src.numel()),), dtype=torch.float32, device=coords.device),
        size=(n_nodes, n_nodes),
        device=coords.device,
    ).coalesce()
    return SparseRadiusGraph(src=src, dst=dst, distances=distances, adjacency=adjacency)
