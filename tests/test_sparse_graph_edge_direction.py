from __future__ import annotations

import types

import torch

from rna3d_local.se3.sparse_graph import build_sparse_radius_graph


def _inject_module(name: str, module: types.ModuleType) -> None:
    import sys

    sys.modules[name] = module


def test_torch_cluster_radius_graph_is_interpreted_as_receiver_src(monkeypatch) -> None:
    edge_index = torch.tensor(
        [
            [1, 0, 2, 1],  # source neighbors (j)
            [0, 1, 1, 2],  # target centers (i)
        ],
        dtype=torch.long,
    )

    torch_cluster = types.ModuleType("torch_cluster")

    def radius_graph(_coords, r, loop, max_num_neighbors):  # noqa: ARG001
        return edge_index

    torch_cluster.radius_graph = radius_graph  # type: ignore[attr-defined]
    _inject_module("torch_cluster", torch_cluster)

    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    graph = build_sparse_radius_graph(
        coords=coords,
        radius_angstrom=2.5,
        max_neighbors=8,
        backend="torch_sparse",
        chunk_size=8,
        stage="TEST",
        location="tests/test_sparse_graph_edge_direction.py:test_torch_cluster_radius_graph_is_interpreted_as_receiver_src",
    )
    # Our convention: src is receiver/center (edge_index[1]) and dst is neighbor/sender (edge_index[0]).
    assert torch.equal(graph.src.cpu(), edge_index[1])
    assert torch.equal(graph.dst.cpu(), edge_index[0])


def test_torch_geometric_radius_graph_is_interpreted_as_receiver_src(monkeypatch) -> None:
    edge_index = torch.tensor(
        [
            [1, 0, 2, 1],  # source neighbors (j)
            [0, 1, 1, 2],  # target centers (i)
        ],
        dtype=torch.long,
    )

    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric_nn = types.ModuleType("torch_geometric.nn")

    def radius_graph(_coords, r, loop, max_num_neighbors):  # noqa: ARG001
        return edge_index

    torch_geometric_nn.radius_graph = radius_graph  # type: ignore[attr-defined]
    torch_geometric.nn = torch_geometric_nn  # type: ignore[attr-defined]
    _inject_module("torch_geometric", torch_geometric)
    _inject_module("torch_geometric.nn", torch_geometric_nn)

    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    graph = build_sparse_radius_graph(
        coords=coords,
        radius_angstrom=2.5,
        max_neighbors=8,
        backend="torch_geometric",
        chunk_size=8,
        stage="TEST",
        location="tests/test_sparse_graph_edge_direction.py:test_torch_geometric_radius_graph_is_interpreted_as_receiver_src",
    )
    assert torch.equal(graph.src.cpu(), edge_index[1])
    assert torch.equal(graph.dst.cpu(), edge_index[0])

