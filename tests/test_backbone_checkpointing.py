from __future__ import annotations

import sys
import types

import torch

import rna3d_local.se3.egnn_backbone as egnn_mod
import rna3d_local.se3.ipa_backbone as ipa_mod
from rna3d_local.se3.egnn_backbone import EgnnBackbone
from rna3d_local.se3.ipa_backbone import IpaBackbone


def _toy_inputs() -> dict[str, torch.Tensor]:
    node_features = torch.randn(6, 8, dtype=torch.float32, requires_grad=True)
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.8, 0.1, 0.0],
            [5.6, 0.2, 0.1],
            [8.4, 0.3, 0.1],
            [11.2, 0.4, 0.2],
            [14.0, 0.5, 0.2],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    base_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    empty_src = torch.zeros((0,), dtype=torch.long)
    empty_dst = torch.zeros((0,), dtype=torch.long)
    empty_prob = torch.zeros((0,), dtype=torch.float32)
    residue_index = torch.arange(6, dtype=torch.long)
    chain_index = torch.zeros(6, dtype=torch.long)
    chem_exposure = torch.full((6,), 0.5, dtype=torch.float32)
    return {
        "node_features": node_features,
        "coords": coords,
        "base_features": base_features,
        "bpp_pair_src": empty_src,
        "bpp_pair_dst": empty_dst,
        "bpp_pair_prob": empty_prob,
        "msa_pair_src": empty_src,
        "msa_pair_dst": empty_dst,
        "msa_pair_prob": empty_prob,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "chem_exposure": chem_exposure,
    }


def test_egnn_backbone_uses_checkpoint_once_per_layer(monkeypatch) -> None:
    call_counter = {"n": 0}

    def _fake_checkpoint(function, *args, **kwargs):  # noqa: ANN001
        call_counter["n"] += 1
        assert kwargs.get("use_reentrant") is False
        return function(*args)

    monkeypatch.setattr(egnn_mod, "checkpoint", _fake_checkpoint)
    num_layers = 3
    model = EgnnBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=num_layers,
        graph_backend="torch_sparse",
        radius_angstrom=10.0,
        max_neighbors=4,
        graph_chunk_size=16,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_backbone_checkpointing.py:test_egnn_backbone_uses_checkpoint_once_per_layer",
        use_gradient_checkpointing=True,
    )
    model.train()
    tensors = _toy_inputs()
    h_out, x_out = model(
        node_features=tensors["node_features"],
        coords=tensors["coords"],
        bpp_pair_src=tensors["bpp_pair_src"],
        bpp_pair_dst=tensors["bpp_pair_dst"],
        bpp_pair_prob=tensors["bpp_pair_prob"],
        msa_pair_src=tensors["msa_pair_src"],
        msa_pair_dst=tensors["msa_pair_dst"],
        msa_pair_prob=tensors["msa_pair_prob"],
        residue_index=tensors["residue_index"],
        chain_index=tensors["chain_index"],
        chem_exposure=tensors["chem_exposure"],
        chain_break_offset=1000,
    )
    assert call_counter["n"] == num_layers
    assert h_out.shape == (6, 8)
    assert x_out.shape == (6, 3)


def test_ipa_backbone_uses_checkpoint_once_per_layer(monkeypatch) -> None:
    call_counter = {"n": 0}

    def _fake_checkpoint(function, *args, **kwargs):  # noqa: ANN001
        call_counter["n"] += 1
        assert kwargs.get("use_reentrant") is False
        return function(*args)

    monkeypatch.setattr(ipa_mod, "checkpoint", _fake_checkpoint)
    num_layers = 2
    model = IpaBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=num_layers,
        heads=2,
        graph_backend="torch_sparse",
        radius_angstrom=10.0,
        max_neighbors=4,
        graph_chunk_size=16,
        ipa_edge_chunk_size=64,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_backbone_checkpointing.py:test_ipa_backbone_uses_checkpoint_once_per_layer",
        use_gradient_checkpointing=True,
    )
    model.train()
    tensors = _toy_inputs()
    h_out, x_out = model(
        node_features=tensors["node_features"],
        coords=tensors["coords"],
        bpp_pair_src=tensors["bpp_pair_src"],
        bpp_pair_dst=tensors["bpp_pair_dst"],
        bpp_pair_prob=tensors["bpp_pair_prob"],
        msa_pair_src=tensors["msa_pair_src"],
        msa_pair_dst=tensors["msa_pair_dst"],
        msa_pair_prob=tensors["msa_pair_prob"],
        base_features=tensors["base_features"],
        residue_index=tensors["residue_index"],
        chain_index=tensors["chain_index"],
        chem_exposure=tensors["chem_exposure"],
        chain_break_offset=1000,
    )
    assert call_counter["n"] == num_layers
    assert h_out.shape == (6, 8)
    assert x_out.shape == (6, 3)


def test_backbones_checkpointing_preserves_gradients(monkeypatch) -> None:
    torch_cluster = types.ModuleType("torch_cluster")

    def _radius_graph(coords, r, loop, max_num_neighbors):  # noqa: ARG001
        n = int(coords.shape[0])
        src = []
        dst = []
        for center in range(n):
            left = max(0, center - 2)
            right = min(n, center + 3)
            for neigh in range(left, right):
                if neigh == center:
                    continue
                src.append(neigh)
                dst.append(center)
        return torch.tensor([src, dst], dtype=torch.long, device=coords.device)

    torch_cluster.radius_graph = _radius_graph  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_cluster", torch_cluster)

    tensors_ref = _toy_inputs()
    tensors_chk = _toy_inputs()
    for key in ["node_features", "coords"]:
        tensors_chk[key] = tensors_ref[key].detach().clone().requires_grad_(True)
    for key in ["base_features", "bpp_pair_src", "bpp_pair_dst", "bpp_pair_prob", "msa_pair_src", "msa_pair_dst", "msa_pair_prob", "residue_index", "chain_index", "chem_exposure"]:
        tensors_chk[key] = tensors_ref[key].detach().clone()

    torch.manual_seed(123)
    egnn_ref = EgnnBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=2,
        graph_backend="torch_sparse",
        radius_angstrom=10.0,
        max_neighbors=4,
        graph_chunk_size=16,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_backbone_checkpointing.py:test_backbones_checkpointing_preserves_gradients:egnn_ref",
        use_gradient_checkpointing=False,
    )
    egnn_chk = EgnnBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=2,
        graph_backend="torch_sparse",
        radius_angstrom=10.0,
        max_neighbors=4,
        graph_chunk_size=16,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_backbone_checkpointing.py:test_backbones_checkpointing_preserves_gradients:egnn_chk",
        use_gradient_checkpointing=True,
    )
    egnn_chk.load_state_dict(egnn_ref.state_dict())
    egnn_ref.train()
    egnn_chk.train()
    out_ref_h, out_ref_x = egnn_ref(
        node_features=tensors_ref["node_features"],
        coords=tensors_ref["coords"],
        bpp_pair_src=tensors_ref["bpp_pair_src"],
        bpp_pair_dst=tensors_ref["bpp_pair_dst"],
        bpp_pair_prob=tensors_ref["bpp_pair_prob"],
        msa_pair_src=tensors_ref["msa_pair_src"],
        msa_pair_dst=tensors_ref["msa_pair_dst"],
        msa_pair_prob=tensors_ref["msa_pair_prob"],
        residue_index=tensors_ref["residue_index"],
        chain_index=tensors_ref["chain_index"],
        chem_exposure=tensors_ref["chem_exposure"],
        chain_break_offset=1000,
    )
    out_chk_h, out_chk_x = egnn_chk(
        node_features=tensors_chk["node_features"],
        coords=tensors_chk["coords"],
        bpp_pair_src=tensors_chk["bpp_pair_src"],
        bpp_pair_dst=tensors_chk["bpp_pair_dst"],
        bpp_pair_prob=tensors_chk["bpp_pair_prob"],
        msa_pair_src=tensors_chk["msa_pair_src"],
        msa_pair_dst=tensors_chk["msa_pair_dst"],
        msa_pair_prob=tensors_chk["msa_pair_prob"],
        residue_index=tensors_chk["residue_index"],
        chain_index=tensors_chk["chain_index"],
        chem_exposure=tensors_chk["chem_exposure"],
        chain_break_offset=1000,
    )
    (out_ref_h.sum() + out_ref_x.sum()).backward()
    (out_chk_h.sum() + out_chk_x.sum()).backward()
    for (name_ref, param_ref), (name_chk, param_chk) in zip(egnn_ref.named_parameters(), egnn_chk.named_parameters(), strict=True):
        assert name_ref == name_chk
        assert param_ref.grad is not None
        assert param_chk.grad is not None
        assert torch.allclose(param_ref.grad, param_chk.grad, atol=1e-5, rtol=1e-5)

    ipa_ref = IpaBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=2,
        heads=2,
        graph_backend="torch_sparse",
        radius_angstrom=10.0,
        max_neighbors=4,
        graph_chunk_size=16,
        ipa_edge_chunk_size=64,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_backbone_checkpointing.py:test_backbones_checkpointing_preserves_gradients:ipa_ref",
        use_gradient_checkpointing=False,
    )
    ipa_chk = IpaBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=2,
        heads=2,
        graph_backend="torch_sparse",
        radius_angstrom=10.0,
        max_neighbors=4,
        graph_chunk_size=16,
        ipa_edge_chunk_size=64,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_backbone_checkpointing.py:test_backbones_checkpointing_preserves_gradients:ipa_chk",
        use_gradient_checkpointing=True,
    )
    ipa_chk.load_state_dict(ipa_ref.state_dict())
    ipa_ref.train()
    ipa_chk.train()

    tensors_ref_ipa = _toy_inputs()
    tensors_chk_ipa = _toy_inputs()
    for key in ["node_features", "coords"]:
        tensors_chk_ipa[key] = tensors_ref_ipa[key].detach().clone().requires_grad_(True)
    for key in ["base_features", "bpp_pair_src", "bpp_pair_dst", "bpp_pair_prob", "msa_pair_src", "msa_pair_dst", "msa_pair_prob", "residue_index", "chain_index", "chem_exposure"]:
        tensors_chk_ipa[key] = tensors_ref_ipa[key].detach().clone()

    out_ref_h, out_ref_x = ipa_ref(
        node_features=tensors_ref_ipa["node_features"],
        coords=tensors_ref_ipa["coords"],
        bpp_pair_src=tensors_ref_ipa["bpp_pair_src"],
        bpp_pair_dst=tensors_ref_ipa["bpp_pair_dst"],
        bpp_pair_prob=tensors_ref_ipa["bpp_pair_prob"],
        msa_pair_src=tensors_ref_ipa["msa_pair_src"],
        msa_pair_dst=tensors_ref_ipa["msa_pair_dst"],
        msa_pair_prob=tensors_ref_ipa["msa_pair_prob"],
        base_features=tensors_ref_ipa["base_features"],
        residue_index=tensors_ref_ipa["residue_index"],
        chain_index=tensors_ref_ipa["chain_index"],
        chem_exposure=tensors_ref_ipa["chem_exposure"],
        chain_break_offset=1000,
    )
    out_chk_h, out_chk_x = ipa_chk(
        node_features=tensors_chk_ipa["node_features"],
        coords=tensors_chk_ipa["coords"],
        bpp_pair_src=tensors_chk_ipa["bpp_pair_src"],
        bpp_pair_dst=tensors_chk_ipa["bpp_pair_dst"],
        bpp_pair_prob=tensors_chk_ipa["bpp_pair_prob"],
        msa_pair_src=tensors_chk_ipa["msa_pair_src"],
        msa_pair_dst=tensors_chk_ipa["msa_pair_dst"],
        msa_pair_prob=tensors_chk_ipa["msa_pair_prob"],
        base_features=tensors_chk_ipa["base_features"],
        residue_index=tensors_chk_ipa["residue_index"],
        chain_index=tensors_chk_ipa["chain_index"],
        chem_exposure=tensors_chk_ipa["chem_exposure"],
        chain_break_offset=1000,
    )
    (out_ref_h.sum() + out_ref_x.sum()).backward()
    (out_chk_h.sum() + out_chk_x.sum()).backward()
    for (name_ref, param_ref), (name_chk, param_chk) in zip(ipa_ref.named_parameters(), ipa_chk.named_parameters(), strict=True):
        assert name_ref == name_chk
        assert param_ref.grad is not None
        assert param_chk.grad is not None
        assert torch.allclose(param_ref.grad, param_chk.grad, atol=1e-5, rtol=1e-5)
