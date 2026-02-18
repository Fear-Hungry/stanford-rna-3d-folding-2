from __future__ import annotations

import pytest
import torch

from rna3d_local.se3.geometry import build_rna_local_frames
from rna3d_local.se3.geometry import rotation_matrix_from_6d
from rna3d_local.se3.ipa_backbone import IpaBackbone


def test_build_rna_local_frames_single_residue_respects_base_family() -> None:
    c1 = torch.zeros((1, 3), dtype=torch.float32)
    purine = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pyrimidine = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    _frame_pur, _p_pur, _c4_pur, n_pur = build_rna_local_frames(c1_coords=c1, base_features=purine)
    _frame_pyr, _p_pyr, _c4_pyr, n_pyr = build_rna_local_frames(c1_coords=c1, base_features=pyrimidine)
    assert float(n_pur[0, 2].item()) > 0.0
    assert float(n_pyr[0, 2].item()) < 0.0


def test_build_rna_local_frames_is_finite_and_orthonormal() -> None:
    c1 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.7, 0.8, 0.4],
            [5.2, 1.6, 0.9],
            [7.8, 2.2, 1.5],
        ],
        dtype=torch.float32,
    )
    base = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    frames, p_proxy, c4_proxy, n_proxy = build_rna_local_frames(c1_coords=c1, base_features=base)
    assert frames.shape == (4, 3, 3)
    assert p_proxy.shape == (4, 3)
    assert c4_proxy.shape == (4, 3)
    assert n_proxy.shape == (4, 3)
    assert torch.isfinite(frames).all()
    gram = torch.matmul(frames, frames.transpose(1, 2))
    identity = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand_as(gram)
    assert torch.allclose(gram, identity, atol=1e-4, rtol=1e-4)

    # Convention check: local basis vectors are stored in columns (local->global).
    # x-axis should align with the (C4' - P) proxy direction used in geometry.py.
    axis_x_dir = c4_proxy - p_proxy
    axis_x_dir = axis_x_dir / torch.clamp(torch.linalg.norm(axis_x_dir, dim=-1, keepdim=True), min=1e-6)
    x_col = frames[:, :, 0]
    dot = torch.sum(x_col * axis_x_dir, dim=-1)
    assert torch.all(dot > 0.90)


def test_rotation_matrix_from_6d_is_orthonormal() -> None:
    x = torch.randn(12, 6, dtype=torch.float32)
    rot = rotation_matrix_from_6d(x)
    assert rot.shape == (12, 3, 3)
    assert torch.isfinite(rot).all()
    gram = torch.matmul(rot.transpose(1, 2), rot)
    identity = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand_as(gram)
    assert torch.allclose(gram, identity, atol=1e-4, rtol=1e-4)


def test_ipa_backbone_requires_valid_base_features_and_runs() -> None:
    model = IpaBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        graph_backend="torch_sparse",
        radius_angstrom=8.0,
        max_neighbors=3,
        graph_chunk_size=8,
        ipa_edge_chunk_size=64,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_ipa_geometry.py:test_ipa_backbone_requires_valid_base_features_and_runs",
    )
    node_features = torch.randn(4, 8)
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.1, 0.0],
            [5.8, 0.2, 0.1],
            [8.4, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    bpp_src = torch.tensor([0, 1], dtype=torch.long)
    bpp_dst = torch.tensor([1, 2], dtype=torch.long)
    bpp_prob = torch.tensor([0.6, 0.4], dtype=torch.float32)
    msa_src = torch.tensor([0], dtype=torch.long)
    msa_dst = torch.tensor([2], dtype=torch.long)
    msa_prob = torch.tensor([0.3], dtype=torch.float32)
    residue_index = torch.arange(4, dtype=torch.long)
    chain_index = torch.zeros(4, dtype=torch.long)
    chem_exposure = torch.full((4,), 0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="base_features"):
        model(
            node_features=node_features,
            coords=coords,
            bpp_pair_src=bpp_src,
            bpp_pair_dst=bpp_dst,
            bpp_pair_prob=bpp_prob,
            msa_pair_src=msa_src,
            msa_pair_dst=msa_dst,
            msa_pair_prob=msa_prob,
            base_features=torch.zeros((4, 3), dtype=torch.float32),
            residue_index=residue_index,
            chain_index=chain_index,
            chem_exposure=chem_exposure,
            chain_break_offset=1000,
        )
    base_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    h_out, x_out = model(
        node_features=node_features,
        coords=coords,
        bpp_pair_src=bpp_src,
        bpp_pair_dst=bpp_dst,
        bpp_pair_prob=bpp_prob,
        msa_pair_src=msa_src,
        msa_pair_dst=msa_dst,
        msa_pair_prob=msa_prob,
        base_features=base_features,
        residue_index=residue_index,
        chain_index=chain_index,
        chem_exposure=chem_exposure,
        chain_break_offset=1000,
    )
    assert h_out.shape == (4, 8)
    assert x_out.shape == (4, 3)
    assert torch.isfinite(h_out).all()
    assert torch.isfinite(x_out).all()


def test_ipa_backbone_chunking_preserves_outputs() -> None:
    torch.manual_seed(17)
    model_ref = IpaBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        graph_backend="torch_sparse",
        radius_angstrom=12.0,
        max_neighbors=6,
        graph_chunk_size=16,
        ipa_edge_chunk_size=256,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_ipa_geometry.py:test_ipa_backbone_chunking_preserves_outputs:ref",
    )
    model_chunked = IpaBackbone(
        input_dim=8,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        graph_backend="torch_sparse",
        radius_angstrom=12.0,
        max_neighbors=6,
        graph_chunk_size=16,
        ipa_edge_chunk_size=2,
        graph_pair_edges="none",
        graph_pair_min_prob=0.0,
        graph_pair_max_per_node=0,
        stage="TEST",
        location="tests/test_ipa_geometry.py:test_ipa_backbone_chunking_preserves_outputs:chunked",
    )
    model_chunked.load_state_dict(model_ref.state_dict())
    model_ref.eval()
    model_chunked.eval()

    node_features = torch.randn(6, 8, dtype=torch.float32)
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.7, 0.2, 0.1],
            [5.4, 0.4, 0.2],
            [8.1, 0.6, 0.3],
            [10.8, 0.8, 0.4],
            [13.5, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    bpp_src = torch.zeros((0,), dtype=torch.long)
    bpp_dst = torch.zeros((0,), dtype=torch.long)
    bpp_prob = torch.zeros((0,), dtype=torch.float32)
    msa_src = torch.zeros((0,), dtype=torch.long)
    msa_dst = torch.zeros((0,), dtype=torch.long)
    msa_prob = torch.zeros((0,), dtype=torch.float32)
    residue_index = torch.arange(6, dtype=torch.long)
    chain_index = torch.zeros(6, dtype=torch.long)
    chem_exposure = torch.full((6,), 0.5, dtype=torch.float32)
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
    with torch.no_grad():
        h_ref, x_ref = model_ref(
            node_features=node_features,
            coords=coords,
            bpp_pair_src=bpp_src,
            bpp_pair_dst=bpp_dst,
            bpp_pair_prob=bpp_prob,
            msa_pair_src=msa_src,
            msa_pair_dst=msa_dst,
            msa_pair_prob=msa_prob,
            base_features=base_features,
            residue_index=residue_index,
            chain_index=chain_index,
            chem_exposure=chem_exposure,
            chain_break_offset=1000,
        )
        h_chk, x_chk = model_chunked(
            node_features=node_features,
            coords=coords,
            bpp_pair_src=bpp_src,
            bpp_pair_dst=bpp_dst,
            bpp_pair_prob=bpp_prob,
            msa_pair_src=msa_src,
            msa_pair_dst=msa_dst,
            msa_pair_prob=msa_prob,
            base_features=base_features,
            residue_index=residue_index,
            chain_index=chain_index,
            chem_exposure=chem_exposure,
            chain_break_offset=1000,
        )
    assert torch.allclose(h_ref, h_chk, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_ref, x_chk, atol=1e-5, rtol=1e-5)
