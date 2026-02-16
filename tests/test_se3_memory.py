from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from rna3d_local.errors import PipelineError
from rna3d_local.se3.sparse_graph import build_sparse_radius_graph
from rna3d_local.se3_pipeline import sample_se3_ensemble, train_se3_generator


def _seq(length: int) -> str:
    alphabet = "ACGU"
    return "".join(alphabet[idx % 4] for idx in range(length))


def _write_targets(path: Path, *, length: int) -> None:
    pl.DataFrame([{"target_id": "LONG1", "sequence": _seq(length), "temporal_cutoff": "2024-01-01"}]).write_csv(path)


def _write_pairings(path: Path, *, length: int) -> None:
    rows = [{"target_id": "LONG1", "resid": resid, "pair_prob": 0.35} for resid in range(1, length + 1)]
    pl.DataFrame(rows).write_parquet(path)


def _write_chemical(path: Path, *, length: int) -> None:
    rows = [
        {
            "target_id": "LONG1",
            "resid": resid,
            "reactivity_dms": 0.3 + (0.0005 * resid),
            "reactivity_2a3": 0.4 + (0.0005 * resid),
            "p_open": 0.45,
            "p_paired": 0.55,
        }
        for resid in range(1, length + 1)
    ]
    pl.DataFrame(rows).write_parquet(path)


def _write_labels(path: Path, *, length: int) -> None:
    rows = [
        {"target_id": "LONG1", "resid": resid, "x": float(resid), "y": float(resid + 1), "z": float(resid + 2)}
        for resid in range(1, length + 1)
    ]
    pl.DataFrame(rows).write_parquet(path)


def test_sparse_radius_graph_limits_neighbors() -> None:
    coords = torch.stack(
        [
            torch.arange(0, 48, dtype=torch.float32) * 3.0,
            torch.zeros(48, dtype=torch.float32),
            torch.zeros(48, dtype=torch.float32),
        ],
        dim=1,
    )
    graph = build_sparse_radius_graph(
        coords=coords,
        radius_angstrom=15.0,
        max_neighbors=4,
        backend="torch_sparse",
        chunk_size=16,
        stage="TEST",
        location="tests/test_se3_memory.py:test_sparse_radius_graph_limits_neighbors",
    )
    assert graph.adjacency.is_sparse
    degree = torch.zeros((48,), dtype=torch.int64)
    degree.index_add_(0, graph.src.cpu(), torch.ones_like(graph.src.cpu(), dtype=torch.int64))
    assert int(degree.max().item()) <= 4
    assert int(degree.min().item()) >= 1


def test_torch_geometric_backend_contract() -> None:
    coords = torch.stack(
        [
            torch.arange(0, 12, dtype=torch.float32) * 3.0,
            torch.zeros(12, dtype=torch.float32),
            torch.zeros(12, dtype=torch.float32),
        ],
        dim=1,
    )
    try:
        graph = build_sparse_radius_graph(
            coords=coords,
            radius_angstrom=14.0,
            max_neighbors=6,
            backend="torch_geometric",
            chunk_size=16,
            stage="TEST",
            location="tests/test_se3_memory.py:test_torch_geometric_backend_contract",
        )
    except PipelineError as exc:
        assert "torch_geometric" in str(exc)
        return
    assert int(graph.src.numel()) > 0


def test_torch_sparse_backend_does_not_use_dense_cdist(monkeypatch) -> None:
    def _raise_cdist(*_args, **_kwargs):
        raise AssertionError("torch.cdist nao deve ser chamado no backend torch_sparse")

    monkeypatch.setattr(torch, "cdist", _raise_cdist)
    coords = torch.stack(
        [
            torch.arange(0, 24, dtype=torch.float32) * 2.8,
            torch.sin(torch.arange(0, 24, dtype=torch.float32) * 0.2),
            torch.cos(torch.arange(0, 24, dtype=torch.float32) * 0.2),
        ],
        dim=1,
    )
    graph = build_sparse_radius_graph(
        coords=coords,
        radius_angstrom=15.0,
        max_neighbors=6,
        backend="torch_sparse",
        chunk_size=8,
        stage="TEST",
        location="tests/test_se3_memory.py:test_torch_sparse_backend_does_not_use_dense_cdist",
    )
    assert int(graph.src.numel()) > 0


def test_train_and_sample_se3_with_linear_memory_config(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    pairings = tmp_path / "pairings.parquet"
    chem = tmp_path / "chem.parquet"
    labels = tmp_path / "labels.parquet"
    config = tmp_path / "config.json"
    length = 96
    _write_targets(targets, length=length)
    _write_pairings(pairings, length=length)
    _write_chemical(chem, length=length)
    _write_labels(labels, length=length)
    config.write_text(
        json.dumps(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "ipa_heads": 4,
                "diffusion_steps": 4,
                "flow_steps": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "method": "diffusion",
                "sequence_tower": "mamba_like",
                "sequence_heads": 4,
                "use_gradient_checkpointing": True,
                "graph_backend": "torch_sparse",
                "radius_angstrom": 14.0,
                "max_neighbors": 12,
                "graph_chunk_size": 64,
                "thermo_backend": "mock",
                "msa_backend": "mock",
            }
        ),
        encoding="utf-8",
    )
    trained = train_se3_generator(
        repo_root=tmp_path,
        targets_path=targets,
        pairings_path=pairings,
        chemical_features_path=chem,
        labels_path=labels,
        config_path=config,
        out_dir=tmp_path / "model",
        seed=17,
    )
    sampled = sample_se3_ensemble(
        repo_root=tmp_path,
        model_dir=trained.model_dir,
        targets_path=targets,
        pairings_path=pairings,
        chemical_features_path=chem,
        out_path=tmp_path / "candidates.parquet",
        method="diffusion",
        n_samples=4,
        seed=17,
    )
    manifest = json.loads(sampled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["params"]["sequence_tower"] == "mamba_like"
    assert manifest["params"]["graph_backend"] == "torch_sparse"
    assert int(manifest["stats"]["n_targets"]) == 1
