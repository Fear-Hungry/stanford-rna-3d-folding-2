from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from rna3d_local.errors import PipelineError
from rna3d_local.se3.sparse_graph import compute_chain_relative_features
from rna3d_local.training.msa_covariance import compute_msa_covariance


def test_compute_msa_covariance_mock_backend_shapes() -> None:
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GG|AAUU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    out = compute_msa_covariance(
        targets=targets,
        backend="mock",
        mmseqs_bin="mmseqs",
        mmseqs_db="",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=16,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_mock_backend_shapes",
    )
    assert set(out.keys()) == {"T1", "T2"}
    assert int(out["T1"].cov_marginal.numel()) == 6
    assert int(out["T2"].cov_marginal.numel()) == 6
    assert int(out["T2"].pair_src.numel()) == int(out["T2"].pair_dst.numel())


def test_compute_msa_covariance_mmseqs_missing_binary_fails() -> None:
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU", "temporal_cutoff": "2024-01-01"}])
    with pytest.raises(PipelineError, match="mmseqs2"):
        compute_msa_covariance(
            targets=targets,
            backend="mmseqs2",
            mmseqs_bin="/bin/nao_existe_mmseqs",
            mmseqs_db="/tmp/db_nao_existe",
            cache_dir=None,
            chain_separator="|",
            max_msa_sequences=16,
            max_cov_positions=64,
            max_cov_pairs=512,
            stage="TEST",
            location="tests/test_msa_covariance.py:test_compute_msa_covariance_mmseqs_missing_binary_fails",
        )


def test_compute_chain_relative_features_applies_break_offset() -> None:
    src = torch.tensor([0, 1, 1], dtype=torch.long)
    dst = torch.tensor([1, 2, 3], dtype=torch.long)
    residue_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    chain_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    rel, break_mask = compute_chain_relative_features(
        src=src,
        dst=dst,
        residue_index=residue_index,
        chain_index=chain_index,
        chain_break_offset=1000,
    )
    assert float(rel[0].item()) < 0.01
    assert float(break_mask[0].item()) == 0.0
    assert float(break_mask[1].item()) == 1.0


def test_compute_msa_covariance_mock_backend_parallel_consistent() -> None:
    targets = pl.DataFrame(
        [
            {"target_id": "T1", "sequence": "ACGUAC", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T2", "sequence": "GGAAUU", "temporal_cutoff": "2024-01-01"},
            {"target_id": "T3", "sequence": "AUGCAU", "temporal_cutoff": "2024-01-01"},
        ]
    )
    serial = compute_msa_covariance(
        targets=targets,
        backend="mock",
        mmseqs_bin="mmseqs",
        mmseqs_db="",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=16,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_mock_backend_parallel_consistent:serial",
        num_workers=1,
    )
    parallel = compute_msa_covariance(
        targets=targets,
        backend="mock",
        mmseqs_bin="mmseqs",
        mmseqs_db="",
        cache_dir=None,
        chain_separator="|",
        max_msa_sequences=16,
        max_cov_positions=64,
        max_cov_pairs=512,
        stage="TEST",
        location="tests/test_msa_covariance.py:test_compute_msa_covariance_mock_backend_parallel_consistent:parallel",
        num_workers=4,
    )
    assert sorted(serial.keys()) == sorted(parallel.keys())
    for target_id in serial:
        assert int(serial[target_id].pair_src.numel()) == int(parallel[target_id].pair_src.numel())
