from __future__ import annotations

import json

import polars as pl
import pytest

from rna3d_local.encoder import encode_sequences
from rna3d_local.errors import PipelineError
from rna3d_local.training.config_se3 import load_se3_train_config
from rna3d_local.training.msa_covariance import compute_msa_covariance
from rna3d_local.training.thermo_2d import compute_thermo_bpp


def test_encode_sequences_blocks_mock_outside_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RNA3D_ALLOW_MOCK_BACKENDS", raising=False)
    with pytest.raises(PipelineError, match="bloqueado por contrato"):
        encode_sequences(
            ["ACGU"],
            encoder="mock",
            embedding_dim=8,
            model_path=None,
            stage="EMBEDDING_INDEX",
            location="tests/test_mock_policy.py:test_encode_sequences_blocks_mock_outside_test",
        )


def test_thermo_mock_blocks_outside_test_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RNA3D_ALLOW_MOCK_BACKENDS", raising=False)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU"}])
    with pytest.raises(PipelineError, match="bloqueado por contrato"):
        compute_thermo_bpp(
            targets=targets,
            backend="mock",
            rnafold_bin="RNAfold",
            linearfold_bin="linearfold",
            cache_dir=None,
            chain_separator="|",
            stage="PREPARE_PHASE1_DATALAB",
            location="tests/test_mock_policy.py:test_thermo_mock_blocks_outside_test_stage",
        )


def test_msa_mock_blocks_outside_test_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RNA3D_ALLOW_MOCK_BACKENDS", raising=False)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU"}])
    with pytest.raises(PipelineError, match="bloqueado por contrato"):
        compute_msa_covariance(
            targets=targets,
            backend="mock",
            mmseqs_bin="mmseqs",
            mmseqs_db="",
            cache_dir=None,
            chain_separator="|",
            max_msa_sequences=16,
            max_cov_positions=64,
            max_cov_pairs=512,
            stage="PREPARE_PHASE1_DATALAB",
            location="tests/test_mock_policy.py:test_msa_mock_blocks_outside_test_stage",
        )


def test_mock_still_allowed_for_test_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RNA3D_ALLOW_MOCK_BACKENDS", raising=False)
    targets = pl.DataFrame([{"target_id": "T1", "sequence": "ACGU"}])
    out = compute_thermo_bpp(
        targets=targets,
        backend="mock",
        rnafold_bin="RNAfold",
        linearfold_bin="linearfold",
        cache_dir=None,
        chain_separator="|",
        stage="TEST",
        location="tests/test_mock_policy.py:test_mock_still_allowed_for_test_stage",
    )
    assert int(out["T1"].paired_marginal.numel()) == 4


def test_train_config_blocks_mock_backends_outside_test(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("RNA3D_ALLOW_MOCK_BACKENDS", raising=False)
    config = tmp_path / "config.json"
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
                "thermo_backend": "mock",
                "msa_backend": "mock",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="bloqueado por contrato"):
        load_se3_train_config(
            config,
            stage="TRAIN_SE3",
            location="tests/test_mock_policy.py:test_train_config_blocks_mock_backends_outside_test",
        )
