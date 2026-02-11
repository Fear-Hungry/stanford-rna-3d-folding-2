from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.datasets import export_train_solution_for_targets, prepare_train_labels_parquet
from rna3d_local.errors import PipelineError
from rna3d_local.rnapro import RnaProConfig, train_rnapro
from rna3d_local.template_db import build_template_db


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _prepare_labels_parquet(tmp_path: Path, rows: list[dict], *, out_dir_name: str = "labels_parquet") -> Path:
    train_labels_csv = tmp_path / "train_labels.csv"
    out_dir = tmp_path / out_dir_name
    _write_csv(train_labels_csv, rows)
    prepare_train_labels_parquet(
        repo_root=tmp_path,
        train_labels_csv=train_labels_csv,
        out_dir=out_dir,
        rows_per_file=1,
        compression="zstd",
    )
    return out_dir


def test_prepare_labels_parquet_creates_manifest_and_parts(tmp_path: Path) -> None:
    train_labels_csv = tmp_path / "train_labels.csv"
    out_dir = tmp_path / "labels_parquet"
    _write_csv(
        train_labels_csv,
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T2_1", "resname": "G", "resid": 1, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ],
    )

    manifest = prepare_train_labels_parquet(
        repo_root=tmp_path,
        train_labels_csv=train_labels_csv,
        out_dir=out_dir,
        rows_per_file=2,
        compression="zstd",
    )
    assert manifest.exists()
    parts = sorted(out_dir.glob("part-*.parquet"))
    assert len(parts) == 2
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stats"]["n_rows"] == 3
    assert payload["stats"]["n_files"] == 2


def test_pipeline_uses_labels_parquet_dir(tmp_path: Path) -> None:
    train_sequences = tmp_path / "train_sequences.csv"
    external_templates = tmp_path / "external_templates.csv"
    labels_parquet_dir = _prepare_labels_parquet(
        tmp_path,
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ],
    )
    _write_csv(
        train_sequences,
        [
            {"target_id": "T1", "sequence": "AC", "temporal_cutoff": "2020-01-01"},
        ],
    )
    _write_csv(
        external_templates,
        [
            {"template_id": "E1", "sequence": "AC", "release_date": "2019-01-01", "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0, "source": "ext"},
            {"template_id": "E1", "sequence": "AC", "release_date": "2019-01-01", "resid": 2, "resname": "C", "x": 1.0, "y": 0.0, "z": 0.0, "source": "ext"},
        ],
    )

    template_res = build_template_db(
        repo_root=tmp_path,
        train_sequences_path=train_sequences,
        train_labels_parquet_dir=labels_parquet_dir,
        external_templates_path=external_templates,
        out_dir=tmp_path / "template_db",
    )
    assert template_res.templates_path.exists()
    assert template_res.index_path.exists()

    model = train_rnapro(
        repo_root=tmp_path,
        train_sequences_path=train_sequences,
        train_labels_parquet_dir=labels_parquet_dir,
        out_dir=tmp_path / "rnapro_model",
        config=RnaProConfig(n_models=1, feature_dim=8, kmer_size=2, seed=7, min_coverage=0.2),
    )
    assert model.exists()

    solution_out = export_train_solution_for_targets(
        out_path=tmp_path / "solution.parquet",
        target_ids=["T1"],
        train_labels_parquet_dir=labels_parquet_dir,
    )
    assert solution_out.exists()


def test_export_train_solution_fails_when_parquet_dir_is_invalid(tmp_path: Path) -> None:
    empty_parquet_dir = tmp_path / "empty_labels_parquet"
    empty_parquet_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(PipelineError) as e:
        export_train_solution_for_targets(
            out_path=tmp_path / "solution.parquet",
            target_ids=["T1"],
            train_labels_parquet_dir=empty_parquet_dir,
        )
    msg = str(e.value)
    assert msg.startswith("[DATA]")
    assert "part-*.parquet" in msg


def test_train_rnapro_fails_when_parquet_dir_is_invalid(tmp_path: Path) -> None:
    train_sequences = tmp_path / "train_sequences.csv"
    empty_parquet_dir = tmp_path / "empty_labels_parquet"
    empty_parquet_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        train_sequences,
        [{"target_id": "T1", "sequence": "AC", "temporal_cutoff": "2020-01-01"}],
    )

    with pytest.raises(PipelineError) as e:
        train_rnapro(
            repo_root=tmp_path,
            train_sequences_path=train_sequences,
            train_labels_parquet_dir=empty_parquet_dir,
            out_dir=tmp_path / "rnapro_model",
            config=RnaProConfig(n_models=1, feature_dim=8, kmer_size=2, seed=7, min_coverage=0.2),
        )
    msg = str(e.value)
    assert msg.startswith("[RNAPRO_TRAIN]")
    assert "part-*.parquet" in msg


def test_template_db_fails_when_parquet_dir_is_invalid(tmp_path: Path) -> None:
    train_sequences = tmp_path / "train_sequences.csv"
    external_templates = tmp_path / "external_templates.csv"
    empty_parquet_dir = tmp_path / "empty_labels_parquet"
    empty_parquet_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        train_sequences,
        [{"target_id": "T1", "sequence": "AC", "temporal_cutoff": "2020-01-01"}],
    )
    _write_csv(
        external_templates,
        [
            {"template_id": "E1", "sequence": "AC", "release_date": "2019-01-01", "resid": 1, "resname": "A", "x": 0.0, "y": 0.0, "z": 0.0, "source": "ext"},
            {"template_id": "E1", "sequence": "AC", "release_date": "2019-01-01", "resid": 2, "resname": "C", "x": 1.0, "y": 0.0, "z": 0.0, "source": "ext"},
        ],
    )

    with pytest.raises(PipelineError) as e:
        build_template_db(
            repo_root=tmp_path,
            train_sequences_path=train_sequences,
            train_labels_parquet_dir=empty_parquet_dir,
            external_templates_path=external_templates,
            out_dir=tmp_path / "template_db",
        )
    msg = str(e.value)
    assert msg.startswith("[TEMPLATE_DB]")
    assert "part-*.parquet" in msg


def test_export_train_solution_supports_target_ids_with_underscore(tmp_path: Path) -> None:
    labels_parquet_dir = _prepare_labels_parquet(
        tmp_path,
        [
            {"ID": "T_A_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T_A_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T_B_1", "resname": "G", "resid": 1, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ],
    )

    out = export_train_solution_for_targets(
        out_path=tmp_path / "solution.parquet",
        target_ids=["T_A"],
        train_labels_parquet_dir=labels_parquet_dir,
    )
    df = pl.read_parquet(out).select("ID").sort("ID")
    assert df.get_column("ID").to_list() == ["T_A_1", "T_A_2"]
