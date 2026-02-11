from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rna3d_local.bigdata import LabelStoreConfig, collect_streaming, scan_labels, sink_partitioned_parquet
from rna3d_local.errors import PipelineError


def test_scan_labels_prefers_parquet_and_fails_if_invalid(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "labels_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(PipelineError) as e:
        scan_labels(
            config=LabelStoreConfig(
                labels_parquet_dir=parquet_dir,
                required_columns=("ID", "resname", "resid"),
                stage="DATA",
                location="tests:test_scan_labels_prefers_parquet_and_fails_if_invalid",
            )
        )
    assert "part-*.parquet" in str(e.value)


def test_scan_labels_uses_parquet_only(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "labels_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    part_path = parquet_dir / "part-00000.parquet"
    pl.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ]
    ).write_parquet(part_path)
    lf = scan_labels(
        config=LabelStoreConfig(
            labels_parquet_dir=parquet_dir,
            required_columns=("ID", "resname", "resid"),
            stage="DATA",
            location="tests:test_scan_labels_uses_parquet_only",
        )
    )
    df = collect_streaming(lf=lf.select("ID", "resname", "resid"), stage="DATA", location="tests:test_scan_labels_uses_parquet_only")
    assert df.height == 2


def test_sink_partitioned_parquet_manifest_shape(tmp_path: Path) -> None:
    out_dir = tmp_path / "parts"
    lf = pl.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
            {"ID": "T1_3", "resname": "G", "resid": 3, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 1},
        ]
    ).lazy()
    info = sink_partitioned_parquet(
        lf=lf,
        out_dir=out_dir,
        rows_per_file=2,
        compression="zstd",
        stage="DATA",
        location="tests:test_sink_partitioned_parquet_manifest_shape",
    )
    assert info["n_rows"] == 3
    assert info["n_files"] == 2
    assert len(info["parts"]) == 2
    for p in info["parts"]:
        assert p.exists()
