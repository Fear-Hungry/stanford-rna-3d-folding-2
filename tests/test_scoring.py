from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rna3d_local.scoring import score_submission
from rna3d_local.errors import PipelineError


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_fake_metric(path: Path) -> None:
    _write_text(
        path,
        (
            "def score(solution, submission, row_id_column_name, usalign_bin_hint=None):\n"
            "    if row_id_column_name not in solution.columns or row_id_column_name not in submission.columns:\n"
            "        raise ValueError('missing row id column')\n"
            "    return float(len(solution))\n"
        ),
    )


def _write_metric_with_coord_compare(path: Path) -> None:
    _write_text(
        path,
        (
            "def score(solution, submission, row_id_column_name, usalign_bin_hint=None):\n"
            "    _ = bool(solution.iloc[0]['x_1'] > -1e6)\n"
            "    return 1.0\n"
        ),
    )


def test_score_submission_per_target_with_parquet_solution(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    sol = tmp_path / "sol.parquet"
    metric_py = tmp_path / "metric.py"
    usalign = tmp_path / "USalign"

    _write_text(
        sample,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0,0,0\n"
        "T1_2,C,2,0,0,0\n"
        "T2_1,G,1,0,0,0\n"
        "T2_2,U,2,0,0,0\n",
    )
    _write_text(
        sub,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0.1,0.1,0.1\n"
        "T1_2,C,2,0.2,0.2,0.2\n"
        "T2_1,G,1,0.3,0.3,0.3\n"
        "T2_2,U,2,0.4,0.4,0.4\n",
    )
    pd.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 1.0, "y_1": 1.0, "z_1": 1.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 2.0, "y_1": 2.0, "z_1": 2.0, "chain": "A", "copy": 1},
            {"ID": "T2_1", "resname": "G", "resid": 1, "x_1": 3.0, "y_1": 3.0, "z_1": 3.0, "chain": "A", "copy": 1},
            {"ID": "T2_2", "resname": "U", "resid": 2, "x_1": 4.0, "y_1": 4.0, "z_1": 4.0, "chain": "A", "copy": 1},
        ]
    ).to_parquet(sol, index=False)
    _write_fake_metric(metric_py)
    usalign.write_text("dummy", encoding="utf-8")

    result = score_submission(
        sample_submission=sample,
        solution=sol,
        submission=sub,
        metric_py=metric_py,
        usalign_bin=usalign,
        per_target=True,
        keep_tmp=False,
    )
    assert result.per_target is not None
    assert set(result.per_target.keys()) == {"T1", "T2"}
    assert result.per_target["T1"] == 2.0
    assert result.per_target["T2"] == 2.0
    assert result.score == 2.0


def test_score_submission_global_mode_with_parquet_solution(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    sol = tmp_path / "sol.parquet"
    metric_py = tmp_path / "metric.py"
    usalign = tmp_path / "USalign"

    _write_text(
        sample,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0,0,0\n"
        "T1_2,C,2,0,0,0\n",
    )
    _write_text(
        sub,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0.1,0.1,0.1\n"
        "T1_2,C,2,0.2,0.2,0.2\n",
    )
    pd.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 1.0, "y_1": 1.0, "z_1": 1.0, "chain": "A", "copy": 1},
            {"ID": "T1_2", "resname": "C", "resid": 2, "x_1": 2.0, "y_1": 2.0, "z_1": 2.0, "chain": "A", "copy": 1},
        ]
    ).to_parquet(sol, index=False)
    _write_fake_metric(metric_py)
    usalign.write_text("dummy", encoding="utf-8")

    result = score_submission(
        sample_submission=sample,
        solution=sol,
        submission=sub,
        metric_py=metric_py,
        usalign_bin=usalign,
        per_target=False,
        keep_tmp=False,
    )
    assert result.per_target is None
    assert result.score == 2.0


def test_score_submission_handles_nullable_coords_without_metric_typeerror(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    sol = tmp_path / "sol.parquet"
    metric_py = tmp_path / "metric.py"
    usalign = tmp_path / "USalign"

    _write_text(
        sample,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0,0,0\n",
    )
    _write_text(
        sub,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0.1,0.1,0.1\n",
    )
    pd.DataFrame(
        [
            {
                "ID": "T1_1",
                "resname": "A",
                "resid": 1,
                "x_1": None,
                "y_1": 1.0,
                "z_1": 1.0,
                "chain": "A",
                "copy": 1,
            }
        ]
    ).to_parquet(sol, index=False)
    _write_metric_with_coord_compare(metric_py)
    usalign.write_text("dummy", encoding="utf-8")

    result = score_submission(
        sample_submission=sample,
        solution=sol,
        submission=sub,
        metric_py=metric_py,
        usalign_bin=usalign,
        per_target=True,
        keep_tmp=False,
    )
    assert result.per_target is not None
    assert result.score == 1.0


def test_score_submission_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    sol = tmp_path / "sol.parquet"
    metric_py = tmp_path / "metric.py"
    usalign = tmp_path / "USalign"

    _write_text(
        sample,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0,0,0\n",
    )
    _write_text(
        sub,
        "ID,resname,resid,x_1,y_1,z_1\n"
        "T1_1,A,1,0.1,0.1,0.1\n",
    )
    pd.DataFrame(
        [
            {"ID": "T1_1", "resname": "A", "resid": 1, "x_1": 1.0, "y_1": 1.0, "z_1": 1.0, "chain": "A", "copy": 1},
        ]
    ).to_parquet(sol, index=False)
    _write_fake_metric(metric_py)
    usalign.write_text("dummy", encoding="utf-8")

    with pytest.raises(PipelineError) as e:
        score_submission(
            sample_submission=sample,
            solution=sol,
            submission=sub,
            metric_py=metric_py,
            usalign_bin=usalign,
            per_target=True,
            keep_tmp=False,
            chunk_size=0,
        )
    msg = str(e.value)
    assert msg.startswith("[SCORE]")
    assert "chunk_size invalido" in msg
