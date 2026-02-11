from __future__ import annotations

from pathlib import Path

import polars as pl

from rna3d_local.datasets import build_train_cv_fold_dataset, prepare_train_labels_parquet


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_train_fold_writes_target_sequences_csv(tmp_path: Path) -> None:
    repo_root = tmp_path

    # Minimal vendor layout required by build_train_cv_fold_dataset (not executed in this test).
    _write_text(repo_root / "vendor/tm_score_permutechains/metric.py", "def score(*args, **kwargs):\n    return 0.0\n")
    _write_text(repo_root / "vendor/usalign/USalign", "dummy\n")

    input_dir = repo_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    train_sequences = input_dir / "train_sequences.csv"
    _write_text(
        train_sequences,
        "target_id,sequence,temporal_cutoff\n"
        "T1,AC,2022-01-01\n"
        "T2,GU,2022-01-01\n",
    )
    train_labels = input_dir / "train_labels.csv"
    _write_text(
        train_labels,
        "ID,resname,resid,x_1,y_1,z_1,chain,copy\n"
        "T1_1,A,1,0,0,0,A,1\n"
        "T1_2,C,2,1,0,0,A,1\n"
        "T2_1,G,1,0,1,0,A,1\n"
        "T2_2,U,2,1,1,0,A,1\n",
    )

    labels_parquet_dir = repo_root / "data/derived/train_labels_parquet"
    prepare_train_labels_parquet(
        repo_root=repo_root,
        train_labels_csv=train_labels,
        out_dir=labels_parquet_dir,
        rows_per_file=2,
        compression="zstd",
    )

    targets_parquet = repo_root / "data/derived/train_cv_targets/targets.parquet"
    targets_parquet.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"target_id": ["T1", "T2"], "fold_id": [0, 0]}).write_parquet(targets_parquet)

    fold_dir = repo_root / "data/derived/train_cv/fold0"
    build_train_cv_fold_dataset(
        repo_root=repo_root,
        input_dir=input_dir,
        targets_parquet=targets_parquet,
        fold_id=0,
        out_dir=fold_dir,
        train_labels_parquet_dir=labels_parquet_dir,
    )

    seq_out = fold_dir / "target_sequences.csv"
    assert seq_out.exists()
    df = pl.read_csv(seq_out)
    assert df.columns == ["target_id", "sequence", "temporal_cutoff"]
    assert set(df.get_column("target_id").to_list()) == {"T1", "T2"}
