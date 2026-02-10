from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .errors import raise_error
from .splits import assign_folds_from_clusters, cluster_targets_minhash
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_type: str
    created_utc: str
    sample_submission: str
    solution: str
    metric_py: str
    usalign_bin: str
    sha256: dict[str, str]


def _write_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.write_text(json.dumps(manifest.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_public_validation_dataset(*, repo_root: Path, input_dir: Path, out_dir: Path) -> Path:
    """
    Builds a local dataset that reproduces Kaggle Public LB evaluation:
    - sample_submission.csv (contract)
    - validation_labels.csv (solution; matches sample keys)
    """
    location = "src/rna3d_local/datasets.py:build_public_validation_dataset"
    sample = input_dir / "sample_submission.csv"
    solution = input_dir / "validation_labels.csv"
    metric_py = repo_root / "vendor" / "tm_score_permutechains" / "metric.py"
    usalign_bin = repo_root / "vendor" / "usalign" / "USalign"

    for p in (sample, solution, metric_py, usalign_bin):
        if not p.exists():
            raise_error(
                "DATA",
                location,
                "arquivo obrigatorio ausente para dataset public_validation",
                impact="1",
                examples=[str(p)],
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy small CSVs into dataset dir for portability.
    (out_dir / "sample_submission.csv").write_bytes(sample.read_bytes())
    (out_dir / "validation_labels.csv").write_bytes(solution.read_bytes())

    sha = {
        "sample_submission.csv": sha256_file(out_dir / "sample_submission.csv"),
        "validation_labels.csv": sha256_file(out_dir / "validation_labels.csv"),
        "metric.py": sha256_file(metric_py),
        "USalign": sha256_file(usalign_bin),
    }
    manifest = DatasetManifest(
        dataset_type="public_validation",
        created_utc=_utc_now(),
        sample_submission=str((out_dir / "sample_submission.csv").relative_to(repo_root)),
        solution=str((out_dir / "validation_labels.csv").relative_to(repo_root)),
        metric_py=str(metric_py.relative_to(repo_root)),
        usalign_bin=str(usalign_bin.relative_to(repo_root)),
        sha256=sha,
    )
    _write_manifest(out_dir / "manifest.json", manifest)
    return out_dir / "manifest.json"


def build_train_cv_targets(
    *,
    repo_root: Path,
    input_dir: Path,
    out_dir: Path,
    n_folds: int,
    seed: int,
    k: int = 5,
    n_hashes: int = 32,
    bands: int = 8,
) -> Path:
    """
    Builds `targets.parquet` with deterministic cluster_id and fold_id, using train_sequences.csv.
    """
    location = "src/rna3d_local/datasets.py:build_train_cv_targets"
    train_seq = input_dir / "train_sequences.csv"
    if not train_seq.exists():
        raise_error("DATA", location, "train_sequences.csv ausente", impact="1", examples=[str(train_seq)])

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(train_seq, infer_schema_length=1000)
    for col in ("target_id", "sequence"):
        if col not in df.columns:
            raise_error("DATA", location, "coluna obrigatoria ausente em train_sequences.csv", impact="1", examples=[col])

    target_ids = df.get_column("target_id").cast(pl.Utf8).to_list()
    seqs = df.get_column("sequence").cast(pl.Utf8).to_list()
    clustered = cluster_targets_minhash(target_ids=target_ids, sequences=seqs, k=k, n_hashes=n_hashes, bands=bands)
    folds = assign_folds_from_clusters(cluster_ids=clustered.cluster_ids, n_folds=n_folds, seed=seed)

    out = df.with_columns(
        pl.Series("cluster_id", clustered.cluster_ids),
        pl.Series("fold_id", folds).cast(pl.Int32),
    )
    out_path = out_dir / "targets.parquet"
    out.write_parquet(out_path)

    sha = {"targets.parquet": sha256_file(out_path), "train_sequences.csv": sha256_file(train_seq)}
    manifest = {
        "dataset_type": "train_cv_targets",
        "created_utc": _utc_now(),
        "params": {"n_folds": n_folds, "seed": seed, "k": k, "n_hashes": n_hashes, "bands": bands},
        "paths": {
            "targets": str(out_path.relative_to(repo_root)),
            "train_sequences": str(train_seq.relative_to(repo_root)),
        },
        "sha256": sha,
        "stats": {"n_targets": int(out.height), "n_clusters": int(clustered.n_clusters)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir / "manifest.json"


def export_train_solution_for_targets(
    *,
    repo_root: Path,
    input_dir: Path,
    out_path: Path,
    target_ids: list[str],
    native_models: int = 40,
    missing_value: float = -1e18,
) -> Path:
    """
    Export a solution file compatible with the vendored metric (expects x_1..x_40 etc).

    The official train_labels.csv only provides x_1/y_1/z_1; we treat that as native model 1
    and fill x_2..x_40 with missing_value. This is explicit and deterministic.
    """
    location = "src/rna3d_local/datasets.py:export_train_solution_for_targets"
    train_labels = input_dir / "train_labels.csv"
    if not train_labels.exists():
        raise_error("DATA", location, "train_labels.csv ausente", impact="1", examples=[str(train_labels)])
    if not target_ids:
        raise_error("DATA", location, "lista de target_ids vazia", impact="0", examples=[])

    # read lazily to avoid loading full 318MB
    scan = pl.scan_csv(train_labels, infer_schema_length=1000)
    # derive target_id from ID prefix before last underscore
    scan = scan.with_columns(
        pl.col("ID").cast(pl.Utf8).str.split("_").list.get(0).alias("_target_id")
    ).filter(pl.col("_target_id").is_in(target_ids))

    required = ["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"]
    cols = set(scan.collect_schema().names())
    for col in required:
        if col not in cols:
            raise_error("DATA", location, "coluna obrigatoria ausente em train_labels.csv", impact="1", examples=[col])

    base = scan.select(required)

    # Add missing native models columns (x_2..x_N, y_2.., z_2..).
    # The metric iterates native_cnt in 1..40.
    exprs = []
    for i in range(2, native_models + 1):
        exprs.append(pl.lit(missing_value).alias(f"x_{i}"))
        exprs.append(pl.lit(missing_value).alias(f"y_{i}"))
        exprs.append(pl.lit(missing_value).alias(f"z_{i}"))

    wide = base.with_columns(exprs)

    # reorder to match validation_labels-like schema: ID,resname,resid,(x_1,y_1,z_1..x_N,y_N,z_N),chain,copy
    cols = ["ID", "resname", "resid"]
    for i in range(1, native_models + 1):
        cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    cols.extend(["chain", "copy"])
    wide = wide.select(cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # parquet is much faster and smaller than CSV; metric wrapper can read parquet too.
    wide.collect(streaming=True).write_parquet(out_path)
    return out_path


def build_train_cv_fold_dataset(
    *,
    repo_root: Path,
    input_dir: Path,
    targets_parquet: Path,
    fold_id: int,
    out_dir: Path,
) -> Path:
    """
    Build a self-contained local scoring dataset for a single CV fold:
    - sample_submission.csv (template for the fold)
    - solution.parquet (wide, compatible with vendored metric)
    - manifest.json (paths + hashes + vendored metric + usalign)
    """
    location = "src/rna3d_local/datasets.py:build_train_cv_fold_dataset"
    if not targets_parquet.exists():
        raise_error("DATA", location, "targets.parquet nao encontrado", impact="1", examples=[str(targets_parquet)])
    metric_py = repo_root / "vendor" / "tm_score_permutechains" / "metric.py"
    usalign_bin = repo_root / "vendor" / "usalign" / "USalign"
    for p in (metric_py, usalign_bin):
        if not p.exists():
            raise_error("DATA", location, "vendor obrigatorio ausente (rode rna3d_local vendor)", impact="1", examples=[str(p)])

    df = pl.read_parquet(targets_parquet)
    if "target_id" not in df.columns or "fold_id" not in df.columns:
        raise_error("DATA", location, "targets.parquet sem colunas esperadas", impact="1", examples=df.columns[:8])
    target_ids = (
        df.filter(pl.col("fold_id") == int(fold_id))
        .get_column("target_id")
        .cast(pl.Utf8)
        .to_list()
    )
    if not target_ids:
        raise_error("DATA", location, "fold sem targets", impact="0", examples=[str(fold_id)])

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample_submission.csv"
    sol_path = out_dir / "solution.parquet"
    seq_csv = input_dir / "train_sequences.csv"
    make_sample_submission_for_targets(sequences_csv=seq_csv, out_path=sample_path, target_ids=target_ids)
    export_train_solution_for_targets(repo_root=repo_root, input_dir=input_dir, out_path=sol_path, target_ids=target_ids)

    sha = {
        "sample_submission.csv": sha256_file(sample_path),
        "solution.parquet": sha256_file(sol_path),
        "metric.py": sha256_file(metric_py),
        "USalign": sha256_file(usalign_bin),
    }
    manifest = DatasetManifest(
        dataset_type="train_cv_fold",
        created_utc=_utc_now(),
        sample_submission=str(sample_path.relative_to(repo_root)),
        solution=str(sol_path.relative_to(repo_root)),
        metric_py=str(metric_py.relative_to(repo_root)),
        usalign_bin=str(usalign_bin.relative_to(repo_root)),
        sha256=sha,
    )
    _write_manifest(out_dir / "manifest.json", manifest)
    return out_dir / "manifest.json"


def make_sample_submission_for_targets(
    *,
    sequences_csv: Path,
    out_path: Path,
    target_ids: list[str],
    pred_models: int = 5,
) -> Path:
    """
    Create a sample_submission-like template (keys + resname/resid) for a subset of targets.
    Coordinates are set to 0 (template only, not used for scoring truth).
    """
    location = "src/rna3d_local/datasets.py:make_sample_submission_for_targets"
    if not sequences_csv.exists():
        raise_error("DATA", location, "sequences_csv nao encontrado", impact="1", examples=[str(sequences_csv)])
    if not target_ids:
        raise_error("DATA", location, "lista de target_ids vazia", impact="0", examples=[])

    df = pl.read_csv(sequences_csv, infer_schema_length=1000)
    for col in ("target_id", "sequence"):
        if col not in df.columns:
            raise_error("DATA", location, "coluna obrigatoria ausente em sequences_csv", impact="1", examples=[col])
    df = df.select(["target_id", "sequence"]).with_columns(
        pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)
    )
    df = df.filter(pl.col("target_id").is_in(target_ids))
    found = set(df.get_column("target_id").to_list())
    missing = [t for t in target_ids if t not in found]
    if missing:
        raise_error("DATA", location, "target_ids ausentes em sequences_csv", impact=str(len(missing)), examples=missing[:8])

    rows = []
    for tid, seq in zip(df.get_column("target_id").to_list(), df.get_column("sequence").to_list()):
        seq = str(seq)
        for i, base in enumerate(seq, start=1):
            row = {"ID": f"{tid}_{i}", "resname": base, "resid": i}
            for m in range(1, pred_models + 1):
                row[f"x_{m}"] = 0.0
                row[f"y_{m}"] = 0.0
                row[f"z_{m}"] = 0.0
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out_path)
    return out_path
