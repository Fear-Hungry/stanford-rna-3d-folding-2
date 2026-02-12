from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    LabelStoreConfig,
    TableReadConfig,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
    scan_labels,
    scan_table,
    sink_partitioned_parquet,
)
from .errors import raise_error
from .splits import assign_folds_from_clusters, cluster_targets_minhash
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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


def _target_ids_for_fold(*, targets_parquet: Path, fold_id: int, location: str) -> list[str]:
    lf = scan_table(
        config=TableReadConfig(
            path=targets_parquet,
            stage="DATA",
            location=location,
            columns=("target_id", "fold_id"),
        )
    )
    out = collect_streaming(
        lf=lf.filter(pl.col("fold_id") == int(fold_id)).select(pl.col("target_id").cast(pl.Utf8)),
        stage="DATA",
        location=location,
    )
    target_ids = out.get_column("target_id").to_list()
    if not target_ids:
        raise_error("DATA", location, "fold sem targets", impact="0", examples=[str(fold_id)])
    return target_ids


def prepare_train_labels_parquet(
    *,
    repo_root: Path,
    train_labels_csv: Path,
    out_dir: Path,
    rows_per_file: int = 2_000_000,
    compression: str = "zstd",
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> Path:
    """
    Convert train_labels.csv to canonical partitioned parquet (part-*.parquet).
    """
    location = "src/rna3d_local/datasets.py:prepare_train_labels_parquet"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    if not train_labels_csv.exists():
        raise_error("DATA", location, "train_labels.csv ausente", impact="1", examples=[str(train_labels_csv)])
    required = ["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"]
    scan = scan_table(
        config=TableReadConfig(
            path=train_labels_csv,
            stage="DATA",
            location=location,
            columns=tuple(required),
        )
    )
    canonical_scan = scan.select(
        pl.col("ID").cast(pl.Utf8),
        pl.col("resname").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("x_1").cast(pl.Float64),
        pl.col("y_1").cast(pl.Float64),
        pl.col("z_1").cast(pl.Float64),
        pl.col("chain").cast(pl.Utf8),
        pl.col("copy").cast(pl.Int32),
    )
    partition_info = sink_partitioned_parquet(
        lf=canonical_scan,
        out_dir=out_dir,
        rows_per_file=rows_per_file,
        compression=compression,
        stage="DATA",
        location=location,
    )
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)

    part_paths: list[Path] = partition_info["parts"]
    sha_parts = {p.name: sha256_file(p) for p in part_paths}
    manifest = {
        "dataset_type": "train_labels_canonical_parquet",
        "created_utc": _utc_now(),
        "paths": {
            "source_csv": _rel_or_abs(train_labels_csv, repo_root),
            "parts": [_rel_or_abs(p, repo_root) for p in part_paths],
        },
        "params": {
            "rows_per_file": int(rows_per_file),
            "compression": compression,
        },
        "stats": {
            "n_rows": int(partition_info["n_rows"]),
            "n_files": int(partition_info["n_files"]),
            "rows_per_file_actual": partition_info["rows_per_file_actual"],
        },
        "sha256": sha_parts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def prepare_train_labels_clean(
    *,
    repo_root: Path,
    train_labels_parquet_dir: Path,
    out_dir: Path,
    train_sequences_csv: Path,
    rows_per_file: int = 2_000_000,
    compression: str = "zstd",
    require_complete_targets: bool = True,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> Path:
    """
    Build a cleaned parquet labels store by dropping rows with null xyz coordinates.
    This is an explicit data-preparation step and never runs implicitly in the pipeline.
    """
    location = "src/rna3d_local/datasets.py:prepare_train_labels_clean"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    if not train_sequences_csv.exists():
        raise_error("DATA", location, "train_sequences.csv ausente", impact="1", examples=[str(train_sequences_csv)])

    required = ("ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy")
    labels_scan = scan_labels(
        config=LabelStoreConfig(
            labels_parquet_dir=train_labels_parquet_dir,
            required_columns=required,
            stage="DATA",
            location=location,
        )
    )
    rows_in = int(
        collect_streaming(
            lf=labels_scan.select(pl.len().alias("n")),
            stage="DATA",
            location=location,
        ).get_column("n")[0]
    )
    if rows_in == 0:
        raise_error("DATA", location, "labels parquet de entrada vazio", impact="0", examples=[str(train_labels_parquet_dir)])

    cleaned_scan = (
        labels_scan.filter(pl.col("x_1").is_not_null() & pl.col("y_1").is_not_null() & pl.col("z_1").is_not_null())
        .select(
            pl.col("ID").cast(pl.Utf8),
            pl.col("resname").cast(pl.Utf8),
            pl.col("resid").cast(pl.Int32),
            pl.col("x_1").cast(pl.Float64),
            pl.col("y_1").cast(pl.Float64),
            pl.col("z_1").cast(pl.Float64),
            pl.col("chain").cast(pl.Utf8),
            pl.col("copy").cast(pl.Int32),
        )
    )
    partition_info = sink_partitioned_parquet(
        lf=cleaned_scan,
        out_dir=out_dir,
        rows_per_file=rows_per_file,
        compression=compression,
        stage="DATA",
        location=location,
    )
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)

    rows_out = int(partition_info["n_rows"])
    rows_dropped = int(rows_in - rows_out)
    if rows_out <= 0:
        raise_error(
            "DATA",
            location,
            "limpeza removeu todas as linhas de labels",
            impact=str(rows_dropped),
            examples=[str(train_labels_parquet_dir)],
        )

    cleaned_validation_scan = scan_labels(
        config=LabelStoreConfig(
            labels_parquet_dir=out_dir,
            required_columns=required,
            stage="DATA",
            location=location,
        )
    )
    null_stats = collect_streaming(
        lf=cleaned_validation_scan.select(
            pl.col("x_1").is_null().sum().alias("x_1"),
            pl.col("y_1").is_null().sum().alias("y_1"),
            pl.col("z_1").is_null().sum().alias("z_1"),
        ),
        stage="DATA",
        location=location,
    ).row(0, named=True)
    bad_nulls = [f"{k}:{int(v)}" for k, v in null_stats.items() if int(v) > 0]
    if bad_nulls:
        raise_error(
            "DATA",
            location,
            "labels limpos ainda contem coordenadas nulas",
            impact=str(len(bad_nulls)),
            examples=bad_nulls,
        )

    missing_targets_count = 0
    missing_target_examples: list[str] = []
    if require_complete_targets:
        seq_targets = scan_table(
            config=TableReadConfig(
                path=train_sequences_csv,
                stage="DATA",
                location=location,
                columns=("target_id",),
            )
        ).select(pl.col("target_id").cast(pl.Utf8)).unique()
        labels_targets = (
            cleaned_validation_scan.with_columns(pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_[0-9]+$", 1).alias("target_id"))
            .select(pl.col("target_id").cast(pl.Utf8))
            .drop_nulls()
            .unique()
        )
        missing_targets = seq_targets.join(labels_targets, on="target_id", how="anti")
        missing_targets_count = int(
            collect_streaming(
                lf=missing_targets.select(pl.len().alias("n")),
                stage="DATA",
                location=location,
            ).get_column("n")[0]
        )
        if missing_targets_count > 0:
            missing_target_examples = (
                collect_streaming(
                    lf=missing_targets.select("target_id").sort("target_id").head(8),
                    stage="DATA",
                    location=location,
                )
                .get_column("target_id")
                .to_list()
            )
            raise_error(
                "DATA",
                location,
                "targets sem labels apos limpeza de coordenadas nulas",
                impact=str(missing_targets_count),
                examples=missing_target_examples,
            )

    part_paths: list[Path] = partition_info["parts"]
    sha_parts = {p.name: sha256_file(p) for p in part_paths}
    manifest = {
        "dataset_type": "train_labels_clean_nonnull_xyz",
        "created_utc": _utc_now(),
        "paths": {
            "source_labels_parquet_dir": _rel_or_abs(train_labels_parquet_dir, repo_root),
            "source_train_sequences_csv": _rel_or_abs(train_sequences_csv, repo_root),
            "parts": [_rel_or_abs(p, repo_root) for p in part_paths],
        },
        "params": {
            "rows_per_file": int(rows_per_file),
            "compression": compression,
            "require_complete_targets": bool(require_complete_targets),
        },
        "stats": {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "rows_dropped_null_xyz": rows_dropped,
            "n_files": int(partition_info["n_files"]),
            "rows_per_file_actual": partition_info["rows_per_file_actual"],
            "targets_missing_after_clean": int(missing_targets_count),
        },
        "sha256": sha_parts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


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
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> Path:
    """
    Builds `targets.parquet` with deterministic cluster_id and fold_id, using train_sequences.csv.
    """
    location = "src/rna3d_local/datasets.py:build_train_cv_targets"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    train_seq = input_dir / "train_sequences.csv"
    if not train_seq.exists():
        raise_error("DATA", location, "train_sequences.csv ausente", impact="1", examples=[str(train_seq)])

    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=train_seq,
                stage="DATA",
                location=location,
                columns=("target_id", "sequence"),
            )
        ).select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)),
        stage="DATA",
        location=location,
    )
    assert_row_budget(
        stage="DATA",
        location=location,
        rows=int(df.height),
        max_rows_in_memory=max_rows_in_memory,
        label="train_sequences",
    )
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
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)

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
    out_path: Path,
    target_ids: list[str],
    train_labels_parquet_dir: Path,
    native_models: int = 40,
    missing_value: float = -1e18,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> Path:
    """
    Export a solution file compatible with the vendored metric (expects x_1..x_40 etc).

    The official train_labels.csv only provides x_1/y_1/z_1; we treat that as native model 1
    and fill x_2..x_40 with missing_value. This is explicit and deterministic.
    """
    location = "src/rna3d_local/datasets.py:export_train_solution_for_targets"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    if not target_ids:
        raise_error("DATA", location, "lista de target_ids vazia", impact="0", examples=[])

    scan = scan_labels(
        config=LabelStoreConfig(
            labels_parquet_dir=train_labels_parquet_dir,
            required_columns=("ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"),
            stage="DATA",
            location=location,
        )
    )

    # derive target_id from ID prefix before last underscore
    scan = scan.with_columns(
        pl.col("ID").cast(pl.Utf8).str.extract(r"^(.*)_\d+$", 1).alias("_target_id")
    ).filter(pl.col("_target_id").is_in(target_ids))

    target_order = pl.DataFrame(
        {
            "_target_id": [str(t) for t in target_ids],
            "_target_ord": list(range(len(target_ids))),
        }
    ).lazy()
    scan = scan.join(target_order, on="_target_id", how="inner")

    present_targets = collect_streaming(
        lf=scan.select(pl.col("_target_id").cast(pl.Utf8)).unique(),
        stage="DATA",
        location=location,
    ).get_column("_target_id").to_list()
    present_set = set(str(x) for x in present_targets)
    missing_targets = [str(t) for t in target_ids if str(t) not in present_set]
    if missing_targets:
        raise_error(
            "DATA",
            location,
            "targets sem labels canonicos para export da solucao",
            impact=str(len(missing_targets)),
            examples=missing_targets[:8],
        )

    required = ["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"]
    cols = set(scan.collect_schema().names())
    for col in required:
        if col not in cols:
            raise_error("DATA", location, "coluna obrigatoria ausente em labels canonicos", impact="1", examples=[col])

    base = scan.select(
        pl.col("ID").cast(pl.Utf8),
        pl.col("resname").cast(pl.Utf8),
        pl.col("resid").cast(pl.Int32),
        pl.col("x_1").cast(pl.Float64),
        pl.col("y_1").cast(pl.Float64),
        pl.col("z_1").cast(pl.Float64),
        pl.col("chain").cast(pl.Utf8),
        pl.col("copy").cast(pl.Int32),
        pl.col("_target_ord").cast(pl.Int32),
    )

    # Add missing native models columns (x_2..x_N, y_2.., z_2..).
    # The metric iterates native_cnt in 1..40.
    exprs = []
    for i in range(2, native_models + 1):
        exprs.append(pl.lit(missing_value).alias(f"x_{i}"))
        exprs.append(pl.lit(missing_value).alias(f"y_{i}"))
        exprs.append(pl.lit(missing_value).alias(f"z_{i}"))

    base = base.sort(["_target_ord", "resid"])
    wide = base.with_columns(exprs)

    # reorder to match validation_labels-like schema: ID,resname,resid,(x_1,y_1,z_1..x_N,y_N,z_N),chain,copy
    cols = ["ID", "resname", "resid"]
    for i in range(1, native_models + 1):
        cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    cols.extend(["chain", "copy"])
    wide = wide.select([*cols, "_target_ord"]).drop("_target_ord")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    try:
        # Avoid materializing the full wide frame in memory.
        wide.sink_parquet(out_path, compression="zstd")
    except Exception as e:  # noqa: BLE001
        raise_error("DATA", location, "falha ao gravar solution parquet em streaming", impact="1", examples=[f"{type(e).__name__}:{e}"])
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    return out_path


def build_train_cv_fold_dataset(
    *,
    repo_root: Path,
    input_dir: Path,
    targets_parquet: Path,
    fold_id: int,
    out_dir: Path,
    train_labels_parquet_dir: Path,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> Path:
    """
    Build a self-contained local scoring dataset for a single CV fold:
    - sample_submission.csv (template for the fold)
    - solution.parquet (wide, compatible with vendored metric)
    - manifest.json (paths + hashes + vendored metric + usalign)
    """
    location = "src/rna3d_local/datasets.py:build_train_cv_fold_dataset"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    if not targets_parquet.exists():
        raise_error("DATA", location, "targets.parquet nao encontrado", impact="1", examples=[str(targets_parquet)])
    metric_py = repo_root / "vendor" / "tm_score_permutechains" / "metric.py"
    usalign_bin = repo_root / "vendor" / "usalign" / "USalign"
    for p in (metric_py, usalign_bin):
        if not p.exists():
            raise_error("DATA", location, "vendor obrigatorio ausente (rode rna3d_local vendor)", impact="1", examples=[str(p)])

    target_ids = _target_ids_for_fold(targets_parquet=targets_parquet, fold_id=fold_id, location=location)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample_submission.csv"
    sol_path = out_dir / "solution.parquet"
    target_sequences_path = out_dir / "target_sequences.csv"
    seq_csv = input_dir / "train_sequences.csv"
    make_sample_submission_for_targets(
        sequences_csv=seq_csv,
        out_path=sample_path,
        target_ids=target_ids,
        memory_budget_mb=memory_budget_mb,
    )
    export_target_sequences_for_targets(
        sequences_csv=seq_csv,
        out_path=target_sequences_path,
        target_ids=target_ids,
        memory_budget_mb=memory_budget_mb,
    )
    export_train_solution_for_targets(
        out_path=sol_path,
        target_ids=target_ids,
        train_labels_parquet_dir=train_labels_parquet_dir,
        memory_budget_mb=memory_budget_mb,
    )

    sha = {
        "sample_submission.csv": sha256_file(sample_path),
        "target_sequences.csv": sha256_file(target_sequences_path),
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
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> Path:
    """
    Create a sample_submission-like template (keys + resname/resid) for a subset of targets.
    Coordinates are set to 0 (template only, not used for scoring truth).
    """
    location = "src/rna3d_local/datasets.py:make_sample_submission_for_targets"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    if not sequences_csv.exists():
        raise_error("DATA", location, "sequences_csv nao encontrado", impact="1", examples=[str(sequences_csv)])
    if not target_ids:
        raise_error("DATA", location, "lista de target_ids vazia", impact="0", examples=[])

    df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=sequences_csv,
                stage="DATA",
                location=location,
                columns=("target_id", "sequence"),
            )
        )
        .select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8))
        .filter(pl.col("target_id").is_in(target_ids)),
        stage="DATA",
        location=location,
    )
    found = set(df.get_column("target_id").to_list())
    missing = [t for t in target_ids if t not in found]
    if missing:
        raise_error("DATA", location, "target_ids ausentes em sequences_csv", impact=str(len(missing)), examples=missing[:8])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ID", "resname", "resid"]
    for m in range(1, pred_models + 1):
        header.extend([f"x_{m}", f"y_{m}", f"z_{m}"])
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for tid, seq in df.select("target_id", "sequence").iter_rows():
            seq = str(seq)
            for i, base in enumerate(seq, start=1):
                row: list[object] = [f"{tid}_{i}", str(base), i]
                for _ in range(1, pred_models + 1):
                    row.extend([0.0, 0.0, 0.0])
                writer.writerow(row)
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    return out_path


def export_target_sequences_for_targets(
    *,
    sequences_csv: Path,
    out_path: Path,
    target_ids: list[str],
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
) -> Path:
    """
    Export a target_sequences-like CSV for a subset of targets.
    Required columns: target_id,sequence,temporal_cutoff.
    """
    location = "src/rna3d_local/datasets.py:export_target_sequences_for_targets"
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    if not sequences_csv.exists():
        raise_error("DATA", location, "sequences_csv nao encontrado", impact="1", examples=[str(sequences_csv)])
    if not target_ids:
        raise_error("DATA", location, "lista de target_ids vazia", impact="0", examples=[])

    df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=sequences_csv,
                stage="DATA",
                location=location,
                columns=("target_id", "sequence", "temporal_cutoff"),
            )
        )
        .select(
            pl.col("target_id").cast(pl.Utf8),
            pl.col("sequence").cast(pl.Utf8),
            pl.col("temporal_cutoff").cast(pl.Utf8),
        )
        .filter(pl.col("target_id").is_in(target_ids)),
        stage="DATA",
        location=location,
    ).sort("target_id")
    found = set(df.get_column("target_id").to_list())
    missing = [t for t in target_ids if t not in found]
    if missing:
        raise_error("DATA", location, "target_ids ausentes em sequences_csv", impact=str(len(missing)), examples=missing[:8])

    parsed = df.with_columns(pl.col("temporal_cutoff").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_cutoff_dt"))
    if int(parsed.get_column("_cutoff_dt").null_count()) > 0:
        bad = (
            parsed.filter(pl.col("_cutoff_dt").is_null())
            .select(pl.col("target_id").cast(pl.Utf8))
            .head(8)
            .get_column("target_id")
            .to_list()
        )
        raise_error("DATA", location, "temporal_cutoff invalido; esperado YYYY-MM-DD", impact=str(int(parsed.get_column("_cutoff_dt").null_count())), examples=bad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.select("target_id", "sequence", "temporal_cutoff").write_csv(out_path)
    assert_memory_budget(stage="DATA", location=location, budget_mb=memory_budget_mb)
    return out_path
