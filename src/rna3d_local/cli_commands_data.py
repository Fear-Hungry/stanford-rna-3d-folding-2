from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .cli_commands_common import _find_repo_root, _rel_or_abs, _target_ids_for_fold, _utc_now_compact
from .contracts import validate_submission_against_sample
from .datasets import (
    build_public_validation_dataset,
    build_train_cv_targets,
    build_train_cv_fold_dataset,
    export_train_solution_for_targets,
    make_sample_submission_for_targets,
    prepare_train_labels_clean,
    prepare_train_labels_parquet,
)
from .download import download_competition_files
from .errors import raise_error
from .scoring import score_submission, write_score_artifacts
from .utils import sha256_file
from .vendor import vendor_all

def _cmd_download(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_dir = (repo / args.out).resolve()
    res = download_competition_files(competition=args.competition, out_dir=out_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": args.competition,
        "out_dir": _rel_or_abs(out_dir, repo),
        "files": {k: _rel_or_abs(v, repo) for k, v in res.files.items()},
        "sha256": res.sha256,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0

def _cmd_vendor(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    res = vendor_all(repo_root=repo, metric_kernel=args.metric_kernel)
    print(json.dumps(asdict(res), indent=2, default=str, sort_keys=True))
    return 0

def _cmd_build_dataset(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    input_dir = (repo / args.input).resolve()
    out_dir = (repo / args.out).resolve()
    if args.type == "public_validation":
        manifest = build_public_validation_dataset(repo_root=repo, input_dir=input_dir, out_dir=out_dir)
        print(str(manifest))
        return 0
    if args.type == "train_cv_targets":
        manifest = build_train_cv_targets(
            repo_root=repo,
            input_dir=input_dir,
            out_dir=out_dir,
            n_folds=args.n_folds,
            seed=args.seed,
            k=args.k,
            n_hashes=args.n_hashes,
            bands=args.bands,
            memory_budget_mb=int(args.memory_budget_mb),
            max_rows_in_memory=int(args.max_rows_in_memory),
        )
        print(str(manifest))
        return 0
    raise_error("CLI", "src/rna3d_local/cli.py:_cmd_build_dataset", "tipo de dataset desconhecido", impact="1", examples=[args.type])
    raise AssertionError("unreachable")

def _cmd_prepare_labels_parquet(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    manifest = prepare_train_labels_parquet(
        repo_root=repo,
        train_labels_csv=(repo / args.train_labels_csv).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        rows_per_file=int(args.rows_per_file),
        compression=str(args.compression),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(manifest))
    return 0

def _cmd_prepare_labels_clean(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    manifest = prepare_train_labels_clean(
        repo_root=repo,
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        train_sequences_csv=(repo / args.train_sequences).resolve(),
        rows_per_file=int(args.rows_per_file),
        compression=str(args.compression),
        require_complete_targets=bool(args.require_complete_targets),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(manifest))
    return 0

def _cmd_build_train_fold(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    input_dir = (repo / args.input).resolve()
    targets_parquet = (repo / args.targets).resolve()
    out_dir = (repo / args.out).resolve()
    manifest = build_train_cv_fold_dataset(
        repo_root=repo,
        input_dir=input_dir,
        targets_parquet=targets_parquet,
        fold_id=int(args.fold),
        out_dir=out_dir,
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(manifest))
    return 0

def _cmd_export_train_solution(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    targets_path = (repo / args.targets).resolve()
    if not targets_path.exists():
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_export_train_solution", "targets.parquet nao encontrado", impact="1", examples=[str(targets_path)])
    target_ids = _target_ids_for_fold(
        targets_path=targets_path,
        fold_id=int(args.fold),
        stage="DATA",
        location="src/rna3d_local/cli.py:_cmd_export_train_solution",
    )
    out_path = (repo / args.out).resolve()
    out = export_train_solution_for_targets(
        out_path=out_path,
        target_ids=target_ids,
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(out))
    return 0

def _cmd_make_sample(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    targets_path = (repo / args.targets).resolve()
    sequences_csv = (repo / args.sequences).resolve()
    if not targets_path.exists():
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_make_sample", "targets.parquet nao encontrado", impact="1", examples=[str(targets_path)])
    target_ids = _target_ids_for_fold(
        targets_path=targets_path,
        fold_id=int(args.fold),
        stage="DATA",
        location="src/rna3d_local/cli.py:_cmd_make_sample",
    )
    out_path = (repo / args.out).resolve()
    out = make_sample_submission_for_targets(
        sequences_csv=sequences_csv,
        out_path=out_path,
        target_ids=target_ids,
        memory_budget_mb=int(args.memory_budget_mb),
    )
    print(str(out))
    return 0

def _cmd_check_submission(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    sample = (repo / args.sample).resolve()
    submission = (repo / args.submission).resolve()
    validate_submission_against_sample(sample_path=sample, submission_path=submission)
    print("OK")
    return 0

def _load_dataset_manifest(repo: Path, dataset_dir: Path) -> dict:
    location = "src/rna3d_local/cli.py:_load_dataset_manifest"
    mf = dataset_dir / "manifest.json"
    if not mf.exists():
        raise_error("SCORE", location, "manifest.json nao encontrado no dataset_dir", impact="1", examples=[str(mf)])
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise_error("SCORE", location, "falha ao ler manifest.json", impact="1", examples=[f"{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def _table_columns(*, path: Path, location: str) -> list[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return list(pl.scan_csv(path, infer_schema_length=1000, ignore_errors=False).collect_schema().names())
        if suffix == ".parquet":
            return list(pl.scan_parquet(path).collect_schema().names())
    except Exception as e:  # noqa: BLE001
        raise_error("SCORE", location, "falha ao ler schema da tabela", impact="1", examples=[f"{path.name}:{type(e).__name__}:{e}"])
    raise_error("SCORE", location, "formato nao suportado para leitura de schema", impact="1", examples=[str(path)])
    raise AssertionError("unreachable")


def _coord_model_count_from_columns(*, columns: list[str], location: str) -> int:
    by_model: dict[int, set[str]] = {}
    for col in columns:
        m = re.match(r"^([xyz])_(\d+)$", str(col))
        if m is None:
            continue
        axis = str(m.group(1))
        model = int(m.group(2))
        by_model.setdefault(model, set()).add(axis)
    if not by_model:
        raise_error("SCORE", location, "sample sem colunas de coordenadas x_k/y_k/z_k", impact="1", examples=columns[:8])
    incomplete = [m for m, axes in sorted(by_model.items()) if axes != {"x", "y", "z"}]
    if incomplete:
        raise_error(
            "SCORE",
            location,
            "sample com bloco de coordenadas incompleto (esperado x_k,y_k,z_k)",
            impact=str(len(incomplete)),
            examples=[str(v) for v in incomplete[:8]],
        )
    return int(len(by_model))


def _schema_sha(*, columns: list[str]) -> str:
    return hashlib.sha256(",".join([str(c) for c in columns]).encode("utf-8")).hexdigest()


def _score_meta(
    *,
    repo: Path,
    dataset_dir: Path,
    manifest: dict,
    sample_path: Path,
    solution_path: Path,
    metric_py: Path,
    usalign_bin: Path,
    submission_path: Path,
    location: str,
) -> dict:
    sample_columns = _table_columns(path=sample_path, location=location)
    solution_columns = _table_columns(path=solution_path, location=location)
    sample_schema_sha = _schema_sha(columns=sample_columns)
    solution_schema_sha = _schema_sha(columns=solution_columns)
    n_models = _coord_model_count_from_columns(columns=sample_columns, location=location)

    sha_block = manifest.get("sha256") if isinstance(manifest, dict) else None
    metric_sha = str((sha_block or {}).get("metric.py") or "").strip()
    usalign_sha = str((sha_block or {}).get("USalign") or "").strip()
    if not metric_sha:
        metric_sha = sha256_file(metric_py)
    if not usalign_sha:
        usalign_sha = sha256_file(usalign_bin)

    official_sample = repo / "input" / "stanford-rna-3d-folding-2" / "sample_submission.csv"
    official_schema_sha = None
    if official_sample.exists():
        official_cols = _table_columns(path=official_sample, location=location)
        official_schema_sha = _schema_sha(columns=official_cols)

    if n_models == 5 and official_schema_sha is not None and sample_schema_sha == official_schema_sha:
        regime_id = "kaggle_official_5model"
    else:
        regime_id = f"custom_n{n_models}_{sample_schema_sha[:12]}"

    return {
        "dataset_dir": _rel_or_abs(dataset_dir, repo),
        "submission": _rel_or_abs(submission_path, repo),
        "dataset_type": str(manifest.get("dataset_type", "unknown")),
        "sample_columns": sample_columns,
        "sample_schema_sha": sample_schema_sha,
        "solution_schema_sha": solution_schema_sha,
        "n_models": int(n_models),
        "metric_sha256": metric_sha,
        "usalign_sha256": usalign_sha,
        "regime_id": regime_id,
        "official_sample_schema_sha": official_schema_sha,
    }

def _cmd_score(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    if args.dataset == "public_validation":
        dataset_dir = repo / "data" / "derived" / "public_validation"
    else:
        dataset_dir = (repo / args.dataset_dir).resolve()

    mf = _load_dataset_manifest(repo, dataset_dir)
    sample = repo / mf["sample_submission"]
    solution = repo / mf["solution"]
    metric_py = repo / mf["metric_py"]
    usalign_bin = repo / mf["usalign_bin"]
    submission = (repo / args.submission).resolve()

    result = score_submission(
        sample_submission=sample,
        solution=solution,
        submission=submission,
        metric_py=metric_py,
        usalign_bin=usalign_bin,
        per_target=bool(args.per_target),
        keep_tmp=bool(args.keep_tmp),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
        chunk_size=int(args.chunk_size),
    )

    out_dir = (repo / args.out_dir).resolve()
    if args.out_dir == "runs/auto":
        out_dir = repo / "runs" / f"{_utc_now_compact()}_score"
    meta = _score_meta(
        repo=repo,
        dataset_dir=dataset_dir,
        manifest=mf,
        sample_path=sample,
        solution_path=solution,
        metric_py=metric_py,
        usalign_bin=usalign_bin,
        submission_path=submission,
        location="src/rna3d_local/cli.py:_cmd_score",
    )
    write_score_artifacts(out_dir=out_dir, result=result, meta=meta)
    print(json.dumps({"score": result.score, "out_dir": _rel_or_abs(out_dir, repo)}, indent=2, sort_keys=True))
    return 0
