from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

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
    meta = {"dataset_dir": _rel_or_abs(dataset_dir, repo), "submission": _rel_or_abs(submission, repo)}
    write_score_artifacts(out_dir=out_dir, result=result, meta=meta)
    print(json.dumps({"score": result.score, "out_dir": _rel_or_abs(out_dir, repo)}, indent=2, sort_keys=True))
    return 0
