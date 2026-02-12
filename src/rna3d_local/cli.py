from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .bigdata import DEFAULT_MAX_ROWS_IN_MEMORY, DEFAULT_MEMORY_BUDGET_MB, TableReadConfig, collect_streaming, scan_table
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
from .drfold2 import predict_drfold2
from .download import download_competition_files
from .ensemble import blend_predictions
from .errors import PipelineError, raise_error
from .export import export_submission_from_long
from .gating import assert_submission_allowed
from .kaggle_cli import run_kaggle
from .retrieval import retrieve_template_candidates
from .rnapro import RnaProConfig, infer_rnapro, train_rnapro
from .scoring import score_submission, write_score_artifacts
from .tbm_predictor import predict_tbm
from .template_pt import convert_templates_to_pt_files
from .template_db import build_template_db
from .vendor import DEFAULT_METRIC_KERNEL, vendor_all


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise_error("CLI", "src/rna3d_local/cli.py:_find_repo_root", "repo_root nao encontrado (pyproject.toml ausente)", impact="1", examples=[str(start)])
    raise AssertionError("unreachable")


def _rel_or_abs(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _parse_float_list_arg(*, raw: str, arg_name: str, location: str) -> tuple[float, ...]:
    items = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not items:
        raise_error("CLI", location, f"{arg_name} vazio", impact="1", examples=[str(raw)])
    out: list[float] = []
    for tok in items:
        try:
            out.append(float(tok))
        except ValueError:
            raise_error("CLI", location, f"{arg_name} contem valor invalido", impact="1", examples=[tok])
    return tuple(out)


def _target_ids_for_fold(*, targets_path: Path, fold_id: int, stage: str, location: str) -> list[str]:
    lf = scan_table(
        config=TableReadConfig(
            path=targets_path,
            stage=stage,
            location=location,
            columns=("target_id", "fold_id"),
        )
    )
    out = collect_streaming(
        lf=lf.filter(pl.col("fold_id") == int(fold_id)).select(pl.col("target_id").cast(pl.Utf8)),
        stage=stage,
        location=location,
    )
    target_ids = out.get_column("target_id").to_list()
    if not target_ids:
        raise_error(stage, location, "fold sem targets", impact="0", examples=[str(fold_id)])
    return target_ids


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


def _cmd_build_template_db(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    res = build_template_db(
        repo_root=repo,
        train_sequences_path=(repo / args.train_sequences).resolve(),
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        external_templates_path=(repo / args.external_templates).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        max_train_templates=args.max_train_templates,
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "manifest": _rel_or_abs(res.manifest_path, repo),
                "templates": _rel_or_abs(res.templates_path, repo),
                "index": _rel_or_abs(res.index_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_retrieve_templates(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    res = retrieve_template_candidates(
        repo_root=repo,
        template_index_path=(repo / args.template_index).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        top_k=int(args.top_k),
        kmer_size=int(args.kmer_size),
        length_weight=float(args.length_weight),
        refine_pool_size=int(args.refine_pool_size),
        refine_alignment_weight=float(args.refine_alignment_weight),
        refine_open_gap_score=float(args.refine_open_gap_score),
        refine_extend_gap_score=float(args.refine_extend_gap_score),
        chunk_size=int(args.chunk_size),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "candidates": _rel_or_abs(res.candidates_path, repo),
                "manifest": _rel_or_abs(res.manifest_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_predict_tbm(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    location = "src/rna3d_local/cli.py:_cmd_predict_tbm"
    gap_open_scores = _parse_float_list_arg(raw=str(args.gap_open_scores), arg_name="gap_open_scores", location=location)
    gap_extend_scores = _parse_float_list_arg(raw=str(args.gap_extend_scores), arg_name="gap_extend_scores", location=location)
    res = predict_tbm(
        repo_root=repo,
        retrieval_candidates_path=(repo / args.retrieval).resolve(),
        templates_path=(repo / args.templates).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        n_models=int(args.n_models),
        min_coverage=float(args.min_coverage),
        rerank_pool_size=int(args.rerank_pool_size),
        gap_open_scores=gap_open_scores,
        gap_extend_scores=gap_extend_scores,
        max_variants_per_template=int(args.max_variants_per_template),
        perturbation_scale=float(args.perturbation_scale),
        chunk_size=int(args.chunk_size),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "predictions": _rel_or_abs(res.predictions_path, repo),
                "manifest": _rel_or_abs(res.manifest_path, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_predict_drfold2(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_path = (repo / args.out).resolve()
    work_dir = (repo / args.work_dir).resolve()
    if args.work_dir == "runs/auto_drfold2":
        work_dir = repo / "runs" / f"{_utc_now_compact()}_drfold2_work"
    predictions_path, manifest_path = predict_drfold2(
        repo_root=repo,
        drfold2_root=(repo / args.drfold_root).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=out_path,
        work_dir=work_dir,
        n_models=int(args.n_models),
        python_bin=str(args.python_bin),
        target_limit=None if args.target_limit is None else int(args.target_limit),
        chunk_size=int(args.chunk_size),
        reuse_existing_targets=bool(args.reuse_existing_targets),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(
        json.dumps(
            {
                "predictions": _rel_or_abs(predictions_path, repo),
                "manifest": _rel_or_abs(manifest_path, repo),
                "work_dir": _rel_or_abs(work_dir, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_train_rnapro(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_dir = (repo / args.out_dir).resolve()
    if args.out_dir == "runs/auto_rnapro":
        out_dir = repo / "runs" / f"{_utc_now_compact()}_rnapro_train"
    cfg = RnaProConfig(
        feature_dim=int(args.feature_dim),
        kmer_size=int(args.kmer_size),
        n_models=int(args.n_models),
        seed=int(args.seed),
        min_coverage=float(args.min_coverage),
    )
    model_path = train_rnapro(
        repo_root=repo,
        train_sequences_path=(repo / args.train_sequences).resolve(),
        train_labels_parquet_dir=(repo / args.train_labels_parquet_dir).resolve(),
        out_dir=out_dir,
        config=cfg,
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"model": _rel_or_abs(model_path, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_predict_rnapro(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = infer_rnapro(
        repo_root=repo,
        model_dir=(repo / args.model_dir).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        n_models=None if args.n_models is None else int(args.n_models),
        min_coverage=None if args.min_coverage is None else float(args.min_coverage),
        rerank_pool_multiplier=int(args.rerank_pool_multiplier),
        chunk_size=int(args.chunk_size),
        use_template=str(args.use_template),
        template_features_dir=None if args.template_features_dir is None else (repo / args.template_features_dir).resolve(),
        template_source=str(args.template_source),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_convert_templates_to_pt(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    manifest_path = convert_templates_to_pt_files(
        repo_root=repo,
        templates_submission_path=(repo / args.templates_submission).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        n_models=None if args.n_models is None else int(args.n_models),
        template_source=str(args.template_source),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"manifest": _rel_or_abs(manifest_path, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_ensemble_predict(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = blend_predictions(
        tbm_predictions_path=(repo / args.tbm).resolve(),
        rnapro_predictions_path=(repo / args.rnapro).resolve(),
        out_path=(repo / args.out).resolve(),
        tbm_weight=float(args.tbm_weight),
        rnapro_weight=float(args.rnapro_weight),
        dynamic_by_coverage=bool(args.dynamic_by_coverage),
        coverage_power=float(args.coverage_power),
        coverage_floor=float(args.coverage_floor),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_export_submission(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = export_submission_from_long(
        sample_submission_path=(repo / args.sample).resolve(),
        predictions_long_path=(repo / args.predictions).resolve(),
        out_submission_path=(repo / args.out).resolve(),
        memory_budget_mb=int(args.memory_budget_mb),
        max_rows_in_memory=int(args.max_rows_in_memory),
    )
    print(json.dumps({"submission": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_submit_kaggle(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    report = (repo / args.gating_report).resolve()
    if args.gating_report == "runs/auto":
        report = repo / "runs" / f"{_utc_now_compact()}_gating_report.json"

    assert_submission_allowed(
        sample_path=(repo / args.sample).resolve(),
        submission_path=(repo / args.submission).resolve(),
        report_path=report,
        is_smoke=bool(args.is_smoke),
        is_partial=bool(args.is_partial),
        score_json_path=None if args.score_json is None else (repo / args.score_json).resolve(),
        baseline_score=None if args.baseline_score is None else float(args.baseline_score),
        allow_regression=bool(args.allow_regression),
    )
    run_kaggle(
        ["competitions", "submit", "-c", args.competition, "-f", str((repo / args.submission).resolve()), "-m", args.message],
        cwd=None,
        location="src/rna3d_local/cli.py:_cmd_submit_kaggle",
    )
    print(
        json.dumps(
            {
                "submitted": _rel_or_abs((repo / args.submission).resolve(), repo),
                "gating_report": _rel_or_abs(report, repo),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rna3d_local", add_help=True)
    sp = p.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("download", help="Download competition files via Kaggle API")
    d.add_argument("--competition", default="stanford-rna-3d-folding-2")
    d.add_argument("--out", default="input/stanford-rna-3d-folding-2")
    d.set_defaults(fn=_cmd_download)

    v = sp.add_parser("vendor", help="Vendor Kaggle metric + USalign")
    v.add_argument("--metric-kernel", default=DEFAULT_METRIC_KERNEL)
    v.set_defaults(fn=_cmd_vendor)

    b = sp.add_parser("build-dataset", help="Build derived local datasets")
    b.add_argument("--type", choices=["public_validation", "train_cv_targets"], required=True)
    b.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    b.add_argument("--out", required=True)
    b.add_argument("--n-folds", type=int, default=5)
    b.add_argument("--seed", type=int, default=123)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--n-hashes", type=int, default=32)
    b.add_argument("--bands", type=int, default=8)
    b.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    b.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    b.set_defaults(fn=_cmd_build_dataset)

    lp = sp.add_parser("prepare-labels-parquet", help="Convert train_labels.csv to canonical partitioned parquet")
    lp.add_argument("--train-labels-csv", default="input/stanford-rna-3d-folding-2/train_labels.csv")
    lp.add_argument("--out-dir", default="data/derived/train_labels_parquet")
    lp.add_argument("--rows-per-file", type=int, default=2_000_000)
    lp.add_argument("--compression", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"], default="zstd")
    lp.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    lp.set_defaults(fn=_cmd_prepare_labels_parquet)

    lpc = sp.add_parser(
        "prepare-train-labels-clean",
        help="Create cleaned labels parquet by explicitly dropping rows with null xyz",
    )
    lpc.add_argument("--train-labels-parquet-dir", required=True, help="Input labels parquet dir (part-*.parquet)")
    lpc.add_argument("--out-dir", default="data/derived/train_labels_parquet_nonnull_xyz")
    lpc.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    lpc.add_argument("--rows-per-file", type=int, default=2_000_000)
    lpc.add_argument("--compression", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"], default="zstd")
    grp = lpc.add_mutually_exclusive_group()
    grp.add_argument(
        "--require-complete-targets",
        dest="require_complete_targets",
        action="store_true",
        default=True,
        help="Fail if any target from train_sequences is missing after cleaning (default)",
    )
    grp.add_argument(
        "--allow-incomplete-targets",
        dest="require_complete_targets",
        action="store_false",
        help="Allow missing targets after cleaning",
    )
    lpc.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    lpc.set_defaults(fn=_cmd_prepare_labels_clean)

    bf = sp.add_parser("build-train-fold", help="Build a scoring dataset for one CV fold (sample + solution + manifest + target_sequences)")
    bf.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    bf.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    bf.add_argument("--fold", type=int, required=True)
    bf.add_argument("--out", required=True)
    bf.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    bf.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    bf.set_defaults(fn=_cmd_build_train_fold)

    e = sp.add_parser("export-train-solution", help="Export train_labels subset as wide solution (parquet)")
    e.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    e.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    e.set_defaults(fn=_cmd_export_train_solution)

    m = sp.add_parser("make-sample", help="Create sample_submission template for a CV fold (CSV)")
    m.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    m.add_argument("--fold", type=int, required=True)
    m.add_argument("--sequences", required=True, help="train_sequences.csv (or other sequences CSV)")
    m.add_argument("--out", required=True)
    m.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    m.set_defaults(fn=_cmd_make_sample)

    c = sp.add_parser("check-submission", help="Strict contract validation vs sample_submission")
    c.add_argument("--sample", required=True)
    c.add_argument("--submission", required=True)
    c.set_defaults(fn=_cmd_check_submission)

    s = sp.add_parser("score", help="Score a submission locally with vendored Kaggle metric")
    s.add_argument("--dataset", choices=["public_validation"], default=None)
    s.add_argument("--dataset-dir", default="data/derived/public_validation")
    s.add_argument("--submission", required=True)
    s.add_argument("--out-dir", default="runs/auto")
    s.add_argument("--per-target", action="store_true")
    s.add_argument("--keep-tmp", action="store_true")
    s.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    s.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    s.add_argument("--chunk-size", type=int, default=100_000)
    s.set_defaults(fn=_cmd_score)

    tdb = sp.add_parser("build-template-db", help="Build template database (local train + external templates)")
    tdb.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tdb.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    tdb.add_argument("--external-templates", required=True, help="CSV/Parquet with template_id,sequence,release_date,resid,resname,x,y,z")
    tdb.add_argument("--out-dir", default="data/derived/template_db")
    tdb.add_argument("--max-train-templates", type=int, default=None)
    tdb.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    tdb.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    tdb.set_defaults(fn=_cmd_build_template_db)

    rt = sp.add_parser("retrieve-templates", help="Retrieve temporal-valid template candidates per target")
    rt.add_argument("--template-index", default="data/derived/template_db/template_index.parquet")
    rt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    rt.add_argument("--out", default="data/derived/template_db/retrieval_candidates.parquet")
    rt.add_argument("--top-k", type=int, default=20)
    rt.add_argument("--kmer-size", type=int, default=3)
    rt.add_argument("--length-weight", type=float, default=0.15)
    rt.add_argument("--refine-pool-size", type=int, default=64)
    rt.add_argument("--refine-alignment-weight", type=float, default=0.25)
    rt.add_argument("--refine-open-gap-score", type=float, default=-5.0)
    rt.add_argument("--refine-extend-gap-score", type=float, default=-1.0)
    rt.add_argument("--chunk-size", type=int, default=200_000)
    rt.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    rt.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    rt.set_defaults(fn=_cmd_retrieve_templates)

    pt = sp.add_parser("predict-tbm", help="Generate TBM predictions in long format")
    pt.add_argument("--retrieval", default="data/derived/template_db/retrieval_candidates.parquet")
    pt.add_argument("--templates", default="data/derived/template_db/templates.parquet")
    pt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    pt.add_argument("--out", required=True)
    pt.add_argument("--n-models", type=int, default=5)
    pt.add_argument("--min-coverage", type=float, default=0.35)
    pt.add_argument("--rerank-pool-size", type=int, default=64)
    pt.add_argument("--gap-open-scores", type=str, default="-5.0")
    pt.add_argument("--gap-extend-scores", type=str, default="-1.0")
    pt.add_argument("--max-variants-per-template", type=int, default=1)
    pt.add_argument("--perturbation-scale", type=float, default=0.0)
    pt.add_argument("--chunk-size", type=int, default=200_000)
    pt.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    pt.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    pt.set_defaults(fn=_cmd_predict_tbm)

    pd2 = sp.add_parser("predict-drfold2", help="Run local DRfold2 inference and export long predictions")
    pd2.add_argument("--drfold-root", required=True, help="Path to local DRfold2 repository (with model_hub and Arena)")
    pd2.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    pd2.add_argument("--out", required=True)
    pd2.add_argument("--work-dir", default="runs/auto_drfold2")
    pd2.add_argument("--n-models", type=int, default=5)
    pd2.add_argument("--python-bin", default="python")
    pd2.add_argument("--target-limit", type=int, default=None, help="Optional limit for smoke validation")
    pd2.add_argument("--chunk-size", type=int, default=200_000)
    pd2.add_argument("--reuse-existing-targets", action="store_true", help="Reuse existing per-target DRfold2 outputs in work-dir")
    pd2.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    pd2.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    pd2.set_defaults(fn=_cmd_predict_drfold2)

    tr = sp.add_parser("train-rnapro", help="Train RNAPro proxy model locally")
    tr.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tr.add_argument("--train-labels-parquet-dir", required=True, help="Canonical labels parquet dir (part-*.parquet)")
    tr.add_argument("--out-dir", default="runs/auto_rnapro")
    tr.add_argument("--feature-dim", type=int, default=256)
    tr.add_argument("--kmer-size", type=int, default=4)
    tr.add_argument("--n-models", type=int, default=5)
    tr.add_argument("--seed", type=int, default=123)
    tr.add_argument("--min-coverage", type=float, default=0.30)
    tr.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    tr.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    tr.set_defaults(fn=_cmd_train_rnapro)

    ct = sp.add_parser(
        "convert-templates-to-pt",
        help="Convert wide template submission (CSV/Parquet) into per-target template_features.pt artifacts",
    )
    ct.add_argument("--templates-submission", required=True, help="Wide template submission (ID,resname,resid,x_i,y_i,z_i)")
    ct.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    ct.add_argument("--out-dir", required=True)
    ct.add_argument("--n-models", type=int, default=None, help="If set, enforce exact number of models in submission header")
    ct.add_argument("--template-source", choices=["tbm", "mmseqs2", "external"], default="tbm")
    ct.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ct.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ct.set_defaults(fn=_cmd_convert_templates_to_pt)

    ir = sp.add_parser("predict-rnapro", help="Run RNAPro inference in long format")
    ir.add_argument("--model-dir", required=True)
    ir.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    ir.add_argument("--out", required=True)
    ir.add_argument("--n-models", type=int, default=None)
    ir.add_argument("--min-coverage", type=float, default=None)
    ir.add_argument("--rerank-pool-multiplier", type=int, default=8)
    ir.add_argument("--chunk-size", type=int, default=200_000)
    ir.add_argument("--use-template", choices=["none", "ca_precomputed"], default="none")
    ir.add_argument("--template-features-dir", default=None, help="Required when --use-template ca_precomputed")
    ir.add_argument("--template-source", choices=["tbm", "mmseqs2", "external"], default="tbm")
    ir.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ir.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ir.set_defaults(fn=_cmd_predict_rnapro)

    ep = sp.add_parser("ensemble-predict", help="Blend TBM and RNAPro predictions")
    ep.add_argument("--tbm", required=True)
    ep.add_argument("--rnapro", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--tbm-weight", type=float, default=0.6)
    ep.add_argument("--rnapro-weight", type=float, default=0.4)
    ep.add_argument("--dynamic-by-coverage", action="store_true")
    ep.add_argument("--coverage-power", type=float, default=1.0)
    ep.add_argument("--coverage-floor", type=float, default=1e-6)
    ep.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ep.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ep.set_defaults(fn=_cmd_ensemble_predict)

    ex = sp.add_parser("export-submission", help="Export strict Kaggle submission from long predictions")
    ex.add_argument("--sample", default="input/stanford-rna-3d-folding-2/sample_submission.csv")
    ex.add_argument("--predictions", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--memory-budget-mb", type=int, default=DEFAULT_MEMORY_BUDGET_MB)
    ex.add_argument("--max-rows-in-memory", type=int, default=DEFAULT_MAX_ROWS_IN_MEMORY)
    ex.set_defaults(fn=_cmd_export_submission)

    sk = sp.add_parser("submit-kaggle", help="Submit to Kaggle after strict local gating")
    sk.add_argument("--competition", default="stanford-rna-3d-folding-2")
    sk.add_argument("--sample", default="input/stanford-rna-3d-folding-2/sample_submission.csv")
    sk.add_argument("--submission", required=True)
    sk.add_argument("--message", required=True)
    sk.add_argument("--gating-report", default="runs/auto")
    sk.add_argument("--score-json", default=None)
    sk.add_argument("--baseline-score", type=float, default=None)
    sk.add_argument("--allow-regression", action="store_true")
    sk.add_argument("--is-smoke", action="store_true")
    sk.add_argument("--is-partial", action="store_true")
    sk.set_defaults(fn=_cmd_submit_kaggle)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except PipelineError as e:
        print(str(e))
        return 2
