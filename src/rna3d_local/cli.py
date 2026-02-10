from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .contracts import validate_submission_against_sample
from .datasets import (
    build_public_validation_dataset,
    build_train_cv_targets,
    build_train_cv_fold_dataset,
    export_train_solution_for_targets,
    make_sample_submission_for_targets,
)
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
        )
        print(str(manifest))
        return 0
    raise_error("CLI", "src/rna3d_local/cli.py:_cmd_build_dataset", "tipo de dataset desconhecido", impact="1", examples=[args.type])
    raise AssertionError("unreachable")


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
    )
    print(str(manifest))
    return 0


def _cmd_export_train_solution(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    input_dir = (repo / args.input).resolve()
    targets_path = (repo / args.targets).resolve()
    if not targets_path.exists():
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_export_train_solution", "targets.parquet nao encontrado", impact="1", examples=[str(targets_path)])
    import polars as pl

    df = pl.read_parquet(targets_path)
    if "target_id" not in df.columns or "fold_id" not in df.columns:
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_export_train_solution", "targets.parquet sem colunas esperadas", impact="1", examples=df.columns[:8])
    target_ids = df.filter(pl.col("fold_id") == int(args.fold)).get_column("target_id").cast(pl.Utf8).to_list()
    if not target_ids:
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_export_train_solution", "fold sem targets", impact="0", examples=[str(args.fold)])
    out_path = (repo / args.out).resolve()
    out = export_train_solution_for_targets(repo_root=repo, input_dir=input_dir, out_path=out_path, target_ids=target_ids)
    print(str(out))
    return 0


def _cmd_make_sample(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    targets_path = (repo / args.targets).resolve()
    sequences_csv = (repo / args.sequences).resolve()
    if not targets_path.exists():
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_make_sample", "targets.parquet nao encontrado", impact="1", examples=[str(targets_path)])
    import polars as pl

    df = pl.read_parquet(targets_path)
    if "target_id" not in df.columns or "fold_id" not in df.columns:
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_make_sample", "targets.parquet sem colunas esperadas", impact="1", examples=df.columns[:8])
    target_ids = df.filter(pl.col("fold_id") == int(args.fold)).get_column("target_id").cast(pl.Utf8).to_list()
    if not target_ids:
        raise_error("DATA", "src/rna3d_local/cli.py:_cmd_make_sample", "fold sem targets", impact="0", examples=[str(args.fold)])
    out_path = (repo / args.out).resolve()
    out = make_sample_submission_for_targets(sequences_csv=sequences_csv, out_path=out_path, target_ids=target_ids)
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
        train_labels_path=(repo / args.train_labels).resolve(),
        external_templates_path=(repo / args.external_templates).resolve(),
        out_dir=(repo / args.out_dir).resolve(),
        max_train_templates=args.max_train_templates,
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
    res = predict_tbm(
        repo_root=repo,
        retrieval_candidates_path=(repo / args.retrieval).resolve(),
        templates_path=(repo / args.templates).resolve(),
        target_sequences_path=(repo / args.targets).resolve(),
        out_path=(repo / args.out).resolve(),
        n_models=int(args.n_models),
        min_coverage=float(args.min_coverage),
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
        train_labels_path=(repo / args.train_labels).resolve(),
        out_dir=out_dir,
        config=cfg,
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
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_ensemble_predict(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = blend_predictions(
        tbm_predictions_path=(repo / args.tbm).resolve(),
        rnapro_predictions_path=(repo / args.rnapro).resolve(),
        out_path=(repo / args.out).resolve(),
        tbm_weight=float(args.tbm_weight),
        rnapro_weight=float(args.rnapro_weight),
    )
    print(json.dumps({"predictions": _rel_or_abs(out, repo)}, indent=2, sort_keys=True))
    return 0


def _cmd_export_submission(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out = export_submission_from_long(
        sample_submission_path=(repo / args.sample).resolve(),
        predictions_long_path=(repo / args.predictions).resolve(),
        out_submission_path=(repo / args.out).resolve(),
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
    b.set_defaults(fn=_cmd_build_dataset)

    bf = sp.add_parser("build-train-fold", help="Build a scoring dataset for one CV fold (sample + solution + manifest)")
    bf.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    bf.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    bf.add_argument("--fold", type=int, required=True)
    bf.add_argument("--out", required=True)
    bf.set_defaults(fn=_cmd_build_train_fold)

    e = sp.add_parser("export-train-solution", help="Export train_labels subset as wide solution (parquet)")
    e.add_argument("--input", default="input/stanford-rna-3d-folding-2")
    e.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    e.add_argument("--fold", type=int, required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=_cmd_export_train_solution)

    m = sp.add_parser("make-sample", help="Create sample_submission template for a CV fold (CSV)")
    m.add_argument("--targets", required=True, help="Path to targets.parquet from train_cv_targets dataset")
    m.add_argument("--fold", type=int, required=True)
    m.add_argument("--sequences", required=True, help="train_sequences.csv (or other sequences CSV)")
    m.add_argument("--out", required=True)
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
    s.set_defaults(fn=_cmd_score)

    tdb = sp.add_parser("build-template-db", help="Build template database (local train + external templates)")
    tdb.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tdb.add_argument("--train-labels", default="input/stanford-rna-3d-folding-2/train_labels.csv")
    tdb.add_argument("--external-templates", required=True, help="CSV/Parquet with template_id,sequence,release_date,resid,resname,x,y,z")
    tdb.add_argument("--out-dir", default="data/derived/template_db")
    tdb.add_argument("--max-train-templates", type=int, default=None)
    tdb.set_defaults(fn=_cmd_build_template_db)

    rt = sp.add_parser("retrieve-templates", help="Retrieve temporal-valid template candidates per target")
    rt.add_argument("--template-index", default="data/derived/template_db/template_index.parquet")
    rt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    rt.add_argument("--out", default="data/derived/template_db/retrieval_candidates.parquet")
    rt.add_argument("--top-k", type=int, default=20)
    rt.add_argument("--kmer-size", type=int, default=3)
    rt.set_defaults(fn=_cmd_retrieve_templates)

    pt = sp.add_parser("predict-tbm", help="Generate TBM predictions in long format")
    pt.add_argument("--retrieval", default="data/derived/template_db/retrieval_candidates.parquet")
    pt.add_argument("--templates", default="data/derived/template_db/templates.parquet")
    pt.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    pt.add_argument("--out", required=True)
    pt.add_argument("--n-models", type=int, default=5)
    pt.add_argument("--min-coverage", type=float, default=0.35)
    pt.set_defaults(fn=_cmd_predict_tbm)

    tr = sp.add_parser("train-rnapro", help="Train RNAPro proxy model locally")
    tr.add_argument("--train-sequences", default="input/stanford-rna-3d-folding-2/train_sequences.csv")
    tr.add_argument("--train-labels", default="input/stanford-rna-3d-folding-2/train_labels.csv")
    tr.add_argument("--out-dir", default="runs/auto_rnapro")
    tr.add_argument("--feature-dim", type=int, default=256)
    tr.add_argument("--kmer-size", type=int, default=4)
    tr.add_argument("--n-models", type=int, default=5)
    tr.add_argument("--seed", type=int, default=123)
    tr.add_argument("--min-coverage", type=float, default=0.30)
    tr.set_defaults(fn=_cmd_train_rnapro)

    ir = sp.add_parser("predict-rnapro", help="Run RNAPro inference in long format")
    ir.add_argument("--model-dir", required=True)
    ir.add_argument("--targets", default="input/stanford-rna-3d-folding-2/test_sequences.csv")
    ir.add_argument("--out", required=True)
    ir.add_argument("--n-models", type=int, default=None)
    ir.add_argument("--min-coverage", type=float, default=None)
    ir.set_defaults(fn=_cmd_predict_rnapro)

    ep = sp.add_parser("ensemble-predict", help="Blend TBM and RNAPro predictions")
    ep.add_argument("--tbm", required=True)
    ep.add_argument("--rnapro", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--tbm-weight", type=float, default=0.6)
    ep.add_argument("--rnapro-weight", type=float, default=0.4)
    ep.set_defaults(fn=_cmd_ensemble_predict)

    ex = sp.add_parser("export-submission", help="Export strict Kaggle submission from long predictions")
    ex.add_argument("--sample", default="input/stanford-rna-3d-folding-2/sample_submission.csv")
    ex.add_argument("--predictions", required=True)
    ex.add_argument("--out", required=True)
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
