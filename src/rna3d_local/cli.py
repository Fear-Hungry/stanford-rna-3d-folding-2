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
from .errors import PipelineError, raise_error
from .scoring import score_submission, write_score_artifacts
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


def _cmd_download(args: argparse.Namespace) -> int:
    repo = _find_repo_root(Path.cwd())
    out_dir = (repo / args.out).resolve()
    res = download_competition_files(competition=args.competition, out_dir=out_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": args.competition,
        "out_dir": str(out_dir.relative_to(repo)),
        "files": {k: str(v.relative_to(repo)) for k, v in res.files.items()},
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
    meta = {"dataset_dir": str(dataset_dir.relative_to(repo)), "submission": str(submission.relative_to(repo))}
    write_score_artifacts(out_dir=out_dir, result=result, meta=meta)
    print(json.dumps({"score": result.score, "out_dir": str(out_dir.relative_to(repo))}, indent=2, sort_keys=True))
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

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except PipelineError as e:
        print(str(e))
        return 2
