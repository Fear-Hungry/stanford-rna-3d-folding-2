from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pytest

from rna3d_local.cli import _enforce_non_ensemble_predictions, build_parser
from rna3d_local.errors import PipelineError


def _subparser(root: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in root._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action.choices[name]
    raise AssertionError(f"subparser not found: {name}")


def test_cli_hides_research_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "research-sync-literature" not in help_text
    assert "research-run" not in help_text
    assert "research-verify" not in help_text
    assert "research-report" not in help_text


def test_submit_kaggle_has_no_permissive_flags() -> None:
    parser = build_parser()
    submit = _subparser(parser, "submit-kaggle")
    options = set(submit._option_string_actions.keys())  # noqa: SLF001
    assert "--predictions-long" in options
    assert submit._option_string_actions["--require-min-cv-count"].default == 3  # noqa: SLF001
    assert "--allow-regression" not in options
    assert "--allow-public-validation-without-cv" not in options
    assert "--allow-target-patch" not in options
    assert "--allow-calibration-extrapolation" not in options
    assert "--allow-missing-robust-report" not in options
    assert "--allow-missing-readiness-report" not in options


def test_submit_kaggle_requires_reports() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "submit-kaggle",
                "--submission",
                "submission.csv",
                "--notebook-ref",
                "owner/notebook",
                "--notebook-version",
                "1",
                "--message",
                "msg",
            ]
        )


def test_robust_and_readiness_have_no_allow_flags() -> None:
    parser = build_parser()
    robust = _subparser(parser, "evaluate-robust")
    readiness = _subparser(parser, "evaluate-submit-readiness")
    calibrate = _subparser(parser, "calibrate-kaggle-local")

    robust_opts = set(robust._option_string_actions.keys())  # noqa: SLF001
    readiness_opts = set(readiness._option_string_actions.keys())  # noqa: SLF001
    calibrate_opts = set(calibrate._option_string_actions.keys())  # noqa: SLF001

    assert "--predictions-long" in robust_opts
    assert "--predictions-long" in readiness_opts
    assert "--allow-public-validation-without-cv" not in robust_opts
    assert "--allow-calibration-extrapolation" not in robust_opts
    assert "--allow-missing-baseline" not in readiness_opts
    assert "--allow-missing-public-score" not in readiness_opts
    assert "--allow-calibration-extrapolation" not in readiness_opts
    assert "--allow-public-validation-without-cv" not in readiness_opts
    assert "--allow-calibration-extrapolation" not in calibrate_opts


def test_add_labels_candidate_pool_parser_exists() -> None:
    parser = build_parser()
    add_labels = _subparser(parser, "add-labels-candidate-pool")
    opts = set(add_labels._option_string_actions.keys())  # noqa: SLF001

    assert "--candidates" in opts
    assert "--solution" in opts
    assert "--out" in opts
    assert "--label-col" in opts
    assert "--label-source-col" in opts
    assert "--label-source" in opts
    assert "--label-method" in opts
    assert "--metric-py" in opts
    assert "--usalign-bin" in opts
    assert "--memory-budget-mb" in opts
    assert "--max-rows-in-memory" in opts
    assert add_labels._option_string_actions["--label-method"].default == "tm_score_usalign"  # noqa: SLF001


def test_predict_drfold2_parser_has_target_ids_file() -> None:
    parser = build_parser()
    pd2 = _subparser(parser, "predict-drfold2")
    opts = set(pd2._option_string_actions.keys())  # noqa: SLF001
    assert "--target-ids-file" in opts


def test_ensemble_predict_command_is_blocked() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "ensemble-predict",
            "--tbm",
            "runs/tbm.parquet",
            "--rnapro",
            "runs/rnapro.parquet",
            "--out",
            "runs/ensemble.parquet",
        ]
    )
    with pytest.raises(PipelineError, match="bloqueado"):
        args.fn(args)


def test_strategy_gate_blocks_ensemble_branch(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.parquet"
    pl.DataFrame(
        [
            {"branch": "ensemble", "target_id": "T1"},
            {"branch": "tbm", "target_id": "T2"},
        ]
    ).write_parquet(predictions)
    with pytest.raises(PipelineError):
        _enforce_non_ensemble_predictions(
            predictions_long_path=predictions,
            stage="GATE",
            location="tests/test_cli_strict_surface.py:test_strategy_gate_blocks_ensemble_branch",
        )


def test_strategy_gate_allows_non_ensemble_branch(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.parquet"
    pl.DataFrame(
        [
            {"branch": "tbm", "target_id": "T1"},
            {"branch": "rnapro", "target_id": "T1"},
        ]
    ).write_parquet(predictions)
    out = _enforce_non_ensemble_predictions(
        predictions_long_path=predictions,
        stage="GATE",
        location="tests/test_cli_strict_surface.py:test_strategy_gate_allows_non_ensemble_branch",
    )
    assert out["allowed"] is True
    assert out["policy"] == "non_ensemble_only"
