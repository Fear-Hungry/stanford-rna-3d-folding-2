from __future__ import annotations

import argparse

import pytest

from rna3d_local.cli import build_parser


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

    assert "--allow-public-validation-without-cv" not in robust_opts
    assert "--allow-calibration-extrapolation" not in robust_opts
    assert "--allow-missing-baseline" not in readiness_opts
    assert "--allow-missing-public-score" not in readiness_opts
    assert "--allow-calibration-extrapolation" not in readiness_opts
    assert "--allow-public-validation-without-cv" not in readiness_opts
    assert "--allow-calibration-extrapolation" not in calibrate_opts
