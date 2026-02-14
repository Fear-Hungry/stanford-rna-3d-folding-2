from __future__ import annotations

from .cli_commands import _enforce_non_ensemble_predictions, _enforce_submit_hardening
from .cli_parser import build_parser
from .errors import PipelineError


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except PipelineError as e:
        print(str(e))
        return 2


__all__ = [
    "build_parser",
    "main",
    "_enforce_non_ensemble_predictions",
    "_enforce_submit_hardening",
]
