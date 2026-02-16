from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..errors import raise_error
from ..io_tables import read_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rnapro_runner")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-models", required=True, type=int)
    ap.add_argument("--chain-separator", default="|")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    stage = "RNAPRO_RUNNER"
    location = "src/rna3d_local/runners/rnapro.py:main"

    model_dir = Path(args.model_dir).resolve()
    targets_path = Path(args.targets).resolve()
    out_path = Path(args.out).resolve()
    n_models = int(args.n_models)
    if n_models <= 0:
        raise_error(stage, location, "n_models invalido", impact="1", examples=[str(args.n_models)])

    # NOTE: RNAPro is not shipped as a PyPI package. We expect either:
    # - a vendored source tree under <model_dir>/src/RNAPro (from Kaggle dataset theoviel/rnapro-src), or
    # - an installed wheel provided by the user in assets/wheels.
    src_root = model_dir / "src" / "RNAPro"
    if src_root.exists():
        sys.path.insert(0, str(src_root.parent))
    else:
        raise_error(
            stage,
            location,
            "RNAPro source tree ausente (esperado model_dir/src/RNAPro). Use fetch-pretrained-assets para baixar o dataset de codigo.",
            impact="1",
            examples=[str(src_root)],
        )

    try:
        import RNAPro  # type: ignore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "falha ao importar RNAPro (dependencias/versao)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    targets = read_table(targets_path, stage=stage, location=location)
    if "target_id" not in targets.columns or "sequence" not in targets.columns:
        raise_error(stage, location, "targets schema invalido (faltam colunas)", impact="1", examples=["target_id", "sequence"])

    # TODO: integrate official RNAPro inference once we vendor the wheel/source entrypoint.
    # For now, fail-fast to avoid any silent fallback or mock output.
    raise_error(
        stage,
        location,
        "RNAPro runner ainda nao foi integrado (necessita entrypoint oficial de inferencia).",
        impact=str(int(targets.height)),
        examples=[str(src_root), "implementar chamada oficial + extracao C1'"],
    )

    # Unreachable.


if __name__ == "__main__":
    raise SystemExit(main())
