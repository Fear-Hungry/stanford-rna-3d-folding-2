from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .kaggle_cli import run_kaggle
from .utils import sha256_file


@dataclass(frozen=True)
class DownloadResult:
    output_dir: Path
    files: dict[str, Path]
    sha256: dict[str, str]


DEFAULT_COMPETITION_FILES = (
    "sample_submission.csv",
    "test_sequences.csv",
    "train_sequences.csv",
    "train_labels.csv",
    "validation_labels.csv",
)


def download_competition_files(
    *,
    competition: str,
    out_dir: Path,
    files: tuple[str, ...] = DEFAULT_COMPETITION_FILES,
) -> DownloadResult:
    location = "src/rna3d_local/download.py:download_competition_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    for fname in files:
        # Kaggle writes directly to out_dir; no unzip needed for CSVs.
        run_kaggle(
            ["competitions", "download", "-c", competition, "-f", fname, "-p", str(out_dir)],
            cwd=None,
            location=location,
        )
        fpath = out_dir / fname
        if not fpath.exists():
            raise_error(
                "DOWNLOAD",
                location,
                "arquivo esperado nao foi baixado",
                impact="1",
                examples=[str(fpath)],
            )
        resolved[fname] = fpath

    hashes = {k: sha256_file(v) for k, v in resolved.items()}
    return DownloadResult(output_dir=out_dir, files=resolved, sha256=hashes)


def download_dataset_usalign(*, out_dir: Path) -> Path:
    """
    Downloads `metric/usalign` as zip into out_dir and returns path to zip.
    """
    location = "src/rna3d_local/download.py:download_dataset_usalign"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle(["datasets", "download", "-d", "metric/usalign", "-p", str(out_dir)], cwd=None, location=location)
    zips = list(out_dir.glob("*.zip"))
    if not zips:
        raise_error(
            "VENDOR",
            location,
            "zip do usalign nao encontrado apos download",
            impact="1",
            examples=[str(out_dir)],
        )
    if len(zips) != 1:
        raise_error(
            "VENDOR",
            location,
            "multiples zips encontrados; especifique diretorio limpo",
            impact=str(len(zips)),
            examples=[p.name for p in zips[:8]],
        )
    return zips[0]


def download_metric_py_from_kernel(*, kernel_ref: str, out_dir: Path) -> Path:
    """
    Downloads kernel output (expects metric.py) into out_dir and returns its path.
    """
    location = "src/rna3d_local/download.py:download_metric_py_from_kernel"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle(["kernels", "output", kernel_ref, "-p", str(out_dir)], cwd=None, location=location)
    metric = out_dir / "metric.py"
    if not metric.exists():
        # show what was downloaded
        files = [p.name for p in out_dir.glob("*") if p.is_file()]
        raise_error(
            "VENDOR",
            location,
            "metric.py nao encontrado no output do kernel",
            impact=str(len(files)),
            examples=files[:8],
        )
    return metric

