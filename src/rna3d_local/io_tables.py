from __future__ import annotations

from pathlib import Path

import polars as pl

from .errors import raise_error


def read_table(path: Path, *, stage: str, location: str) -> pl.DataFrame:
    if not path.exists():
        raise_error(stage, location, "arquivo obrigatorio ausente", impact="1", examples=[str(path)])
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10_000)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise_error(stage, location, "formato de arquivo nao suportado", impact="1", examples=[str(path)])


def write_table(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
        return
    if suffix in {".parquet", ".pq"}:
        df.write_parquet(path, compression="zstd")
        return
    raise ValueError(f"unsupported suffix: {suffix}")
