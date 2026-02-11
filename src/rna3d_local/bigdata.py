from __future__ import annotations

import os
import resource
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .errors import raise_error

DEFAULT_MEMORY_BUDGET_MB = 24_576
DEFAULT_MAX_ROWS_IN_MEMORY = 10_000_000


def current_rss_mb() -> float:
    """
    Returns current resident memory (MB).
    Prefer /proc/self/status VmRSS on Linux; fallback to ru_maxrss.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    # VmRSS is in kB
                    return float(parts[1]) / 1024.0
    r = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: ru_maxrss in KB; macOS: bytes.
    if os.name == "posix":
        if r.ru_maxrss > 1024 * 1024 * 1024:
            return float(r.ru_maxrss) / (1024.0 * 1024.0)
        return float(r.ru_maxrss) / 1024.0
    return float(r.ru_maxrss) / 1024.0


def assert_memory_budget(
    *,
    stage: str,
    location: str,
    budget_mb: int | float,
    context_examples: list[str] | None = None,
) -> None:
    if budget_mb <= 0:
        raise_error(stage, location, "memory_budget_mb invalido (deve ser > 0)", impact="1", examples=[str(budget_mb)])
    rss_mb = current_rss_mb()
    if rss_mb > float(budget_mb):
        raise_error(
            stage,
            location,
            "memoria acima do budget configurado",
            impact=f"rss_mb={rss_mb:.2f} budget_mb={float(budget_mb):.2f}",
            examples=context_examples or [],
        )


def assert_row_budget(
    *,
    stage: str,
    location: str,
    rows: int,
    max_rows_in_memory: int,
    label: str,
) -> None:
    if max_rows_in_memory <= 0:
        raise_error(stage, location, "max_rows_in_memory invalido (deve ser > 0)", impact="1", examples=[str(max_rows_in_memory)])
    if rows > max_rows_in_memory:
        raise_error(
            stage,
            location,
            f"linhas em memoria acima do limite para {label}",
            impact=f"rows={rows} max_rows_in_memory={max_rows_in_memory}",
            examples=[label],
        )


@dataclass(frozen=True)
class LabelStoreConfig:
    labels_parquet_dir: Path
    required_columns: tuple[str, ...]
    stage: str
    location: str


@dataclass(frozen=True)
class TableReadConfig:
    path: Path
    stage: str
    location: str
    columns: tuple[str, ...] | None = None
    infer_schema_length: int = 1000


def _validate_required_columns(*, lf: pl.LazyFrame, required_columns: tuple[str, ...], stage: str, location: str, source: str) -> None:
    cols = set(lf.collect_schema().names())
    missing = [c for c in required_columns if c not in cols]
    if missing:
        raise_error(
            stage,
            location,
            "colunas obrigatorias ausentes",
            impact=f"source={source} missing={len(missing)}",
            examples=missing[:8],
        )


def resolve_label_parts(*, config: LabelStoreConfig) -> list[Path]:
    if not config.labels_parquet_dir.exists():
        raise_error(
            config.stage,
            config.location,
            "diretorio de labels parquet nao encontrado",
            impact="1",
            examples=[str(config.labels_parquet_dir)],
        )
    parts = sorted(config.labels_parquet_dir.glob("part-*.parquet"))
    if not parts:
        raise_error(
            config.stage,
            config.location,
            "diretorio de labels parquet sem arquivos part-*.parquet",
            impact="1",
            examples=[str(config.labels_parquet_dir)],
        )
    return parts


def scan_labels(*, config: LabelStoreConfig) -> pl.LazyFrame:
    parts = resolve_label_parts(config=config)
    lf = pl.scan_parquet([str(p) for p in parts])
    _validate_required_columns(
        lf=lf,
        required_columns=config.required_columns,
        stage=config.stage,
        location=config.location,
        source="labels_parquet_dir",
    )
    return lf


def scan_table(*, config: TableReadConfig) -> pl.LazyFrame:
    if not config.path.exists():
        raise_error(config.stage, config.location, "arquivo nao encontrado", impact="1", examples=[str(config.path)])
    suffix = config.path.suffix.lower()
    if suffix == ".parquet":
        lf = pl.scan_parquet(config.path)
    elif suffix == ".csv":
        lf = pl.scan_csv(config.path, infer_schema_length=config.infer_schema_length)
    else:
        raise_error(
            config.stage,
            config.location,
            "formato nao suportado (use CSV ou Parquet)",
            impact="1",
            examples=[str(config.path)],
        )
        raise AssertionError("unreachable")
    if config.columns is not None:
        _validate_required_columns(
            lf=lf,
            required_columns=config.columns,
            stage=config.stage,
            location=config.location,
            source=str(config.path),
        )
        lf = lf.select([pl.col(c) for c in config.columns])
    return lf


def collect_streaming(*, lf: pl.LazyFrame, stage: str, location: str) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except Exception as e:  # noqa: BLE001
        raise_error(stage, location, "falha ao coletar frame em streaming", impact="1", examples=[f"{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def sink_partitioned_parquet(
    *,
    lf: pl.LazyFrame,
    out_dir: Path,
    rows_per_file: int,
    compression: str,
    stage: str,
    location: str,
) -> dict:
    if rows_per_file <= 0:
        raise_error(stage, location, "rows_per_file invalido (deve ser > 0)", impact="1", examples=[str(rows_per_file)])
    allowed_compression = {"zstd", "snappy", "gzip", "brotli", "lz4", "none"}
    if compression not in allowed_compression:
        raise_error(stage, location, "compression invalida", impact="1", examples=[compression])

    out_dir.mkdir(parents=True, exist_ok=True)
    old_parts = sorted(out_dir.glob("part-*.parquet"))
    if old_parts:
        raise_error(
            stage,
            location,
            "diretorio de saida ja contem part-*.parquet; use um diretorio novo",
            impact=str(len(old_parts)),
            examples=[str(old_parts[0])],
        )

    temp_parquet = out_dir / "_tmp_sink.parquet"
    if temp_parquet.exists():
        temp_parquet.unlink()
    lf.sink_parquet(temp_parquet, compression=compression)

    scanner = ds.dataset(temp_parquet, format="parquet").scanner(batch_size=131_072)
    part_paths: list[Path] = []
    part_rows: list[int] = []
    writer: pq.ParquetWriter | None = None
    file_index = 0
    cur_rows = 0
    total_rows = 0
    part_path: Path | None = None
    try:
        try:
            for batch in scanner.to_batches():
                offset = 0
                while offset < batch.num_rows:
                    if writer is None:
                        part_path = out_dir / f"part-{file_index:05d}.parquet"
                        writer = pq.ParquetWriter(str(part_path), batch.schema, compression=compression)
                        cur_rows = 0
                    cap = rows_per_file - cur_rows
                    take = min(cap, batch.num_rows - offset)
                    piece = batch.slice(offset, take)
                    writer.write_batch(piece)
                    cur_rows += take
                    total_rows += take
                    offset += take
                    if cur_rows >= rows_per_file:
                        writer.close()
                        writer = None
                        if part_path is None:
                            raise_error(stage, location, "estado interno invalido ao escrever particao", impact="1", examples=[])
                        part_paths.append(part_path)
                        part_rows.append(cur_rows)
                        file_index += 1
                        part_path = None
            if writer is not None:
                writer.close()
                writer = None
                if part_path is None:
                    raise_error(stage, location, "estado interno invalido ao finalizar particao", impact="1", examples=[])
                part_paths.append(part_path)
                part_rows.append(cur_rows)
        finally:
            if writer is not None:
                writer.close()
        if total_rows == 0:
            raise_error(stage, location, "nenhuma linha convertida para parquet", impact="0", examples=[str(out_dir)])
    finally:
        if temp_parquet.exists():
            temp_parquet.unlink()
    return {
        "parts": part_paths,
        "rows_per_file_actual": part_rows,
        "n_rows": int(total_rows),
        "n_files": int(len(part_paths)),
    }


__all__ = [
    "DEFAULT_MEMORY_BUDGET_MB",
    "DEFAULT_MAX_ROWS_IN_MEMORY",
    "LabelStoreConfig",
    "TableReadConfig",
    "assert_memory_budget",
    "assert_row_budget",
    "collect_streaming",
    "current_rss_mb",
    "resolve_label_parts",
    "scan_labels",
    "scan_table",
    "sink_partitioned_parquet",
]
