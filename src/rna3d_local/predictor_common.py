from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table


def load_targets_with_contract(*, targets_path: Path, stage: str, location: str) -> pl.DataFrame:
    targets = read_table(targets_path, stage=stage, location=location)
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    if "ligand_SMILES" not in targets.columns:
        targets = targets.with_columns(pl.lit("").alias("ligand_SMILES"))
    targets = targets.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("sequence").cast(pl.Utf8),
        pl.col("ligand_SMILES").cast(pl.Utf8),
    )
    bad = targets.filter((pl.col("target_id").str.len_chars() == 0) | (pl.col("sequence").str.len_chars() == 0))
    if bad.height > 0:
        examples = bad.select("target_id").head(8).get_column("target_id").to_list()
        raise_error(stage, location, "targets com target_id/sequence vazios", impact=str(bad.height), examples=[str(x) for x in examples])
    dup = targets.group_by("target_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = dup.select("target_id").head(8).get_column("target_id").to_list()
        raise_error(stage, location, "target_id duplicado em targets", impact=str(dup.height), examples=[str(x) for x in examples])
    return targets


def ensure_model_artifacts(*, model_dir: Path, required_files: list[str], stage: str, location: str) -> None:
    if not model_dir.exists():
        raise_error(stage, location, "diretorio de modelo ausente", impact="1", examples=[str(model_dir)])
    missing: list[str] = []
    for rel in required_files:
        path = model_dir / rel
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise_error(stage, location, "arquivos obrigatorios de modelo ausentes", impact=str(len(missing)), examples=missing[:8])


def load_model_entrypoint(*, model_dir: Path, stage: str, location: str) -> list[str]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise_error(stage, location, "config.json do modelo ausente", impact="1", examples=[str(config_path)])
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "config.json invalido (nao e JSON)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    entrypoint = payload.get("entrypoint")
    if not isinstance(entrypoint, list):
        raise_error(stage, location, "config.json sem entrypoint (lista) para runner offline", impact="1", examples=[str(config_path)])
    cleaned = [str(item) for item in entrypoint if isinstance(item, str) and str(item).strip()]
    if len(cleaned) != len(entrypoint) or not cleaned:
        raise_error(stage, location, "config.json com entrypoint invalido (itens vazios/nao-string)", impact="1", examples=[str(entrypoint)[:200]])
    return cleaned


def render_entrypoint(
    entrypoint: list[str],
    *,
    model_dir: Path,
    targets_path: Path,
    out_path: Path,
    n_models: int,
) -> list[str]:
    mapping = {
        "{model_dir}": str(model_dir),
        "{targets}": str(targets_path),
        "{out}": str(out_path),
        "{n_models}": str(int(n_models)),
    }
    rendered: list[str] = []
    for arg in entrypoint:
        text = str(arg)
        for key, value in mapping.items():
            text = text.replace(key, value)
        rendered.append(text)
    return rendered


def run_external_entrypoint(
    *,
    model_dir: Path,
    entrypoint: list[str],
    stage: str,
    location: str,
    timeout_seconds: int = 8 * 60 * 60,
) -> None:
    if int(timeout_seconds) <= 0:
        raise_error(stage, location, "timeout_seconds invalido para runner externo", impact="1", examples=[str(timeout_seconds)])
    try:
        proc = subprocess.run(
            [str(item) for item in entrypoint],
            cwd=str(model_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=int(timeout_seconds),
        )
    except FileNotFoundError as exc:
        raise_error(stage, location, "runner externo falhou (comando nao encontrado)", impact="1", examples=[str(exc)])
    except subprocess.TimeoutExpired:
        raise_error(stage, location, "runner externo excedeu timeout", impact="1", examples=[f"timeout_seconds={timeout_seconds}"])
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "runner externo falhou ao executar", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    if proc.returncode != 0:
        stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout_txt = proc.stdout.decode("utf-8", errors="replace").strip()
        snippet = stderr_txt[:240] if stderr_txt else (stdout_txt[:240] if stdout_txt else f"returncode={proc.returncode}")
        raise_error(stage, location, "runner externo retornou erro", impact="1", examples=[snippet])


def _normalize_sequence(seq: str, *, stage: str, location: str, target_id: str) -> str:
    raw = str(seq or "").strip().upper().replace("T", "U")
    cleaned = "".join(ch for ch in raw if ch not in {"|", " ", "\t", "\n", "\r"})
    if not cleaned:
        raise_error(stage, location, "sequencia vazia apos normalizacao", impact="1", examples=[target_id])
    bad = sorted({ch for ch in cleaned if ch not in {"A", "C", "G", "U"}})
    if bad:
        raise_error(stage, location, "sequencia contem simbolos invalidos", impact=str(len(bad)), examples=[f"{target_id}:{''.join(bad[:8])}"])
    return cleaned


def validate_long_predictions(
    *,
    predictions: pl.DataFrame,
    targets: pl.DataFrame,
    n_models: int,
    stage: str,
    location: str,
    label: str,
) -> None:
    if int(n_models) <= 0:
        raise_error(stage, location, "n_models invalido para validacao", impact="1", examples=[str(n_models)])
    require_columns(predictions, ["target_id", "model_id", "resid", "resname", "x", "y", "z"], stage=stage, location=location, label=label)
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")

    pred = predictions.select(
        pl.col("target_id").cast(pl.Utf8),
        pl.col("model_id").cast(pl.Int32),
        pl.col("resid").cast(pl.Int32),
        pl.col("resname").cast(pl.Utf8),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("z").cast(pl.Float64),
    )
    bad_coord = pred.filter(~pl.col("x").is_finite() | ~pl.col("y").is_finite() | ~pl.col("z").is_finite())
    if bad_coord.height > 0:
        examples = bad_coord.select("target_id", "model_id", "resid").head(8).rows()
        raise_error(stage, location, "coordenadas nao-finitas em predictions", impact=str(int(bad_coord.height)), examples=[str(x) for x in examples])
    if pred.get_column("target_id").null_count() > 0:
        raise_error(
            stage,
            location,
            "target_id nulo em predictions",
            impact=str(int(pred.get_column("target_id").null_count())),
            examples=["target_id"],
        )

    key_dup = pred.group_by(["target_id", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if key_dup.height > 0:
        examples = (
            key_dup.with_columns(
                (pl.col("target_id") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8) + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "predictions com chave duplicada", impact=str(int(key_dup.height)), examples=[str(x) for x in examples])

    target_rows = targets.select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)).iter_rows()
    target_seqs: dict[str, str] = {}
    for tid, seq in target_rows:
        target_seqs[str(tid)] = _normalize_sequence(str(seq), stage=stage, location=location, target_id=str(tid))

    pred_targets = set(pred.get_column("target_id").unique().to_list())
    expected_targets = set(target_seqs.keys())
    extra = sorted(pred_targets - expected_targets)
    missing = sorted(expected_targets - pred_targets)
    if extra or missing:
        examples = [f"extra:{item}" for item in extra[:4]] + [f"missing:{item}" for item in missing[:4]]
        raise_error(
            stage,
            location,
            "targets em predictions nao batem com targets",
            impact=str(len(extra) + len(missing)),
            examples=examples,
        )

    per_target = (
        pred.group_by("target_id")
        .agg(
            pl.col("model_id").n_unique().alias("n_models"),
            pl.col("model_id").min().alias("min_model"),
            pl.col("model_id").max().alias("max_model"),
        )
        .sort("target_id")
    )
    bad_models = per_target.filter(
        (pl.col("n_models") != int(n_models))
        | (pl.col("min_model") != 1)
        | (pl.col("max_model") != int(n_models))
    )
    if bad_models.height > 0:
        examples = bad_models.select("target_id", "n_models", "min_model", "max_model").head(8).rows()
        raise_error(stage, location, "cobertura de model_id invalida por target", impact=str(int(bad_models.height)), examples=[str(x) for x in examples])

    expected_len_df = pl.DataFrame(
        [{"target_id": tid, "expected_len": int(len(seq))} for tid, seq in target_seqs.items()],
        schema={"target_id": pl.Utf8, "expected_len": pl.Int32},
    )
    per_group = (
        pred.group_by(["target_id", "model_id"])
        .agg(
            pl.len().alias("n_rows"),
            pl.col("resid").n_unique().alias("n_unique"),
            pl.col("resid").min().alias("min_resid"),
            pl.col("resid").max().alias("max_resid"),
        )
        .join(expected_len_df, on="target_id", how="left")
    )
    bad_group = per_group.filter(
        (pl.col("expected_len").is_null())
        | (pl.col("n_rows") != pl.col("expected_len"))
        | (pl.col("n_unique") != pl.col("expected_len"))
        | (pl.col("min_resid") != 1)
        | (pl.col("max_resid") != pl.col("expected_len"))
    )
    if bad_group.height > 0:
        examples = (
            bad_group.with_columns(
                (
                    pl.col("target_id")
                    + pl.lit(":")
                    + pl.col("model_id").cast(pl.Utf8)
                    + pl.lit(":n=")
                    + pl.col("n_rows").cast(pl.Utf8)
                    + pl.lit(":expected=")
                    + pl.col("expected_len").cast(pl.Utf8)
                ).alias("k")
            )
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "cobertura de resid invalida por target/model", impact=str(int(bad_group.height)), examples=[str(x) for x in examples])

    expected_rows: list[dict[str, object]] = []
    for tid, seq in target_seqs.items():
        for resid, base in enumerate(seq, start=1):
            expected_rows.append({"target_id": tid, "resid": int(resid), "expected_resname": base})
    expected = pl.DataFrame(expected_rows).with_columns(pl.col("resid").cast(pl.Int32)).sort(["target_id", "resid"])
    joined = pred.join(expected, on=["target_id", "resid"], how="inner")
    if joined.height != pred.height:
        raise_error(stage, location, "predictions com mismatch de chaves ao validar resname", impact=str(abs(pred.height - joined.height)), examples=[label])
    bad_resname = joined.filter(pl.col("resname") != pl.col("expected_resname"))
    if bad_resname.height > 0:
        examples = bad_resname.select("target_id", "model_id", "resid", "resname", "expected_resname").head(8).rows()
        raise_error(stage, location, "resname divergente da sequencia", impact=str(int(bad_resname.height)), examples=[str(x) for x in examples])
