from __future__ import annotations

import csv
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .bigdata import DEFAULT_MAX_ROWS_IN_MEMORY, DEFAULT_MEMORY_BUDGET_MB, assert_memory_budget, assert_row_budget, collect_streaming, scan_table, TableReadConfig
from .errors import raise_error
from .utils import sha256_file

_STAGE = "TEMPLATE_PT"
_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_target_and_resid(*, rid: str, location: str) -> tuple[str, int]:
    raw = str(rid or "").strip()
    if "_" not in raw:
        raise_error(_STAGE, location, "ID invalido (esperado <target_id>_<resid>)", impact="1", examples=[raw])
    tid, resid_raw = raw.rsplit("_", 1)
    if not tid:
        raise_error(_STAGE, location, "ID invalido (target_id vazio)", impact="1", examples=[raw])
    try:
        resid = int(resid_raw)
    except ValueError:
        raise_error(_STAGE, location, "ID invalido (resid nao inteiro)", impact="1", examples=[raw])
    if resid <= 0:
        raise_error(_STAGE, location, "ID invalido (resid deve ser > 0)", impact="1", examples=[raw])
    return tid, resid


def _parse_model_ids(*, columns: Iterable[str], location: str) -> list[int]:
    colset = {str(c) for c in columns}
    required = {"ID", "resname", "resid"}
    missing_base = [c for c in sorted(required) if c not in colset]
    if missing_base:
        raise_error(_STAGE, location, "submission sem colunas base obrigatorias", impact=str(len(missing_base)), examples=missing_base)

    def _ids(prefix: str) -> set[int]:
        out: set[int] = set()
        for c in colset:
            if not c.startswith(prefix):
                continue
            raw = c[len(prefix) :]
            if not raw.isdigit():
                continue
            out.add(int(raw))
        return out

    x_ids = _ids("x_")
    y_ids = _ids("y_")
    z_ids = _ids("z_")
    if not x_ids or not y_ids or not z_ids:
        raise_error(_STAGE, location, "submission sem colunas de coordenadas x_i/y_i/z_i", impact="1", examples=sorted(colset)[:8])
    if x_ids != y_ids or x_ids != z_ids:
        raise_error(
            _STAGE,
            location,
            "conjunto de model_id divergente entre x_i/y_i/z_i",
            impact=f"x={len(x_ids)} y={len(y_ids)} z={len(z_ids)}",
            examples=[f"x_only={sorted(x_ids - y_ids - z_ids)[:4]}", f"y_only={sorted(y_ids - x_ids - z_ids)[:4]}", f"z_only={sorted(z_ids - x_ids - y_ids)[:4]}"],
        )
    model_ids = sorted(x_ids)
    expected = list(range(1, len(model_ids) + 1))
    if model_ids != expected:
        raise_error(
            _STAGE,
            location,
            "model_id nao consecutivo (esperado 1..N)",
            impact=f"n_models={len(model_ids)}",
            examples=[f"model_ids={model_ids[:8]}"],
        )
    return model_ids


def _iter_submission_rows(*, submission_path: Path, location: str) -> tuple[list[str], Iterable[dict]]:
    suffix = submission_path.suffix.lower()
    if suffix == ".csv":
        f = submission_path.open("r", encoding="utf-8", newline="")
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            f.close()
            raise_error(_STAGE, location, "CSV de submissao sem cabecalho", impact="1", examples=[str(submission_path)])
        fieldnames = [str(c) for c in reader.fieldnames]

        def _gen_csv() -> Iterable[dict]:
            try:
                for row in reader:
                    yield row
            finally:
                f.close()

        return fieldnames, _gen_csv()

    if suffix == ".parquet":
        if not submission_path.exists():
            raise_error(_STAGE, location, "arquivo de submissao nao encontrado", impact="1", examples=[str(submission_path)])
        pf = pq.ParquetFile(str(submission_path))
        fieldnames = list(pf.schema.names)
        if not fieldnames:
            raise_error(_STAGE, location, "Parquet de submissao sem colunas", impact="1", examples=[str(submission_path)])

        def _gen_parquet() -> Iterable[dict]:
            for batch in pf.iter_batches(batch_size=65_536):
                for row in batch.to_pylist():
                    yield row

        return fieldnames, _gen_parquet()

    raise_error(_STAGE, location, "formato de submissao nao suportado (use CSV/Parquet)", impact="1", examples=[str(submission_path)])
    raise AssertionError("unreachable")


def _load_target_sequences(*, target_sequences_path: Path, location: str) -> dict[str, str]:
    targets = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=target_sequences_path,
                stage=_STAGE,
                location=location,
                columns=("target_id", "sequence"),
            )
        ).select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)),
        stage=_STAGE,
        location=location,
    )
    if targets.height == 0:
        raise_error(_STAGE, location, "target_sequences vazio", impact="0", examples=[str(target_sequences_path)])
    dup = (
        targets.group_by("target_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select("target_id")
        .head(8)
    )
    if dup.height > 0:
        raise_error(
            _STAGE,
            location,
            "target_id duplicado em target_sequences",
            impact=str(int(dup.height)),
            examples=[str(v) for v in dup.get_column("target_id").to_list()],
        )
    out: dict[str, str] = {}
    bad_count = 0
    bad_examples: list[str] = []
    for tid, seq in targets.iter_rows():
        s = str(seq or "")
        if not s:
            bad_count += 1
            if len(bad_examples) < 8:
                bad_examples.append(str(tid))
            continue
        out[str(tid)] = s
    if bad_count > 0:
        raise_error(_STAGE, location, "target_sequences com sequence vazia", impact=str(bad_count), examples=bad_examples)
    return out


def _parse_float_value(*, raw: object, field_name: str, row_id: str, location: str) -> float:
    try:
        val = float(raw)
    except Exception:  # noqa: BLE001
        raise_error(_STAGE, location, "valor de coordenada invalido", impact="1", examples=[f"{row_id}:{field_name}={raw}"])
    if not math.isfinite(val):
        raise_error(_STAGE, location, "valor de coordenada nao-finito", impact="1", examples=[f"{row_id}:{field_name}={raw}"])
    return val


def _load_template_payload(*, path: Path, stage: str, location: str) -> dict:
    if not path.exists():
        raise_error(stage, location, "template_features.pt ausente", impact="1", examples=[str(path)])
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        raise_error(stage, location, "falha ao ler template_features.pt", impact="1", examples=[f"{type(e).__name__}:{e}", str(path)])
    if not isinstance(payload, dict):
        raise_error(stage, location, "template_features.pt com payload invalido", impact="1", examples=[str(path)])
    return payload


def load_template_features_for_target(
    *,
    template_features_dir: Path,
    target_id: str,
    stage: str,
    location: str,
) -> dict:
    """
    Load + validate one target payload from `<dir>/<target_id>/template_features.pt`.
    """
    p = template_features_dir / str(target_id) / "template_features.pt"
    payload = _load_template_payload(path=p, stage=stage, location=location)
    req = ("target_id", "sequence", "model_ids", "coordinates", "mask")
    missing = [k for k in req if k not in payload]
    if missing:
        raise_error(stage, location, "template_features.pt sem chaves obrigatorias", impact=str(len(missing)), examples=missing)

    pt_tid = str(payload.get("target_id"))
    if pt_tid != str(target_id):
        raise_error(stage, location, "target_id divergente em template_features.pt", impact="1", examples=[str(target_id), pt_tid])

    coords = np.asarray(payload.get("coordinates"))
    mask = np.asarray(payload.get("mask"))
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise_error(stage, location, "coordinates com shape invalido (esperado [n_models,n_resid,3])", impact="1", examples=[str(coords.shape)])
    if mask.ndim != 2 or mask.shape[0] != coords.shape[0] or mask.shape[1] != coords.shape[1]:
        raise_error(stage, location, "mask com shape invalido", impact="1", examples=[str(mask.shape), str(coords.shape)])
    if int(coords.shape[0]) <= 0 or int(coords.shape[1]) <= 0:
        raise_error(stage, location, "coordinates vazio", impact=str(int(coords.shape[0] * coords.shape[1])), examples=[str(p)])

    model_ids_raw = payload.get("model_ids")
    if not isinstance(model_ids_raw, (list, tuple)) or len(model_ids_raw) != int(coords.shape[0]):
        raise_error(stage, location, "model_ids invalido em template_features.pt", impact="1", examples=[str(p)])
    try:
        model_ids = [int(x) for x in model_ids_raw]
    except Exception:  # noqa: BLE001
        raise_error(stage, location, "model_ids nao inteiro em template_features.pt", impact="1", examples=[str(model_ids_raw)[:120]])
    if model_ids != list(range(1, len(model_ids) + 1)):
        raise_error(stage, location, "model_ids nao consecutivo em template_features.pt", impact=str(len(model_ids)), examples=[str(model_ids[:8])])

    coords = coords.astype(np.float32, copy=False)
    mask = mask.astype(bool, copy=False)
    finite_ok = np.isfinite(coords).all()
    if not bool(finite_ok):
        raise_error(stage, location, "coordinates contem valores nao-finitos", impact="1", examples=[str(p)])

    return {
        "target_id": pt_tid,
        "sequence": str(payload.get("sequence") or ""),
        "model_ids": model_ids,
        "coordinates": coords,
        "mask": mask,
        "path": p,
    }


def convert_templates_to_pt_files(
    *,
    repo_root: Path,
    templates_submission_path: Path,
    target_sequences_path: Path,
    out_dir: Path,
    n_models: int | None = None,
    template_source: str = "tbm",
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> Path:
    """
    Convert a wide submission (`ID,resname,resid,x_1,y_1,z_1,...`) into per-target
    `template_features.pt` artifacts consumable by RNAPro precomputed-template mode.
    """
    location = "src/rna3d_local/template_pt.py:convert_templates_to_pt_files"
    assert_memory_budget(stage=_STAGE, location=location, budget_mb=memory_budget_mb)
    if n_models is not None and int(n_models) <= 0:
        raise_error(_STAGE, location, "n_models invalido (deve ser > 0)", impact="1", examples=[str(n_models)])
    if template_source not in {"tbm", "mmseqs2", "external"}:
        raise_error(_STAGE, location, "template_source invalido", impact="1", examples=[str(template_source)])

    target_seq = _load_target_sequences(target_sequences_path=target_sequences_path, location=location)
    header, row_iter = _iter_submission_rows(submission_path=templates_submission_path, location=location)
    model_ids = _parse_model_ids(columns=header, location=location)
    if n_models is not None and int(n_models) != len(model_ids):
        raise_error(
            _STAGE,
            location,
            "n_models divergente do cabecalho da submissao",
            impact=f"expected={int(n_models)} got={len(model_ids)}",
            examples=[str(templates_submission_path)],
        )
    use_n_models = len(model_ids)

    if out_dir.exists():
        existing = list(out_dir.iterdir())
        if existing:
            raise_error(
                _STAGE,
                location,
                "out_dir ja contem arquivos; use um diretorio novo para evitar sobrescrita",
                impact=str(len(existing)),
                examples=[str(existing[0])],
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    current_tid: str | None = None
    current_seq: str = ""
    current_expected_resid = 1
    current_coords: np.ndarray | None = None
    current_mask: np.ndarray | None = None
    seen_targets: set[str] = set()
    rows_total = 0

    def _flush_target() -> None:
        nonlocal current_tid, current_seq, current_expected_resid, current_coords, current_mask
        if current_tid is None:
            return
        expected_rows = len(current_seq)
        written_rows = current_expected_resid - 1
        if written_rows != expected_rows:
            raise_error(
                _STAGE,
                location,
                "target com quantidade de residuos divergente da sequencia",
                impact=f"target={current_tid} expected={expected_rows} got={written_rows}",
                examples=[current_tid],
            )
        if current_mask is None or current_coords is None:
            raise_error(_STAGE, location, "estado interno invalido no flush", impact="1", examples=[str(current_tid)])
        if not bool(current_mask.all()):
            missing = int((~current_mask).sum())
            raise_error(
                _STAGE,
                location,
                "target com coordenadas faltantes para algum modelo/residuo",
                impact=f"target={current_tid} missing={missing}",
                examples=[current_tid],
            )

        target_dir = out_dir / current_tid
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_pt = target_dir / "template_features.pt.tmp"
        final_pt = target_dir / "template_features.pt"
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "target_id": current_tid,
            "sequence": current_seq,
            "n_models": use_n_models,
            "model_ids": model_ids,
            "coordinates": current_coords.astype(np.float32, copy=False),
            "mask": current_mask.astype(bool, copy=False),
            "template_source": template_source,
        }
        with tmp_pt.open("wb") as f:
            pickle.dump(payload, f, protocol=4)
        if final_pt.exists():
            final_pt.unlink()
        tmp_pt.replace(final_pt)
        manifest_rows.append(
            {
                "target_id": current_tid,
                "n_residues": len(current_seq),
                "n_models": use_n_models,
                "template_features_pt": _rel(final_pt, repo_root),
                "sha256": sha256_file(final_pt),
            }
        )
        seen_targets.add(current_tid)
        assert_memory_budget(stage=_STAGE, location=location, budget_mb=memory_budget_mb)

        current_tid = None
        current_seq = ""
        current_expected_resid = 1
        current_coords = None
        current_mask = None

    for row in row_iter:
        rows_total += 1
        rid = str(row.get("ID", "")).strip()
        tid, resid_from_id = _parse_target_and_resid(rid=rid, location=location)

        if tid not in target_seq:
            raise_error(_STAGE, location, "target da submissao nao existe em target_sequences", impact="1", examples=[tid])

        if current_tid is None or tid != current_tid:
            if tid in seen_targets:
                raise_error(
                    _STAGE,
                    location,
                    "target apareceu em blocos nao contiguos na submissao",
                    impact="1",
                    examples=[tid],
                )
            _flush_target()
            current_tid = tid
            current_seq = target_seq[tid]
            assert_row_budget(
                stage=_STAGE,
                location=location,
                rows=int(len(current_seq) * use_n_models),
                max_rows_in_memory=max_rows_in_memory,
                label=f"target_buffer:{tid}",
            )
            current_coords = np.zeros((use_n_models, len(current_seq), 3), dtype=np.float32)
            current_mask = np.zeros((use_n_models, len(current_seq)), dtype=bool)
            current_expected_resid = 1

        assert current_coords is not None
        assert current_mask is not None
        raw_resid = row.get("resid")
        try:
            resid_col = int(raw_resid)
        except Exception:  # noqa: BLE001
            raise_error(_STAGE, location, "coluna resid invalida", impact="1", examples=[f"{rid}:{raw_resid}"])
        if resid_col != resid_from_id:
            raise_error(_STAGE, location, "resid divergente entre ID e coluna resid", impact="1", examples=[rid])
        if resid_col != current_expected_resid:
            raise_error(
                _STAGE,
                location,
                "residuos fora de ordem ou duplicados dentro do target",
                impact=f"target={tid} expected_resid={current_expected_resid} got={resid_col}",
                examples=[rid],
            )
        if resid_col > len(current_seq):
            raise_error(
                _STAGE,
                location,
                "resid excede tamanho da sequencia do target",
                impact=f"target={tid} seq_len={len(current_seq)} resid={resid_col}",
                examples=[rid],
            )
        resname = str(row.get("resname", "")).strip().upper()
        expected_base = current_seq[resid_col - 1].upper()
        if resname != expected_base:
            raise_error(
                _STAGE,
                location,
                "resname divergente da sequencia alvo",
                impact="1",
                examples=[f"{rid}:{resname}!={expected_base}"],
            )

        row_idx = resid_col - 1
        for i, model_id in enumerate(model_ids):
            x = _parse_float_value(raw=row.get(f"x_{model_id}"), field_name=f"x_{model_id}", row_id=rid, location=location)
            y = _parse_float_value(raw=row.get(f"y_{model_id}"), field_name=f"y_{model_id}", row_id=rid, location=location)
            z = _parse_float_value(raw=row.get(f"z_{model_id}"), field_name=f"z_{model_id}", row_id=rid, location=location)
            current_coords[i, row_idx, 0] = float(x)
            current_coords[i, row_idx, 1] = float(y)
            current_coords[i, row_idx, 2] = float(z)
            current_mask[i, row_idx] = True
        current_expected_resid += 1
        if rows_total % 250_000 == 0:
            assert_memory_budget(stage=_STAGE, location=location, budget_mb=memory_budget_mb)

    _flush_target()
    missing_targets = sorted(set(target_seq.keys()) - seen_targets)
    if missing_targets:
        raise_error(
            _STAGE,
            location,
            "submission nao contem todos os targets esperados",
            impact=str(len(missing_targets)),
            examples=missing_targets[:8],
        )
    if rows_total == 0:
        raise_error(_STAGE, location, "submission sem linhas", impact="0", examples=[str(templates_submission_path)])

    manifest_path = out_dir / "template_features_manifest.json"
    manifest = {
        "created_utc": _utc_now(),
        "schema_version": _SCHEMA_VERSION,
        "paths": {
            "templates_submission": _rel(templates_submission_path, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "out_dir": _rel(out_dir, repo_root),
        },
        "params": {
            "n_models": use_n_models,
            "template_source": template_source,
            "memory_budget_mb": int(memory_budget_mb),
            "max_rows_in_memory": int(max_rows_in_memory),
        },
        "stats": {
            "n_rows_submission": int(rows_total),
            "n_targets": int(len(seen_targets)),
            "targets": manifest_rows,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
