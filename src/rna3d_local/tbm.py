from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import parse_date_column, require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class TbmResult:
    predictions_path: Path
    manifest_path: Path


def _parse_ranking_table(retrieval: pl.DataFrame) -> pl.DataFrame:
    if "rerank_rank" in retrieval.columns:
        return retrieval.sort(["target_id", "rerank_rank"])
    if "rank" in retrieval.columns:
        return retrieval.sort(["target_id", "rank"])
    if "final_score" in retrieval.columns:
        return retrieval.sort(["target_id", "final_score"], descending=[False, True])
    raise ValueError("retrieval table has no ranking columns")


def _missing_resid_examples(*, coords: dict[int, tuple[str, float, float, float]], seq_len: int, limit: int = 3) -> list[int]:
    missing: list[int] = []
    for resid in range(1, int(seq_len) + 1):
        if resid not in coords:
            missing.append(resid)
            if len(missing) >= int(limit):
                break
    return missing


def predict_tbm(
    *,
    repo_root: Path,
    retrieval_path: Path,
    templates_path: Path,
    targets_path: Path,
    out_path: Path,
    n_models: int,
) -> TbmResult:
    stage = "PREDICT_TBM"
    location = "src/rna3d_local/tbm.py:predict_tbm"
    if n_models <= 0:
        raise_error(stage, location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    retrieval = read_table(retrieval_path, stage=stage, location=location)
    require_columns(retrieval, ["target_id", "template_uid"], stage=stage, location=location, label="retrieval")
    ranked = _parse_ranking_table(retrieval)

    templates = read_table(templates_path, stage=stage, location=location)
    require_columns(templates, ["template_uid", "resid", "resname", "x", "y", "z"], stage=stage, location=location, label="templates")
    template_map: dict[str, dict[int, tuple[str, float, float, float]]] = {}
    for row in templates.select("template_uid", "resid", "resname", "x", "y", "z").iter_rows():
        uid = str(row[0])
        resid = int(row[1])
        template_map.setdefault(uid, {})
        if resid in template_map[uid]:
            raise_error(stage, location, "resid duplicado no template", impact="1", examples=[f"{uid}:{resid}"])
        template_map[uid][resid] = (str(row[2]), float(row[3]), float(row[4]), float(row[5]))

    targets = read_table(targets_path, stage=stage, location=location)
    require_columns(targets, ["target_id", "sequence", "temporal_cutoff"], stage=stage, location=location, label="targets")
    targets = parse_date_column(targets, "temporal_cutoff", stage=stage, location=location, label="targets")
    target_rows = targets.select("target_id", "sequence").iter_rows()

    candidate_map: dict[str, list[str]] = {}
    for row in ranked.select("target_id", "template_uid").iter_rows():
        target_id = str(row[0])
        candidate_map.setdefault(target_id, [])
        if str(row[1]) not in candidate_map[target_id]:
            candidate_map[target_id].append(str(row[1]))

    out_rows: list[dict[str, object]] = []
    skipped_targets: list[str] = []
    for target_id, sequence in target_rows:
        tid = str(target_id)
        seq = str(sequence)
        choices = candidate_map.get(tid, [])
        if len(choices) == 0:
            skipped_targets.append(f"{tid}:sem_candidatos")
            continue

        valid_choices: list[str] = []
        rejected: list[str] = []
        for template_uid in choices:
            if template_uid not in template_map:
                rejected.append(f"{template_uid}:sem_coordenadas")
                continue
            coords = template_map[template_uid]
            missing = _missing_resid_examples(coords=coords, seq_len=len(seq), limit=3)
            if missing:
                rejected.append(f"{template_uid}:missing={','.join(str(item) for item in missing)}")
                continue
            valid_choices.append(template_uid)
            if len(valid_choices) >= n_models:
                break

        if len(valid_choices) < n_models:
            skipped_targets.append(f"{tid}:validos={len(valid_choices)}<n_models={n_models}")
            selected = valid_choices[:n_models]
            if not selected:
                continue
        else:
            selected = valid_choices[:n_models]
        for model_id, template_uid in enumerate(selected, start=1):
            coords = template_map[template_uid]
            for resid, base in enumerate(seq, start=1):
                resname, x, y, z = coords[resid]
                out_rows.append(
                    {
                        "target_id": tid,
                        "model_id": model_id,
                        "resid": resid,
                        "resname": str(base),
                        "x": x,
                        "y": y,
                        "z": z,
                        "template_uid": template_uid,
                        "template_resname": resname,
                    }
                )

    if out_rows:
        out = pl.DataFrame(out_rows).sort(["target_id", "model_id", "resid"])
    else:
        out = pl.DataFrame(
            schema={
                "target_id": pl.Utf8,
                "model_id": pl.Int32,
                "resid": pl.Int32,
                "resname": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "template_uid": pl.Utf8,
                "template_resname": pl.Utf8,
            }
        )
    write_table(out, out_path)
    manifest_path = out_path.parent / "tbm_manifest.json"
    manifest = {
        "created_utc": utc_now_iso(),
        "paths": {
            "retrieval": rel_or_abs(retrieval_path, repo_root),
            "templates": rel_or_abs(templates_path, repo_root),
            "targets": rel_or_abs(targets_path, repo_root),
            "predictions": rel_or_abs(out_path, repo_root),
        },
        "params": {"n_models": int(n_models)},
        "stats": {
            "n_rows": int(out.height),
            "n_targets_with_tbm": int(out.get_column("target_id").n_unique()) if "target_id" in out.columns else 0,
            "n_targets_skipped": int(len(skipped_targets)),
            "examples_targets_skipped": [str(item) for item in skipped_targets[:8]],
        },
        "sha256": {"predictions.parquet": sha256_file(out_path)},
    }
    write_json(manifest_path, manifest)
    return TbmResult(predictions_path=out_path, manifest_path=manifest_path)
