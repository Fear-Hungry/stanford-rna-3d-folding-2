from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .alignment import compute_coverage, map_target_to_template_positions, project_target_coordinates
from .errors import raise_error
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TbmPredictionResult:
    predictions_path: Path
    manifest_path: Path


def _build_template_maps(templates_df: pl.DataFrame, *, location: str) -> tuple[dict[str, str], dict[str, dict[int, tuple[float, float, float]]]]:
    required = {"template_uid", "sequence", "resid", "x", "y", "z"}
    miss = [c for c in required if c not in templates_df.columns]
    if miss:
        raise_error("TBM", location, "templates sem coluna obrigatoria", impact=str(len(miss)), examples=miss[:8])

    seq_map: dict[str, str] = {}
    coord_map: dict[str, dict[int, tuple[float, float, float]]] = {}
    for row in templates_df.select("template_uid", "sequence", "resid", "x", "y", "z").to_dicts():
        uid = str(row["template_uid"])
        seq = str(row["sequence"])
        resid = int(row["resid"])
        coord = (float(row["x"]), float(row["y"]), float(row["z"]))
        prev = seq_map.get(uid)
        if prev is None:
            seq_map[uid] = seq
        elif prev != seq:
            raise_error(
                "TBM",
                location,
                "sequence inconsistente dentro do mesmo template_uid",
                impact="1",
                examples=[uid],
            )
        coord_map.setdefault(uid, {})
        if resid in coord_map[uid]:
            raise_error(
                "TBM",
                location,
                "resid duplicado no template",
                impact="1",
                examples=[f"{uid}:{resid}"],
            )
        coord_map[uid][resid] = coord
    return seq_map, coord_map


def predict_tbm(
    *,
    repo_root: Path,
    retrieval_candidates_path: Path,
    templates_path: Path,
    target_sequences_path: Path,
    out_path: Path,
    n_models: int,
    min_coverage: float = 0.35,
) -> TbmPredictionResult:
    """
    Generate per-residue predictions from top template candidates.
    """
    location = "src/rna3d_local/tbm_predictor.py:predict_tbm"
    for p in (retrieval_candidates_path, templates_path, target_sequences_path):
        if not p.exists():
            raise_error("TBM", location, "arquivo obrigatorio ausente", impact="1", examples=[str(p)])
    if n_models <= 0:
        raise_error("TBM", location, "n_models deve ser > 0", impact="1", examples=[str(n_models)])
    if min_coverage <= 0 or min_coverage > 1:
        raise_error("TBM", location, "min_coverage invalido (0,1]", impact="1", examples=[str(min_coverage)])

    retrieval_df = pl.read_parquet(retrieval_candidates_path)
    for col in ("target_id", "template_uid", "rank", "similarity"):
        if col not in retrieval_df.columns:
            raise_error("TBM", location, "retrieval sem coluna obrigatoria", impact="1", examples=[col])
    retrieval_df = retrieval_df.sort(["target_id", "rank"])

    templates_df = pl.read_parquet(templates_path)
    template_sequences, template_coords = _build_template_maps(templates_df, location=location)

    targets = pl.read_csv(target_sequences_path, infer_schema_length=1000)
    for col in ("target_id", "sequence"):
        if col not in targets.columns:
            raise_error("TBM", location, "target_sequences sem coluna obrigatoria", impact="1", examples=[col])
    targets = targets.select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)).sort("target_id")

    rows: list[dict] = []
    missing_candidates: list[str] = []
    low_coverage: list[str] = []
    for row in targets.to_dicts():
        tid = str(row["target_id"])
        seq = str(row["sequence"])
        candidates = retrieval_df.filter(pl.col("target_id") == tid).sort("rank")
        if candidates.height < n_models:
            missing_candidates.append(f"{tid}:{candidates.height}")
            continue
        target_length = len(seq)
        for model_id, cand in enumerate(candidates.head(n_models).to_dicts(), start=1):
            uid = str(cand["template_uid"])
            tpl_seq = template_sequences.get(uid)
            tpl_coords = template_coords.get(uid)
            if tpl_seq is None or tpl_coords is None:
                raise_error("TBM", location, "template_uid sem dados de template", impact="1", examples=[uid])

            mapping = map_target_to_template_positions(
                target_sequence=seq,
                template_sequence=tpl_seq,
                location=location,
            )
            valid_mapped = {
                tpos: spos
                for tpos, spos in mapping.items()
                if spos in tpl_coords
            }
            cov = compute_coverage(mapped_positions=len(valid_mapped), target_length=target_length, location=location)
            if cov < min_coverage:
                low_coverage.append(f"{tid}:{uid}:{cov:.4f}")
                continue
            coords = project_target_coordinates(
                target_length=target_length,
                mapping=valid_mapped,
                template_coordinates=tpl_coords,
                location=location,
            )
            for resid, (base, xyz) in enumerate(zip(seq, coords, strict=True), start=1):
                rows.append(
                    {
                        "branch": "tbm",
                        "target_id": tid,
                        "ID": f"{tid}_{resid}",
                        "resid": resid,
                        "resname": base,
                        "model_id": model_id,
                        "x": float(xyz[0]),
                        "y": float(xyz[1]),
                        "z": float(xyz[2]),
                        "template_uid": uid,
                        "template_rank": int(cand["rank"]),
                        "similarity": float(cand["similarity"]),
                        "coverage": float(cov),
                    }
                )

    if missing_candidates:
        raise_error(
            "TBM",
            location,
            "alvos sem candidatos suficientes para n_models",
            impact=str(len(missing_candidates)),
            examples=missing_candidates[:8],
        )
    if low_coverage:
        raise_error(
            "TBM",
            location,
            "cobertura abaixo do minimo",
            impact=str(len(low_coverage)),
            examples=low_coverage[:8],
        )
    if not rows:
        raise_error("TBM", location, "nenhuma predicao gerada", impact="0", examples=[])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pl.DataFrame(rows).sort(["target_id", "model_id", "resid"])
    out_df.write_parquet(out_path)

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "retrieval_candidates": _rel(retrieval_candidates_path, repo_root),
            "templates": _rel(templates_path, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "predictions_long": _rel(out_path, repo_root),
        },
        "params": {"n_models": n_models, "min_coverage": min_coverage},
        "stats": {"n_rows": int(out_df.height), "n_targets": int(out_df.get_column("target_id").n_unique())},
        "sha256": {"predictions_long.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "tbm_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TbmPredictionResult(predictions_path=out_path, manifest_path=manifest_path)
