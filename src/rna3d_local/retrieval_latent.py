from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import parse_date_column, require_columns
from .encoder import encode_sequences
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class RetrievalResult:
    candidates_path: Path
    manifest_path: Path


def _sequence_alignment_refine(target_sequence: str, template_sequence: str) -> float:
    target = str(target_sequence).strip().upper()
    template = str(template_sequence).strip().upper()
    if not target or not template:
        return 0.0
    span = min(len(target), len(template))
    matches = 0
    for idx in range(span):
        if target[idx] == template[idx]:
            matches += 1
    return float(matches) / float(max(len(target), len(template)))


def _search_faiss(index_path: Path, queries: np.ndarray, top_k: int, *, stage: str, location: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "faiss indisponivel para busca ANN", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    if not index_path.exists():
        raise_error(stage, location, "indice faiss ausente", impact="1", examples=[str(index_path)])
    index = faiss.read_index(str(index_path))
    scores, ids = index.search(queries.astype(np.float32), int(top_k))
    return scores, ids


def retrieve_templates_latent(
    *,
    repo_root: Path,
    template_index_path: Path,
    template_embeddings_path: Path,
    targets_path: Path,
    out_path: Path,
    top_k: int,
    encoder: str,
    embedding_dim: int,
    model_path: Path | None,
    ann_engine: str,
    faiss_index_path: Path | None,
    family_prior_path: Path | None,
    weight_embed: float = 0.70,
    weight_llm: float = 0.20,
    weight_seq: float = 0.10,
) -> RetrievalResult:
    stage = "RETRIEVAL_LATENT"
    location = "src/rna3d_local/retrieval_latent.py:retrieve_templates_latent"
    if top_k <= 0:
        raise_error(stage, location, "top_k deve ser > 0", impact="1", examples=[str(top_k)])
    if abs((weight_embed + weight_llm + weight_seq) - 1.0) > 1e-6:
        raise_error(
            stage,
            location,
            "pesos de fusao devem somar 1.0",
            impact="1",
            examples=[f"{weight_embed}+{weight_llm}+{weight_seq}"],
        )

    template_index = read_table(template_index_path, stage=stage, location=location)
    require_columns(
        template_index,
        ["template_uid", "sequence", "release_date"],
        stage=stage,
        location=location,
        label="template_index",
    )
    template_index = parse_date_column(template_index, "release_date", stage=stage, location=location, label="template_index")
    template_embeddings = read_table(template_embeddings_path, stage=stage, location=location)
    require_columns(template_embeddings, ["template_uid", "embedding"], stage=stage, location=location, label="template_embeddings")

    merged = template_index.join(template_embeddings, on="template_uid", how="inner")
    if merged.height != template_index.height:
        raise_error(
            stage,
            location,
            "template_embeddings sem cobertura completa do template_index",
            impact=str(abs(template_index.height - merged.height)),
            examples=[],
        )
    templates = merged.sort("template_uid")
    template_uids = templates.get_column("template_uid").cast(pl.Utf8).to_list()
    template_sequences = templates.get_column("sequence").cast(pl.Utf8).to_list()
    template_release_dates = templates.get_column("release_date").to_list()
    template_matrix = np.asarray([np.asarray(item, dtype=np.float32) for item in templates.get_column("embedding").to_list()], dtype=np.float32)
    if template_matrix.shape[1] != embedding_dim:
        raise_error(
            stage,
            location,
            "embedding_dim divergente no template_embeddings",
            impact="1",
            examples=[f"expected={embedding_dim}", f"actual={template_matrix.shape[1]}"],
        )

    targets = read_table(targets_path, stage=stage, location=location)
    require_columns(targets, ["target_id", "sequence"], stage=stage, location=location, label="targets")
    default_cutoff = date(2100, 1, 1)
    if "temporal_cutoff" in targets.columns:
        parsed_cutoff = targets.with_columns(
            pl.col("temporal_cutoff").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("temporal_cutoff")
        )
        bad_cutoff = int(parsed_cutoff.get_column("temporal_cutoff").null_count())
        if bad_cutoff > 0:
            examples = parsed_cutoff.filter(pl.col("temporal_cutoff").is_null()).head(8).to_dicts()
            print(
                f"[{stage}] [{location}] targets com temporal_cutoff invalido; usando cutoff padrao para evitar crash em rerun oculto | "
                f"impacto={bad_cutoff} | exemplos={examples}",
            )
        targets = parsed_cutoff.with_columns(pl.col("temporal_cutoff").fill_null(pl.lit(default_cutoff)))
    else:
        targets = targets.with_columns(pl.lit(default_cutoff).cast(pl.Date).alias("temporal_cutoff"))
    target_ids = targets.get_column("target_id").cast(pl.Utf8).to_list()
    target_sequences = targets.get_column("sequence").cast(pl.Utf8).to_list()
    query_matrix = encode_sequences(
        [str(seq) for seq in target_sequences],
        encoder=encoder,
        embedding_dim=embedding_dim,
        model_path=model_path,
        stage=stage,
        location=location,
    )

    family_prior: dict[tuple[str, str], float] = {}
    if family_prior_path is not None:
        priors = read_table(family_prior_path, stage=stage, location=location)
        require_columns(priors, ["target_id", "template_uid", "family_prior_score"], stage=stage, location=location, label="family_prior")
        for row in priors.select("target_id", "template_uid", "family_prior_score").iter_rows():
            family_prior[(str(row[0]), str(row[1]))] = float(row[2])

    ann_mode = ann_engine.strip().lower()
    candidate_scores: dict[str, list[tuple[int, float]]] = {}
    search_k = min(max(int(top_k) * 8, int(top_k)), template_matrix.shape[0])
    if ann_mode == "faiss_ivfpq":
        if faiss_index_path is None:
            raise_error(stage, location, "faiss_index_path obrigatorio para ann_engine=faiss_ivfpq", impact="1", examples=["faiss_index_path=None"])
        scores, indices = _search_faiss(faiss_index_path, query_matrix, search_k, stage=stage, location=location)
        for idx, target_id in enumerate(target_ids):
            pairs = []
            for score, position in zip(scores[idx].tolist(), indices[idx].tolist()):
                if int(position) < 0:
                    continue
                pairs.append((int(position), float(score)))
            candidate_scores[target_id] = pairs
    elif ann_mode == "numpy_bruteforce":
        for idx, target_id in enumerate(target_ids):
            q = query_matrix[idx]
            sims = template_matrix @ q
            top_positions = np.argsort(-sims)[:search_k]
            candidate_scores[target_id] = [(int(position), float(sims[position])) for position in top_positions.tolist()]
    else:
        raise_error(stage, location, "ann_engine invalido", impact="1", examples=[ann_engine])

    rows: list[dict[str, object]] = []
    no_candidate_targets: list[str] = []
    target_cutoffs = {
        str(row["target_id"]): row["temporal_cutoff"]
        for row in targets.select("target_id", "temporal_cutoff").iter_rows(named=True)
    }
    target_sequence_map = {
        str(row["target_id"]): str(row["sequence"])
        for row in targets.select("target_id", "sequence").iter_rows(named=True)
    }
    for target_id in target_ids:
        cutoff = target_cutoffs[target_id]
        target_sequence = target_sequence_map[target_id]
        scored: list[dict[str, object]] = []
        for position, cosine_score in candidate_scores.get(target_id, []):
            release_date = template_release_dates[position]
            if release_date is None or release_date > cutoff:
                continue
            template_uid = template_uids[position]
            template_sequence = str(template_sequences[position])
            family_score = float(family_prior.get((target_id, template_uid), 0.0))
            refine_score = _sequence_alignment_refine(target_sequence, template_sequence)
            final_score = (weight_embed * cosine_score) + (weight_llm * family_score) + (weight_seq * refine_score)
            scored.append(
                {
                    "target_id": target_id,
                    "template_uid": template_uid,
                    "cosine_score": float(cosine_score),
                    "family_prior_score": float(family_score),
                    "alignment_refine_score": float(refine_score),
                    "final_score": float(final_score),
                    "template_release_date": release_date,
                    "target_temporal_cutoff": cutoff,
                }
            )
        scored = sorted(scored, key=lambda item: (-float(item["final_score"]), -float(item["cosine_score"]), str(item["template_uid"])))[:top_k]
        if not scored:
            no_candidate_targets.append(target_id)
            continue
        for rank, row in enumerate(scored, start=1):
            row["rank"] = rank
            rows.append(row)

    if rows:
        out = pl.DataFrame(rows).sort(["target_id", "rank"])
    else:
        out = pl.DataFrame(
            schema={
                "target_id": pl.Utf8,
                "template_uid": pl.Utf8,
                "cosine_score": pl.Float64,
                "family_prior_score": pl.Float64,
                "alignment_refine_score": pl.Float64,
                "final_score": pl.Float64,
                "template_release_date": pl.Date,
                "target_temporal_cutoff": pl.Date,
                "rank": pl.Int64,
            }
        )
    write_table(out, out_path)
    manifest_path = out_path.parent / "retrieval_manifest.json"
    manifest = {
        "created_utc": utc_now_iso(),
        "params": {
            "top_k": int(top_k),
            "encoder": encoder,
            "embedding_dim": int(embedding_dim),
            "ann_engine": ann_mode,
            "weights": {"embed": float(weight_embed), "llm": float(weight_llm), "seq": float(weight_seq)},
        },
        "paths": {
            "template_index": rel_or_abs(template_index_path, repo_root),
            "template_embeddings": rel_or_abs(template_embeddings_path, repo_root),
            "targets": rel_or_abs(targets_path, repo_root),
            "family_prior": None if family_prior_path is None else rel_or_abs(family_prior_path, repo_root),
            "candidates": rel_or_abs(out_path, repo_root),
        },
        "stats": {
            "n_rows": int(out.height),
            "n_targets_with_candidates": int(out.get_column("target_id").n_unique()) if "target_id" in out.columns else 0,
            "n_targets_without_candidates": int(len(no_candidate_targets)),
            "examples_targets_without_candidates": [str(item) for item in no_candidate_targets[:8]],
        },
        "sha256": {"candidates.parquet": sha256_file(out_path)},
    }
    write_json(manifest_path, manifest)
    return RetrievalResult(candidates_path=out_path, manifest_path=manifest_path)
