from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class DescriptionFamilyResult:
    target_family_path: Path
    family_prior_path: Path | None
    manifest_path: Path


def _rule_based_family(description: str) -> tuple[str, float, list[str]]:
    text = str(description or "").lower()
    patterns: list[tuple[str, list[str]]] = [
        ("crispr_cas13", ["cas13", "crispr"]),
        ("ribozyme", ["ribozyme", "ribozima"]),
        ("riboswitch", ["riboswitch"]),
        ("srp", ["signal recognition particle", "srp"]),
        ("trna_like", ["trna"]),
    ]
    for family, words in patterns:
        hits = [word for word in words if word in text]
        if hits:
            return family, 0.95, hits
    return "unknown", 0.2, []


def _llama_cpp_family(*, model_path: Path, description: str, stage: str, location: str) -> tuple[str, float, list[str]]:
    if not model_path.exists():
        raise_error(stage, location, "modelo gguf ausente", impact="1", examples=[str(model_path)])
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise_error(stage, location, "llama_cpp indisponivel", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    llm = Llama(model_path=str(model_path), n_ctx=1024, verbose=False)
    prompt = (
        "Return strict JSON with keys family_label, confidence, keywords.\n"
        "Description:\n"
        f"{description}\n"
    )
    out = llm.create_completion(prompt=prompt, max_tokens=120, temperature=0.0, stop=["\n\n"])
    text = str(out["choices"][0]["text"]).strip()
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        snippet = re.sub(r"\s+", " ", text)[:160]
        raise_error(stage, location, "saida LLM nao e JSON valido", impact="1", examples=[f"{type(exc).__name__}:{snippet}"])
    family = str(payload.get("family_label", "")).strip().lower()
    confidence = float(payload.get("confidence", 0.0))
    keywords = payload.get("keywords", [])
    if not family:
        raise_error(stage, location, "saida LLM sem family_label", impact="1", examples=[text[:120]])
    if not isinstance(keywords, list):
        raise_error(stage, location, "saida LLM com keywords invalidas", impact="1", examples=[str(type(keywords).__name__)])
    return family, confidence, [str(item) for item in keywords[:8]]


def infer_description_family(
    *,
    repo_root: Path,
    targets_path: Path,
    out_dir: Path,
    backend: str,
    llm_model_path: Path | None,
    template_family_map_path: Path | None = None,
) -> DescriptionFamilyResult:
    stage = "DESCRIPTION_FAMILY"
    location = "src/rna3d_local/description_family.py:infer_description_family"
    targets = read_table(targets_path, stage=stage, location=location)
    require_columns(targets, ["target_id", "description"], stage=stage, location=location, label="targets")
    backend_mode = str(backend).strip().lower()
    if backend_mode not in {"llama_cpp", "rules"}:
        raise_error(stage, location, "backend invalido", impact="1", examples=[backend])

    records: list[dict[str, object]] = []
    for row in targets.select(["target_id", "description"]).iter_rows(named=True):
        target_id = str(row["target_id"])
        description = str(row["description"] or "")
        if backend_mode == "llama_cpp":
            if llm_model_path is None:
                raise_error(stage, location, "llm_model_path obrigatorio para backend llama_cpp", impact="1", examples=["llm_model_path=None"])
            family, confidence, keywords = _llama_cpp_family(
                model_path=llm_model_path,
                description=description,
                stage=stage,
                location=location,
            )
        else:
            family, confidence, keywords = _rule_based_family(description)
        records.append(
            {
                "target_id": target_id,
                "family_label": family,
                "confidence": float(confidence),
                "keywords": ",".join(keywords),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    target_family_path = out_dir / "target_family.parquet"
    target_family = pl.DataFrame(records).sort("target_id")
    write_table(target_family, target_family_path)

    family_prior_path: Path | None = None
    family_prior_sha: str | None = None
    if template_family_map_path is not None:
        template_map = read_table(template_family_map_path, stage=stage, location=location)
        require_columns(template_map, ["template_uid", "family_label"], stage=stage, location=location, label="template_family_map")
        family_prior = (
            target_family.join(
                template_map.select(pl.col("template_uid").cast(pl.Utf8), pl.col("family_label").cast(pl.Utf8)),
                on="family_label",
                how="inner",
            )
            .select(
                pl.col("target_id").cast(pl.Utf8),
                pl.col("template_uid").cast(pl.Utf8),
                pl.col("confidence").cast(pl.Float64).alias("family_prior_score"),
            )
            .sort(["target_id", "template_uid"])
        )
        family_prior_path = out_dir / "family_prior.parquet"
        write_table(family_prior, family_prior_path)
        family_prior_sha = sha256_file(family_prior_path)

    manifest_path = out_dir / "description_family_manifest.json"
    manifest = {
        "created_utc": utc_now_iso(),
        "backend": backend_mode,
        "paths": {
            "targets": rel_or_abs(targets_path, repo_root),
            "target_family": rel_or_abs(target_family_path, repo_root),
            "template_family_map": None if template_family_map_path is None else rel_or_abs(template_family_map_path, repo_root),
            "family_prior": None if family_prior_path is None else rel_or_abs(family_prior_path, repo_root),
        },
        "stats": {"n_targets": int(target_family.height)},
        "sha256": {
            "target_family.parquet": sha256_file(target_family_path),
            "family_prior.parquet": family_prior_sha,
        },
    }
    write_json(manifest_path, manifest)
    return DescriptionFamilyResult(target_family_path=target_family_path, family_prior_path=family_prior_path, manifest_path=manifest_path)
