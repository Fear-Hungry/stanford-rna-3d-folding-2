from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .se3.sequence_parser import parse_sequence_with_chains
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class HomologyFoldsResult:
    clusters_path: Path
    train_folds_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class _SequenceEntry:
    entity_type: str
    entity_id: str
    global_id: str
    sequence: str
    sequence_length: int


def _normalize_domain_label(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _infer_domain_from_description(description: str) -> str:
    text = str(description or "").lower()
    rules: list[tuple[str, list[str]]] = [
        ("crispr_cas", ["crispr", "cas13", "cas12", "cas9", "cas"]),
        ("ribozyme", ["ribozyme", "ribozima", "self-cleaving"]),
        ("trna", ["trna", "transfer rna"]),
        ("riboswitch", ["riboswitch"]),
        ("rrna_ribosome", ["ribosome", "ribosomal", "rrna"]),
        ("srp", ["signal recognition particle", "srp"]),
    ]
    for label, needles in rules:
        if any(needle in text for needle in needles):
            return label
    return "unknown"


def _resolve_id_column(df: pl.DataFrame, candidates: list[str], *, stage: str, location: str, label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise_error(stage, location, f"{label} sem coluna de id suportada", impact="1", examples=candidates[:8])


def _read_entries(
    *,
    path: Path,
    entity_type: str,
    id_candidates: list[str],
    chain_separator: str,
    stage: str,
    location: str,
) -> list[_SequenceEntry]:
    table = read_table(path, stage=stage, location=location)
    require_columns(table, ["sequence"], stage=stage, location=location, label=f"{entity_type}_table")
    id_column = _resolve_id_column(table, id_candidates, stage=stage, location=location, label=f"{entity_type}_table")
    rows = table.select(pl.col(id_column).cast(pl.Utf8).alias("entity_id"), pl.col("sequence").cast(pl.Utf8))
    bad_id = rows.filter(pl.col("entity_id").is_null() | (pl.col("entity_id").str.strip_chars() == ""))
    if bad_id.height > 0:
        examples = bad_id.get_column("entity_id").head(8).to_list()
        raise_error(stage, location, f"{entity_type} com id vazio", impact=str(int(bad_id.height)), examples=[str(item) for item in examples])
    dup = rows.group_by("entity_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = dup.get_column("entity_id").head(8).to_list()
        raise_error(stage, location, f"{entity_type} com id duplicado", impact=str(int(dup.height)), examples=[str(item) for item in examples])
    entries: list[_SequenceEntry] = []
    for entity_id, sequence in rows.iter_rows():
        parsed = parse_sequence_with_chains(
            sequence=str(sequence),
            chain_separator=chain_separator,
            stage=stage,
            location=location,
            target_id=str(entity_id),
        )
        normalized = "".join(parsed.residues)
        if len(normalized) < 4:
            raise_error(stage, location, f"{entity_type} com sequencia curta demais", impact="1", examples=[str(entity_id)])
        entries.append(
            _SequenceEntry(
                entity_type=str(entity_type),
                entity_id=str(entity_id),
                global_id=f"{entity_type}::{entity_id}",
                sequence=normalized,
                sequence_length=len(normalized),
            )
        )
    if not entries:
        raise_error(stage, location, f"{entity_type} sem entradas", impact="0", examples=[])
    return entries


def _write_fasta(entries: list[_SequenceEntry], path: Path) -> None:
    lines: list[str] = []
    for entry in entries:
        lines.append(f">{entry.global_id}")
        lines.append(entry.sequence)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _identity_coverage(seq_a: str, seq_b: str) -> tuple[float, float]:
    min_len = min(len(seq_a), len(seq_b))
    max_len = max(len(seq_a), len(seq_b))
    matches = sum(1 for idx in range(min_len) if seq_a[idx] == seq_b[idx])
    identity = float(matches) / float(min_len)
    coverage = float(min_len) / float(max_len)
    return identity, coverage


def _build_train_domain_map(
    *,
    train_targets_path: Path,
    domain_labels_path: Path | None,
    domain_column: str,
    description_column: str,
    strict_domain_stratification: bool,
    stage: str,
    location: str,
) -> dict[str, str]:
    train_table = read_table(train_targets_path, stage=stage, location=location)
    id_column = _resolve_id_column(train_table, ["target_id", "ID", "id"], stage=stage, location=location, label="train_table")
    train_ids_df = train_table.select(pl.col(id_column).cast(pl.Utf8).alias("target_id"))
    bad_id = train_ids_df.filter(pl.col("target_id").is_null() | (pl.col("target_id").str.strip_chars() == ""))
    if bad_id.height > 0:
        raise_error(stage, location, "train com target_id vazio", impact=str(int(bad_id.height)), examples=["target_id"])
    dup = train_ids_df.group_by("target_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = dup.get_column("target_id").head(8).to_list()
        raise_error(stage, location, "train com target_id duplicado", impact=str(int(dup.height)), examples=[str(x) for x in examples])
    train_ids = train_ids_df.get_column("target_id").to_list()
    train_id_set = set(str(item) for item in train_ids)

    labels_raw: dict[str, str] = {}
    source = "none"
    if domain_labels_path is not None:
        source = "domain_labels_file"
        labels = read_table(domain_labels_path, stage=stage, location=location)
        labels_id_col = _resolve_id_column(labels, ["target_id", "ID", "id"], stage=stage, location=location, label="domain_labels")
        if domain_column not in labels.columns:
            raise_error(stage, location, "domain_labels sem coluna de dominio", impact="1", examples=[domain_column])
        subset = labels.select(
            pl.col(labels_id_col).cast(pl.Utf8).alias("target_id"),
            pl.col(domain_column).cast(pl.Utf8).alias("domain_label"),
        )
        label_dup = subset.group_by("target_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
        if label_dup.height > 0:
            examples = label_dup.get_column("target_id").head(8).to_list()
            raise_error(stage, location, "domain_labels com target_id duplicado", impact=str(int(label_dup.height)), examples=[str(x) for x in examples])
        labels_raw = {
            str(row["target_id"]): str(row["domain_label"] or "")
            for row in subset.iter_rows(named=True)
            if str(row["target_id"]) in train_id_set
        }
    elif domain_column in train_table.columns:
        source = "train_domain_column"
        subset = train_table.select(
            pl.col(id_column).cast(pl.Utf8).alias("target_id"),
            pl.col(domain_column).cast(pl.Utf8).alias("domain_label"),
        )
        labels_raw = {str(row["target_id"]): str(row["domain_label"] or "") for row in subset.iter_rows(named=True)}
    elif description_column in train_table.columns:
        source = "description_rules"
        subset = train_table.select(
            pl.col(id_column).cast(pl.Utf8).alias("target_id"),
            pl.col(description_column).cast(pl.Utf8).alias("description"),
        )
        labels_raw = {
            str(row["target_id"]): _infer_domain_from_description(str(row["description"] or ""))
            for row in subset.iter_rows(named=True)
        }
    elif strict_domain_stratification:
        raise_error(
            stage,
            location,
            "estratificacao por dominio sem fonte de dominio",
            impact="1",
            examples=[f"domain_column={domain_column}", f"description_column={description_column}"],
        )

    labels: dict[str, str] = {}
    missing: list[str] = []
    blank: list[str] = []
    for target_id in train_ids:
        value = labels_raw.get(str(target_id))
        if value is None:
            missing.append(str(target_id))
            continue
        normalized = _normalize_domain_label(value)
        if not normalized:
            blank.append(str(target_id))
            continue
        labels[str(target_id)] = normalized
    if strict_domain_stratification:
        if missing:
            raise_error(stage, location, "target sem rotulo de dominio", impact=str(len(missing)), examples=missing[:8])
        if blank:
            raise_error(stage, location, "rotulo de dominio vazio", impact=str(len(blank)), examples=blank[:8])
        unknown = [target_id for target_id, label in labels.items() if label == "unknown"]
        if unknown and len(unknown) == len(train_ids):
            raise_error(stage, location, "inferencia de dominio retornou apenas unknown", impact=str(len(unknown)), examples=unknown[:8])
    if not labels and strict_domain_stratification:
        raise_error(stage, location, "nenhum dominio valido encontrado", impact="0", examples=[source])
    return labels


def _cluster_python(entries: list[_SequenceEntry], *, identity_threshold: float, coverage_threshold: float) -> dict[str, str]:
    ordered = sorted(entries, key=lambda item: (-item.sequence_length, item.global_id))
    unassigned = {item.global_id: item for item in ordered}
    mapping: dict[str, str] = {}
    while unassigned:
        rep_id = sorted(unassigned.keys())[0]
        rep = unassigned.pop(rep_id)
        mapping[rep.global_id] = rep.global_id
        members = list(unassigned.values())
        for candidate in members:
            identity, coverage = _identity_coverage(rep.sequence, candidate.sequence)
            if identity >= identity_threshold and coverage >= coverage_threshold:
                mapping[candidate.global_id] = rep.global_id
                del unassigned[candidate.global_id]
    return mapping


def _cluster_mmseqs2(
    *,
    entries: list[_SequenceEntry],
    fasta_path: Path,
    mmseqs_bin: str,
    identity_threshold: float,
    coverage_threshold: float,
    stage: str,
    location: str,
) -> dict[str, str]:
    with TemporaryDirectory(prefix="rna3d_mmseqs_cluster_") as tmp_dir:
        tmp = Path(tmp_dir)
        out_prefix = tmp / "cluster"
        cmd = [
            str(mmseqs_bin),
            "easy-cluster",
            str(fasta_path),
            str(out_prefix),
            str(tmp / "work"),
            "--min-seq-id",
            f"{identity_threshold:.6f}",
            "-c",
            f"{coverage_threshold:.6f}",
            "--cov-mode",
            "0",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            raise_error(stage, location, "binario mmseqs2 nao encontrado", impact="1", examples=[str(mmseqs_bin)])
        except Exception as exc:
            raise_error(stage, location, "falha ao executar mmseqs2 easy-cluster", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
        if proc.returncode != 0:
            stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
            raise_error(
                stage,
                location,
                "mmseqs2 easy-cluster retornou erro",
                impact="1",
                examples=[stderr_txt[:240] if stderr_txt else f"returncode={proc.returncode}"],
            )
        cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
        if not cluster_tsv.exists():
            raise_error(stage, location, "mmseqs2 nao gerou cluster.tsv", impact="1", examples=[str(cluster_tsv)])
        mapping: dict[str, str] = {}
        for line in cluster_tsv.read_text(encoding="utf-8", errors="replace").splitlines():
            clean = line.strip()
            if not clean:
                continue
            parts = clean.split("\t")
            if len(parts) < 2:
                continue
            rep_id = parts[0].strip()
            member_id = parts[1].strip()
            if rep_id and member_id:
                mapping[member_id] = rep_id
        return mapping


def _parse_cdhit_clusters(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    current_members: list[str] = []
    current_rep: str | None = None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines + [">Cluster END"]:
        if line.startswith(">Cluster"):
            if current_members:
                rep = current_rep if current_rep is not None else current_members[0]
                for member in current_members:
                    mapping[member] = rep
            current_members = []
            current_rep = None
            continue
        stripped = line.strip()
        if not stripped:
            continue
        start = stripped.find(">")
        end = stripped.find("...", start + 1)
        if start < 0 or end < 0:
            continue
        member_id = stripped[start + 1 : end].strip()
        if member_id:
            current_members.append(member_id)
            if stripped.endswith("*"):
                current_rep = member_id
    return mapping


def _cluster_cdhit_est(
    *,
    entries: list[_SequenceEntry],
    fasta_path: Path,
    cdhit_bin: str,
    identity_threshold: float,
    coverage_threshold: float,
    stage: str,
    location: str,
) -> dict[str, str]:
    with TemporaryDirectory(prefix="rna3d_cdhit_cluster_") as tmp_dir:
        tmp = Path(tmp_dir)
        out_fasta = tmp / "cluster.fa"
        cmd = [
            str(cdhit_bin),
            "-i",
            str(fasta_path),
            "-o",
            str(out_fasta),
            "-c",
            f"{identity_threshold:.6f}",
            "-aS",
            f"{coverage_threshold:.6f}",
            "-d",
            "0",
            "-M",
            "0",
            "-T",
            "1",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            raise_error(stage, location, "binario cd-hit-est nao encontrado", impact="1", examples=[str(cdhit_bin)])
        except Exception as exc:
            raise_error(stage, location, "falha ao executar cd-hit-est", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
        if proc.returncode != 0:
            stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
            raise_error(
                stage,
                location,
                "cd-hit-est retornou erro",
                impact="1",
                examples=[stderr_txt[:240] if stderr_txt else f"returncode={proc.returncode}"],
            )
        cluster_file = Path(f"{out_fasta}.clstr")
        if not cluster_file.exists():
            raise_error(stage, location, "cd-hit-est nao gerou arquivo .clstr", impact="1", examples=[str(cluster_file)])
        return _parse_cdhit_clusters(cluster_file)


def _build_cluster_mapping(
    *,
    entries: list[_SequenceEntry],
    backend: str,
    identity_threshold: float,
    coverage_threshold: float,
    mmseqs_bin: str,
    cdhit_bin: str,
    stage: str,
    location: str,
) -> dict[str, str]:
    with TemporaryDirectory(prefix="rna3d_cluster_input_") as tmp_dir:
        fasta_path = Path(tmp_dir) / "all_sequences.fasta"
        _write_fasta(entries, fasta_path)
        if backend == "python":
            mapping = _cluster_python(
                entries,
                identity_threshold=identity_threshold,
                coverage_threshold=coverage_threshold,
            )
        elif backend == "mmseqs2":
            mapping = _cluster_mmseqs2(
                entries=entries,
                fasta_path=fasta_path,
                mmseqs_bin=mmseqs_bin,
                identity_threshold=identity_threshold,
                coverage_threshold=coverage_threshold,
                stage=stage,
                location=location,
            )
        elif backend == "cdhit_est":
            mapping = _cluster_cdhit_est(
                entries=entries,
                fasta_path=fasta_path,
                cdhit_bin=cdhit_bin,
                identity_threshold=identity_threshold,
                coverage_threshold=coverage_threshold,
                stage=stage,
                location=location,
            )
        else:
            raise_error(stage, location, "backend de clustering invalido", impact="1", examples=[backend])
    missing = [entry.global_id for entry in entries if entry.global_id not in mapping]
    if missing:
        raise_error(
            stage,
            location,
            "mapeamento de clusters incompleto",
            impact=str(len(missing)),
            examples=missing[:8],
        )
    return mapping


def _assign_folds_for_train(
    *,
    train_rows: list[dict[str, object]],
    n_folds: int,
    strict_domain_stratification: bool,
    stage: str,
    location: str,
) -> tuple[
    list[dict[str, object]],
    int,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    if n_folds <= 1:
        raise_error(stage, location, "n_folds deve ser > 1", impact="1", examples=[str(n_folds)])
    train_ids = [str(row["entity_id"]) for row in train_rows]
    if len(train_ids) < n_folds:
        raise_error(
            stage,
            location,
            "n_folds maior que numero de alvos de treino",
            impact=str(n_folds - len(train_ids)),
            examples=[f"n_train={len(train_ids)}", f"n_folds={n_folds}"],
        )
    bad_domain = [str(row["entity_id"]) for row in train_rows if not _normalize_domain_label(str(row.get("domain_label", "")))]
    if strict_domain_stratification and bad_domain:
        raise_error(stage, location, "target de treino sem domain_label valido", impact=str(len(bad_domain)), examples=bad_domain[:8])
    cluster_to_targets: dict[str, list[str]] = {}
    target_to_domain: dict[str, str] = {}
    cluster_to_domains: dict[str, dict[str, int]] = {}
    domain_total_counts: dict[str, int] = {}
    domain_cluster_sets: dict[str, set[str]] = {}
    for row in train_rows:
        cluster_id = str(row["cluster_id"])
        target_id = str(row["entity_id"])
        domain_label = _normalize_domain_label(str(row.get("domain_label", "")))
        target_to_domain[target_id] = domain_label
        cluster_to_targets.setdefault(cluster_id, []).append(target_id)
        if domain_label:
            domain_total_counts[domain_label] = int(domain_total_counts.get(domain_label, 0) + 1)
            cluster_domain = cluster_to_domains.setdefault(cluster_id, {})
            cluster_domain[domain_label] = int(cluster_domain.get(domain_label, 0) + 1)
            domain_cluster_sets.setdefault(domain_label, set()).add(cluster_id)
    cluster_sizes = sorted(cluster_to_targets.items(), key=lambda item: (-len(item[1]), item[0]))
    fold_load = [0 for _ in range(n_folds)]
    fold_domain_counts: list[dict[str, int]] = [{} for _ in range(n_folds)]
    cluster_to_fold: dict[str, int] = {}
    for cluster_id, members in cluster_sizes:
        cluster_domains = cluster_to_domains.get(cluster_id, {})

        def _objective(idx: int) -> tuple[float, float, float, int]:
            gain = 0.0
            domain_penalty = 0.0
            for domain_label, count in cluster_domains.items():
                if fold_domain_counts[idx].get(domain_label, 0) == 0:
                    gain += 1.0
                target_share = float(domain_total_counts[domain_label]) / float(n_folds)
                projected = float(fold_domain_counts[idx].get(domain_label, 0) + count)
                domain_penalty += abs(projected - target_share)
            projected_load = float(fold_load[idx] + len(members))
            return (-gain, projected_load, domain_penalty, idx)

        fold_id = min(range(n_folds), key=_objective)
        cluster_to_fold[cluster_id] = int(fold_id)
        fold_load[fold_id] += len(members)
        for domain_label, count in cluster_domains.items():
            fold_domain_counts[fold_id][domain_label] = int(fold_domain_counts[fold_id].get(domain_label, 0) + count)
    fold_rows: list[dict[str, object]] = []
    for cluster_id, members in cluster_to_targets.items():
        fold_id = cluster_to_fold[cluster_id]
        for target_id in sorted(members):
            fold_rows.append(
                {
                    "target_id": target_id,
                    "cluster_id": cluster_id,
                    "domain_label": target_to_domain.get(target_id, ""),
                    "fold_id": int(fold_id),
                }
            )
    fold_rows = sorted(fold_rows, key=lambda item: (item["fold_id"], item["target_id"]))
    seen_targets = {str(row["target_id"]) for row in fold_rows}
    missing_targets = sorted(set(train_ids) - seen_targets)
    if missing_targets:
        raise_error(stage, location, "targets de treino sem fold", impact=str(len(missing_targets)), examples=missing_targets[:8])
    cluster_fold_count: dict[str, int] = {}
    for cluster_id, members in cluster_to_targets.items():
        fold_values = {row["fold_id"] for row in fold_rows if row["cluster_id"] == cluster_id}
        cluster_fold_count[cluster_id] = int(len(fold_values))
    max_folds_per_cluster = max(cluster_fold_count.values()) if cluster_fold_count else 0
    if max_folds_per_cluster > 1:
        leaking_clusters = [cluster_id for cluster_id, count in cluster_fold_count.items() if count > 1]
        raise_error(
            stage,
            location,
            "leakage de homologia detectado: cluster em multiplos folds",
            impact=str(len(leaking_clusters)),
            examples=leaking_clusters[:8],
        )
    domain_fold_coverage: dict[str, int] = {}
    domain_max_possible_coverage: dict[str, int] = {}
    for domain_label, clusters in domain_cluster_sets.items():
        expected = min(int(n_folds), int(len(clusters)))
        actual = len({int(row["fold_id"]) for row in fold_rows if str(row["domain_label"]) == domain_label})
        domain_fold_coverage[domain_label] = int(actual)
        domain_max_possible_coverage[domain_label] = int(expected)
    if strict_domain_stratification:
        bad_coverage = [
            f"{domain_label}:{domain_fold_coverage.get(domain_label, 0)}/{domain_max_possible_coverage.get(domain_label, 0)}"
            for domain_label in sorted(domain_max_possible_coverage)
            if domain_fold_coverage.get(domain_label, 0) < domain_max_possible_coverage.get(domain_label, 0)
        ]
        if bad_coverage:
            raise_error(
                stage,
                location,
                "cobertura de dominio por fold abaixo do maximo factivel",
                impact=str(len(bad_coverage)),
                examples=bad_coverage[:8],
            )
    fold_domain_counts_out = {f"fold_{idx}": {key: int(value) for key, value in sorted(fold_domain_counts[idx].items())} for idx in range(n_folds)}
    fold_counts = {f"fold_{idx}": int(fold_load[idx]) for idx in range(n_folds)}
    return (
        fold_rows,
        int(max_folds_per_cluster),
        fold_counts,
        domain_fold_coverage,
        domain_max_possible_coverage,
        fold_domain_counts_out,
        {key: int(value) for key, value in sorted(domain_total_counts.items())},
    )


def build_homology_folds(
    *,
    repo_root: Path,
    train_targets_path: Path,
    pdb_sequences_path: Path,
    out_dir: Path,
    backend: str,
    identity_threshold: float,
    coverage_threshold: float,
    n_folds: int,
    chain_separator: str,
    mmseqs_bin: str,
    cdhit_bin: str,
    domain_labels_path: Path | None,
    domain_column: str,
    description_column: str,
    strict_domain_stratification: bool,
) -> HomologyFoldsResult:
    stage = "BUILD_HOMOLOGY_FOLDS"
    location = "src/rna3d_local/homology_folds.py:build_homology_folds"
    if identity_threshold <= 0 or identity_threshold > 1:
        raise_error(stage, location, "identity_threshold invalido", impact="1", examples=[str(identity_threshold)])
    if coverage_threshold <= 0 or coverage_threshold > 1:
        raise_error(stage, location, "coverage_threshold invalido", impact="1", examples=[str(coverage_threshold)])
    train_domain_map = _build_train_domain_map(
        train_targets_path=train_targets_path,
        domain_labels_path=domain_labels_path,
        domain_column=domain_column,
        description_column=description_column,
        strict_domain_stratification=bool(strict_domain_stratification),
        stage=stage,
        location=location,
    )
    train_entries = _read_entries(
        path=train_targets_path,
        entity_type="train",
        id_candidates=["target_id", "ID", "id"],
        chain_separator=chain_separator,
        stage=stage,
        location=location,
    )
    pdb_entries = _read_entries(
        path=pdb_sequences_path,
        entity_type="pdb",
        id_candidates=["template_id", "pdb_id", "target_id", "ID", "id"],
        chain_separator=chain_separator,
        stage=stage,
        location=location,
    )
    entries = train_entries + pdb_entries
    mapping = _build_cluster_mapping(
        entries=entries,
        backend=str(backend),
        identity_threshold=float(identity_threshold),
        coverage_threshold=float(coverage_threshold),
        mmseqs_bin=str(mmseqs_bin),
        cdhit_bin=str(cdhit_bin),
        stage=stage,
        location=location,
    )
    reps = sorted(set(mapping.values()))
    rep_to_cluster = {rep: f"C{idx:05d}" for idx, rep in enumerate(reps, start=1)}
    cluster_rows: list[dict[str, object]] = []
    for entry in entries:
        rep_global = mapping[entry.global_id]
        domain_label = train_domain_map.get(entry.entity_id) if entry.entity_type == "train" else None
        cluster_rows.append(
            {
                "entity_type": entry.entity_type,
                "entity_id": entry.entity_id,
                "global_id": entry.global_id,
                "representative_global_id": rep_global,
                "cluster_id": rep_to_cluster[rep_global],
                "sequence_length": int(entry.sequence_length),
                "domain_label": None if domain_label is None else str(domain_label),
            }
        )
    clusters_df = pl.DataFrame(cluster_rows).sort(["cluster_id", "entity_type", "entity_id"])
    train_cluster_rows = [row for row in cluster_rows if row["entity_type"] == "train"]
    (
        fold_rows,
        max_folds_per_cluster,
        fold_counts,
        domain_fold_coverage,
        domain_max_possible_coverage,
        fold_domain_counts,
        domain_total_counts,
    ) = _assign_folds_for_train(
        train_rows=train_cluster_rows,
        n_folds=int(n_folds),
        strict_domain_stratification=bool(strict_domain_stratification),
        stage=stage,
        location=location,
    )
    train_folds_df = pl.DataFrame(fold_rows).sort(["fold_id", "target_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = out_dir / "clusters.parquet"
    train_folds_path = out_dir / "train_folds.parquet"
    manifest_path = out_dir / "homology_folds_manifest.json"
    write_table(clusters_df, clusters_path)
    write_table(train_folds_df, train_folds_path)
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "paths": {
                "train_targets": rel_or_abs(train_targets_path, repo_root),
                "pdb_sequences": rel_or_abs(pdb_sequences_path, repo_root),
                "clusters": rel_or_abs(clusters_path, repo_root),
                "train_folds": rel_or_abs(train_folds_path, repo_root),
            },
            "params": {
                "backend": str(backend),
                "identity_threshold": float(identity_threshold),
                "coverage_threshold": float(coverage_threshold),
                "n_folds": int(n_folds),
                "chain_separator": str(chain_separator),
                "mmseqs_bin": str(mmseqs_bin),
                "cdhit_bin": str(cdhit_bin),
                "domain_labels_path": None if domain_labels_path is None else rel_or_abs(domain_labels_path, repo_root),
                "domain_column": str(domain_column),
                "description_column": str(description_column),
                "strict_domain_stratification": bool(strict_domain_stratification),
            },
            "stats": {
                "n_train": int(len(train_entries)),
                "n_pdb": int(len(pdb_entries)),
                "n_total": int(len(entries)),
                "n_clusters": int(len(reps)),
                "n_clusters_train": int(len({row["cluster_id"] for row in train_cluster_rows})),
                "fold_counts": fold_counts,
                "max_folds_per_cluster_train": int(max_folds_per_cluster),
                "domain_counts_train": domain_total_counts,
                "domain_fold_coverage_train": {key: int(value) for key, value in sorted(domain_fold_coverage.items())},
                "domain_fold_coverage_train_max_possible": {key: int(value) for key, value in sorted(domain_max_possible_coverage.items())},
                "domain_counts_by_fold_train": fold_domain_counts,
            },
            "sha256": {
                "clusters.parquet": sha256_file(clusters_path),
                "train_folds.parquet": sha256_file(train_folds_path),
            },
        },
    )
    return HomologyFoldsResult(clusters_path=clusters_path, train_folds_path=train_folds_path, manifest_path=manifest_path)
