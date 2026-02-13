from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from Bio.PDB import PDBParser

from .bigdata import (
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    TableReadConfig,
    assert_memory_budget,
    assert_row_budget,
    collect_streaming,
    scan_table,
)
from .errors import raise_error
from .utils import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _validate_target_rows(*, targets: pl.DataFrame, location: str) -> list[tuple[str, str]]:
    if targets.height == 0:
        raise_error("DRFOLD2", location, "targets vazio", impact="0", examples=[])
    dup = (
        targets.group_by("target_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select("target_id")
        .head(8)
    )
    if dup.height > 0:
        raise_error(
            "DRFOLD2",
            location,
            "targets com target_id duplicado",
            impact=str(int(dup.height)),
            examples=[str(v) for v in dup.get_column("target_id").to_list()],
        )
    rows: list[tuple[str, str]] = []
    bad_count = 0
    bad_examples: list[str] = []
    for tid, seq in targets.iter_rows():
        t = str(tid)
        s = str(seq or "").strip().upper()
        if not t or not s:
            bad_count += 1
            if len(bad_examples) < 8:
                bad_examples.append(f"{t}:{s}")
            continue
        rows.append((t, s))
    if bad_count > 0:
        raise_error("DRFOLD2", location, "targets com campos vazios", impact=str(bad_count), examples=bad_examples)
    return rows


def extract_target_coordinates_from_pdb(*, pdb_path: Path, target_sequence: str, location: str) -> list[tuple[float, float, float]]:
    if not pdb_path.exists():
        raise_error("DRFOLD2", location, "arquivo PDB ausente", impact="1", examples=[str(pdb_path)])
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("drfold2", str(pdb_path))
    except Exception as e:  # noqa: BLE001
        raise_error("DRFOLD2", location, "falha ao parsear PDB", impact="1", examples=[f"{type(e).__name__}:{e}", str(pdb_path)])
    models = list(structure.get_models())
    if not models:
        raise_error("DRFOLD2", location, "PDB sem modelo", impact="1", examples=[str(pdb_path)])
    model = models[0]
    residues = []
    for chain in model.get_chains():
        for residue in chain.get_residues():
            hetflag, _resseq, _icode = residue.id
            if str(hetflag).strip():
                continue
            residues.append(residue)
    if len(residues) != len(target_sequence):
        raise_error(
            "DRFOLD2",
            location,
            "PDB com numero de residuos divergente da sequencia alvo",
            impact=f"expected={len(target_sequence)} got={len(residues)}",
            examples=[str(pdb_path)],
        )
    coords: list[tuple[float, float, float]] = []
    for r in residues:
        atom = None
        for name in ("C1'", "C4'", "P", "O3'"):
            if r.has_id(name):
                atom = r[name]
                break
        if atom is None:
            atom_list = list(r.get_atoms())
            if not atom_list:
                raise_error("DRFOLD2", location, "residuo sem atomos no PDB", impact="1", examples=[str(pdb_path), str(r.id)])
            atom = atom_list[0]
        xyz = atom.get_coord()
        coords.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return coords


def _required_model_paths(*, relax_dir: Path, n_models: int) -> list[Path]:
    return [relax_dir / f"model_{i}.pdb" for i in range(1, n_models + 1)]


def _ensure_drfold2_ready(*, drfold2_root: Path, n_models: int, location: str) -> Path:
    if not drfold2_root.exists():
        raise_error("DRFOLD2", location, "drfold2_root nao encontrado", impact="1", examples=[str(drfold2_root)])
    infer_script = drfold2_root / "DRfold_infer.py"
    if not infer_script.exists():
        raise_error("DRFOLD2", location, "DRfold_infer.py ausente", impact="1", examples=[str(infer_script)])
    model_hub = drfold2_root / "model_hub"
    if not model_hub.exists():
        raise_error("DRFOLD2", location, "model_hub ausente em DRfold2", impact="1", examples=[str(model_hub)])
    arena_bin = drfold2_root / "Arena" / "Arena"
    if not arena_bin.exists():
        raise_error("DRFOLD2", location, "binario Arena ausente em DRfold2", impact="1", examples=[str(arena_bin)])
    if n_models > 5:
        raise_error("DRFOLD2", location, "n_models invalido para DRfold2 (max=5)", impact="1", examples=[str(n_models)])
    return infer_script


def predict_drfold2(
    *,
    repo_root: Path,
    drfold2_root: Path,
    target_sequences_path: Path,
    out_path: Path,
    work_dir: Path,
    n_models: int = 5,
    python_bin: str = "python",
    target_limit: int | None = None,
    chunk_size: int = 200_000,
    reuse_existing_targets: bool = False,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    max_rows_in_memory: int = DEFAULT_MAX_ROWS_IN_MEMORY,
) -> tuple[Path, Path]:
    location = "src/rna3d_local/drfold2.py:predict_drfold2"
    assert_memory_budget(stage="DRFOLD2", location=location, budget_mb=memory_budget_mb)
    if n_models <= 0:
        raise_error("DRFOLD2", location, "n_models invalido (deve ser > 0)", impact="1", examples=[str(n_models)])
    if chunk_size <= 0:
        raise_error("DRFOLD2", location, "chunk_size invalido (deve ser > 0)", impact="1", examples=[str(chunk_size)])
    if target_limit is not None and int(target_limit) <= 0:
        raise_error("DRFOLD2", location, "target_limit invalido (deve ser > 0)", impact="1", examples=[str(target_limit)])

    infer_script = _ensure_drfold2_ready(drfold2_root=drfold2_root, n_models=int(n_models), location=location)

    targets_df = collect_streaming(
        lf=scan_table(
            config=TableReadConfig(
                path=target_sequences_path,
                stage="DRFOLD2",
                location=location,
                columns=("target_id", "sequence"),
            )
        ).select(pl.col("target_id").cast(pl.Utf8), pl.col("sequence").cast(pl.Utf8)),
        stage="DRFOLD2",
        location=location,
    ).sort("target_id")
    targets = _validate_target_rows(targets=targets_df, location=location)
    if target_limit is not None:
        targets = targets[: int(target_limit)]
    assert_row_budget(
        stage="DRFOLD2",
        location=location,
        rows=int(len(targets)),
        max_rows_in_memory=max_rows_in_memory,
        label="targets",
    )

    if work_dir.exists() and not reuse_existing_targets:
        existing = list(work_dir.iterdir())
        if existing:
            raise_error(
                "DRFOLD2",
                location,
                "work_dir ja contem arquivos; use um diretorio novo ou --reuse-existing-targets",
                impact=str(len(existing)),
                examples=[str(existing[0])],
            )
    work_dir.mkdir(parents=True, exist_ok=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out_path.exists():
        tmp_out_path.unlink()
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    rows_written = 0
    finalized = False
    per_target_stats: list[dict] = []

    def _flush() -> None:
        nonlocal writer, rows_written
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(str(tmp_out_path), table.schema, compression="zstd")
        writer.write_table(table)
        rows_written += len(buffer)
        buffer.clear()

    try:
        for tid, seq in targets:
            t0 = time.time()
            target_dir = work_dir / tid
            target_dir.mkdir(parents=True, exist_ok=True)
            fasta_path = target_dir / "target.fasta"
            fasta_path.write_text(f">{tid}\n{seq}\n", encoding="utf-8")
            relax_dir = target_dir / "relax"
            required_pdbs = _required_model_paths(relax_dir=relax_dir, n_models=int(n_models))
            if not (reuse_existing_targets and all(p.exists() for p in required_pdbs)):
                cmd = [python_bin, str(infer_script), str(fasta_path), str(target_dir)]
                if int(n_models) > 1:
                    cmd.append("1")
                proc = subprocess.run(
                    cmd,
                    cwd=str(drfold2_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip().splitlines()[-8:]
                    raise_error(
                        "DRFOLD2",
                        location,
                        "falha ao executar DRfold2",
                        impact=str(proc.returncode),
                        examples=tail if tail else [str(cmd)],
                    )
            missing = [str(p) for p in required_pdbs if not p.exists()]
            if missing:
                raise_error(
                    "DRFOLD2",
                    location,
                    "saida DRfold2 incompleta (model_i.pdb ausente)",
                    impact=str(len(missing)),
                    examples=missing[:8],
                )

            for model_id, pdb_path in enumerate(required_pdbs, start=1):
                coords = extract_target_coordinates_from_pdb(
                    pdb_path=pdb_path,
                    target_sequence=seq,
                    location=location,
                )
                if len(coords) != len(seq):
                    raise_error(
                        "DRFOLD2",
                        location,
                        "coordenadas DRfold2 com comprimento invalido",
                        impact=f"target={tid} expected={len(seq)} got={len(coords)}",
                        examples=[str(pdb_path)],
                    )
                for resid, (base, xyz) in enumerate(zip(seq, coords, strict=True), start=1):
                    buffer.append(
                        {
                            "branch": "drfold2",
                            "target_id": tid,
                            "ID": f"{tid}_{resid}",
                            "resid": resid,
                            "resname": base,
                            "model_id": int(model_id),
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "template_uid": f"drfold2:{tid}:model_{model_id}",
                            "similarity": 1.0,
                            "coverage": 1.0,
                        }
                    )
                if len(buffer) >= int(chunk_size):
                    _flush()
                    assert_memory_budget(stage="DRFOLD2", location=location, budget_mb=memory_budget_mb)

            per_target_stats.append(
                {
                    "target_id": tid,
                    "sequence_len": len(seq),
                    "n_models": int(n_models),
                    "elapsed_sec": round(float(time.time() - t0), 3),
                }
            )
            assert_memory_budget(stage="DRFOLD2", location=location, budget_mb=memory_budget_mb)

        _flush()
        if writer is not None:
            writer.close()
            writer = None
        assert_memory_budget(stage="DRFOLD2", location=location, budget_mb=memory_budget_mb)

        if rows_written == 0:
            raise_error("DRFOLD2", location, "nenhuma predicao DRfold2 gerada", impact="0", examples=[])
        if out_path.exists():
            out_path.unlink()
        tmp_out_path.replace(out_path)
        finalized = True
    finally:
        if writer is not None:
            writer.close()
        if not finalized and tmp_out_path.exists():
            tmp_out_path.unlink()

    manifest = {
        "created_utc": _utc_now(),
        "paths": {
            "drfold2_root": _rel(drfold2_root, repo_root),
            "target_sequences": _rel(target_sequences_path, repo_root),
            "work_dir": _rel(work_dir, repo_root),
            "predictions_long": _rel(out_path, repo_root),
        },
        "params": {
            "n_models": int(n_models),
            "python_bin": str(python_bin),
            "target_limit": None if target_limit is None else int(target_limit),
            "chunk_size": int(chunk_size),
            "reuse_existing_targets": bool(reuse_existing_targets),
        },
        "stats": {
            "n_targets": int(len(targets)),
            "n_rows": int(rows_written),
            "per_target": per_target_stats,
        },
        "sha256": {"predictions_long.parquet": sha256_file(out_path)},
    }
    manifest_path = out_path.parent / "drfold2_predict_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path, manifest_path
