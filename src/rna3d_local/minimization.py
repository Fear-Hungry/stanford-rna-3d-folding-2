from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import require_columns
from .errors import raise_error
from .io_tables import read_table, write_table
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class MinimizeEnsembleResult:
    predictions_path: Path
    manifest_path: Path


_OPENMM_LONG_SKIP_LEN = 350
_FOUNDATION_SOURCE_TOKENS = ("chai", "boltz", "rnapro", "foundation")


def _is_foundation_source(source_name: str) -> bool:
    name = str(source_name).strip().lower()
    return any(token in name for token in _FOUNDATION_SOURCE_TOKENS)


def _build_covalent_pairs(resids: list[int], chain_index: list[int] | None = None) -> list[tuple[int, int]]:
    if chain_index is not None and int(len(chain_index)) != int(len(resids)):
        raise ValueError(f"chain_index_len={len(chain_index)} resid_len={len(resids)}")
    pairs: list[tuple[int, int]] = []
    for idx in range(len(resids) - 1):
        same_chain = True
        if chain_index is not None:
            same_chain = int(chain_index[idx + 1]) == int(chain_index[idx])
        if same_chain and int(resids[idx + 1]) - int(resids[idx]) == 1:
            pairs.append((idx, idx + 1))
    return pairs


def _minimize_openmm(
    *,
    coords_angstrom: np.ndarray,
    residue_index: np.ndarray,
    chain_index: np.ndarray | None,
    max_iterations: int,
    bond_length_angstrom: float,
    bond_force_k: float,
    angle_force_k: float,
    angle_target_deg: float,
    vdw_min_distance_angstrom: float,
    vdw_epsilon: float,
    position_restraint_k: float,
    openmm_platform: str | None,
    stage: str,
    location: str,
) -> np.ndarray:
    try:
        import openmm as mm  # type: ignore
        from openmm import unit  # type: ignore
    except Exception:
        try:
            from simtk import openmm as mm  # type: ignore
            from simtk import unit  # type: ignore
        except Exception as exc:
            raise_error(stage, location, "backend=openmm indisponivel (dependencia ausente)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])

    system = mm.System()
    count = int(coords_angstrom.shape[0])
    for _ in range(count):
        system.addParticle(12.0)

    try:
        covalent_pairs = _build_covalent_pairs(
            [int(item) for item in residue_index.tolist()],
            None if chain_index is None else [int(item) for item in chain_index.tolist()],
        )
    except ValueError as exc:
        raise_error(
            stage,
            location,
            "chain_index/residue_index com comprimento inconsistente na minimizacao",
            impact="1",
            examples=[str(exc)],
        )
    bond_force = mm.HarmonicBondForce()
    for i, j in covalent_pairs:
        bond_force.addBond(int(i), int(j), float(bond_length_angstrom) * 0.1, float(bond_force_k))
    system.addForce(bond_force)

    if len(covalent_pairs) >= 2:
        angle_force = mm.HarmonicAngleForce()
        angle_target = float(np.deg2rad(float(angle_target_deg)))
        for idx in range(len(covalent_pairs) - 1):
            i0, i1 = covalent_pairs[idx]
            j0, j1 = covalent_pairs[idx + 1]
            if i1 != j0:
                continue
            angle_force.addAngle(int(i0), int(i1), int(j1), angle_target, float(angle_force_k))
        system.addForce(angle_force)

    repulsion = mm.CustomNonbondedForce("epsilon*step(sigma-r)*((sigma/r)^12 - 1)")
    repulsion.addGlobalParameter("epsilon", float(vdw_epsilon))
    repulsion.addGlobalParameter("sigma", float(vdw_min_distance_angstrom) * 0.1)
    repulsion.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    repulsion.setCutoffDistance(float(vdw_min_distance_angstrom) * 0.1)
    for _ in range(count):
        repulsion.addParticle([])
    for i, j in covalent_pairs:
        repulsion.addExclusion(int(i), int(j))
    system.addForce(repulsion)

    coords_nm = np.asarray(coords_angstrom, dtype=np.float64) * 0.1
    restraint_force = mm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint_force.addGlobalParameter("k", float(position_restraint_k))
    restraint_force.addPerParticleParameter("x0")
    restraint_force.addPerParticleParameter("y0")
    restraint_force.addPerParticleParameter("z0")
    for atom_idx in range(count):
        restraint_force.addParticle(
            int(atom_idx),
            [float(coords_nm[atom_idx, 0]), float(coords_nm[atom_idx, 1]), float(coords_nm[atom_idx, 2])],
        )
    system.addForce(restraint_force)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    if openmm_platform is None or str(openmm_platform).strip() == "":
        context = mm.Context(system, integrator)
    else:
        try:
            platform = mm.Platform.getPlatformByName(str(openmm_platform))
        except Exception as exc:
            raise_error(stage, location, "openmm_platform invalido", impact="1", examples=[f"{openmm_platform}:{type(exc).__name__}"])
        context = mm.Context(system, integrator, platform)
    context.setPositions(coords_nm * unit.nanometer)
    try:
        mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=int(max_iterations))
    except Exception as exc:
        raise_error(stage, location, "falha na minimizacao openmm", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    try:
        state = context.getState(getPositions=True)
        out_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    finally:
        del context
        del integrator
        del system
        import gc
        gc.collect()
    return np.asarray(out_nm, dtype=np.float64) * 10.0


def _minimize_pyrosetta(
    *,
    coords_angstrom: np.ndarray,
    stage: str,
    location: str,
) -> np.ndarray:
    try:
        import pyrosetta  # type: ignore  # noqa: F401
    except Exception as exc:
        raise_error(stage, location, "backend=pyrosetta indisponivel (dependencia ausente)", impact="1", examples=[f"{type(exc).__name__}:{exc}"])
    raise_error(
        stage,
        location,
        "backend=pyrosetta requer entrada full-atom RNA (input atual C1' only)",
        impact="1",
        examples=["forneca PDB full-atom para usar rna_minimize"],
    )
    return coords_angstrom


def minimize_ensemble(
    *,
    repo_root: Path,
    predictions_path: Path,
    out_path: Path,
    backend: str,
    max_iterations: int,
    bond_length_angstrom: float,
    bond_force_k: float,
    angle_force_k: float,
    angle_target_deg: float,
    vdw_min_distance_angstrom: float,
    vdw_epsilon: float,
    position_restraint_k: float,
    openmm_platform: str | None,
) -> MinimizeEnsembleResult:
    stage = "MINIMIZE_ENSEMBLE"
    location = "src/rna3d_local/minimization.py:minimize_ensemble"
    backend_name = str(backend).strip().lower()
    max_iterations_int = int(max_iterations)
    minimization_enabled = bool(max_iterations_int > 0)
    if backend_name not in {"openmm", "pyrosetta"}:
        raise_error(stage, location, "backend de minimizacao invalido", impact="1", examples=[str(backend)])
    if max_iterations_int < 0:
        raise_error(stage, location, "max_iterations deve ser >= 0", impact="1", examples=[str(max_iterations)])
    if max_iterations_int > 100:
        raise_error(stage, location, "max_iterations deve ser <= 100 para relaxacao curta", impact="1", examples=[str(max_iterations)])
    if float(bond_length_angstrom) <= 0.0:
        raise_error(stage, location, "bond_length_angstrom invalido", impact="1", examples=[str(bond_length_angstrom)])
    if float(bond_force_k) <= 0.0:
        raise_error(stage, location, "bond_force_k invalido", impact="1", examples=[str(bond_force_k)])
    if float(angle_force_k) <= 0.0:
        raise_error(stage, location, "angle_force_k invalido", impact="1", examples=[str(angle_force_k)])
    if float(angle_target_deg) <= 0.0 or float(angle_target_deg) >= 180.0:
        raise_error(stage, location, "angle_target_deg deve estar em (0,180)", impact="1", examples=[str(angle_target_deg)])
    if float(vdw_min_distance_angstrom) <= 0.0:
        raise_error(stage, location, "vdw_min_distance_angstrom invalido", impact="1", examples=[str(vdw_min_distance_angstrom)])
    if float(vdw_epsilon) <= 0.0:
        raise_error(stage, location, "vdw_epsilon invalido", impact="1", examples=[str(vdw_epsilon)])
    if float(position_restraint_k) <= 0.0:
        raise_error(stage, location, "position_restraint_k invalido", impact="1", examples=[str(position_restraint_k)])

    pred = read_table(predictions_path, stage=stage, location=location)
    require_columns(pred, ["target_id", "model_id", "resid", "resname", "x", "y", "z"], stage=stage, location=location, label="predictions_long")
    dup = pred.group_by(["target_id", "model_id", "resid"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dup.height > 0:
        examples = (
            dup.with_columns((pl.col("target_id") + pl.lit(":") + pl.col("model_id").cast(pl.Utf8) + pl.lit(":") + pl.col("resid").cast(pl.Utf8)).alias("k"))
            .get_column("k")
            .head(8)
            .to_list()
        )
        raise_error(stage, location, "predictions long com chave duplicada", impact=str(int(dup.height)), examples=[str(x) for x in examples])

    out_parts: list[pl.DataFrame] = []
    max_shift = 0.0
    mean_shift_sum = 0.0
    mean_shift_count = 0
    skipped_openmm_long = 0
    skipped_foundation_sources = 0
    for group_key, group_df in pred.group_by(["target_id", "model_id"], maintain_order=True):
        target_id = str(group_key[0]) if isinstance(group_key, tuple) else str(group_key)
        model_id = int(group_key[1]) if isinstance(group_key, tuple) else 0
        sort_col = "resid"
        if "residue_index_1d" in group_df.columns:
            residue_index_1d = group_df.get_column("residue_index_1d")
            if int(residue_index_1d.null_count()) == 0:
                sort_col = "residue_index_1d"
        part = group_df.sort(sort_col)
        coords = part.select(pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64), pl.col("z").cast(pl.Float64)).to_numpy()
        if not np.isfinite(coords).all():
            raise_error(stage, location, "coordenadas invalidas antes da minimizacao", impact="1", examples=[f"{target_id}:{model_id}"])
        residue_index = part.get_column(sort_col).cast(pl.Int64).to_numpy()
        chain_index: np.ndarray | None = None
        if "chain_index" in part.columns:
            chain_col = part.get_column("chain_index")
            if int(chain_col.null_count()) == 0:
                chain_index = chain_col.cast(pl.Int64).to_numpy()
        source_name = "unknown"
        if "source" in part.columns:
            source_values = part.get_column("source").drop_nulls().head(1).to_list()
            if source_values:
                source_name = str(source_values[0])
        sequence_length = int(part.height)
        refinement_steps = int(max_iterations_int)
        refinement_skip_reason: str | None = None
        if minimization_enabled and _is_foundation_source(source_name):
            minimized = np.asarray(coords, dtype=np.float64)
            skipped_foundation_sources += 1
            refinement_steps = 0
            refinement_skip_reason = "foundation_source"
            print(
                f"[{stage}] [{location}] fonte foundation: minimizacao pulada para preservar geometria original | "
                f"impacto=1 | exemplos={target_id}:{model_id}:source={source_name}",
                file=sys.stderr,
            )
        elif minimization_enabled and backend_name == "openmm" and sequence_length > _OPENMM_LONG_SKIP_LEN:
            minimized = np.asarray(coords, dtype=np.float64)
            skipped_openmm_long += 1
            refinement_steps = 0
            refinement_skip_reason = "openmm_long_skip"
            print(
                f"[{stage}] [{location}] alvo longo: OpenMM pulado para evitar timeout/OOM | impacto=1 | exemplos={target_id}:{model_id}:len={sequence_length}",
                file=sys.stderr,
            )
        elif minimization_enabled and backend_name == "openmm":
            minimized = _minimize_openmm(
                coords_angstrom=np.asarray(coords, dtype=np.float64),
                residue_index=np.asarray(residue_index, dtype=np.int64),
                chain_index=(None if chain_index is None else np.asarray(chain_index, dtype=np.int64)),
                max_iterations=max_iterations_int,
                bond_length_angstrom=float(bond_length_angstrom),
                bond_force_k=float(bond_force_k),
                angle_force_k=float(angle_force_k),
                angle_target_deg=float(angle_target_deg),
                vdw_min_distance_angstrom=float(vdw_min_distance_angstrom),
                vdw_epsilon=float(vdw_epsilon),
                position_restraint_k=float(position_restraint_k),
                openmm_platform=openmm_platform,
                stage=stage,
                location=location,
            )
        elif minimization_enabled:
            minimized = _minimize_pyrosetta(coords_angstrom=np.asarray(coords, dtype=np.float64), stage=stage, location=location)
        else:
            minimized = np.asarray(coords, dtype=np.float64)
            refinement_steps = 0
            refinement_skip_reason = "max_iterations_zero"
        if minimized.shape != coords.shape:
            raise_error(stage, location, "backend de minimizacao retornou shape invalido", impact="1", examples=[f"{target_id}:{model_id}:{minimized.shape}!={coords.shape}"])
        if not np.isfinite(minimized).all():
            raise_error(stage, location, "backend de minimizacao retornou coordenadas nao-finitas", impact="1", examples=[f"{target_id}:{model_id}"])
        shift = np.sqrt(np.sum(np.square(minimized - coords), axis=1))
        max_shift = max(max_shift, float(np.max(shift)) if shift.size > 0 else 0.0)
        mean_shift_sum += float(np.sum(shift))
        mean_shift_count += int(shift.size)
        out_part = part.with_columns(
            [
                pl.Series("x", minimized[:, 0]),
                pl.Series("y", minimized[:, 1]),
                pl.Series("z", minimized[:, 2]),
                pl.lit(str(backend_name)).alias("refinement_backend"),
                pl.lit(int(refinement_steps)).alias("refinement_steps"),
                pl.lit(float(position_restraint_k)).alias("refinement_position_restraint_k"),
                pl.lit(bool(refinement_skip_reason is not None)).alias("refinement_skipped"),
                pl.lit(None if refinement_skip_reason is None else str(refinement_skip_reason)).alias("refinement_skip_reason"),
            ]
        )
        out_parts.append(out_part)

    if not out_parts:
        raise_error(stage, location, "nenhuma estrutura encontrada para minimizacao", impact="0", examples=[])
    out = pl.concat(out_parts, how="vertical").sort(["target_id", "model_id", "resid"])
    write_table(out, out_path)
    manifest_path = out_path.parent / "minimize_ensemble_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": utc_now_iso(),
            "backend": backend_name,
            "paths": {
                "predictions_in": rel_or_abs(predictions_path, repo_root),
                "predictions_out": rel_or_abs(out_path, repo_root),
            },
            "params": {
                "max_iterations": max_iterations_int,
                "minimization_enabled": minimization_enabled,
                "bond_length_angstrom": float(bond_length_angstrom),
                "bond_force_k": float(bond_force_k),
                "angle_force_k": float(angle_force_k),
                "angle_target_deg": float(angle_target_deg),
                "vdw_min_distance_angstrom": float(vdw_min_distance_angstrom),
                "vdw_epsilon": float(vdw_epsilon),
                "position_restraint_k": float(position_restraint_k),
                "openmm_platform": None if openmm_platform is None else str(openmm_platform),
            },
            "stats": {
                "n_rows": int(out.height),
                "n_targets": int(out.get_column("target_id").n_unique()),
                "n_target_models": int(out.select("target_id", "model_id").unique().height),
                "shift_max_angstrom": float(max_shift),
                "shift_mean_angstrom": float(mean_shift_sum / float(max(mean_shift_count, 1))),
                "n_target_models_openmm_skipped_long": int(skipped_openmm_long),
                "n_target_models_skipped_foundation_source": int(skipped_foundation_sources),
            },
            "sha256": {"predictions.parquet": sha256_file(out_path)},
        },
    )
    return MinimizeEnsembleResult(predictions_path=out_path, manifest_path=manifest_path)
