from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..errors import raise_error
from ..se3.graph_builder import TargetGraph
from ..utils import rel_or_abs, utc_now_iso, write_json

_RESNAME_TO_ID = {"A": 0, "C": 1, "G": 2, "U": 3}
_ID_TO_RESNAME = {value: key for key, value in _RESNAME_TO_ID.items()}


def _require_zarr(*, stage: str, location: str):
    try:
        import zarr  # type: ignore
    except Exception:
        raise_error(
            stage,
            location,
            "dependencia zarr ausente para store lazy",
            impact="1",
            examples=["pip install zarr"],
        )
    return zarr


def ensure_zarr_available(*, stage: str, location: str) -> None:
    _require_zarr(stage=stage, location=location)


def _chunk_2d(length: int, width: int) -> tuple[int, int]:
    return (max(1, min(int(length), 1024)), int(width))


def _chunk_1d(length: int) -> tuple[int]:
    return (max(1, min(int(length), 4096)),)


def _create_dataset(group: Any, *, name: str, data: np.ndarray, chunks: tuple[int, ...]) -> None:
    try:
        group.create_dataset(name, data=data, chunks=chunks, overwrite=True)
    except TypeError:
        group.create_dataset(
            name,
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
            data=data,
            overwrite=True,
        )


@dataclass(frozen=True)
class ZarrTrainingStoreBuildResult:
    store_path: Path
    manifest_path: Path
    n_targets: int
    n_residues_total: int
    input_dim: int
    max_target_length: int


def build_training_store_zarr(
    *,
    repo_root: Path,
    graphs: list[TargetGraph],
    store_path: Path,
    manifest_path: Path,
    stage: str,
    location: str,
) -> ZarrTrainingStoreBuildResult:
    zarr = _require_zarr(stage=stage, location=location)
    if not graphs:
        raise_error(stage, location, "nenhum grafo disponivel para build do store zarr", impact="0", examples=[])
    input_dim = int(graphs[0].node_features.shape[1])
    if input_dim <= 0:
        raise_error(stage, location, "input_dim invalido no build do store zarr", impact="1", examples=[str(input_dim)])
    for graph in graphs:
        if int(graph.node_features.shape[1]) != input_dim:
            raise_error(
                stage,
                location,
                "input_dim inconsistente entre targets no store zarr",
                impact="1",
                examples=[graph.target_id],
            )
        if graph.coords_true is None:
            raise_error(stage, location, "coords_true ausente no grafo de treino para store zarr", impact="1", examples=[graph.target_id])
    store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(store_path), mode="w")
    targets_group = root.create_group("targets")

    target_entries: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    n_residues_total = 0
    max_target_length = 0
    for index, graph in enumerate(graphs):
        length = int(graph.node_features.shape[0])
        max_target_length = max(max_target_length, length)
        n_residues_total += length
        source = str(graph.chem_source)
        source_counts[source] = int(source_counts.get(source, 0) + 1)
        subgroup = targets_group.create_group(str(index))
        node_features = graph.node_features.detach().cpu().numpy().astype(np.float32, copy=False)
        coords_init = graph.coords_init.detach().cpu().numpy().astype(np.float32, copy=False)
        coords_true = graph.coords_true.detach().cpu().numpy().astype(np.float32, copy=False) if graph.coords_true is not None else None
        residue_index = graph.residue_index.detach().cpu().numpy().astype(np.int32, copy=False)
        chain_index = graph.chain_index.detach().cpu().numpy().astype(np.int16, copy=False)
        chem_exposure = graph.chem_exposure.detach().cpu().numpy().astype(np.float32, copy=False)
        bpp_pair_src = graph.bpp_pair_src.detach().cpu().numpy().astype(np.int32, copy=False)
        bpp_pair_dst = graph.bpp_pair_dst.detach().cpu().numpy().astype(np.int32, copy=False)
        bpp_pair_prob = graph.bpp_pair_prob.detach().cpu().numpy().astype(np.float32, copy=False)
        msa_pair_src = graph.msa_pair_src.detach().cpu().numpy().astype(np.int32, copy=False)
        msa_pair_dst = graph.msa_pair_dst.detach().cpu().numpy().astype(np.int32, copy=False)
        msa_pair_prob = graph.msa_pair_prob.detach().cpu().numpy().astype(np.float32, copy=False)
        resids = np.asarray(graph.resids, dtype=np.int32)
        resname_ids = np.asarray([_RESNAME_TO_ID[item] for item in graph.resnames], dtype=np.int8)
        if coords_true is None:
            raise_error(stage, location, "coords_true ausente no store zarr", impact="1", examples=[graph.target_id])

        _create_dataset(subgroup, name="node_features", data=node_features, chunks=_chunk_2d(length, input_dim))
        _create_dataset(subgroup, name="coords_init", data=coords_init, chunks=_chunk_2d(length, 3))
        _create_dataset(subgroup, name="coords_true", data=coords_true, chunks=_chunk_2d(length, 3))
        _create_dataset(subgroup, name="residue_index", data=residue_index, chunks=_chunk_1d(length))
        _create_dataset(subgroup, name="chain_index", data=chain_index, chunks=_chunk_1d(length))
        _create_dataset(subgroup, name="chem_exposure", data=chem_exposure, chunks=_chunk_1d(length))
        _create_dataset(subgroup, name="bpp_pair_src", data=bpp_pair_src, chunks=_chunk_1d(int(max(1, bpp_pair_src.shape[0]))))
        _create_dataset(subgroup, name="bpp_pair_dst", data=bpp_pair_dst, chunks=_chunk_1d(int(max(1, bpp_pair_dst.shape[0]))))
        _create_dataset(subgroup, name="bpp_pair_prob", data=bpp_pair_prob, chunks=_chunk_1d(int(max(1, bpp_pair_prob.shape[0]))))
        _create_dataset(subgroup, name="msa_pair_src", data=msa_pair_src, chunks=_chunk_1d(int(max(1, msa_pair_src.shape[0]))))
        _create_dataset(subgroup, name="msa_pair_dst", data=msa_pair_dst, chunks=_chunk_1d(int(max(1, msa_pair_dst.shape[0]))))
        _create_dataset(subgroup, name="msa_pair_prob", data=msa_pair_prob, chunks=_chunk_1d(int(max(1, msa_pair_prob.shape[0]))))
        _create_dataset(subgroup, name="resids", data=resids, chunks=_chunk_1d(length))
        _create_dataset(subgroup, name="resname_ids", data=resname_ids, chunks=_chunk_1d(length))

        target_entries.append(
            {
                "index": int(index),
                "target_id": str(graph.target_id),
                "length": int(length),
                "chem_source": source,
            }
        )

    root.attrs["schema_version"] = "se3_training_store_v1"
    root.attrs["n_targets"] = int(len(graphs))
    root.attrs["input_dim"] = int(input_dim)

    payload = {
        "created_utc": utc_now_iso(),
        "schema_version": "se3_training_store_v1",
        "store_format": "zarr",
        "paths": {
            "store": rel_or_abs(store_path, repo_root),
            "manifest": rel_or_abs(manifest_path, repo_root),
        },
        "stats": {
            "n_targets": int(len(graphs)),
            "n_residues_total": int(n_residues_total),
            "input_dim": int(input_dim),
            "max_target_length": int(max_target_length),
            "chemical_mapping_source_counts": source_counts,
        },
        "targets": target_entries,
    }
    write_json(manifest_path, payload)
    return ZarrTrainingStoreBuildResult(
        store_path=store_path,
        manifest_path=manifest_path,
        n_targets=int(len(graphs)),
        n_residues_total=int(n_residues_total),
        input_dim=int(input_dim),
        max_target_length=int(max_target_length),
    )


class ZarrTrainingStore:
    def __init__(
        self,
        *,
        store_path: Path,
        manifest_path: Path,
        stage: str,
        location: str,
    ) -> None:
        zarr = _require_zarr(stage=stage, location=location)
        if not store_path.exists():
            raise_error(stage, location, "store zarr de treino ausente", impact="1", examples=[str(store_path)])
        if not manifest_path.exists():
            raise_error(stage, location, "manifest do store zarr ausente", impact="1", examples=[str(manifest_path)])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        targets = payload.get("targets")
        if not isinstance(targets, list) or not targets:
            raise_error(stage, location, "manifest do store zarr sem targets validos", impact="1", examples=[str(manifest_path)])
        stats = payload.get("stats", {})
        input_dim = int(stats.get("input_dim", 0))
        if input_dim <= 0:
            raise_error(stage, location, "manifest do store zarr sem input_dim valido", impact="1", examples=[str(input_dim)])
        self._root = zarr.open_group(str(store_path), mode="r")
        self._target_entries = targets
        self._input_dim = input_dim
        self._store_path = store_path
        self._manifest_path = manifest_path
        self._stage = stage
        self._location = location

    @property
    def input_dim(self) -> int:
        return int(self._input_dim)

    @property
    def n_targets(self) -> int:
        return int(len(self._target_entries))

    @property
    def n_residues_total(self) -> int:
        return int(sum(int(item.get("length", 0)) for item in self._target_entries))

    @property
    def max_target_length(self) -> int:
        return int(max(int(item.get("length", 0)) for item in self._target_entries))

    @property
    def chemical_source_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self._target_entries:
            source = str(item.get("chem_source", "unknown"))
            out[source] = int(out.get(source, 0) + 1)
        return out

    def __len__(self) -> int:
        return int(len(self._target_entries))

    def load_graph(self, index: int) -> TargetGraph:
        if index < 0 or index >= len(self._target_entries):
            raise_error(self._stage, self._location, "indice fora do intervalo no store zarr", impact="1", examples=[str(index)])
        entry = self._target_entries[index]
        group_key = str(entry.get("index", index))
        group_path = f"targets/{group_key}"
        if group_path not in self._root:
            raise_error(self._stage, self._location, "grupo de target ausente no store zarr", impact="1", examples=[group_path])
        group = self._root[group_path]

        required = [
            "node_features",
            "coords_init",
            "coords_true",
            "residue_index",
            "chain_index",
            "chem_exposure",
            "bpp_pair_src",
            "bpp_pair_dst",
            "bpp_pair_prob",
            "msa_pair_src",
            "msa_pair_dst",
            "msa_pair_prob",
            "resids",
            "resname_ids",
        ]
        missing = [name for name in required if name not in group]
        if missing:
            raise_error(self._stage, self._location, "datasets ausentes no store zarr", impact=str(len(missing)), examples=missing[:8])

        resname_ids = np.asarray(group["resname_ids"][:], dtype=np.int8)
        try:
            resnames = [_ID_TO_RESNAME[int(item)] for item in resname_ids.tolist()]
        except Exception:
            raise_error(self._stage, self._location, "resname_ids invalido no store zarr", impact="1", examples=[group_path])

        return TargetGraph(
            target_id=str(entry.get("target_id")),
            resids=[int(item) for item in np.asarray(group["resids"][:], dtype=np.int32).tolist()],
            resnames=resnames,
            chain_index=torch.from_numpy(np.asarray(group["chain_index"][:], dtype=np.int16)).to(dtype=torch.long),
            residue_index=torch.from_numpy(np.asarray(group["residue_index"][:], dtype=np.int32)).to(dtype=torch.long),
            chem_exposure=torch.from_numpy(np.asarray(group["chem_exposure"][:], dtype=np.float32)).to(dtype=torch.float32),
            chem_source=str(entry.get("chem_source", "unknown")),
            node_features=torch.from_numpy(np.asarray(group["node_features"][:], dtype=np.float32)).to(dtype=torch.float32),
            bpp_pair_src=torch.from_numpy(np.asarray(group["bpp_pair_src"][:], dtype=np.int32)).to(dtype=torch.long),
            bpp_pair_dst=torch.from_numpy(np.asarray(group["bpp_pair_dst"][:], dtype=np.int32)).to(dtype=torch.long),
            bpp_pair_prob=torch.from_numpy(np.asarray(group["bpp_pair_prob"][:], dtype=np.float32)).to(dtype=torch.float32),
            msa_pair_src=torch.from_numpy(np.asarray(group["msa_pair_src"][:], dtype=np.int32)).to(dtype=torch.long),
            msa_pair_dst=torch.from_numpy(np.asarray(group["msa_pair_dst"][:], dtype=np.int32)).to(dtype=torch.long),
            msa_pair_prob=torch.from_numpy(np.asarray(group["msa_pair_prob"][:], dtype=np.float32)).to(dtype=torch.float32),
            coords_init=torch.from_numpy(np.asarray(group["coords_init"][:], dtype=np.float32)).to(dtype=torch.float32),
            coords_true=torch.from_numpy(np.asarray(group["coords_true"][:], dtype=np.float32)).to(dtype=torch.float32),
        )
