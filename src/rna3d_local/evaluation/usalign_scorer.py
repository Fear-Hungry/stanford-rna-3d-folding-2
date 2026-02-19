from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..errors import raise_error
from ..io_tables import read_table
from ..utils import utc_now_iso, write_json

_COORD_MODEL_IDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class LocalBestOf5ScoreResult:
    score: float
    n_targets: int
    score_json_path: Path
    report_path: Path


@dataclass(frozen=True)
class _TargetRows:
    target_id: str
    rows: list[dict[str, object]]


def _ensure_executable_binary(path: Path, *, stage: str, location: str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise_error(stage, location, f"binario {label} ausente", impact="1", examples=[str(resolved)])
    if not resolved.is_file():
        raise_error(stage, location, f"binario {label} invalido (nao e arquivo)", impact="1", examples=[str(resolved)])
    if os.access(resolved, os.X_OK):
        return resolved
    try:
        resolved.chmod(0o755)
    except Exception as exc:  # noqa: BLE001
        raise_error(
            stage,
            location,
            f"binario {label} sem permissao de execucao e falha ao aplicar chmod",
            impact="1",
            examples=[str(resolved), f"{type(exc).__name__}:{exc}"],
        )
    if not os.access(resolved, os.X_OK):
        raise_error(
            stage,
            location,
            f"binario {label} segue sem permissao de execucao apos chmod",
            impact="1",
            examples=[str(resolved)],
        )
    return resolved


class USalignBestOf5Scorer:
    def __init__(
        self,
        *,
        usalign_path: Path,
        timeout_seconds: int = 900,
        ground_truth_mode: str = "single",
        missing_coord_threshold: float = -1e17,
    ) -> None:
        self.usalign_path = Path(usalign_path).resolve()
        self.timeout_seconds = int(timeout_seconds)
        self.ground_truth_mode = str(ground_truth_mode).strip().lower()
        self.missing_coord_threshold = float(missing_coord_threshold)
        stage = "SCORE_LOCAL_BESTOF5"
        location = "src/rna3d_local/evaluation/usalign_scorer.py:USalignBestOf5Scorer.__init__"
        self.usalign_path = _ensure_executable_binary(
            self.usalign_path,
            stage=stage,
            location=location,
            label="USalign",
        )
        if self.timeout_seconds <= 0:
            raise_error(stage, location, "timeout_seconds invalido (<=0)", impact="1", examples=[str(self.timeout_seconds)])
        if self.ground_truth_mode not in {"single", "best_of_gt_copies"}:
            raise_error(stage, location, "ground_truth_mode invalido", impact="1", examples=[self.ground_truth_mode])
        if self.missing_coord_threshold >= 0:
            raise_error(stage, location, "missing_coord_threshold invalido (esperado negativo)", impact="1", examples=[str(self.missing_coord_threshold)])

    def score_submission(
        self,
        *,
        ground_truth_path: Path,
        submission_path: Path,
        score_json_path: Path,
        report_path: Path | None = None,
    ) -> LocalBestOf5ScoreResult:
        stage = "SCORE_LOCAL_BESTOF5"
        location = "src/rna3d_local/evaluation/usalign_scorer.py:USalignBestOf5Scorer.score_submission"
        ground_truth = read_table(ground_truth_path, stage=stage, location=location)
        submission = read_table(submission_path, stage=stage, location=location)
        gt_copy_ids: list[int] | None = None
        if self.ground_truth_mode == "best_of_gt_copies":
            gt_copy_ids = self._extract_gt_copy_ids(ground_truth, stage=stage, location=location)
        gt = self._canonicalize_ground_truth(ground_truth, gt_copy_ids=gt_copy_ids, stage=stage, location=location)
        sub = self._canonicalize_submission(submission, stage=stage, location=location)
        self._validate_global_contract(gt=gt, submission=sub, stage=stage, location=location)
        per_target_gt = self._group_by_target(gt, stage=stage, location=location)
        per_target_sub = self._group_by_target(sub, stage=stage, location=location)

        target_scores: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="rna3d_usalign_") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            for target_id, target in per_target_gt.items():
                if target_id not in per_target_sub:
                    raise_error(
                        stage,
                        location,
                        "target sem predicao correspondente na submissao",
                        impact="1",
                        examples=[target_id],
                    )
                pred_target = per_target_sub[target_id]
                best_tm, model_scores = self._score_target_bestof5(
                    ground_truth_target=target,
                    submission_target=pred_target,
                    gt_copy_ids=gt_copy_ids,
                    tmp_dir=tmp_dir,
                    stage=stage,
                    location=location,
                )
                target_scores.append(
                    {
                        "target_id": target.target_id,
                        "best_tm_score": float(best_tm),
                        "model_tm_scores": [float(item) for item in model_scores],
                    }
                )
        if not target_scores:
            raise_error(stage, location, "nenhum alvo encontrado para score local", impact="0", examples=[])
        score = float(sum(float(item["best_tm_score"]) for item in target_scores) / float(len(target_scores)))
        output_report_path = report_path if report_path is not None else score_json_path.with_name("local_bestof5_report.json")
        score_json_payload = {
            "created_utc": utc_now_iso(),
            "metric": "usalign_tm_best_of_5",
            "score": float(score),
            "n_targets": int(len(target_scores)),
            "score_source": "local_usalign",
        }
        report_payload = {
            "created_utc": utc_now_iso(),
            "metric": "usalign_tm_best_of_5",
            "score": float(score),
            "n_targets": int(len(target_scores)),
            "paths": {
                "ground_truth": str(ground_truth_path.resolve()),
                "submission": str(submission_path.resolve()),
                "usalign": str(self.usalign_path),
                "score_json": str(score_json_path.resolve()),
                "report": str(output_report_path.resolve()),
            },
            "targets": target_scores,
        }
        write_json(score_json_path, score_json_payload)
        write_json(output_report_path, report_payload)
        return LocalBestOf5ScoreResult(
            score=float(score),
            n_targets=int(len(target_scores)),
            score_json_path=score_json_path,
            report_path=output_report_path,
        )

    def _extract_gt_copy_ids(self, df: pl.DataFrame, *, stage: str, location: str) -> list[int]:
        copy_ids: set[int] = set()
        for column in df.columns:
            if "_" not in column:
                continue
            axis, suffix = column.split("_", 1)
            if axis not in {"x", "y", "z"}:
                continue
            if not suffix.isdigit():
                continue
            copy_ids.add(int(suffix))
        candidates = sorted(copy_ids)
        valid: list[int] = []
        for copy_id in candidates:
            if f"x_{copy_id}" in df.columns and f"y_{copy_id}" in df.columns and f"z_{copy_id}" in df.columns:
                valid.append(int(copy_id))
        if not valid:
            raise_error(stage, location, "tripletos de coordenadas x_i,y_i,z_i ausentes no ground_truth", impact="1", examples=df.columns[:8])
        return valid

    def _canonicalize_ground_truth(
        self,
        df: pl.DataFrame,
        *,
        gt_copy_ids: list[int] | None,
        stage: str,
        location: str,
    ) -> pl.DataFrame:
        required_from_id = ["ID"]
        required_from_target_resid = ["target_id", "resid"]
        has_id = all(column in df.columns for column in required_from_id)
        has_target_resid = all(column in df.columns for column in required_from_target_resid)
        if not has_id and not has_target_resid:
            raise_error(
                stage,
                location,
                "ground_truth sem chave obrigatoria (ID ou target_id+resid)",
                impact="2",
                examples=["ID", "target_id+resid"],
            )
        if has_id:
            out = df.with_columns(pl.col("ID").cast(pl.Utf8).alias("ID"))
        else:
            out = df.with_columns(
                pl.concat_str([pl.col("target_id").cast(pl.Utf8), pl.lit("_"), pl.col("resid").cast(pl.Int64).cast(pl.Utf8)]).alias("ID")
            )
        base_exprs: list[pl.Expr] = [pl.col("ID").cast(pl.Utf8)]
        if "resname" in out.columns:
            base_exprs.append(pl.col("resname").cast(pl.Utf8).alias("resname"))
        else:
            base_exprs.append(pl.lit("A", dtype=pl.Utf8).alias("resname"))

        if self.ground_truth_mode == "single":
            x_col: str
            y_col: str
            z_col: str
            if all(column in df.columns for column in ["x", "y", "z"]):
                x_col, y_col, z_col = "x", "y", "z"
            elif all(column in df.columns for column in ["x_1", "y_1", "z_1"]):
                x_col, y_col, z_col = "x_1", "y_1", "z_1"
            else:
                raise_error(
                    stage,
                    location,
                    "ground_truth sem coordenadas suportadas (x,y,z ou x_1,y_1,z_1)",
                    impact="1",
                    examples=df.columns[:8],
                )
            out_single = out.select(
                *base_exprs,
                pl.col(x_col).cast(pl.Float64, strict=False).alias("x"),
                pl.col(y_col).cast(pl.Float64, strict=False).alias("y"),
                pl.col(z_col).cast(pl.Float64, strict=False).alias("z"),
            )
            self._validate_coord_columns(out_single, coord_columns=["x", "y", "z"], stage=stage, location=location, label="ground_truth")
            return out_single

        if not gt_copy_ids:
            raise_error(stage, location, "gt_copy_ids ausente para ground_truth_mode=best_of_gt_copies", impact="1", examples=["gt_copy_ids"])
        coord_exprs: list[pl.Expr] = []
        coord_columns: list[str] = []
        for copy_id in gt_copy_ids:
            coord_columns.extend([f"x_{copy_id}", f"y_{copy_id}", f"z_{copy_id}"])
            coord_exprs.extend(
                [
                    pl.col(f"x_{copy_id}").cast(pl.Float64, strict=False).alias(f"x_{copy_id}"),
                    pl.col(f"y_{copy_id}").cast(pl.Float64, strict=False).alias(f"y_{copy_id}"),
                    pl.col(f"z_{copy_id}").cast(pl.Float64, strict=False).alias(f"z_{copy_id}"),
                ]
            )
        out_multi = out.select(*base_exprs, *coord_exprs)
        self._validate_coord_columns(out_multi, coord_columns=coord_columns, stage=stage, location=location, label="ground_truth")
        return out_multi

    def _canonicalize_submission(self, df: pl.DataFrame, *, stage: str, location: str) -> pl.DataFrame:
        missing = ["ID"] + [f"{axis}_{model_id}" for model_id in _COORD_MODEL_IDS for axis in ("x", "y", "z")]
        missing = [column for column in missing if column not in df.columns]
        if missing:
            raise_error(stage, location, "submissao sem coluna obrigatoria para Best-of-5", impact=str(len(missing)), examples=missing[:8])
        out = df.with_columns(pl.col("ID").cast(pl.Utf8).alias("ID"))
        cast_exprs = [pl.col("ID").cast(pl.Utf8)]
        for model_id in _COORD_MODEL_IDS:
            for axis in ("x", "y", "z"):
                cast_exprs.append(pl.col(f"{axis}_{model_id}").cast(pl.Float64, strict=False).alias(f"{axis}_{model_id}"))
        out = out.select(*cast_exprs)
        coord_cols = [f"{axis}_{model_id}" for model_id in _COORD_MODEL_IDS for axis in ("x", "y", "z")]
        self._validate_coord_columns(out, coord_columns=coord_cols, stage=stage, location=location, label="submission")
        return out

    def _validate_coord_columns(
        self,
        df: pl.DataFrame,
        *,
        coord_columns: list[str],
        stage: str,
        location: str,
        label: str,
    ) -> None:
        bad_rows: list[str] = []
        for column in coord_columns:
            series = df.get_column(column)
            values = series.to_list()
            for row_idx, raw in enumerate(values):
                if raw is None:
                    bad_rows.append(f"{column}@{row_idx}:null")
                    if len(bad_rows) >= 8:
                        break
                    continue
                try:
                    value = float(raw)
                except Exception:
                    bad_rows.append(f"{column}@{row_idx}:non-numeric")
                    if len(bad_rows) >= 8:
                        break
                    continue
                if value != value or value in {float("inf"), float("-inf")}:
                    bad_rows.append(f"{column}@{row_idx}:non-finite")
                if len(bad_rows) >= 8:
                    break
            if len(bad_rows) >= 8:
                break
        if bad_rows:
            raise_error(stage, location, f"{label} com coordenadas invalidas", impact=str(len(bad_rows)), examples=bad_rows)

    def _validate_global_contract(self, *, gt: pl.DataFrame, submission: pl.DataFrame, stage: str, location: str) -> None:
        gt_ids = gt.get_column("ID").to_list()
        submission_ids = submission.get_column("ID").to_list()
        gt_dup = gt.group_by("ID").len().filter(pl.col("len") > 1)
        if gt_dup.height > 0:
            examples = [str(row["ID"]) for row in gt_dup.head(8).iter_rows(named=True)]
            raise_error(stage, location, "ground_truth com ID duplicado", impact=str(gt_dup.height), examples=examples)
        sub_dup = submission.group_by("ID").len().filter(pl.col("len") > 1)
        if sub_dup.height > 0:
            examples = [str(row["ID"]) for row in sub_dup.head(8).iter_rows(named=True)]
            raise_error(stage, location, "submissao com ID duplicado", impact=str(sub_dup.height), examples=examples)
        gt_set = set(str(item) for item in gt_ids)
        sub_set = set(str(item) for item in submission_ids)
        missing = sorted(gt_set - sub_set)
        extra = sorted(sub_set - gt_set)
        if missing or extra:
            examples = [f"missing:{item}" for item in missing[:4]] + [f"extra:{item}" for item in extra[:4]]
            raise_error(
                stage,
                location,
                "chaves da submissao nao batem com ground_truth",
                impact=str(len(missing) + len(extra)),
                examples=examples,
            )

    def _group_by_target(self, df: pl.DataFrame, *, stage: str, location: str) -> dict[str, _TargetRows]:
        parsed = df.with_columns(
            pl.col("ID").map_elements(lambda key: self._target_id_from_key(key, stage=stage, location=location), return_dtype=pl.Utf8).alias("target_id"),
            pl.col("ID").map_elements(lambda key: self._resid_from_key(key, stage=stage, location=location), return_dtype=pl.Int64).alias("resid"),
        )
        grouped: dict[str, _TargetRows] = {}
        for target_id, target_df in parsed.group_by("target_id", maintain_order=True):
            key = target_id[0] if isinstance(target_id, tuple) else target_id
            rows = target_df.sort("resid").to_dicts()
            grouped[str(key)] = _TargetRows(target_id=str(key), rows=rows)
        return grouped

    def _score_target_bestof5(
        self,
        *,
        ground_truth_target: _TargetRows,
        submission_target: _TargetRows,
        gt_copy_ids: list[int] | None,
        tmp_dir: Path,
        stage: str,
        location: str,
    ) -> tuple[float, list[float]]:
        gt_resids = [int(row["resid"]) for row in ground_truth_target.rows]
        pred_resids = [int(row["resid"]) for row in submission_target.rows]
        if gt_resids != pred_resids:
            mismatch_examples: list[str] = []
            for idx, (gt_resid, pred_resid) in enumerate(zip(gt_resids, pred_resids)):
                if gt_resid != pred_resid:
                    mismatch_examples.append(f"{ground_truth_target.target_id}@{idx}:{gt_resid}!={pred_resid}")
                if len(mismatch_examples) >= 8:
                    break
            if len(gt_resids) != len(pred_resids):
                mismatch_examples.append(f"{ground_truth_target.target_id}:n_gt={len(gt_resids)}:n_pred={len(pred_resids)}")
            raise_error(
                stage,
                location,
                "residuos da submissao nao alinham com ground_truth no alvo",
                impact=str(max(len(gt_resids), len(pred_resids))),
                examples=mismatch_examples[:8],
            )
        gt_modes: list[tuple[str, str, str, str | None]] = []
        if self.ground_truth_mode == "single":
            gt_modes = [("x", "y", "z", None)]
        else:
            gt_modes = [(f"x_{copy_id}", f"y_{copy_id}", f"z_{copy_id}", str(copy_id)) for copy_id in (gt_copy_ids or [])]
        if not gt_modes:
            raise_error(stage, location, "nenhuma copia de ground_truth disponivel para score", impact="1", examples=[ground_truth_target.target_id])
        model_scores: list[float] = []
        for model_id in _COORD_MODEL_IDS:
            pred_path = tmp_dir / f"{ground_truth_target.target_id}_pred_{model_id}.pdb"
            best_for_model: float | None = None
            for x_col, y_col, z_col, gt_copy_id in gt_modes:
                keep_indices = self._keep_indices_for_gt(rows=ground_truth_target.rows, x_col=x_col, y_col=y_col, z_col=z_col)
                if len(keep_indices) < 2:
                    continue
                gt_rows = [ground_truth_target.rows[idx] for idx in keep_indices]
                pred_rows = [submission_target.rows[idx] for idx in keep_indices]
                suffix = "" if gt_copy_id is None else f"_{gt_copy_id}"
                gt_path = tmp_dir / f"{ground_truth_target.target_id}_gt{suffix}.pdb"
                self._write_pdb(
                    rows=gt_rows,
                    x_col=x_col,
                    y_col=y_col,
                    z_col=z_col,
                    out_path=gt_path,
                    stage=stage,
                    location=location,
                    target_id=ground_truth_target.target_id,
                )
                self._write_pdb(
                    rows=pred_rows,
                    x_col=f"x_{model_id}",
                    y_col=f"y_{model_id}",
                    z_col=f"z_{model_id}",
                    out_path=pred_path,
                    stage=stage,
                    location=location,
                    target_id=ground_truth_target.target_id,
                )
                score = self._run_usalign(pred_pdb=pred_path, true_pdb=gt_path, stage=stage, location=location, target_id=ground_truth_target.target_id)
                if best_for_model is None or float(score) > float(best_for_model):
                    best_for_model = float(score)
            if best_for_model is None:
                raise_error(
                    stage,
                    location,
                    "ground_truth sem coordenadas validas suficientes (sentinela) para score",
                    impact="1",
                    examples=[ground_truth_target.target_id],
                )
            model_scores.append(float(best_for_model))
        return max(model_scores), model_scores

    def _keep_indices_for_gt(self, *, rows: list[dict[str, object]], x_col: str, y_col: str, z_col: str) -> list[int]:
        keep: list[int] = []
        thr = float(self.missing_coord_threshold)
        for idx, row in enumerate(rows):
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                z = float(row[z_col])
            except Exception:
                continue
            if x <= thr or y <= thr or z <= thr:
                continue
            keep.append(int(idx))
        return keep

    def _write_pdb(
        self,
        *,
        rows: list[dict[str, object]],
        x_col: str,
        y_col: str,
        z_col: str,
        out_path: Path,
        stage: str,
        location: str,
        target_id: str,
    ) -> None:
        with out_path.open("w", encoding="utf-8") as handle:
            for atom_serial, row in enumerate(rows, start=1):
                resid = int(row["resid"])
                x = float(row[x_col])
                y = float(row[y_col])
                z = float(row[z_col])
                resname = str(row.get("resname", "A")).strip().upper()[:1] or "A"
                handle.write(
                    f"ATOM  {atom_serial:5d}  C1' {resname:>3s} A{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            handle.write("TER\nEND\n")
        if not out_path.exists():
            raise_error(stage, location, "falha ao escrever PDB temporario", impact="1", examples=[target_id, str(out_path)])

    def _run_usalign(self, *, pred_pdb: Path, true_pdb: Path, stage: str, location: str, target_id: str) -> float:
        command = [
            str(self.usalign_path),
            str(pred_pdb),
            str(true_pdb),
            "-outfmt",
            "2",
            "-mol",
            "RNA",
            "-atom",
            " C1'",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise_error(
                stage,
                location,
                "timeout no USalign durante score local",
                impact="1",
                examples=[target_id, f"timeout_seconds={self.timeout_seconds}"],
            )
        except Exception as exc:
            raise_error(
                stage,
                location,
                "falha ao executar binario USalign",
                impact="1",
                examples=[target_id, str(exc)],
            )
        if completed.returncode != 0:
            raise_error(
                stage,
                location,
                "USalign retornou erro no score local",
                impact="1",
                examples=[target_id, f"returncode={completed.returncode}", completed.stderr.strip()[:120]],
            )
        tm_score = self._parse_tm_score(completed.stdout)
        if tm_score is None:
            raise_error(
                stage,
                location,
                "nao foi possivel extrair TM-score do output do USalign",
                impact="1",
                examples=[target_id, completed.stdout.strip()[:200]],
            )
        return float(tm_score)

    def _parse_tm_score(self, stdout: str) -> float | None:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) >= 2:
            parts = re.split(r"\s+", lines[1])
            if len(parts) >= 4:
                try:
                    value = float(parts[3])
                    if value >= 0.0:
                        return float(value)
                except Exception:
                    pass
        matches = re.findall(r"TM-score=\s*([0-9]*\.?[0-9]+)", stdout)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                return None
        return None

    def _target_id_from_key(self, key: str, *, stage: str, location: str) -> str:
        text = str(key)
        if "_" not in text:
            raise_error(stage, location, "ID sem separador de residuo (_)", impact="1", examples=[text])
        return text.rsplit("_", 1)[0]

    def _resid_from_key(self, key: str, *, stage: str, location: str) -> int:
        text = str(key)
        if "_" not in text:
            raise_error(stage, location, "ID sem separador de residuo (_)", impact="1", examples=[text])
        chunk = text.rsplit("_", 1)[1]
        try:
            return int(chunk)
        except Exception:
            raise_error(stage, location, "resid invalido no ID", impact="1", examples=[text])
        return 0


def score_local_bestof5(
    *,
    ground_truth_path: Path,
    submission_path: Path,
    usalign_path: Path,
    score_json_path: Path,
    report_path: Path | None = None,
    timeout_seconds: int = 900,
    ground_truth_mode: str = "single",
    missing_coord_threshold: float = -1e17,
) -> LocalBestOf5ScoreResult:
    scorer = USalignBestOf5Scorer(
        usalign_path=usalign_path,
        timeout_seconds=int(timeout_seconds),
        ground_truth_mode=str(ground_truth_mode),
        missing_coord_threshold=float(missing_coord_threshold),
    )
    return scorer.score_submission(
        ground_truth_path=ground_truth_path,
        submission_path=submission_path,
        score_json_path=score_json_path,
        report_path=report_path,
    )
