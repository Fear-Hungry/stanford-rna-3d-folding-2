from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Reduz fragmentacao de VRAM em execucoes com tamanhos de alvo heterogeneos.
_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)

from .errors import raise_error
from .submission import check_submission
from .utils import sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class SubmitNotebookResult:
    report_path: Path


def _kaggle_submit_supports_kernel_flags(*, stage: str, location: str) -> None:
    try:
        completed = subprocess.run(
            ["kaggle", "competitions", "submit", "-h"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise_error(stage, location, "kaggle CLI nao encontrado no PATH", impact="1", examples=[str(exc)])
    help_txt = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        snippet = (help_txt.strip() or f"returncode={completed.returncode}")[:240]
        raise_error(stage, location, "falha ao executar 'kaggle competitions submit -h'", impact="1", examples=[snippet])
    if ("--kernel" not in help_txt) or ("--version" not in help_txt):
        version_txt = ""
        try:
            v = subprocess.run(["kaggle", "--version"], check=False, capture_output=True, text=True)
            version_txt = ((v.stdout or "") + " " + (v.stderr or "")).strip()
        except Exception:
            version_txt = ""
        examples = ["kaggle competitions submit -h sem --kernel/--version; atualize o pacote kaggle"]
        if version_txt:
            examples.append(version_txt[:120])
        raise_error(stage, location, "kaggle CLI sem suporte a submit via notebook (-k/-v)", impact="1", examples=examples)


def submit_kaggle_notebook(
    *,
    competition: str,
    notebook_ref: str,
    notebook_version: str,
    notebook_file: str,
    sample_path: Path,
    submission_path: Path,
    notebook_output_path: Path,
    score_json_path: Path,
    baseline_score: float,
    message: str,
    execute_submit: bool,
) -> SubmitNotebookResult:
    stage = "SUBMIT_NOTEBOOK"
    location = "src/rna3d_local/submit_kaggle_notebook.py:submit_kaggle_notebook"
    check_submission(sample_path=sample_path, submission_path=submission_path)
    if not notebook_output_path.exists():
        raise_error(stage, location, "arquivo de output do notebook ausente", impact="1", examples=[str(notebook_output_path)])
    if not score_json_path.exists():
        raise_error(stage, location, "score_json ausente", impact="1", examples=[str(score_json_path)])
    payload = json.loads(score_json_path.read_text(encoding="utf-8"))
    if "score" not in payload:
        raise_error(stage, location, "score_json sem campo score", impact="1", examples=[str(score_json_path)])
    score = float(payload["score"])
    if score <= float(baseline_score):
        raise_error(
            stage,
            location,
            "score local nao supera baseline (melhora estrita obrigatoria)",
            impact="1",
            examples=[f"score={score}", f"baseline={baseline_score}"],
        )
    local_hash = sha256_file(submission_path)
    notebook_hash = sha256_file(notebook_output_path)
    if local_hash != notebook_hash:
        raise_error(
            stage,
            location,
            "hash divergente entre submissao local e arquivo do notebook",
            impact="1",
            examples=[local_hash, notebook_hash],
        )

    if bool(execute_submit):
        _kaggle_submit_supports_kernel_flags(stage=stage, location=location)

    version_str = str(notebook_version).strip()
    if not version_str:
        raise_error(stage, location, "notebook_version vazio", impact="1", examples=[repr(str(notebook_version))])

    command = [
        "kaggle",
        "competitions",
        "submit",
        competition,
        "-k",
        notebook_ref,
        "-f",
        notebook_file,
        "-v",
        version_str,
        "-m",
        message,
    ]
    report = {
        "created_utc": utc_now_iso(),
        "competition": competition,
        "notebook_ref": notebook_ref,
        "notebook_version": version_str,
        "submission": str(submission_path),
        "notebook_output": str(notebook_output_path),
        "score": score,
        "baseline_score": float(baseline_score),
        "hash_match": True,
        "execute_submit": bool(execute_submit),
        "command": command,
        "status": "READY",
    }

    if execute_submit:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        report["returncode"] = int(completed.returncode)
        report["stdout"] = completed.stdout[-2000:]
        report["stderr"] = completed.stderr[-2000:]
        if completed.returncode != 0:
            raise_error(
                stage,
                location,
                "kaggle competitions submit retornou erro",
                impact="1",
                examples=[f"returncode={completed.returncode}", completed.stderr[-200:]],
            )
        report["status"] = "SUBMITTED"

    report_path = submission_path.parent / "submit_notebook_report.json"
    write_json(report_path, report)
    return SubmitNotebookResult(report_path=report_path)
