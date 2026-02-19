from __future__ import annotations

import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import raise_error
from ..utils import utc_now_iso, write_json


@dataclass(frozen=True)
class RunExperimentResult:
    run_dir: Path
    recipe_resolved_path: Path
    meta_path: Path
    report_path: Path


@dataclass(frozen=True)
class SafePredictResult:
    ok: bool
    prediction: Any | None
    error_kind: str | None
    model_name: str
    sequence_len: int
    error_message: str | None = None


def _load_torch_module() -> Any:
    import torch  # type: ignore

    return torch


def _safe_predict_error_message(*, location: str, cause: str, impact: str, examples: list[str]) -> str:
    joined = ",".join(str(item) for item in examples)
    return f"[INFER] [{location}] {cause} | impacto={impact} | exemplos={joined}"


def safe_predict(
    model_runner_class: type,
    sequence: str,
    *args: object,
    stage: str = "INFER",
    location: str = "src/rna3d_local/experiments/runner.py:safe_predict",
    **kwargs: object,
) -> SafePredictResult:
    sequence_len = int(len(sequence))
    model_name = str(getattr(model_runner_class, "__name__", model_runner_class))
    model = None
    torch_module: Any | None = None
    try:
        try:
            torch_module = _load_torch_module()
        except Exception as exc:
            raise_error(stage, location, "falha ao importar torch para inferencia", impact="1", examples=[f"{model_name}:{type(exc).__name__}:{exc}"])
        model = model_runner_class()
        infer_ctx = torch_module.inference_mode() if hasattr(torch_module, "inference_mode") else nullcontext()
        use_cuda = bool(getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available())
        autocast_ctx = (
            torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
            if use_cuda and hasattr(torch_module, "autocast")
            else nullcontext()
        )
        with infer_ctx, autocast_ctx:
            prediction = model.predict(sequence, *args, **kwargs)
        return SafePredictResult(
            ok=True,
            prediction=prediction,
            error_kind=None,
            model_name=model_name,
            sequence_len=sequence_len,
            error_message=None,
        )
    except Exception as exc:  # noqa: BLE001
        oom_type = None
        if torch_module is not None and getattr(torch_module, "cuda", None) is not None:
            oom_type = getattr(torch_module.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            message = _safe_predict_error_message(
                location=location,
                cause="OOM durante inferencia; fallback explicito habilitado",
                impact="1",
                examples=[f"{model_name}:L={sequence_len}:{type(exc).__name__}:{exc}"],
            )
            print(message, file=sys.stderr)
            return SafePredictResult(
                ok=False,
                prediction=None,
                error_kind="oom",
                model_name=model_name,
                sequence_len=sequence_len,
                error_message=message,
            )
        raise_error(
            stage,
            location,
            "falha na inferencia do modelo",
            impact="1",
            examples=[f"{model_name}:L={sequence_len}:{type(exc).__name__}:{exc}"],
        )
    finally:
        if model is not None:
            to_fn = getattr(model, "to", None)
            if callable(to_fn):
                try:
                    to_fn("cpu")
                except Exception:
                    pass
            del model
        gc.collect()
        if torch_module is not None and getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass


def _sanitize_token(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")
    return clean if clean else "step"


def _parse_var_overrides(items: list[str], *, stage: str, location: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    bad: list[str] = []
    for raw in items:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            bad.append(text)
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            bad.append(text)
            continue
        overrides[key] = value
    if bad:
        raise_error(stage, location, "override --var invalido (esperado KEY=VALUE)", impact=str(len(bad)), examples=bad[:8])
    return overrides


def _format_value(value: Any, variables: dict[str, str], *, stage: str, location: str, context: str) -> str:
    if value is None:
        raise_error(stage, location, "valor nulo em recipe onde string era esperada", impact="1", examples=[context])
    try:
        return str(value).format(**variables)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise_error(stage, location, "placeholder ausente na recipe", impact="1", examples=[context, missing])
    except Exception as exc:
        raise_error(stage, location, "falha ao expandir placeholder da recipe", impact="1", examples=[context, f"{type(exc).__name__}:{exc}"])
    return ""


def _git_head(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        head = proc.stdout.strip()
        return head if head else None
    except Exception:
        return None


def _allocate_run_dir(*, runs_dir: Path, base_name: str, stage: str, location: str) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _sanitize_token(base_name)
    candidate = (runs_dir / f"{timestamp}_{base}").resolve()
    if not candidate.exists():
        return candidate
    for version in range(2, 100):
        alt = Path(str(candidate) + f"_v{version}")
        if not alt.exists():
            return alt
    raise_error(stage, location, "nao foi possivel alocar run_dir unico", impact="1", examples=[str(candidate)])
    return candidate


def run_experiment(
    *,
    repo_root: Path,
    recipe_path: Path,
    runs_dir: Path,
    tag_override: str | None,
    var_overrides: list[str],
    dry_run: bool,
) -> RunExperimentResult | None:
    stage = "EXPERIMENT"
    location = "src/rna3d_local/experiments/runner.py:run_experiment"
    recipe_path = Path(recipe_path).resolve()
    if not recipe_path.exists():
        raise_error(stage, location, "recipe ausente", impact="1", examples=[str(recipe_path)])
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise_error(stage, location, "falha ao ler recipe JSON", impact="1", examples=[str(recipe_path), f"{type(exc).__name__}:{exc}"])
        return None
    if not isinstance(recipe, dict):
        raise_error(stage, location, "recipe JSON deve ser objeto", impact="1", examples=[type(recipe).__name__])
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not steps:
        raise_error(stage, location, "recipe sem steps (lista vazia)", impact="1", examples=[str(recipe_path)])
    tag = str(tag_override).strip() if tag_override is not None else str(recipe.get("tag", "")).strip()
    if not tag:
        tag = recipe_path.stem
    variables_raw = recipe.get("variables", {})
    if variables_raw is None:
        variables_raw = {}
    if not isinstance(variables_raw, dict):
        raise_error(stage, location, "recipe.variables deve ser objeto", impact="1", examples=[type(variables_raw).__name__])
    variables: dict[str, str] = {str(k): str(v) for k, v in variables_raw.items()}
    variables.update(_parse_var_overrides(var_overrides, stage=stage, location=location))

    run_dir = _allocate_run_dir(runs_dir=runs_dir, base_name=tag, stage=stage, location=location)
    variables["repo_root"] = str(repo_root.resolve())
    variables["recipe_path"] = str(recipe_path)
    variables["run_dir"] = str(run_dir)

    resolved_steps: list[dict[str, object]] = []
    expanded_commands: list[str] = []
    for step_index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            raise_error(stage, location, "step invalido (esperado objeto)", impact="1", examples=[f"index={step_index}"])
        name = str(raw_step.get("name", f"step_{step_index}")).strip() or f"step_{step_index}"
        argv = raw_step.get("argv")
        if not isinstance(argv, list) or not argv:
            raise_error(stage, location, "step sem argv (lista)", impact="1", examples=[name])
        resolved_argv = [
            _format_value(item, variables, stage=stage, location=location, context=f"{name}:argv")
            for item in argv
        ]
        env_overrides_raw = raw_step.get("env", {})
        if env_overrides_raw is None:
            env_overrides_raw = {}
        if not isinstance(env_overrides_raw, dict):
            raise_error(stage, location, "step.env invalido (esperado objeto)", impact="1", examples=[name, type(env_overrides_raw).__name__])
        resolved_env_overrides = {
            str(k): _format_value(v, variables, stage=stage, location=location, context=f"{name}:env:{k}")
            for k, v in env_overrides_raw.items()
        }
        cwd_raw = raw_step.get("cwd", None)
        cwd = None if cwd_raw in {None, ""} else _format_value(cwd_raw, variables, stage=stage, location=location, context=f"{name}:cwd")
        log_path = run_dir / f"step_{step_index:02d}_{_sanitize_token(name)}.log"
        resolved_steps.append(
            {
                "index": int(step_index),
                "name": name,
                "argv": resolved_argv,
                "cwd": cwd,
                "env_overrides": resolved_env_overrides,
                "log": str(log_path),
            }
        )
        expanded_commands.append(" ".join(resolved_argv))
    artifacts_raw = recipe.get("artifacts", [])
    if artifacts_raw is None:
        artifacts_raw = []
    if not isinstance(artifacts_raw, list):
        raise_error(stage, location, "recipe.artifacts invalido (esperado lista)", impact="1", examples=[type(artifacts_raw).__name__])
    resolved_artifacts = [
        _format_value(item, variables, stage=stage, location=location, context="artifacts") for item in artifacts_raw
    ]

    if dry_run:
        payload = {
            "recipe": str(recipe_path),
            "tag": tag,
            "run_dir": str(run_dir),
            "variables": variables,
            "steps": resolved_steps,
            "artifacts": resolved_artifacts,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return None

    run_dir.mkdir(parents=True, exist_ok=False)
    recipe_resolved_path = run_dir / "recipe.json"
    meta_path = run_dir / "meta.json"
    report_path = run_dir / "run_report.json"
    write_json(
        meta_path,
        {
            "created_utc": utc_now_iso(),
            "recipe_path": str(recipe_path),
            "run_dir": str(run_dir),
            "git_head": _git_head(repo_root),
            "python": {
                "executable": sys.executable,
                "version": sys.version.split()[0],
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
        },
    )
    resolved_recipe_payload = {
        "id": recipe.get("id", None),
        "tag": tag,
        "description": recipe.get("description", ""),
        "variables": variables,
        "steps": resolved_steps,
        "artifacts": resolved_artifacts,
        "recipe_source": str(recipe_path),
    }
    write_json(recipe_resolved_path, resolved_recipe_payload)

    step_reports: list[dict[str, object]] = []
    for resolved in resolved_steps:
        name = str(resolved["name"])
        argv = [str(item) for item in resolved["argv"]]  # type: ignore[list-item]
        cwd = resolved.get("cwd", None)
        log_path = Path(str(resolved["log"]))
        env = os.environ.copy()
        env_overrides = dict(resolved.get("env_overrides") or {})
        env.update({str(k): str(v) for k, v in env_overrides.items()})
        workdir = str(repo_root) if not cwd else str(cwd)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# step={name}\n")
            handle.write(f"# cwd={workdir}\n")
            handle.write(f"# argv={' '.join(argv)}\n")
            if env_overrides:
                handle.write(f"# env_overrides={json.dumps(env_overrides, sort_keys=True)}\n")
            handle.write("\n")
            proc = subprocess.run(argv, cwd=workdir, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = float(time.time() - started)
        report = {
            "name": name,
            "argv": argv,
            "cwd": workdir,
            "log": str(log_path),
            "returncode": int(proc.returncode),
            "elapsed_seconds": elapsed,
        }
        step_reports.append(report)
        if proc.returncode != 0:
            write_json(
                report_path,
                {
                    "created_utc": utc_now_iso(),
                    "status": "failed",
                    "run_dir": str(run_dir),
                    "recipe": str(recipe_path),
                    "steps": step_reports,
                    "artifacts_expected": resolved_artifacts,
                    "artifacts_missing": [],
                },
            )
            raise_error(
                stage,
                location,
                "comando falhou durante execucao do experimento",
                impact="1",
                examples=[f"step={name}", f"returncode={proc.returncode}", str(log_path)],
            )

    missing_artifacts = [path for path in resolved_artifacts if path and not Path(path).exists()]
    if missing_artifacts:
        write_json(
            report_path,
            {
                "created_utc": utc_now_iso(),
                "status": "failed",
                "run_dir": str(run_dir),
                "recipe": str(recipe_path),
                "steps": step_reports,
                "artifacts_expected": resolved_artifacts,
                "artifacts_missing": missing_artifacts,
            },
        )
        raise_error(stage, location, "artefatos esperados ausentes ao final do experimento", impact=str(len(missing_artifacts)), examples=missing_artifacts[:8])

    write_json(
        report_path,
        {
            "created_utc": utc_now_iso(),
            "status": "ok",
            "run_dir": str(run_dir),
            "recipe": str(recipe_path),
            "steps": step_reports,
            "artifacts_expected": resolved_artifacts,
            "artifacts_missing": [],
        },
    )
    return RunExperimentResult(
        run_dir=run_dir,
        recipe_resolved_path=recipe_resolved_path,
        meta_path=meta_path,
        report_path=report_path,
    )
