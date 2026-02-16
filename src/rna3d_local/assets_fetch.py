from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import raise_error
from .errors import PipelineError
from .utils import rel_or_abs, sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    urls: list[str]
    dest_relpath: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class FetchPretrainedAssetsResult:
    assets_dir: Path
    manifest_path: Path
    payload: dict[str, object]


def _content_length(url: str, *, stage: str, location: str, timeout_seconds: int) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")  # noqa: S310
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:  # noqa: S310
            raw = resp.headers.get("Content-Length")
            if raw is None:
                return None
            n = int(str(raw))
            return n if n >= 0 else None
    except Exception:
        return None


def _ensure_disk_budget(*, dest_dir: Path, needed_bytes: int, stage: str, location: str) -> None:
    if needed_bytes <= 0:
        return
    usage = shutil.disk_usage(dest_dir)
    # Reserve a small safety margin (256MB) to avoid filling the FS.
    margin = 256 * 1024 * 1024
    if int(usage.free) < int(needed_bytes) + int(margin):
        raise_error(
            stage,
            location,
            "espaco em disco insuficiente para baixar assets",
            impact="1",
            examples=[f"free={int(usage.free)}", f"needed={int(needed_bytes)}", str(dest_dir)],
        )


def _download_one(
    spec: DownloadSpec,
    *,
    assets_dir: Path,
    timeout_seconds: int,
    max_bytes: int | None,
    dry_run: bool,
    stage: str,
    location: str,
) -> dict[str, object]:
    dest = assets_dir / spec.dest_relpath
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        existing_sha = sha256_file(dest)
        if spec.expected_sha256 is not None and existing_sha != spec.expected_sha256:
            raise_error(
                stage,
                location,
                "arquivo ja existe mas sha256 diverge do esperado",
                impact="1",
                examples=[str(dest), f"expected={spec.expected_sha256}", f"actual={existing_sha}"],
            )
        return {"path": str(spec.dest_relpath), "status": "exists", "sha256": existing_sha, "size_bytes": int(dest.stat().st_size)}

    # Pre-check disk budget using the first URL with content-length, if available.
    known_size = None
    for url in spec.urls:
        known_size = _content_length(url, stage=stage, location=location, timeout_seconds=int(timeout_seconds))
        if known_size is not None:
            break
    if known_size is not None:
        _ensure_disk_budget(dest_dir=dest.parent, needed_bytes=int(known_size), stage=stage, location=location)
        if max_bytes is not None and int(known_size) > int(max_bytes):
            raise_error(stage, location, "asset excede max_bytes", impact="1", examples=[spec.name, f"size={known_size}", f"max={max_bytes}"])

    if dry_run:
        return {
            "path": str(spec.dest_relpath),
            "status": "dry_run",
            "urls": list(spec.urls),
            "size_bytes": None if known_size is None else int(known_size),
        }

    last_exc: str | None = None
    for url in spec.urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rna3d_local/0.1"}, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:  # noqa: S310
                status = getattr(resp, "status", None)
                if status is not None and int(status) >= 400:
                    raise RuntimeError(f"http_status={int(status)}")
                h = hashlib.sha256()
                downloaded = 0
                with tmp.open("wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        h.update(chunk)
                        downloaded += len(chunk)
                        if max_bytes is not None and int(downloaded) > int(max_bytes):
                            raise_error(stage, location, "download excedeu max_bytes", impact="1", examples=[spec.name, f"bytes={downloaded}", f"max={max_bytes}"])
            sha = h.hexdigest()
            if spec.expected_sha256 is not None and sha != spec.expected_sha256:
                raise_error(stage, location, "sha256 do download nao bate com esperado", impact="1", examples=[spec.name, f"expected={spec.expected_sha256}", f"actual={sha}"])
            tmp.replace(dest)
            return {"path": str(spec.dest_relpath), "status": "downloaded", "sha256": sha, "size_bytes": int(dest.stat().st_size), "url": url}
        except PipelineError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_exc = f"{type(exc).__name__}:{exc}"
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = f"{type(exc).__name__}:{exc}"
            continue
        finally:
            if tmp.exists() and not dest.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    raise_error(
        stage,
        location,
        "falha ao baixar asset (todas URLs falharam)",
        impact=str(len(spec.urls)),
        examples=[spec.name, last_exc or "-", *spec.urls[:3]],
    )


def _run_kaggle_dataset_download(
    dataset_ref: str,
    *,
    out_dir: Path,
    dry_run: bool,
    stage: str,
    location: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", str(dataset_ref), "-p", str(out_dir), "--unzip"]
    if dry_run:
        return {"status": "dry_run", "cmd": cmd, "out_dir": str(out_dir)}
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60 * 60)
    except FileNotFoundError:
        raise_error(stage, location, "kaggle CLI nao encontrado no PATH", impact="1", examples=["kaggle"])
    except subprocess.TimeoutExpired:
        raise_error(stage, location, "kaggle download excedeu timeout", impact="1", examples=[dataset_ref])
    if proc.returncode != 0:
        stderr_txt = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout_txt = proc.stdout.decode("utf-8", errors="replace").strip()
        snippet = stderr_txt[:240] if stderr_txt else (stdout_txt[:240] if stdout_txt else f"returncode={proc.returncode}")
        raise_error(stage, location, "kaggle datasets download falhou", impact="1", examples=[dataset_ref, snippet])
    return {"status": "downloaded", "cmd": cmd, "out_dir": str(out_dir)}


def fetch_pretrained_assets(
    *,
    repo_root: Path,
    assets_dir: Path,
    include: list[str],
    dry_run: bool = False,
    timeout_seconds: int = 60,
    max_bytes: int | None = None,
) -> FetchPretrainedAssetsResult:
    stage = "FETCH_ASSETS"
    location = "src/rna3d_local/assets_fetch.py:fetch_pretrained_assets"
    assets_dir = assets_dir.resolve()
    includes = sorted({str(x).strip().lower() for x in include if str(x).strip()})
    if not includes:
        raise_error(stage, location, "include vazio (nada para baixar)", impact="1", examples=["--include ribonanzanet2"])

    manifest_rows: list[dict[str, object]] = []

    if "ribonanzanet2" in includes:
        # Kaggle dataset mirror commonly used for local/offline work.
        # Files (for planning): RibonanzaNet-DDPM-v2.pt (~567MB), diffusion_config.yaml
        dataset_ref = "shujun717/ribonanzanet2-ddpm-v2"
        out_dir = assets_dir / "encoders" / "ribonanzanet2"
        expected_paths = [out_dir / "RibonanzaNet-DDPM-v2.pt", out_dir / "diffusion_config.yaml"]
        if not dry_run and all(path.exists() for path in expected_paths):
            out = {"status": "exists", "cmd": ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(out_dir), "--unzip"], "out_dir": str(out_dir)}
        else:
            out = _run_kaggle_dataset_download(
                dataset_ref,
                out_dir=out_dir,
                dry_run=bool(dry_run),
                stage=stage,
                location=location,
            )
        files = [
            {"name": "RibonanzaNet-DDPM-v2.pt", "size_bytes": 567412078},
            {"name": "diffusion_config.yaml", "size_bytes": 675},
        ]
        if not dry_run:
            enriched: list[dict[str, object]] = []
            for item in files:
                path = (assets_dir / "encoders" / "ribonanzanet2" / str(item["name"])).resolve()
                if not path.exists():
                    raise_error(stage, location, "arquivo esperado ausente apos kaggle download", impact="1", examples=[str(path)])
                enriched.append({"name": str(item["name"]), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
            files = enriched
        manifest_rows.append(
            {
                "name": "ribonanzanet2_ddpm_v2",
                "kind": "kaggle_dataset",
                "dataset_ref": dataset_ref,
                "files": files,
                **out,
            }
        )

    if "boltz1" in includes:
        specs = [
            DownloadSpec(
                name="boltz1_conf.ckpt",
                urls=[
                    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
                    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
                ],
                dest_relpath="models/boltz1/boltz1_conf.ckpt",
            ),
            DownloadSpec(
                name="ccd.pkl",
                urls=["https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"],
                dest_relpath="models/boltz1/ccd.pkl",
            ),
        ]
        for spec in specs:
            manifest_rows.append({"name": spec.name, "kind": "http", **_download_one(spec, assets_dir=assets_dir, timeout_seconds=int(timeout_seconds), max_bytes=max_bytes, dry_run=bool(dry_run), stage=stage, location=location)})

    if "chai1" in includes:
        base = "https://chaiassets.com/chai1-inference-depencencies"
        specs = [
            DownloadSpec(
                name="conformers_v1.apkl",
                urls=[f"{base}/conformers_v1.apkl"],
                dest_relpath="models/chai1/conformers_v1.apkl",
            ),
            DownloadSpec(
                name="esm_traced_3b_fp16.pt",
                urls=[f"{base}/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"],
                dest_relpath="models/chai1/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
            ),
        ]
        for comp in [
            "feature_embedding.pt",
            "bond_loss_input_proj.pt",
            "token_embedder.pt",
            "trunk.pt",
            "diffusion_module.pt",
            "confidence_head.pt",
        ]:
            specs.append(
                DownloadSpec(
                    name=comp,
                    urls=[f"{base}/models_v2/{comp}"],
                    dest_relpath=f"models/chai1/models_v2/{comp}",
                )
            )
        for spec in specs:
            manifest_rows.append({"name": spec.name, "kind": "http", **_download_one(spec, assets_dir=assets_dir, timeout_seconds=int(timeout_seconds), max_bytes=max_bytes, dry_run=bool(dry_run), stage=stage, location=location)})

    if "rnapro" in includes:
        # RNAPro weights are often gated on HF/NGC; default to a Kaggle dataset mirror when available.
        dataset_ref = "andrewlee23023/rna-pro-public-best"
        out_dir = assets_dir / "models" / "rnapro"
        ckpt = out_dir / "rnapro-public-best-500m.ckpt"
        if not dry_run and ckpt.exists():
            out = {"status": "exists", "cmd": ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(out_dir), "--unzip"], "out_dir": str(out_dir)}
        else:
            out = _run_kaggle_dataset_download(
                dataset_ref,
                out_dir=out_dir,
                dry_run=bool(dry_run),
                stage=stage,
                location=location,
            )
        # Standardize expected filename for downstream runners.
        if not dry_run:
            if not ckpt.exists():
                raise_error(stage, location, "kaggle dataset rnapro sem ckpt esperado", impact="1", examples=[str(ckpt)])
        files = [{"name": "rnapro-public-best-500m.ckpt", "size_bytes": 4919894111}]
        if not dry_run:
            files = [{"name": ckpt.name, "size_bytes": int(ckpt.stat().st_size), "sha256": sha256_file(ckpt)}]
        manifest_rows.append({"name": "rnapro-public-best-500m.ckpt", "kind": "kaggle_dataset", "dataset_ref": dataset_ref, "files": files, **out})

    manifest_path = assets_dir / "runtime" / "fetch_manifest.json"
    payload = {
        "created_utc": utc_now_iso(),
        "assets_dir": rel_or_abs(assets_dir, repo_root),
        "include": includes,
        "dry_run": bool(dry_run),
        "items": manifest_rows,
    }
    if not dry_run:
        write_json(manifest_path, payload)
    return FetchPretrainedAssetsResult(assets_dir=assets_dir, manifest_path=manifest_path, payload=payload)
