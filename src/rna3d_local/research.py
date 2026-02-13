from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .contracts import validate_submission_against_sample
from .errors import PipelineError, raise_error
from .utils import sha256_file


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_slug(value: str) -> str:
    keep = []
    for ch in str(value).lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("-")
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:120] or "item"


def _json_dumps_pretty(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps_pretty(payload), encoding="utf-8")


def _results_fingerprint(df: pl.DataFrame) -> str:
    cols = ["instance_id", "seed", "solver_status", "objective", "feasible", "gap"]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        return ""
    stable = (
        df.select(
            pl.col("instance_id").cast(pl.Utf8),
            pl.col("seed").cast(pl.Int64),
            pl.col("solver_status").cast(pl.Utf8),
            pl.col("objective").cast(pl.Float64),
            pl.col("feasible").cast(pl.Boolean),
            pl.col("gap").cast(pl.Float64),
        )
        .sort(["instance_id", "seed"])
        .to_dicts()
    )
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _http_get_json(url: str, *, timeout_s: int, location: str) -> dict[str, Any]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rna3d-local-research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = r.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise_error("RESEARCH", location, "payload JSON invalido (esperado objeto)", impact="1", examples=[url])
        return payload
    except Exception as e:  # noqa: BLE001
        raise_error("RESEARCH", location, "falha ao buscar JSON remoto", impact="1", examples=[f"{url}", f"{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def _http_get_text(url: str, *, timeout_s: int, location: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rna3d-local-research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        raise_error("RESEARCH", location, "falha ao buscar texto remoto", impact="1", examples=[f"{url}", f"{type(e).__name__}:{e}"])
    raise AssertionError("unreachable")


def _download_pdf(
    *,
    url: str,
    out_path: Path,
    timeout_s: int,
    max_pdf_mb: int,
    strict: bool,
    paper_id: str,
    location: str,
) -> tuple[str, str | None]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_pdf_mb) * 1024 * 1024
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rna3d-local-research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r, out_path.open("wb") as f:
            total = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise_error(
                        "RESEARCH",
                        location,
                        "download de PDF excedeu limite configurado",
                        impact=str(total),
                        examples=[paper_id, url, f"max_bytes={max_bytes}"],
                    )
                f.write(chunk)
        return ("downloaded", sha256_file(out_path))
    except Exception as e:  # noqa: BLE001
        if strict:
            raise_error(
                "RESEARCH",
                location,
                "falha no download de PDF OA",
                impact="1",
                examples=[paper_id, url, f"{type(e).__name__}:{e}"],
            )
        return ("download_failed", None)


def _extract_s2_papers(*, topic: str, limit: int, timeout_s: int) -> list[dict[str, Any]]:
    location = "src/rna3d_local/research.py:_extract_s2_papers"
    fields = "title,abstract,year,authors,url,openAccessPdf,externalIds"
    query = urllib.parse.urlencode({"query": topic, "limit": int(limit), "fields": fields})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{query}"
    payload = _http_get_json(url, timeout_s=timeout_s, location=location)
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise_error("RESEARCH", location, "resposta sem lista de papers", impact="1", examples=[url])
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("paperId") or row.get("externalIds", {}).get("DOI") or row.get("title") or "")
        if not pid:
            continue
        authors = row.get("authors") or []
        names = [str(a.get("name") or "") for a in authors if isinstance(a, dict) and str(a.get("name") or "")]
        oa = row.get("openAccessPdf") if isinstance(row.get("openAccessPdf"), dict) else {}
        ext_ids = row.get("externalIds") if isinstance(row.get("externalIds"), dict) else {}
        out.append(
            {
                "paper_id": f"s2:{pid}",
                "title": str(row.get("title") or ""),
                "authors": "; ".join(names),
                "year": row.get("year"),
                "doi": str(ext_ids.get("DOI") or ""),
                "url": str(row.get("url") or ""),
                "abstract": str(row.get("abstract") or ""),
                "pdf_url": str(oa.get("url") or ""),
                "license": "unknown",
                "source": "semanticscholar",
            }
        )
    return out


def _extract_openalex_papers(*, topic: str, limit: int, timeout_s: int) -> list[dict[str, Any]]:
    location = "src/rna3d_local/research.py:_extract_openalex_papers"
    query = urllib.parse.urlencode({"search": topic, "per-page": int(limit)})
    url = f"https://api.openalex.org/works?{query}"
    payload = _http_get_json(url, timeout_s=timeout_s, location=location)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise_error("RESEARCH", location, "resposta sem lista de works", impact="1", examples=[url])
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or row.get("doi") or row.get("display_name") or "")
        if not pid:
            continue
        aa = row.get("authorships") or []
        names: list[str] = []
        for it in aa:
            if not isinstance(it, dict):
                continue
            auth = it.get("author") if isinstance(it.get("author"), dict) else {}
            nm = str(auth.get("display_name") or "")
            if nm:
                names.append(nm)
        loc = row.get("best_oa_location") if isinstance(row.get("best_oa_location"), dict) else {}
        pdf_url = str(loc.get("pdf_url") or "")
        out.append(
            {
                "paper_id": f"openalex:{pid}",
                "title": str(row.get("display_name") or ""),
                "authors": "; ".join(names),
                "year": row.get("publication_year"),
                "doi": str(row.get("doi") or ""),
                "url": str(row.get("id") or ""),
                "abstract": "",
                "pdf_url": pdf_url,
                "license": str(loc.get("license") or "unknown"),
                "source": "openalex",
            }
        )
    return out


def _extract_arxiv_papers(*, topic: str, limit: int, timeout_s: int) -> list[dict[str, Any]]:
    location = "src/rna3d_local/research.py:_extract_arxiv_papers"
    query = urllib.parse.urlencode(
        {
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": int(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{query}"
    xml_text = _http_get_text(url, timeout_s=timeout_s, location=location)
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:  # noqa: BLE001
        raise_error("RESEARCH", location, "falha ao parsear XML do arXiv", impact="1", examples=[f"{type(e).__name__}:{e}"])
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        id_text = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        if not id_text:
            continue
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        year = None
        if len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if str(link.attrib.get("title") or "").lower() == "pdf":
                pdf_url = str(link.attrib.get("href") or "").strip()
                break
        if not pdf_url and id_text:
            pdf_url = id_text.replace("abs", "pdf") + ".pdf"
        out.append(
            {
                "paper_id": f"arxiv:{id_text}",
                "title": title,
                "authors": "; ".join(authors),
                "year": year,
                "doi": "",
                "url": id_text,
                "abstract": summary,
                "pdf_url": pdf_url,
                "license": "arxiv",
                "source": "arxiv",
            }
        )
    return out


@dataclass(frozen=True)
class LiteratureSyncResult:
    out_dir: Path
    papers_path: Path
    manifest_path: Path
    related_work_path: Path


def sync_literature(
    *,
    topic: str,
    out_dir: Path,
    limit_per_source: int,
    timeout_s: int,
    download_pdfs: bool,
    strict_pdf_download: bool,
    max_pdf_mb: int,
    strict_sources: bool,
) -> LiteratureSyncResult:
    location = "src/rna3d_local/research.py:sync_literature"
    if not str(topic).strip():
        raise_error("RESEARCH", location, "topic vazio", impact="1", examples=[str(topic)])
    if int(limit_per_source) <= 0:
        raise_error("RESEARCH", location, "limit_per_source invalido", impact=str(limit_per_source), examples=[])
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    papers: list[dict[str, Any]] = []
    source_failures: list[str] = []

    for source_name, fetch_fn in [
        ("semanticscholar", _extract_s2_papers),
        ("openalex", _extract_openalex_papers),
        ("arxiv", _extract_arxiv_papers),
    ]:
        try:
            papers.extend(fetch_fn(topic=topic, limit=limit_per_source, timeout_s=timeout_s))
        except PipelineError as e:
            if strict_sources:
                raise
            source_failures.append(f"{source_name}:{e}")

    if not papers:
        raise_error("RESEARCH", location, "nenhum paper retornado pelas fontes", impact=str(len(source_failures)), examples=source_failures[:8] or [topic])

    pdf_dir = out_dir / "pdf"
    for row in papers:
        row["pdf_local_path"] = ""
        row["pdf_sha256"] = ""
        row["pdf_status"] = "not_requested"
        pdf_url = str(row.get("pdf_url") or "").strip()
        if not download_pdfs:
            row["pdf_status"] = "disabled"
            continue
        if not pdf_url:
            row["pdf_status"] = "no_pdf_url"
            continue
        fname = _safe_slug(str(row.get("paper_id") or "paper")) + ".pdf"
        dst = pdf_dir / fname
        status, sha = _download_pdf(
            url=pdf_url,
            out_path=dst,
            timeout_s=timeout_s,
            max_pdf_mb=max_pdf_mb,
            strict=strict_pdf_download,
            paper_id=str(row.get("paper_id") or ""),
            location=location,
        )
        row["pdf_status"] = status
        if sha:
            row["pdf_sha256"] = sha
            row["pdf_local_path"] = str(dst.relative_to(out_dir))

    papers_df = pl.DataFrame(papers)
    papers_path = out_dir / "papers.parquet"
    papers_df.write_parquet(papers_path)

    related_work_path = out_dir / "related_work.md"
    lines = [f"# Related Work: {topic}", "", f"Generated: {_utc_now_iso()}", ""]
    for row in papers_df.sort(["source", "year", "title"], descending=[False, True, False]).iter_rows(named=True):
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        year = row.get("year")
        source = str(row.get("source") or "")
        url = str(row.get("url") or "")
        doi = str(row.get("doi") or "")
        authors = str(row.get("authors") or "")
        lines.append(f"- [{source}] {year} - {title}")
        if authors:
            lines.append(f"  - authors: {authors}")
        if doi:
            lines.append(f"  - doi: {doi}")
        if url:
            lines.append(f"  - url: {url}")
    related_work_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_source = papers_df.group_by("source").len().sort("source")
    source_counts = {str(r[0]): int(r[1]) for r in by_source.iter_rows()}
    downloaded = int((papers_df.get_column("pdf_status") == "downloaded").sum())
    manifest = {
        "created_utc": _utc_now_iso(),
        "topic": topic,
        "limit_per_source": int(limit_per_source),
        "download_pdfs": bool(download_pdfs),
        "strict_pdf_download": bool(strict_pdf_download),
        "max_pdf_mb": int(max_pdf_mb),
        "total_papers": int(papers_df.height),
        "source_counts": source_counts,
        "pdf_downloaded": downloaded,
        "strict_sources": bool(strict_sources),
        "source_failures": source_failures,
        "papers_path": str(papers_path),
        "related_work_path": str(related_work_path),
    }
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return LiteratureSyncResult(
        out_dir=out_dir,
        papers_path=papers_path,
        manifest_path=manifest_path,
        related_work_path=related_work_path,
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    location = "src/rna3d_local/research.py:_load_config"
    if not config_path.exists():
        raise_error("RESEARCH", location, "arquivo de configuracao ausente", impact="1", examples=[str(config_path)])
    suffix = config_path.suffix.lower()
    try:
        raw = config_path.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        raise_error("RESEARCH", location, "falha ao ler configuracao", impact="1", examples=[f"{type(e).__name__}:{e}"])

    if suffix == ".json":
        try:
            payload = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            raise_error("RESEARCH", location, "JSON de configuracao invalido", impact="1", examples=[f"{type(e).__name__}:{e}"])
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            raise_error("RESEARCH", location, "PyYAML nao disponivel para ler .yaml/.yml", impact="1", examples=[f"{type(e).__name__}:{e}"])
        try:
            payload = yaml.safe_load(raw)
        except Exception as e:  # noqa: BLE001
            raise_error("RESEARCH", location, "YAML de configuracao invalido", impact="1", examples=[f"{type(e).__name__}:{e}"])
    else:
        raise_error("RESEARCH", location, "extensao de configuracao nao suportada", impact="1", examples=[str(config_path)])

    if not isinstance(payload, dict):
        raise_error("RESEARCH", location, "configuracao deve ser objeto/mapa", impact="1", examples=[str(config_path)])
    return payload


def _parse_solver_payload(stdout: str, *, location: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    raise_error("RESEARCH", location, "solver stdout sem JSON de resultado", impact="1", examples=[stdout[-300:]])
    raise AssertionError("unreachable")


def _render_tokens(tokens: list[str], *, seed: int, run_id: str, run_dir: Path, repo_root: Path, location: str) -> list[str]:
    out: list[str] = []
    repl = {
        "{seed}": str(int(seed)),
        "{run_id}": str(run_id),
        "{run_dir}": str(run_dir),
        "{repo_root}": str(repo_root),
    }
    for tok in tokens:
        rendered = str(tok)
        for needle, value in repl.items():
            rendered = rendered.replace(needle, value)
        out.append(rendered)
    return out


@dataclass(frozen=True)
class ResearchRunResult:
    run_dir: Path
    manifest_path: Path
    results_path: Path


def run_experiment(
    *,
    repo_root: Path,
    config_path: Path,
    run_id: str,
    out_base_dir: Path,
    allow_existing_run_dir: bool,
) -> ResearchRunResult:
    location = "src/rna3d_local/research.py:run_experiment"
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    cfg = _load_config(config_path)

    if not str(run_id).strip():
        raise_error("RESEARCH", location, "run_id vazio", impact="1", examples=[str(run_id)])

    solver_cfg = cfg.get("solver")
    if not isinstance(solver_cfg, dict):
        raise_error("RESEARCH", location, "config sem bloco solver", impact="1", examples=[str(config_path)])
    solver_type = str(solver_cfg.get("type") or "")
    if solver_type != "command_json":
        raise_error("RESEARCH", location, "solver.type nao suportado", impact="1", examples=[solver_type])

    command_template = solver_cfg.get("command")
    if not isinstance(command_template, list) or not command_template:
        raise_error("RESEARCH", location, "solver.command invalido (esperado lista nao vazia)", impact="1", examples=[str(command_template)])

    timeout_s = int(solver_cfg.get("timeout_s", 600))
    if timeout_s <= 0:
        raise_error("RESEARCH", location, "solver.timeout_s invalido", impact=str(timeout_s), examples=[])

    seeds_raw = cfg.get("seeds", [123])
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise_error("RESEARCH", location, "seeds invalido (esperado lista nao vazia)", impact="1", examples=[str(seeds_raw)])
    seeds: list[int] = []
    for v in seeds_raw:
        try:
            seeds.append(int(v))
        except Exception:
            raise_error("RESEARCH", location, "seed invalida", impact="1", examples=[str(v)])

    required_outputs = cfg.get("required_outputs", [])
    if not isinstance(required_outputs, list):
        raise_error("RESEARCH", location, "required_outputs deve ser lista", impact="1", examples=[str(required_outputs)])

    out_base_dir = out_base_dir.resolve()
    run_dir = (out_base_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.parquet"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and (not allow_existing_run_dir):
        raise_error("RESEARCH", location, "run_dir ja contem manifest; use --allow-existing-run-dir para sobrescrever", impact="1", examples=[str(run_dir)])

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cmd = _render_tokens(
            [str(x) for x in command_template],
            seed=seed,
            run_id=run_id,
            run_dir=run_dir,
            repo_root=repo_root,
            location=location,
        )
        started = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        runtime_s = time.monotonic() - started
        (logs_dir / f"seed_{seed}.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (logs_dir / f"seed_{seed}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise_error(
                "RESEARCH",
                location,
                "solver command retornou codigo nao-zero",
                impact=str(proc.returncode),
                examples=[f"seed={seed}", shlex.join(cmd), (proc.stderr or "")[:200]],
            )

        payload = _parse_solver_payload(proc.stdout, location=location)
        solver_status = str(payload.get("solver_status") or "")
        if not solver_status:
            raise_error("RESEARCH", location, "payload do solver sem solver_status", impact="1", examples=[f"seed={seed}"])

        feasible = bool(payload.get("feasible", False))
        objective_raw = payload.get("objective")
        objective = None if objective_raw is None else float(objective_raw)
        gap_raw = payload.get("gap")
        gap = None if gap_raw is None else float(gap_raw)

        rows.append(
            {
                "instance_id": str(payload.get("instance_id") or f"seed_{seed}"),
                "seed": int(seed),
                "solver_status": solver_status,
                "objective": objective,
                "feasible": feasible,
                "runtime_s": float(payload.get("runtime_s", runtime_s)),
                "gap": gap,
                "command": shlex.join(cmd),
                "return_code": int(proc.returncode),
            }
        )

    if not rows:
        raise_error("RESEARCH", location, "execucao sem linhas de resultado", impact="0", examples=[run_id])

    res_df = pl.DataFrame(rows)
    res_df.write_parquet(results_path)
    results_sha = sha256_file(results_path)
    results_fingerprint = _results_fingerprint(res_df)

    repro_command_raw = cfg.get("repro_command")
    if repro_command_raw is None:
        repro_command = [
            "python",
            "-m",
            "rna3d_local",
            "research-run",
            "--config",
            str(config_path),
            "--run-id",
            str(run_id),
            "--allow-existing-run-dir",
        ]
    elif isinstance(repro_command_raw, list) and repro_command_raw:
        repro_command = _render_tokens(
            [str(x) for x in repro_command_raw],
            seed=seeds[0],
            run_id=run_id,
            run_dir=run_dir,
            repo_root=repo_root,
            location=location,
        )
    else:
        raise_error("RESEARCH", location, "repro_command invalido (esperado lista nao vazia)", impact="1", examples=[str(repro_command_raw)])

    rendered_required_outputs = []
    for path_token in required_outputs:
        if not isinstance(path_token, str):
            raise_error("RESEARCH", location, "required_outputs contem item nao-string", impact="1", examples=[str(path_token)])
        rendered_required_outputs.append(
            _render_tokens([path_token], seed=seeds[0], run_id=run_id, run_dir=run_dir, repo_root=repo_root, location=location)[0]
        )

    kaggle_gate_cfg = cfg.get("kaggle_gate", {})
    if kaggle_gate_cfg is None:
        kaggle_gate_cfg = {}
    if not isinstance(kaggle_gate_cfg, dict):
        raise_error("RESEARCH", location, "kaggle_gate deve ser objeto/mapa quando informado", impact="1", examples=[str(type(kaggle_gate_cfg))])

    kaggle_gate_rendered: dict[str, Any] = {}
    for key in ("sample_submission", "submission", "score_json"):
        val = kaggle_gate_cfg.get(key)
        if val is None:
            continue
        if not isinstance(val, str) or not val.strip():
            raise_error("RESEARCH", location, f"kaggle_gate.{key} invalido", impact="1", examples=[str(val)])
        kaggle_gate_rendered[key] = _render_tokens(
            [str(val)],
            seed=seeds[0],
            run_id=run_id,
            run_dir=run_dir,
            repo_root=repo_root,
            location=location,
        )[0]

    for key in ("baseline_score", "min_improvement", "max_submission_mb", "max_runtime_s_per_seed"):
        val = kaggle_gate_cfg.get(key)
        if val is None:
            continue
        try:
            kaggle_gate_rendered[key] = float(val)
        except Exception:
            raise_error("RESEARCH", location, f"kaggle_gate.{key} invalido (esperado numerico)", impact="1", examples=[str(val)])

    manifest = {
        "created_utc": _utc_now_iso(),
        "run_id": run_id,
        "experiment_name": str(cfg.get("experiment_name") or run_id),
        "config_path": str(config_path),
        "solver": {
            "type": solver_type,
            "timeout_s": timeout_s,
            "command_template": [str(x) for x in command_template],
        },
        "seeds": seeds,
        "required_outputs": rendered_required_outputs,
        "results_path": str(results_path),
        "results_sha256": results_sha,
        "results_fingerprint": results_fingerprint,
        "repro_command": repro_command,
        "literature_manifest_path": str(cfg.get("literature_manifest_path") or ""),
        "kaggle_gate": kaggle_gate_rendered,
    }
    _write_json(manifest_path, manifest)
    return ResearchRunResult(run_dir=run_dir, manifest_path=manifest_path, results_path=results_path)


@dataclass(frozen=True)
class VerifyResult:
    verify_path: Path
    accepted: bool


def verify_run(
    *,
    repo_root: Path,
    run_dir: Path,
    allowed_statuses: tuple[str, ...],
) -> VerifyResult:
    location = "src/rna3d_local/research.py:verify_run"
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.parquet"

    if not manifest_path.exists():
        raise_error("RESEARCH", location, "run_manifest.json ausente", impact="1", examples=[str(run_dir)])
    if not results_path.exists():
        raise_error("RESEARCH", location, "results.parquet ausente", impact="1", examples=[str(run_dir)])

    manifest = _load_config(manifest_path)
    try:
        res_df = pl.read_parquet(results_path)
    except Exception as e:  # noqa: BLE001
        raise_error("RESEARCH", location, "falha ao ler results.parquet", impact="1", examples=[f"{type(e).__name__}:{e}"])

    required_cols = ["instance_id", "seed", "solver_status", "objective", "feasible", "runtime_s", "gap", "command", "return_code"]
    missing = [c for c in required_cols if c not in res_df.columns]
    if missing:
        raise_error("RESEARCH", location, "results.parquet sem colunas obrigatorias", impact=str(len(missing)), examples=missing)

    allowed = {s.strip().lower() for s in allowed_statuses if str(s).strip()}
    if not allowed:
        raise_error("RESEARCH", location, "allowed_statuses vazio", impact="0", examples=[])

    bad_status = res_df.filter(~pl.col("solver_status").cast(pl.Utf8).str.to_lowercase().is_in(sorted(allowed)))
    bad_rc = res_df.filter(pl.col("return_code") != 0)
    bad_feasible = res_df.filter(~pl.col("feasible").cast(pl.Boolean))

    required_outputs = manifest.get("required_outputs", [])
    if not isinstance(required_outputs, list):
        raise_error("RESEARCH", location, "manifest.required_outputs invalido", impact="1", examples=[str(required_outputs)])

    missing_outputs: list[str] = []
    for raw in required_outputs:
        p = Path(str(raw))
        if not p.is_absolute():
            p = run_dir / p
        if not p.exists():
            missing_outputs.append(str(p))

    repro_command = manifest.get("repro_command")
    if not isinstance(repro_command, list) or not repro_command:
        raise_error("RESEARCH", location, "manifest.repro_command invalido", impact="1", examples=[str(repro_command)])

    repro = subprocess.run(
        [str(x) for x in repro_command],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    repro_ok = repro.returncode == 0

    expected_sha = str(manifest.get("results_sha256") or "")
    current_sha = sha256_file(results_path)
    expected_fp = str(manifest.get("results_fingerprint") or "")
    current_fp = _results_fingerprint(pl.read_parquet(results_path))
    fp_ok = bool(expected_fp) and (expected_fp == current_fp)

    kaggle_gate = manifest.get("kaggle_gate", {})
    if kaggle_gate is None:
        kaggle_gate = {}
    if not isinstance(kaggle_gate, dict):
        raise_error("RESEARCH", location, "manifest.kaggle_gate invalido", impact="1", examples=[str(type(kaggle_gate))])

    kaggle_failures: list[str] = []
    kaggle_details: dict[str, Any] = {}
    if kaggle_gate:
        sample_raw = str(kaggle_gate.get("sample_submission") or "").strip()
        submission_raw = str(kaggle_gate.get("submission") or "").strip()
        score_json_raw = str(kaggle_gate.get("score_json") or "").strip()

        if sample_raw:
            sample_path = Path(sample_raw)
            if not sample_path.is_absolute():
                sample_path = run_dir / sample_path
        else:
            sample_path = None
        if submission_raw:
            submission_path = Path(submission_raw)
            if not submission_path.is_absolute():
                submission_path = run_dir / submission_path
        else:
            submission_path = None
        if score_json_raw:
            score_json_path = Path(score_json_raw)
            if not score_json_path.is_absolute():
                score_json_path = run_dir / score_json_path
        else:
            score_json_path = None

        kaggle_details["sample_submission"] = None if sample_path is None else str(sample_path)
        kaggle_details["submission"] = None if submission_path is None else str(submission_path)
        kaggle_details["score_json"] = None if score_json_path is None else str(score_json_path)

        if sample_path is not None and submission_path is not None:
            if not sample_path.exists():
                kaggle_failures.append(f"sample_submission_missing:{sample_path}")
            if not submission_path.exists():
                kaggle_failures.append(f"submission_missing:{submission_path}")
            if sample_path.exists() and submission_path.exists():
                try:
                    validate_submission_against_sample(sample_path=sample_path, submission_path=submission_path)
                except Exception as e:  # noqa: BLE001
                    kaggle_failures.append(f"submission_contract_fail:{type(e).__name__}")

        baseline_score = float(kaggle_gate.get("baseline_score", 0.0))
        min_improvement = float(kaggle_gate.get("min_improvement", 0.0))
        kaggle_details["baseline_score"] = baseline_score
        kaggle_details["min_improvement"] = min_improvement
        current_score = None
        if score_json_path is not None:
            if not score_json_path.exists():
                kaggle_failures.append(f"score_json_missing:{score_json_path}")
            else:
                try:
                    payload = json.loads(score_json_path.read_text(encoding="utf-8"))
                    current_score = float(payload["score"])
                except Exception as e:  # noqa: BLE001
                    kaggle_failures.append(f"score_json_parse_fail:{type(e).__name__}:{e}")
        kaggle_details["current_score"] = current_score
        if current_score is not None:
            threshold = baseline_score + min_improvement
            kaggle_details["required_score_gt"] = threshold
            if not (current_score > threshold):
                kaggle_failures.append(f"score_not_improved:{current_score}<=:{threshold}")

        max_submission_mb = kaggle_gate.get("max_submission_mb")
        if max_submission_mb is not None and submission_path is not None and submission_path.exists():
            size_mb = submission_path.stat().st_size / (1024 * 1024)
            kaggle_details["submission_size_mb"] = size_mb
            kaggle_details["max_submission_mb"] = float(max_submission_mb)
            if size_mb > float(max_submission_mb):
                kaggle_failures.append(f"submission_too_large:{size_mb:.3f}MB>{float(max_submission_mb):.3f}MB")

        max_runtime = kaggle_gate.get("max_runtime_s_per_seed")
        if max_runtime is not None:
            limit = float(max_runtime)
            slow = res_df.filter(pl.col("runtime_s").cast(pl.Float64) > limit)
            kaggle_details["max_runtime_s_per_seed"] = limit
            kaggle_details["slow_seed_count"] = int(slow.height)
            if slow.height > 0:
                kaggle_failures.extend([f"slow_seed:{r[1]}:{r[5]}" for r in slow.select("instance_id", "seed", "solver_status", "feasible", "objective", "runtime_s").head(4).iter_rows()])

    kaggle_gate_pass = len(kaggle_failures) == 0

    verify_payload = {
        "created_utc": _utc_now_iso(),
        "run_dir": str(run_dir),
        "solver_pass": int(bad_status.height) == 0 and int(bad_rc.height) == 0 and int(bad_feasible.height) == 0,
        "checks_pass": len(missing_outputs) == 0,
        "repro_pass": bool(repro_ok and fp_ok),
        "kaggle_gate_pass": kaggle_gate_pass,
        "details": {
            "bad_status_count": int(bad_status.height),
            "bad_return_code_count": int(bad_rc.height),
            "bad_feasible_count": int(bad_feasible.height),
            "missing_outputs_count": len(missing_outputs),
            "missing_outputs_examples": missing_outputs[:8],
            "repro_command": [str(x) for x in repro_command],
            "repro_return_code": int(repro.returncode),
            "repro_stderr_head": (repro.stderr or "")[:400],
            "results_sha_expected": expected_sha,
            "results_sha_current": current_sha,
            "results_fingerprint_expected": expected_fp,
            "results_fingerprint_current": current_fp,
            "fingerprint_match": fp_ok,
            "kaggle_gate": kaggle_details,
            "kaggle_gate_failures": kaggle_failures[:16],
        },
    }
    accepted = bool(
        verify_payload["solver_pass"]
        and verify_payload["checks_pass"]
        and verify_payload["repro_pass"]
        and verify_payload["kaggle_gate_pass"]
    )
    verify_payload["accepted"] = accepted

    verify_path = run_dir / "verify.json"
    _write_json(verify_path, verify_payload)

    if not accepted:
        examples: list[str] = []
        if int(bad_status.height) > 0:
            examples.extend([f"bad_status:{r[0]}:{r[2]}" for r in bad_status.select("instance_id", "seed", "solver_status").head(4).iter_rows()])
        if int(bad_rc.height) > 0:
            examples.extend([f"bad_rc:{r[0]}:{r[2]}" for r in bad_rc.select("instance_id", "seed", "return_code").head(4).iter_rows()])
        if int(bad_feasible.height) > 0:
            examples.extend([f"bad_feasible:{r[0]}:{r[2]}" for r in bad_feasible.select("instance_id", "seed", "feasible").head(4).iter_rows()])
        examples.extend(missing_outputs[:4])
        if not repro_ok:
            examples.append(f"repro_rc={repro.returncode}")
        if not fp_ok:
            examples.append("fingerprint_mismatch")
        examples.extend(kaggle_failures[:4])
        raise_error("RESEARCH", location, "gate do experimento falhou (solver/checks/repro/kaggle)", impact="1", examples=examples[:8])

    return VerifyResult(verify_path=verify_path, accepted=True)


def generate_report(*, run_dir: Path, out_path: Path) -> Path:
    location = "src/rna3d_local/research.py:generate_report"
    run_dir = run_dir.resolve()
    out_path = out_path.resolve()
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.parquet"
    verify_path = run_dir / "verify.json"

    if not manifest_path.exists():
        raise_error("RESEARCH", location, "run_manifest.json ausente", impact="1", examples=[str(run_dir)])
    if not results_path.exists():
        raise_error("RESEARCH", location, "results.parquet ausente", impact="1", examples=[str(run_dir)])
    if not verify_path.exists():
        raise_error("RESEARCH", location, "verify.json ausente", impact="1", examples=[str(run_dir)])

    manifest = _load_config(manifest_path)
    verify = _load_config(verify_path)
    res_df = pl.read_parquet(results_path)

    if "objective" not in res_df.columns:
        raise_error("RESEARCH", location, "results sem coluna objective", impact="1", examples=[str(results_path)])

    obj_df = res_df.select("seed", "instance_id", "solver_status", "objective", "feasible", "runtime_s", "gap")
    best_obj = None
    if obj_df.height > 0:
        non_null = obj_df.filter(pl.col("objective").is_not_null())
        if non_null.height > 0:
            best_obj = float(non_null.select(pl.col("objective").max()).item(0, 0))

    lines = [
        f"# Research Report - {manifest.get('run_id', 'unknown')}",
        "",
        f"- generated_utc: {_utc_now_iso()}",
        f"- experiment_name: {manifest.get('experiment_name', '')}",
        f"- accepted: {verify.get('accepted', False)}",
        f"- best_objective: {best_obj}",
        "",
        "## Verification",
        "",
        f"- solver_pass: {verify.get('solver_pass', False)}",
        f"- checks_pass: {verify.get('checks_pass', False)}",
        f"- repro_pass: {verify.get('repro_pass', False)}",
        f"- kaggle_gate_pass: {verify.get('kaggle_gate_pass', True)}",
        "",
        "## Results",
        "",
        "| seed | instance_id | solver_status | feasible | objective | runtime_s | gap |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in obj_df.sort("seed").iter_rows(named=True):
        lines.append(
            "| {seed} | {instance_id} | {solver_status} | {feasible} | {objective} | {runtime_s} | {gap} |".format(
                seed=row.get("seed"),
                instance_id=row.get("instance_id"),
                solver_status=row.get("solver_status"),
                feasible=row.get("feasible"),
                objective=row.get("objective"),
                runtime_s=row.get("runtime_s"),
                gap=row.get("gap"),
            )
        )

    lit_manifest = str(manifest.get("literature_manifest_path") or "").strip()
    lines.extend(["", "## Literature", ""])
    if lit_manifest:
        lines.append(f"- manifest: {lit_manifest}")
    else:
        lines.append("- manifest: (not provided)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
