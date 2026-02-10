# CHANGES.md

Log append-only de mudancas implementadas (UTC).

## 2026-02-10 - marcusvinicius/Codex - ADHOC

- Inicializacao do repositorio para suportar `PLAN-001` (estrutura de pacote, regras de gitignore e scaffolding).
- Arquivos: `.gitignore`, `pyproject.toml`, `PLANS.md`, `CHANGES.md`.
- Validacao local: (pendente; sera registrado ao final do PLAN-001).
- Riscos/follow-ups:
  - Completar implementacao do modulo `rna3d_local` + testes + validacao de score com metrica vendored.

## 2026-02-10 - marcusvinicius/Codex - PLAN-001

- Resumo:
  - Implementado pacote `rna3d_local` para download de dados, vendor de metrica oficial, validacao estrita de submissao e score local identico ao Kaggle.
- Arquivos principais:
  - `src/rna3d_local/cli.py`, `src/rna3d_local/contracts.py`, `src/rna3d_local/scoring.py`
  - `src/rna3d_local/download.py`, `src/rna3d_local/vendor.py`, `src/rna3d_local/datasets.py`, `src/rna3d_local/splits.py`
  - `tests/test_contracts.py`, `README.md`, `.gitignore`, `pyproject.toml`
  - `vendor/tm_score_permutechains/metric.py`, `vendor/tm_score_permutechains/SOURCE.json`
- Validacao local executada:
  - `python -m pip install -e '.[dev]'` (ok)
  - `pytest -q` (4 passed)
  - `python -m rna3d_local download --out input/stanford-rna-3d-folding-2` (ok)
  - `python -m rna3d_local vendor` (ok; instala `metric.py` + `USalign`)
  - `python -m rna3d_local build-dataset --type public_validation --out data/derived/public_validation` (ok)
  - `python -m rna3d_local score --dataset public_validation --submission data/derived/public_validation/sample_submission.csv --per-target` (ok; smoke score ~0.0552)
- Riscos/follow-ups:
  - Definir convencao e suporte a CV completo (gerar sample/solution por fold e integrar ao treino) conforme evolucao do pipeline de modelo.

## 2026-02-10 - marcusvinicius/Codex - PLAN-003

- Resumo:
  - Documentado CASP16 (Assessment of nucleic acid structure prediction in CASP16) como benchmark/diagnostico oficial do repositorio.
  - Criado protocolo/runbook de benchmark em arquivo dedicado.
- Arquivos principais:
  - `README.md`, `benchmarks/CASP16.md`, `PLANS.md`
- Validacao local executada:
  - `python -m rna3d_local --help`
  - `python -m rna3d_local score -h`

## 2026-02-10 - marcusvinicius/Codex - PLAN-002

- Resumo:
  - Implementada arquitetura modular para artigo 1 com branch template-aware e branch RNAPro proxy, incluindo retrieval temporal estrito, predicao TBM, treino/inferencia RNAPro, ensemble, export estrito e gating de submissao.
- Arquivos principais:
  - `src/rna3d_local/template_db.py`, `src/rna3d_local/retrieval.py`, `src/rna3d_local/alignment.py`, `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/rnapro/config.py`, `src/rna3d_local/rnapro/train.py`, `src/rna3d_local/rnapro/infer.py`, `src/rna3d_local/rnapro/__init__.py`
  - `src/rna3d_local/ensemble.py`, `src/rna3d_local/export.py`, `src/rna3d_local/gating.py`
  - `src/rna3d_local/cli.py`, `README.md`, `PLANS.md`
  - `tests/test_template_workflow.py`, `tests/test_export_strict.py`
- Validacao local executada:
  - `python -m pytest -q` (8 passed)
  - `python -m rna3d_local --help` (ok; comandos novos disponiveis)
  - Smoke E2E de CLI com dataset sintetico temporario:
    - `build-template-db -> retrieve-templates -> predict-tbm -> train-rnapro -> predict-rnapro -> ensemble-predict -> export-submission -> check-submission` (ok)
- Riscos/follow-ups:
  - A trilha `RNAPro` implementada e um proxy open-source (nao replica componentes proprietarios do paper como Protenix/RNet2).
  - A qualidade final depende da curadoria da base externa de templates e da politica de cutoff temporal usada na ingestao.
