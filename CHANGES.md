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
