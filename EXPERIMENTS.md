# EXPERIMENTS.md

Log append-only de experimentos executados (UTC).

## 2026-02-10T20:00:40Z - marcusvinicius/Codex - PLAN-002

- Objetivo/hipotese:
  - Validar integracao tecnica completa do novo pipeline modular (template-aware + RNAPro proxy + ensemble + export + gating), comparando baseline de repositorio sem esses modulos vs novo fluxo com cobertura de testes.
- Comandos executados + configuracao efetiva:
  - `python -m pytest -q`
  - `python -m rna3d_local --help`
  - Config efetiva nos testes de integracao sinteticos:
    - TBM: `n_models=2`, `min_coverage=0.30`, `kmer_size=2`, `top_k=2`
    - RNAPro proxy: `feature_dim=32`, `kmer_size=2`, `n_models=2`, `seed=7`, `min_coverage=0.30`
    - Ensemble: `tbm_weight=0.7`, `rnapro_weight=0.3`
    - Gating: `baseline_score=0.50`, `current_score=0.52`, `allow_regression=false`
- Parametros/hiperparametros efetivos:
  - Ver itens acima (TBM, RNAPro proxy, ensemble, gating).
- Seeds usadas:
  - `seed=7` (RNAPro proxy nos testes de integracao).
- Versao de codigo e dados:
  - Git commit base: `e6abd58`
  - Dados: datasets sinteticos temporarios criados durante `pytest` (fixtures `tmp_path`), sem alteracao de snapshots oficiais.
- Artefatos gerados:
  - Artefatos temporarios de testes em diretorios `tmp_path` do `pytest` (nao persistidos em `runs/`).
- Metricas/resultado/custo:
  - Resultado: `8 passed` em `0.14s`.
  - Custo: execucao CPU local, sem GPU, baixo uso de memoria (dataset sintetico pequeno).
- Conclusao + proximos passos:
  - Integracao tecnica confirmada para contratos e fluxos E2E.
  - Proximo passo: executar rodada com base externa real de templates e medir score local em dataset publico com artefatos persistidos em `runs/`.

## 2026-02-10T20:04:44Z - marcusvinicius/Codex - PLAN-002

- Objetivo/hipotese:
  - Validar os novos comandos CLI em cadeia ponta-a-ponta com dados sinteticos, incluindo export final em contrato estrito.
- Comandos executados + configuracao efetiva:
  - Pipeline CLI executado em sequencia:
    - `python -m rna3d_local build-template-db ...`
    - `python -m rna3d_local retrieve-templates ...`
    - `python -m rna3d_local predict-tbm ...`
    - `python -m rna3d_local train-rnapro ...`
    - `python -m rna3d_local predict-rnapro ...`
    - `python -m rna3d_local ensemble-predict ...`
    - `python -m rna3d_local export-submission ...`
    - `python -m rna3d_local check-submission ...`
  - Config efetiva usada:
    - `n_models=2`, `kmer_size=2`, `feature_dim=32`, `seed=7`, `min_coverage=0.30`
    - `tbm_weight=0.7`, `rnapro_weight=0.3`
- Parametros/hiperparametros efetivos:
  - Ver itens acima.
- Seeds usadas:
  - `seed=7`.
- Versao de codigo e dados:
  - Git commit base: `e6abd58`
  - Dataset sintetico temporario em `/tmp/tmp.fwEOJD9YVh`.
- Artefatos gerados:
  - `/tmp/tmp.fwEOJD9YVh/template_db/manifest.json`
  - `/tmp/tmp.fwEOJD9YVh/retrieval_manifest.json`
  - `/tmp/tmp.fwEOJD9YVh/tbm_manifest.json`
  - `/tmp/tmp.fwEOJD9YVh/rnapro_model/model.json`
  - `/tmp/tmp.fwEOJD9YVh/submission.csv`
- Metricas/resultado/custo:
  - Resultado: fluxo completo executado com sucesso; `check-submission` retornou `OK`.
  - Custo: ~2.3s CPU local (dataset pequeno).
- Conclusao + proximos passos:
  - CLI novo funcional em cadeia E2E.
  - Proximo passo: repetir o mesmo fluxo com base externa real e avaliar score local oficial antes de submit.
