# PLANS.md

Backlog e planos ativos deste repositorio. Use IDs `PLAN-###`.

## PLAN-001 - Dataset local + score identico ao Kaggle (RNA 3D Folding 2)

- Objetivo: gerar dados locais (manifest + splits) e rodar metrica local usando o mesmo codigo do Kaggle.
- Escopo:
  - Download via Kaggle API (competition files + metric assets).
  - Validacao estrita de submissao vs `sample_submission` (fail-fast).
  - Builder de dataset local (public validation + opcional train CV com clusters).
  - Runner de score local com `TM-Score PermuteChains` + `USalign`.
- Criterios de aceite:
  - `python -m rna3d_local download` baixa `sample_submission.csv` e `validation_labels.csv` (e arquivos de treino) para `input/`.
  - `python -m rna3d_local vendor` instala `vendor/tm_score_permutechains/metric.py` e `vendor/usalign/USalign`.
  - `python -m rna3d_local score --dataset public_validation --submission <file>` calcula score local e salva artefatos em `runs/`.
  - Qualquer mismatch de contrato encerra com erro no padrao do `AGENTS.md`.

## PLAN-002 - Implementar artigo 1 (template-aware) com pipeline modular completo

- Objetivo: adicionar branch template-aware + branch RNAPro proxy com ensemble, export estrito e gating de submissao.
- Escopo:
  - Banco de templates com filtro temporal estrito (local + externo).
  - Retrieval de candidatos por alvo e preditor TBM multi-modelo.
  - Treino/inferencia RNAPro proxy (indexacao de features + predicao por alinhamento).
  - Ensemble TBM+RNAPro.
  - Export de submissao a partir de formato long, em modo estrito por padrao.
  - Gating pre-submissao Kaggle com bloqueio para smoke/partial/regressao.
  - Suite de testes para fluxos de sucesso e falhas de contrato.
- Criterios de aceite:
  - `python -m rna3d_local build-template-db` gera `templates.parquet` + `template_index.parquet` + `manifest.json`.
  - `python -m rna3d_local retrieve-templates` gera candidatos temporais sem fallback.
  - `python -m rna3d_local predict-tbm` e `python -m rna3d_local predict-rnapro` geram preditos long validos.
  - `python -m rna3d_local ensemble-predict` combina preditos com chave exata.
  - `python -m rna3d_local export-submission` gera CSV que passa `check-submission`.
  - `python -m rna3d_local submit-kaggle` bloqueia quando gating local falha.

## PLAN-003 - Benchmark/diagnostico baseado em CASP16 (documentacao + protocolo)

- Objetivo: adotar o artigo **Assessment of nucleic acid structure prediction in CASP16** como referencia oficial de benchmark e diagnostico, com protocolo reprodutivel para comparar modelos e investigar falhas.
- Escopo:
  - Documentar o papel do CASP16 como benchmark/diagnostico no `README.md`.
  - Criar arquivo dedicado `benchmarks/CASP16.md` com:
    - mapeamento de metricas (primaria = metrica Kaggle; secundarias/diagnosticas inspiradas em CASP16);
    - estratificacoes recomendadas (tamanho, stoichiometry/multicopy, ligantes, temporal_cutoff);
    - formato padrao de registro (entradas em `EXPERIMENTS.md` + artefatos `runs/`).
- Fora de escopo:
  - Implementar metricas adicionais alem do score Kaggle (vira trabalho futuro quando houver baseline estavel).
- Criterios de aceite:
  - `README.md` referencia `benchmarks/CASP16.md` explicitamente.
  - `benchmarks/CASP16.md` contem um runbook minimo (comandos) para gerar `per_target.csv` e produzir comparacoes.
