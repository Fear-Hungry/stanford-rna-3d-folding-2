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

