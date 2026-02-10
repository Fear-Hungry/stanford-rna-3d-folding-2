# CASP16 Benchmark e Diagnostico (Nucleic Acids)

Este arquivo define como vamos usar o artigo **Assessment of nucleic acid structure prediction in CASP16** como **benchmark e diagnostico** para este repositorio.

## Objetivo

- Usar CASP16 como guia de avaliacao cega: nao apenas um numero unico, mas um conjunto de diagnosticos para entender falhas.
- Manter comparacoes reprodutiveis: comandos fixos, seeds, artefatos em `runs/`, e log append-only em `EXPERIMENTS.md`.

## Fonte de verdade (score)

- **Metrica primaria (obrigatoria)**: score identico ao Kaggle (TM-score com tratamento de permutacao de cadeias quando aplicavel), rodado localmente via `rna3d_local`.
- Tudo que for “diagnostico adicional” nao substitui a metrica primaria; serve para explicar regressao/ganho.

## O que registrar (benchmark minimo)

Para cada experimento/modelo comparado:
- Um identificador estavel (ex.: `runs/<timestamp>_score/`).
- Dataset avaliado:
  - `public_validation` e/ou
  - `train_cv` por fold (anti-leakage via clusters).
- Artefatos obrigatorios:
  - `runs/<id>/score.json`
  - `runs/<id>/per_target.csv` (gerado com `--per-target`)
- Metadados relevantes do modelo (checkpoint, parametros efetivos, seed).

## Estratificacoes recomendadas (diagnostico)

Quando possivel, produzir relatorios estratificados usando `per_target.csv` + metadados de `train_sequences.csv`:
- `length_bin`: bins por comprimento de `sequence` (ex.: <=80, 81-160, 161-320, >320).
- `stoichiometry`: separar monomer vs multimer (derivado de `stoichiometry`).
- `ligands`: `ligand_ids` vazio vs nao vazio.
- `temporal_cutoff`: bins por ano (proxy de disponibilidade historica e risco de leakage).

## Runbook (como rodar)

### Preparacao (1 vez por maquina)

```bash
python -m pip install -e '.[dev]'
python -m rna3d_local download --out input/stanford-rna-3d-folding-2
python -m rna3d_local vendor
```

### Benchmark “Public Validation” (baseline rapido)

```bash
python -m rna3d_local build-dataset --type public_validation \
  --input input/stanford-rna-3d-folding-2 \
  --out data/derived/public_validation

python -m rna3d_local score --dataset public_validation \
  --submission submission.csv \
  --per-target
```

Outputs:
- `runs/<timestamp>_score/score.json`
- `runs/<timestamp>_score/per_target.csv`

### Benchmark por CV em train (comparacao robusta)

```bash
python -m rna3d_local build-dataset --type train_cv_targets \
  --input input/stanford-rna-3d-folding-2 \
  --out data/derived/train_cv_targets \
  --n-folds 5 --seed 123

python -m rna3d_local build-train-fold \
  --input input/stanford-rna-3d-folding-2 \
  --targets data/derived/train_cv_targets/targets.parquet \
  --fold 0 \
  --out data/derived/train_cv/fold0

python -m rna3d_local score \
  --dataset-dir data/derived/train_cv/fold0 \
  --submission submission_fold0.csv \
  --per-target
```

## Template de comparacao (preencher via EXPERIMENTS.md)

Campos minimos para a entrada em `EXPERIMENTS.md`:
- Data UTC e autor
- `PLAN-003`
- Objetivo/hipotese (o que mudou vs baseline)
- Comandos executados (incluindo `rna3d_local score --per-target`)
- Seeds usadas
- Artefatos em `runs/` (paths)
- Score global + resumo por estrato (se aplicavel)

## Guardrails (nao-negociaveis)

- Nao usar resultados de smoke para decisoes finais.
- Qualquer score “melhor” sem `per_target.csv` e sem estratificacao minima e considerado insuficiente para diagnostico.
- Se houver mismatch de contrato, falhar cedo e corrigir antes de comparar modelos.

