# Artigo 1 Baseline/Benchmark (template-aware + RNAPro proxy)

Este arquivo define como vamos usar o **Artigo 1** (Template-based RNA structure prediction advanced through a blind code competition) como **baseline** e **benchmark operacional**: um pipeline completo que gera uma submissao valida e mede score local **identico ao Kaggle**.

## Fonte de verdade (score)

- A metrica primaria e obrigatoria e o `rna3d_local score` (TM-score via `tm-score-permutechains` + `USalign`, vendorizados).
- Qualquer comparacao com novas solucoes deve usar o mesmo scorer e registrar `--per-target`.

## Pre-requisitos (1 vez por maquina)

```bash
python -m pip install -e '.[dev]'
python -m rna3d_local download --out input/stanford-rna-3d-folding-2
python -m rna3d_local vendor

# Labels canonicos (obrigatorio para consumidores de labels)
python -m rna3d_local prepare-labels-parquet \
  --train-labels-csv input/stanford-rna-3d-folding-2/train_labels.csv \
  --out-dir data/derived/train_labels_parquet \
  --rows-per-file 2000000 \
  --compression zstd
```

## Entrada obrigatoria: `external_templates.csv`

O baseline depende de uma base externa de templates (sem fallback).

Formato (CSV ou Parquet) com colunas obrigatorias:
- `template_id` (string)
- `sequence` (string)
- `release_date` (YYYY-MM-DD)
- `resid` (int)
- `resname` (string)
- `x`,`y`,`z` (float)
- `source` (string, opcional; default `external`)

Regra critica:
- `release_date` deve respeitar o `temporal_cutoff` do alvo (o retrieval filtra templates com `release_date > temporal_cutoff`).

## Configuracao baseline (defaults da CLI)

Para comparacoes, trate os defaults como baseline (registre overrides explicitamente em `EXPERIMENTS.md`):
- Retrieval: `top_k=20`, `kmer_size=3`
- TBM: `n_models=5`, `min_coverage=0.35`
- RNAPro proxy: `feature_dim=256`, `kmer_size=4`, `n_models=5`, `seed=123`, `min_coverage=0.30`
- Ensemble: `tbm_weight=0.6`, `rnapro_weight=0.4`

## Runbook A: benchmark em `public_validation` (Kaggle Public LB)

Este modo produz uma submissao para `test_sequences.csv` e mede score contra `validation_labels.csv` (mesmas chaves do `sample_submission.csv`).

```bash
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_article1_baseline"
OUT_DIR="runs/${RUN_ID}"
mkdir -p "${OUT_DIR}"

# 1) Template DB (train Kaggle + externos)
python -m rna3d_local build-template-db \
  --train-labels-parquet-dir data/derived/train_labels_parquet \
  --external-templates external_templates.csv \
  --out-dir "${OUT_DIR}/template_db"

# 2) Retrieval (targets = test_sequences)
python -m rna3d_local retrieve-templates \
  --template-index "${OUT_DIR}/template_db/template_index.parquet" \
  --targets input/stanford-rna-3d-folding-2/test_sequences.csv \
  --out "${OUT_DIR}/retrieval_candidates.parquet"

# 3) TBM predictions (long)
python -m rna3d_local predict-tbm \
  --retrieval "${OUT_DIR}/retrieval_candidates.parquet" \
  --templates "${OUT_DIR}/template_db/templates.parquet" \
  --targets input/stanford-rna-3d-folding-2/test_sequences.csv \
  --out "${OUT_DIR}/tbm_predictions.parquet"

# 4) RNAPro proxy (train + infer)
python -m rna3d_local train-rnapro \
  --train-labels-parquet-dir data/derived/train_labels_parquet \
  --out-dir "${OUT_DIR}/rnapro_model"

python -m rna3d_local predict-rnapro \
  --model-dir "${OUT_DIR}/rnapro_model" \
  --targets input/stanford-rna-3d-folding-2/test_sequences.csv \
  --out "${OUT_DIR}/rnapro_predictions.parquet"

# 5) Ensemble (long)
python -m rna3d_local ensemble-predict \
  --tbm "${OUT_DIR}/tbm_predictions.parquet" \
  --rnapro "${OUT_DIR}/rnapro_predictions.parquet" \
  --out "${OUT_DIR}/ensemble_predictions.parquet"

# 6) Export + valida contrato
python -m rna3d_local export-submission \
  --sample input/stanford-rna-3d-folding-2/sample_submission.csv \
  --predictions "${OUT_DIR}/ensemble_predictions.parquet" \
  --out "${OUT_DIR}/submission.csv"

python -m rna3d_local check-submission \
  --sample input/stanford-rna-3d-folding-2/sample_submission.csv \
  --submission "${OUT_DIR}/submission.csv"

# 7) Score local identico ao Kaggle (gera runs/<timestamp>_score/)
python -m rna3d_local build-dataset --type public_validation \
  --input input/stanford-rna-3d-folding-2 \
  --out data/derived/public_validation

python -m rna3d_local score \
  --dataset public_validation \
  --submission "${OUT_DIR}/submission.csv" \
  --per-target
```

Artefatos esperados:
- `${OUT_DIR}/submission.csv`
- `runs/<timestamp>_score/score.json`
- `runs/<timestamp>_score/per_target.csv`

## Runbook B: benchmark por CV em `train` (por fold)

Este modo mede o baseline em alvos de `train` (sem leakage via folds por cluster) usando o mesmo scorer.

```bash
# 0) Gerar targets com fold_id (1 vez)
python -m rna3d_local build-dataset --type train_cv_targets \
  --input input/stanford-rna-3d-folding-2 \
  --out data/derived/train_cv_targets \
  --n-folds 5 --seed 123

# 1) Montar dataset de score de um fold (gera sample + solution + target_sequences)
python -m rna3d_local build-train-fold \
  --input input/stanford-rna-3d-folding-2 \
  --targets data/derived/train_cv_targets/targets.parquet \
  --fold 0 \
  --out data/derived/train_cv/fold0 \
  --train-labels-parquet-dir data/derived/train_labels_parquet

# 2) Rodar pipeline Artigo 1 para os targets do fold
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_article1_baseline_fold0"
OUT_DIR="runs/${RUN_ID}"
mkdir -p "${OUT_DIR}"

python -m rna3d_local build-template-db \
  --train-labels-parquet-dir data/derived/train_labels_parquet \
  --external-templates external_templates.csv \
  --out-dir "${OUT_DIR}/template_db"

python -m rna3d_local retrieve-templates \
  --template-index "${OUT_DIR}/template_db/template_index.parquet" \
  --targets data/derived/train_cv/fold0/target_sequences.csv \
  --out "${OUT_DIR}/retrieval_candidates.parquet"

python -m rna3d_local predict-tbm \
  --retrieval "${OUT_DIR}/retrieval_candidates.parquet" \
  --templates "${OUT_DIR}/template_db/templates.parquet" \
  --targets data/derived/train_cv/fold0/target_sequences.csv \
  --out "${OUT_DIR}/tbm_predictions.parquet"

python -m rna3d_local train-rnapro \
  --train-labels-parquet-dir data/derived/train_labels_parquet \
  --out-dir "${OUT_DIR}/rnapro_model"

python -m rna3d_local predict-rnapro \
  --model-dir "${OUT_DIR}/rnapro_model" \
  --targets data/derived/train_cv/fold0/target_sequences.csv \
  --out "${OUT_DIR}/rnapro_predictions.parquet"

python -m rna3d_local ensemble-predict \
  --tbm "${OUT_DIR}/tbm_predictions.parquet" \
  --rnapro "${OUT_DIR}/rnapro_predictions.parquet" \
  --out "${OUT_DIR}/ensemble_predictions.parquet"

python -m rna3d_local export-submission \
  --sample data/derived/train_cv/fold0/sample_submission.csv \
  --predictions "${OUT_DIR}/ensemble_predictions.parquet" \
  --out "${OUT_DIR}/submission_fold0.csv"

python -m rna3d_local check-submission \
  --sample data/derived/train_cv/fold0/sample_submission.csv \
  --submission "${OUT_DIR}/submission_fold0.csv"

# 3) Score do fold
python -m rna3d_local score \
  --dataset-dir data/derived/train_cv/fold0 \
  --submission "${OUT_DIR}/submission_fold0.csv" \
  --per-target
```

## O que registrar (obrigatorio) em `EXPERIMENTS.md`

Para qualquer rodada baseline ou comparacao vs baseline:
- Data UTC e autor
- `PLAN-011`
- Caminhos dos artefatos em `runs/` (incluindo `score.json` e `per_target.csv`)
- Parametros efetivos (defaults + overrides), incluindo `seed`
- Hash/snapshot do `external_templates.csv` (ou referencia a versao imutavel)
- Score global + resumo por estratos (ver `benchmarks/CASP16.md` para diagnostico)

## Guardrails

- Se `external_templates.csv` estiver invalido (colunas faltando, datas invalidas), falhar cedo e corrigir antes de comparar scores.
- Se `check-submission` falhar, a rodada e invalida como benchmark (erro de contrato).
- Nunca usar resultado de smoke/partial como baseline oficial; registre explicitamente `is_smoke`/`is_partial` quando aplicavel.
