# Runbook “Submarino” (Kaggle notebook offline, sem fallback)

Objetivo: construir tudo localmente (com internet), empacotar como Kaggle Datasets e executar o notebook de submissão no Kaggle com `enable_internet=false`, gerando `submission.csv` de forma autônoma e **fail-fast**.

Este runbook segue o que o repositório realmente implementa hoje (CLI `rna3d_local`, wheelhouse `phase2`, assets phase2 e roteamento híbrido).

## Fase 0 — Fixar o “contrato” do Kaggle (antes de empacotar)

No Kaggle, crie um notebook vazio, rode:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

E **use exatamente essa versão de Python** no wheelhouse (o projeto exige `>=3.11`).

## Fase 1 — Estaleiro (local)

### 1) Wheelhouse offline (phase2)

Gera wheels para instalar no Kaggle sem internet (o profile `phase2` inclui `boltz`, `chai_lab`, `rdkit`, etc.). Por design, o wheelhouse **não** puxa `torch/numpy/pandas` por padrão (assume que o Kaggle já fornece).

Para pipeline SE(3) competitivo (`training_protocol=local_16gb`), inclua tambem `torch_geometric` + `torch_cluster` no wheelhouse/dataset offline. O backend de grafo nesse protocolo e fail-fast e exige `graph_backend=torch_geometric`.

```bash
python -m rna3d_local build-wheelhouse \
  --wheels-dir export/kaggle_wheels \
  --python-version 3.12 \
  --platform manylinux2014_x86_64 \
  --profile phase2
```

Se o seu Kaggle não for Py3.12, ajuste `--python-version` para o valor real.

### 2) Assets offline (foundation) + configs

```bash
python -m rna3d_local fetch-pretrained-assets \
  --assets-dir export/kaggle_assets \
  --include ribonanzanet2 \
  --include boltz1 \
  --include chai1 \
  --include rnapro

python -m rna3d_local prepare-rnapro-support-files \
  --model-dir export/kaggle_assets/models/rnapro

python -m rna3d_local write-phase2-model-configs \
  --assets-dir export/kaggle_assets
```

Inclua também o arquivo de “quickstart” de química (Ribonanza QUICK_START) no dataset de assets (o notebook precisa dele para `prepare-chemical-features`).

### 3) Modelos e bancos “seus” (custom)

Exemplos (ajuste paths/flags conforme seus artefatos):

```bash
python -m rna3d_local build-template-db \
  --external-templates external_templates.csv \
  --out-dir export/custom_models/template_db

python -m rna3d_local build-embedding-index \
  --template-index export/custom_models/template_db/template_index.parquet \
  --out-dir export/custom_models/template_db/embedding \
  --encoder ribonanzanet2 \
  --embedding-dim 384
```

Se você quiser usar reranker:

```bash
python -m rna3d_local train-template-reranker ... --out-dir export/custom_models/reranker
```

Se você quiser usar SE(3) no Kaggle, garanta também o artefato do modelo e os arquivos necessários (e note que o pipeline SE(3) depende de inputs adicionais: pairings/química e, por padrão, MSA/BPP).

## Empacotamento (Kaggle Datasets privados)

Crie datasets separados para não estourar limites de tamanho:

1) `rna3d-wheels-and-code`
   - `export/kaggle_wheels/`
   - o código do projeto (ex.: `src/` ou um zip com `src/rna3d_local/`)

2) `rna3d-foundation-assets`
   - `export/kaggle_assets/`
   - arquivo de química QUICK_START (csv/parquet)

3) `rna3d-custom-models`
   - `export/custom_models/` (template_db, embeddings, reranker, se3_model, etc.)

## Fase 2 — Mergulho (Notebook Kaggle, internet OFF)

Premissas:
- Internet do notebook desativada (`enable_internet=false`).
- Datasets acima anexados + dataset da competição.

### Célula 1 — Instalação offline + PYTHONPATH

```bash
%%bash
set -euo pipefail

WHEELS="/kaggle/input/rna3d-wheels-and-code/export/kaggle_wheels"
SRC="/kaggle/input/rna3d-wheels-and-code/src"

python --version

pip install --no-index --find-links "$WHEELS" rna3d-local

# Alternativa (quando você carrega src/ diretamente em vez de wheel do projeto):
mkdir -p /kaggle/working/src
cp -r "$SRC/rna3d_local" /kaggle/working/src/
echo 'export PYTHONPATH=/kaggle/working/src:$PYTHONPATH' >> /kaggle/working/.env
```

Se você optar pela alternativa via `PYTHONPATH`, nas células seguintes execute com:
`source /kaggle/working/.env` antes de chamar `python -m rna3d_local ...`.

### Célula 2 — Pipeline híbrido (exemplo mínimo consistente)

```bash
%%bash
set -euo pipefail
source /kaggle/working/.env || true

INPUT="/kaggle/input/stanford-rna-3d-folding-2"
RUNS="/kaggle/working/runs"
ASSETS="/kaggle/input/rna3d-foundation-assets/export/kaggle_assets"
CUSTOM="/kaggle/input/rna3d-custom-models/export/custom_models"

mkdir -p "$RUNS"

# 0) Sanity-check dos assets phase2
python -m rna3d_local build-phase2-assets --assets-dir "$ASSETS" --manifest "$RUNS/phase2_assets_manifest.json"

# 1) Química + pairings (quickstart vem do dataset anexado)
python -m rna3d_local prepare-chemical-features --quickstart /kaggle/input/rna3d-foundation-assets/QUICK_START.csv --out "$RUNS/chemical_features.parquet"
python -m rna3d_local derive-pairings-from-chemical --chemical-features "$RUNS/chemical_features.parquet" --out "$RUNS/pairings.parquet"

# 2) Retrieval + TBM (faiss opcional; use numpy_bruteforce para evitar dependências)
python -m rna3d_local infer-description-family --targets "$INPUT/test_sequences.csv" --out-dir "$RUNS/description_family" --backend rules --template-family-map "$CUSTOM/template_db/template_family_map.parquet"

python -m rna3d_local retrieve-templates-latent \
  --template-index "$CUSTOM/template_db/template_index.parquet" \
  --template-embeddings "$CUSTOM/template_db/embedding/template_embeddings.parquet" \
  --targets "$INPUT/test_sequences.csv" \
  --out "$RUNS/retrieval_candidates.parquet" \
  --ann-engine numpy_bruteforce \
  --family-prior "$RUNS/description_family/family_prior.parquet"

python -m rna3d_local predict-tbm \
  --retrieval "$RUNS/retrieval_candidates.parquet" \
  --templates "$CUSTOM/template_db/templates.parquet" \
  --targets "$INPUT/test_sequences.csv" \
  --out "$RUNS/tbm_predictions.parquet" \
  --n-models 5

# 3) Foundation models offline (Chai/Boltz/RNAPro)
python -m rna3d_local predict-rnapro-offline --model-dir "$ASSETS/models/rnapro" --targets "$INPUT/test_sequences.csv" --out "$RUNS/rnapro_predictions.parquet" --n-models 5
python -m rna3d_local predict-chai1-offline  --model-dir "$ASSETS/models/chai1"  --targets "$INPUT/test_sequences.csv" --out "$RUNS/chai1_predictions.parquet"  --n-models 5
python -m rna3d_local predict-boltz1-offline --model-dir "$ASSETS/models/boltz1" --targets "$INPUT/test_sequences.csv" --out "$RUNS/boltz1_predictions.parquet" --n-models 5

# 4) Router + Top-5 + export + gate estrito
python -m rna3d_local build-hybrid-candidates \
  --targets "$INPUT/test_sequences.csv" \
  --retrieval "$RUNS/retrieval_candidates.parquet" \
  --tbm "$RUNS/tbm_predictions.parquet" \
  --rnapro "$RUNS/rnapro_predictions.parquet" \
  --chai1 "$RUNS/chai1_predictions.parquet" \
  --boltz1 "$RUNS/boltz1_predictions.parquet" \
  --out "$RUNS/hybrid_candidates.parquet" \
  --routing-out "$RUNS/routing.parquet" \
  --template-score-threshold 0.65 \
  --ultra-long-seq-threshold 1500

python -m rna3d_local select-top5-hybrid --candidates "$RUNS/hybrid_candidates.parquet" --out "$RUNS/hybrid_top5.parquet" --n-models 5
python -m rna3d_local export-submission --sample "$INPUT/sample_submission.csv" --predictions "$RUNS/hybrid_top5.parquet" --out /kaggle/working/submission.csv
python -m rna3d_local check-submission --sample "$INPUT/sample_submission.csv" --submission /kaggle/working/submission.csv
```

Observações importantes:
- `prepare-chemical-features` **não** aceita `test_sequences.csv`; ele exige o arquivo QUICK_START (por resíduo) no schema suportado por `src/rna3d_local/chemical_features.py`.
- Minimização (`minimize-ensemble --backend openmm`) é opcional; `--max-iterations 0` faz bypass explícito e, quando habilitada, exige `openmm` disponível no ambiente offline (não faz parte do wheelhouse `phase2` por padrão).

Checklist fail-fast sugerido (antes de rodar o pipeline pesado):
- `python --version` (deve ser compatível com o wheelhouse e com `requires-python>=3.11`)
- `python -c "import torch; print(torch.__version__)"` (torch deve existir no ambiente)
- `python -c "import torch_geometric, torch_cluster; print('pyg-ok')"` (obrigatorio para SE(3) competitivo)
- `python -m rna3d_local --help` (garante que o pacote foi carregado corretamente)

## Submissão (notebook-only)

O notebook deve gerar `submission.csv` em `/kaggle/working/`.

Para submeter via CLI (fora do Kaggle), use o gate oficial do repo:

1) `python -m rna3d_local check-submission --sample <sample_submission.csv> --submission <submission.csv>`
2) `python -m rna3d_local submit-kaggle-notebook --competition <comp> --notebook-ref <owner/notebook> --notebook-version <Version 1> --sample <sample_submission.csv> --submission <submission.csv> --notebook-output-path <path>`
