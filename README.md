# stanford-rna-3d-folding-2

Guia tecnico para evolucao de pipeline de predicao de estrutura RNA 3D com foco em reproducibilidade, validacao estrita e submissao Kaggle.

## Competicao

- Kaggle: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2

## Quickstart (score local identico ao Kaggle)

Este repositorio inclui um modulo `rna3d_local` que:
- baixa os arquivos oficiais necessarios (`sample_submission.csv`, `validation_labels.csv`, etc.);
- valida submissao em modo estrito (fail-fast, sem fallback silencioso);
- roda a **mesma metrica do Kaggle** (TM-score via `tm-score-permutechains` + `USalign`) localmente.

Comandos:

```bash
python -m pip install -e '.[dev]'

# 1) Baixar dados oficiais (vai para input/, ignorado pelo git)
python -m rna3d_local download --out input/stanford-rna-3d-folding-2

# 2) Instalar metric.py (vendored) + USalign
python -m rna3d_local vendor

# 3) Montar dataset local do Public LB (sample + validation_labels)
python -m rna3d_local build-dataset --type public_validation \
  --input input/stanford-rna-3d-folding-2 \
  --out data/derived/public_validation

# 4) Rodar score local (gera runs/<timestamp>_score/score.json)
python -m rna3d_local score --dataset public_validation --submission submission.csv

# 5) (Recomendado) Gate robusto + calibrado antes de promover para submit
python -m rna3d_local evaluate-robust \
  --score public_validation=runs/<run>/score/score.json \
  --score cv:fold0=runs/<run>/cv_fold0/score.json \
  --score cv:fold1=runs/<run>/cv_fold1/score.json \
  --baseline-robust-score <best_local_robust> \
  --baseline-public-score <best_public_lb> \
  --calibration-method p10
```

Comportamento padrao endurecido:
- `evaluate-robust` exige `cv_count >= 2` por padrao;
- candidatos sem CV (somente `public_validation`) sao bloqueados para promocao competitiva;
- calibracao local->public bloqueia extrapolacao fora do range historico observado (por padrao);
- `submit-kaggle` exige `--robust-report` por padrao.

Notas:
- Se a submissao divergir do `sample_submission` (colunas, chaves, duplicatas, nulos), o fluxo falha imediatamente com erro acionavel no padrao do `AGENTS.md`.
- `data/`, `input/` e `runs/` sao ignorados pelo git (artefatos locais).

### Fluxo recomendado para evitar pico de RAM (labels canonicos em Parquet)

Para cargas maiores, converta `train_labels.csv` uma vez para Parquet particionado e reuse esse artefato no pipeline:

```bash
# 0) Converter labels para formato colunar particionado
python -m rna3d_local prepare-labels-parquet \
  --train-labels-csv input/stanford-rna-3d-folding-2/train_labels.csv \
  --out-dir data/derived/train_labels_parquet \
  --rows-per-file 2000000 \
  --compression zstd

# 1) Consumir parquet canonico no template_db e no treino RNAPro
python -m rna3d_local build-template-db \
  --train-labels-parquet-dir data/derived/train_labels_parquet \
  --external-templates external_templates.csv

python -m rna3d_local train-rnapro \
  --train-labels-parquet-dir data/derived/train_labels_parquet \
  --out-dir runs/auto_rnapro
```

Regra de seguranca:
- `--train-labels-parquet-dir` e obrigatorio nos comandos consumidores de labels; se estiver invalido (sem `part-*.parquet`), o processo falha imediatamente.

Consumidores diretos do artefato `data/derived/train_labels_parquet/`:
- `build-template-db --train-labels-parquet-dir ...`
- `train-rnapro --train-labels-parquet-dir ...`
- `export-train-solution --train-labels-parquet-dir ...`
- `build-train-fold --train-labels-parquet-dir ...`

### Modulo unico reutilizavel de Big Data

As praticas de processamento de dados grandes foram centralizadas em `src/rna3d_local/bigdata.py`, incluindo:
- leitura lazy com projection/predicate pushdown (`scan_table`, `scan_labels`);
- coleta em streaming (`collect_streaming`);
- materializacao particionada em parquet (`sink_partitioned_parquet`);
- guardrails de memoria/linhas (`assert_memory_budget`, `assert_row_budget`).

Esse modulo e a API canonica para novos experimentos/treinos.

## Benchmark e diagnostico (CASP16)

O artigo **Assessment of nucleic acid structure prediction in CASP16** sera a nossa referencia oficial para **benchmark e diagnostico** (alem de um score unico).

Aplicacao pratica neste repositorio:
- metrica primaria alinhada ao Kaggle (score local identico ao Kaggle);
- protocolo de comparacao e diagnostico reprodutivel, inspirado nos tipos de falha observados em CASP16.

Runbook e template: `benchmarks/CASP16.md`.

## Template-aware + RNAPro (implementado)

O **Artigo 1** e o nosso **baseline/benchmark operacional** (ver `PLANS.md` / `PLAN-002` e `PLAN-011`): um pipeline completo template-aware + RNAPro proxy + ensemble que gera uma submissao valida e e avaliado com score local identico ao Kaggle.

Runbook do baseline (comandos + configuracao): `benchmarks/ARTICLE1_BASELINE.md`.

O pipeline modular do artigo 1 esta implementado e exposto na CLI:
- `build-template-db` -> `retrieve-templates` -> `predict-tbm`
- `train-rnapro` -> `predict-rnapro` -> `ensemble-predict`
- `export-submission` -> `check-submission` -> `submit-kaggle` (com gating estrito)

Observacoes:
- Para rodar em dados reais, e necessario fornecer uma base externa de templates (entrada `--external-templates`) e respeitar cutoff temporal sem fallback.
- A integracao tecnica esta coberta por testes sinteticos; qualidade/score deve ser validado via `benchmarks/CASP16.md` antes de qualquer submissao.

### Wrapper operacional `main GPU` (GPU-first)

Para executar o pipeline em modo GPU-first com fail-fast de CUDA:

```bash
scripts/rna3d_main_gpu.sh <comando> [args...]
```

Comportamento:
- comandos GPU-capable recebem flags CUDA automaticamente:
  - `retrieve-templates`, `train-rnapro`, `build-candidate-pool`: `--compute-backend cuda`
  - `predict-tbm`, `predict-rnapro`: `--qa-device cuda --compute-backend cuda`
  - `train-qa-rnrank`, `score-qa-rnrank`, `select-top5-global`, `train-qa-gnn-ranker`, `score-qa-gnn-ranker`: `--device cuda`
- antes de executar comandos GPU-capable, o wrapper valida `torch.cuda.is_available()`; se CUDA estiver indisponivel, encerra com erro explicito.
- comandos CPU-only (`score`, `check-submission`, `export-submission`, `ensemble-predict`, etc.) seguem sem injecao de flags GPU.

Exemplos:

```bash
# ajuda do subcomando (sem exigir CUDA)
scripts/rna3d_main_gpu.sh predict-tbm --help

# inferencia TBM GPU-first
scripts/rna3d_main_gpu.sh predict-tbm \
  --templates runs/.../retrieved_templates.parquet \
  --targets input/stanford-rna-3d-folding-2/test_sequences.csv \
  --out runs/.../tbm_predictions.parquet

# treino QA RNArank em CUDA
scripts/rna3d_main_gpu.sh train-qa-rnrank \
  --candidates runs/.../candidate_pool.parquet \
  --out-model runs/.../qa_rnrank_model.json \
  --out-weights runs/.../qa_rnrank_model.pt
```

## (Opcional) CV em `train` (splits por cluster)

Para criar splits deterministicas e um dataset de score local por fold (usando a mesma metrica):

```bash
# 1) Gerar tabela de targets com cluster_id + fold_id
python -m rna3d_local build-dataset --type train_cv_targets \
  --input input/stanford-rna-3d-folding-2 \
  --out data/derived/train_cv_targets \
  --n-folds 5 --seed 123

# 2) Montar dataset de score para um fold especifico (sample + solution + manifest + target_sequences)
python -m rna3d_local build-train-fold \
  --input input/stanford-rna-3d-folding-2 \
  --targets data/derived/train_cv_targets/targets.parquet \
  --fold 0 \
  --out data/derived/train_cv/fold0 \
  --train-labels-parquet-dir data/derived/train_labels_parquet

# 3) Score do fold (submissao deve conter apenas as chaves do sample do fold)
python -m rna3d_local score \
  --dataset-dir data/derived/train_cv/fold0 \
  --submission submission_fold0.csv \
  --chunk-size 100000 \
  --max-rows-in-memory 1000000
```

No scorer em lotes:
- `--chunk-size` controla quantas linhas sao lidas por lote (CSV/Parquet);
- `--max-rows-in-memory` limita o tamanho maximo de um target em memoria (fail-fast se exceder).

## Objetivo do repositorio

- Consolidar um pipeline de treino/inferencia/export para RNA 3D com validacao local estrita.
- Evitar regressao metodologica usando referencias cientificas atuais.
- Traduzir literatura em decisoes de engenharia rastreaveis.

## Como usar este guia cientifico

- A tabela abaixo resume cada artigo em termos de contribuicao tecnica e aplicabilidade pratica.
- A coluna `Prioridade` indica ganho esperado para este repositorio no curto/medio prazo.
- A coluna `Acao recomendada` vira backlog tecnico em `PLANS.md` quando houver implementacao.

## Tabela comparativa de artigos (alto detalhe)

| Artigo | Ano | Problema-alvo | O que o artigo traz | Como ajuda a resolver RNA 3D | Dados/benchmark | Pontos fortes | Limitacoes/riscos | Maturidade | Prioridade | Acao recomendada no repositorio | Dependencias para adocao | Sinal esperado de ganho |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| Template-based RNA structure prediction advanced through a blind code competition | 2025 | Predicao RNA 3D automatica em blind set | Mostra que estrategia top em Kaggle foi template-discovery sem DL; integra estrategias em RNAPro | Usa descoberta de templates 3D e reconciliacao de candidatos para melhorar fold global | Competicao Kaggle (43 estruturas nao liberadas) + comparacao com CASP16 | Evidencia forte em cenario blind e orientada a engenharia | Dependencia de templates proximos; menor cobertura para folds novos | Prototipo competitivo com viabilidade pratica | Alta | Implementar branch template-aware (busca, selecao, rerank) antes da etapa final de export | Base de templates curada, busca por similaridade, score/rerank robusto | Melhora de TM-score/lDDT em alvos com homologos estruturais |
| Assessment of nucleic acid structure prediction in CASP16 | 2025 | Diagnostico realista de desempenho em NA (RNA/DNA) | Avaliacao ampla mostra que acuracia 3D ainda e inconsistente sem template | Identifica gargalos: pares nao-canonicos, pseudoknots, motivos terciarios e complexos | CASP16 (42 alvos, 65 grupos) | Referencia de avaliacao cega com alta credibilidade | Nao entrega metodo unico implementavel; e analise de estado da arte | Benchmark/diagnostico | Alta | Alinhar validacao local com metricas e erros observados em CASP16 | Ferramentas para TM-score, lDDT, analise de pares/motivos | Menor risco de overfitting em metrica unica; diagnostico mais acionavel |
| The RNA-Puzzles Assessments of RNA-Only Targets in CASP16 | 2025 | Falta de avaliacao quimica/fina alem de fold global | Protocolo de avaliacao com estereoquimica, pares WC/nWC e stacking | Evita aceitar modelos com fold aceitavel, mas quimicamente incorretos | RNA-Puzzles + CASP16 RNA-only | Metodologia de avaliacao detalhada e orientada a funcao | Mais custo computacional e de implementacao de metricas | Benchmark/avaliacao | Alta | Expandir validacao local para metricas estruturais finas e nao so globais | Pipeline de avaliacao local com extracao de contatos/interacoes | Reducao de falsos positivos de qualidade |
| Accurate RNA 3D structure prediction using a language model-based deep learning approach (RhoFold+) | 2022/2024 | Predicao end-to-end com pouca estrutura experimental | RNA language model pretreinado (~23.7M sequencias) + pipeline end-to-end | Melhora generalizacao por conhecimento de sequencia em larga escala + predicao 3D/2D | RNA-Puzzles, CASP15, testes cross-family/cross-type | Forte desempenho em blind benchmarks e automacao | Custo alto de treino/inferencia; dependencia de pretreino e MSA/feature stack | Producao-pesquisa (SOTA academico) | Alta | Rodar baseline local LM-based para comparar com estrategia template-aware | GPU robusta, checkpoint/pretreino, pipeline de inferencia estavel | Ganho em targets com baixa cobertura de templates |
| trRosettaRNA: automated prediction of RNA 3D structure with transformer network | 2023 | Predicao automatica 3D com geometria aprendida | Transformer para geometrias 1D/2D + folding por minimizacao de energia | Combina predicoes geometricas com restricoes energeticas para montar 3D | CASP15 + RNA-Puzzles + benchmarks internos | Pipeline automatica com bom equilibrio entre DL e fisica | Desempenho cai em RNAs sinteticos/fora da distribuicao | Producao-pesquisa | Alta | Incluir como baseline forte e fonte de features geometricas para ensemble/rerank | Geracao de MSA/2D features, rotina de minimizacao energetica | Melhor robustez geral vs metodos tradicionais |
| Accurate Biomolecular Structure Prediction in CASP16 With Optimized Inputs to State-Of-The-Art Predictors | 2025 | Sensibilidade de preditores a qualidade de entrada | Mostra que otimizar entradas (ex. secundaria, recorte de regioes) altera forte acuracia final | Ataca gargalo de input quality sem mudar arquitetura core | CASP16 (dominios, multimeros, RNA monomeros) | Alta relacao custo-beneficio (melhorias por pre-processamento) | Ganhos dependem da qualidade do processamento upstream | Pratica de competicao/ciencia aplicada | Alta | Criar modulo de input optimization (secundaria, MSA, filtros) antes da inferencia | Ferramentas de secundaria, MSA tuning, regras de QA de entrada | Melhora consistente sem retreinar modelos grandes |
| Integrating end-to-end learning with deep geometrical potentials for ab initio RNA structure prediction (DRfold) | 2022/2023 | Predicao ab initio com melhor equilibrio local/global | Aprende frames + restraints geometricas e usa energia hibrida | Junta supervisao por coordenadas com potenciais geometricos para orientar folding | Benchmark nao redundante de estruturas recentes | Melhoria expressiva reportada em TM-score e boa formulacao hibrida | Pode exigir tuning fino e boa calibracao de energia composta | Pesquisa aplicada | Media-Alta | Avaliar integracao de energia hibrida no rerank de candidatos gerados por outros modelos | Implementacao de score energetico + stack geometrico | Melhor ordenacao de candidatos, especialmente em casos sem template |
| NuFold: end-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation | 2025 | Geometria local de RNA (alta flexibilidade ribose/base) | Representacao de nucleobase center flexivel + reciclagem e MSA metagenomico | Melhora modelagem local/all-atom e pode estender para multimeros por linking de sequencias | Benchmarks comparativos com metodos energeticos e DL | Bom desempenho local (geometria fina) e desenho moderno | Complexidade maior de reproducao e custo computacional | Pesquisa aplicada recente | Media | Fazer ablation local de representacao atomica/flexivel para impacto em lDDT local | Infra de treino/inferencia all-atom + MSA metagenomico | Ganho em qualidade local e contatos corretos |
| Geometric deep learning of RNA structure (ARES) | 2021 | Falta de bom scoring/ranking de modelos candidatos | Scorer equivarante (ARES) treinado com poucos dados para ranquear modelos | Melhora selecao de estruturas corretas entre candidatos/decoys | RNA-Puzzles/CASP em cenario de scoring | Muito util como modulo de quality ranking com poucos dados | Nao gera estrutura sozinho; depende de gerador de candidatos | Producao-pesquisa (modulo de QA) | Alta | Integrar scorer de rerank apos geracao de candidatos antes do export | Conjunto de decoys/candidatos + extracao de features coordenadas | Melhor top-1 entre candidatos e menor variancia de submissao |
| Quality assessment of RNA 3D structure models using deep learning and intermediate 2D maps (RNArank) | 2025 | QA local/global para selecionar modelos melhores | Rede residual em Y que prediz mapas 2D intermediarios (contato + desvio) e estima lDDT | Fornece filtro de qualidade acionavel para gating de export/submissao | Benchmarks + CASP15/CASP16 | QA moderno com foco em metricas locais e globais | Metodo recente, precisa validar robustez fora do estudo original | Pesquisa aplicada recente | Alta | Incluir etapa de QA/rerank obrigatoria com bloqueio de export para candidatos fracos | Predicao de mapas intermediarios e calibracao de thresholds | Export mais confiavel e menos submissao cega |
| RNA3DB: A structurally-dissimilar dataset split for training and benchmarking deep learning models for RNA structure prediction | 2024 | Vazamento e split fraco em treino/teste de RNA DL | Dataset/split estruturalmente e sequencialmente nao redundante com metodologia reproducivel | Reduz inflacao de metrica e melhora avaliacao de generalizacao real | PDB RNA chains + split aproximado 70/30 por componentes nao redundantes | Base forte para benchmark honesto e comparavel | Pode reduzir tamanho efetivo de treino; exige governanca de dados | Infra de dados/benchmark | Alta | Adotar split RNA3DB-like para treino/validacao interno e relatorios por familia | Pipeline de clustering/split por similaridade estrutural + controle de leakage | Metricas mais realistas e menor risco de regressao oculta |
| Systematic benchmarking of deep-learning methods for tertiary RNA structure prediction | 2024 | Variacao de desempenho por tipo de RNA/cenario | Benchmark sistematico em diversidade, comprimento, tipo, qualidade de MSA e arquitetura | Mostra onde DL ganha/perde e quais variaveis mais impactam resultado | Datasets diversos + comparacao entre metodos DL e nao-DL | Guia pratico para desenho de ablations e priorizacao de melhorias | Resultados dependem da cobertura dos datasets avaliados | Benchmark estrategico | Alta | Criar matriz interna de benchmark por tipo/tamanho/novidade de alvo | Suite de avaliacao estratificada + logging de metadados dos alvos | Decisao de roadmap baseada em evidencia, nao media agregada |

## Sintese por linha tecnica

- Linha 1 (template-aware): artigos de CASP16 + Kaggle 2025 reforcam que template segue decisivo em muitos alvos.
- Linha 2 (DL end-to-end): RhoFold+, trRosettaRNA, DRfold e NuFold cobrem nucleo de predicao automatica.
- Linha 3 (QA/rerank): ARES e RNArank sao alavancas diretas para melhorar selecao de candidatos.
- Linha 4 (benchmark/dados): RNA3DB e benchmarking sistematico evitam metrica inflada e guiam evolucao.

## Priorizacao pratica para este projeto

1. Curto prazo: template-aware + QA/rerank + validacao local expandida (CASP/RNA-Puzzles style).
2. Medio prazo: baseline(s) DL end-to-end reprodutiveis em GPU local com split anti-leakage.
3. Longo prazo: ensemble hibrido (template + DL + scorer) com gating estrito antes de submissao.

## Riscos tecnicos e lacunas

- Risco de inflacao de score sem split estruturalmente dissimilar.
- Risco de boa metrica global com quimica local ruim (pares nWC/stacking/pseudoknot).
- Risco operacional de submissao cega sem QA local e sem rastreabilidade por tipo de alvo.

## Referencias (DOI)

- https://doi.org/10.64898/2025.12.30.696949
- https://doi.org/10.1002/prot.70072
- https://doi.org/10.1002/prot.70052
- https://doi.org/10.1038/s41592-024-02487-0
- https://doi.org/10.1038/s41467-023-42528-4
- https://doi.org/10.1002/prot.70030
- https://doi.org/10.1038/s41467-023-41303-9
- https://doi.org/10.1038/s41467-025-56261-7
- https://doi.org/10.1126/science.abe5650
- https://doi.org/10.1101/2025.07.25.666746
- https://doi.org/10.1016/j.jmb.2024.168552
- https://doi.org/10.1371/journal.pcbi.1012715
