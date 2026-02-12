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

## 2026-02-10 - marcusvinicius/Codex - PLAN-003

- Resumo:
  - Corrigido bloqueio do benchmark `train_cv/fold0`: validacao de solucao agora aceita `solution` em Parquet (alem de CSV), mantendo fail-fast para formatos invalidos.
  - Incluidos testes de contrato para validar caminho de sucesso/falha com `solution.parquet`.
- Arquivos principais:
  - `src/rna3d_local/contracts.py`
  - `tests/test_contracts.py`
- Validacao local executada:
  - `python -m pytest -q` (10 passed)
  - `python -m rna3d_local score --dataset-dir data/derived/train_cv/fold0 --submission data/derived/train_cv/fold0/sample_submission.csv --per-target` (execucao iniciou sem erro de contrato/parquet; interrompida manualmente por custo/tempo)
- Riscos/follow-ups:
  - Benchmark CV por fold completo com `--per-target` e computacionalmente caro; recomenda-se executar em janela dedicada.

## 2026-02-10 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Atualizada regra operacional em `AGENTS.md` para exigir explicitamente otimizacao para a maquina atual, prevenindo OOM de RAM.
- Arquivos principais:
  - `AGENTS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `rg -n "otimizado para a maquina atual|OOM de RAM" AGENTS.md` (ok)

## 2026-02-10 - marcusvinicius/Codex - PLAN-004

- Resumo:
  - Implementadas otimizacoes de memoria RAM em toda a pipeline com guardrails fail-fast (`memory_budget_mb` + `max_rows_in_memory`) e processamento incremental por chunks.
  - Corrigido bug em `predict_tbm` (buffer de escrita), adicionada escrita atomica (`*.tmp` -> rename) para evitar artefato parcial em falha.
  - Refatorados `retrieve_template_candidates` e `infer_rnapro` para escrita incremental parquet (sem lista global de linhas) e selecao top-k com heap.
  - Refatorado `train_rnapro` para features `float32` e sem `list[dict]` massiva.
  - CLI expandida para expor flags de memoria/chunk nos comandos de dataset/template/retrieval/predicao/ensemble/export/score.
- Arquivos principais:
  - `src/rna3d_local/memory.py`
  - `src/rna3d_local/template_db.py`
  - `src/rna3d_local/retrieval.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/rnapro/train.py`
  - `src/rna3d_local/rnapro/infer.py`
  - `src/rna3d_local/ensemble.py`
  - `src/rna3d_local/export.py`
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/scoring.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_memory_guardrails.py`
- Validacao local executada:
  - `python -m compileall src` (ok)
  - `python -m pytest -q` (14 passed)
  - `python -m rna3d_local predict-tbm --help | rg -n "memory-budget-mb|max-rows-in-memory|chunk-size"` (ok)
  - `python -m rna3d_local predict-rnapro --help | rg -n "memory-budget-mb|max-rows-in-memory|chunk-size"` (ok)
  - `python -m rna3d_local score --help | rg -n "memory-budget-mb"` (ok)
- Riscos/follow-ups:
  - `retrieve_template_candidates` ainda materializa indice com k-mers em RAM; proxima iteracao pode migrar para blocos por cutoff/data para reduzir pico em bases muito grandes.

## 2026-02-10 - marcusvinicius/Codex - PLAN-005

- Resumo:
  - Adicionado fluxo canonico de labels em Parquet particionado (`prepare-labels-parquet`) para reduzir pico de RAM em etapas que usam `train_labels`.
  - Integrado consumo de `train_labels_parquet_dir` em `build-template-db`, `train-rnapro` e `export-train-solution`, com precedencia explicita e sem fallback silencioso quando o diretorio parquet estiver invalido.
  - Atualizado `README.md` com runbook de uso do formato colunar para cargas grandes.
- Arquivos principais:
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/template_db.py`
  - `src/rna3d_local/rnapro/train.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_labels_parquet.py`
  - `README.md`
  - `PLANS.md`
- Validacao local executada:
  - `python -m compileall src` (ok)
  - `python -m pytest -q` (18 passed)
  - `python -m rna3d_local prepare-labels-parquet --help | rg -n "rows-per-file|compression|memory-budget-mb"` (ok)
  - `python -m rna3d_local build-template-db --help | rg -n "train-labels-parquet-dir"` (ok)
  - `python -m rna3d_local train-rnapro --help | rg -n "train-labels-parquet-dir"` (ok)
  - `python -m rna3d_local export-train-solution --help | rg -n "train-labels-parquet-dir|train-labels-csv"` (ok)
  - Smoke real do novo comando:
    - `python -m rna3d_local prepare-labels-parquet --train-labels-csv <tmp>/train_labels.csv --out-dir <tmp>/out --rows-per-file 2 --compression zstd` (ok; gerou `manifest.json` + `part-00000.parquet` + `part-00001.parquet`)
- Riscos/follow-ups:
  - Conversao CSV -> parquet atualmente usa etapa intermediaria `_labels_tmp.parquet`; iteracao futura pode remover esse arquivo temporario com escrita direta em particoes para reduzir I/O em disco.

## 2026-02-10 - marcusvinicius/Codex - PLAN-006

- Resumo:
  - Criada camada central `data_access.py` para padronizar consumo de dados grandes por outros modulos (`scan_labels`, `scan_table`, `collect_streaming`, `sink_partitioned_parquet`).
  - Refatorados `datasets`, `template_db` e `rnapro/train` para usar a API unica de labels com precedencia explicita para `train_labels_parquet_dir` e sem fallback silencioso.
  - Refatorados `retrieval`, `tbm_predictor`, `export` e `ensemble` para leitura via scan lazy + collect streaming em pontos de tabelas grandes.
  - CLI padronizada com helper interno de resolucao de labels (`_resolve_label_inputs`) para reduzir duplicacao de logica.
  - README atualizado com lista clara de consumidores do artefato `data/derived/train_labels_parquet/`.
- Arquivos principais:
  - `src/rna3d_local/data_access.py`
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/template_db.py`
  - `src/rna3d_local/rnapro/train.py`
  - `src/rna3d_local/retrieval.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/export.py`
  - `src/rna3d_local/ensemble.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_data_access.py`
  - `tests/test_labels_parquet.py`
  - `README.md`
  - `PLANS.md`
- Validacao local executada:
  - `python -m compileall src` (ok)
  - `python -m pytest -q` (23 passed)
  - `python -m rna3d_local prepare-labels-parquet --help | rg -n "rows-per-file|compression|memory-budget-mb"` (ok)
  - `python -m rna3d_local build-template-db --help | rg -n "train-labels-parquet-dir"` (ok)
  - `python -m rna3d_local train-rnapro --help | rg -n "train-labels-parquet-dir"` (ok)
  - `python -m rna3d_local export-train-solution --help | rg -n "train-labels-parquet-dir|train-labels-csv"` (ok)
  - Smoke real:
    - `python -m rna3d_local prepare-labels-parquet --train-labels-csv <tmp>/train_labels.csv --out-dir <tmp>/out --rows-per-file 2 --compression zstd` (ok; gerou `manifest.json` + `part-00000.parquet` + `part-00001.parquet`)
- Riscos/follow-ups:
  - Em datasets muito grandes, `retrieval` ainda calcula similaridade em Python puro (jaccard por set), podendo ser gargalo de CPU mesmo com RAM controlada.

## 2026-02-10 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Ajustado `score_submission` para reduzir risco de OOM em fold grande (`fold2`), evitando `groupby` pesado no DataFrame completo e usando mapa de indices por `ID`.
  - Corrigido erro de tipo no metric vendorizado com coordenadas nullable (`pd.NA`) ao converter coordenadas por target para `float64` antes da chamada do `metric.score`.
  - Refatorada validacao de contratos para leitura lazy/colunar em `contracts.py` (scan + leitura apenas da chave), reduzindo pico de memoria antes do score.
- Arquivos principais:
  - `src/rna3d_local/scoring.py`
  - `src/rna3d_local/contracts.py`
  - `tests/test_scoring.py`
  - `tests/test_contracts.py`
- Validacao local executada:
  - `python -m pytest -q tests/test_scoring.py tests/test_contracts.py` (9 passed)
  - `python -m pytest -q tests/test_labels_parquet.py tests/test_memory_guardrails.py` (7 passed)
  - `python -m pytest -q` (23 passed)
  - `python -m rna3d_local prepare-labels-parquet --train-labels-csv input/stanford-rna-3d-folding-2/train_labels.csv --out-dir data/derived/train_labels_parquet --rows-per-file 2000000 --compression zstd --memory-budget-mb 22000` (ok; gerou `manifest.json` + `part-00000..00003.parquet`)
  - `python -m rna3d_local build-train-fold --input input/stanford-rna-3d-folding-2 --targets data/derived/train_cv_targets/targets.parquet --fold 1 --out data/derived/train_cv/fold1_parquet_test --train-labels-parquet-dir data/derived/train_labels_parquet --memory-budget-mb 22000` (ok; `/usr/bin/time -v`: max RSS 1.31 GB)
  - `python -m rna3d_local build-train-fold --input input/stanford-rna-3d-folding-2 --targets data/derived/train_cv_targets/targets.parquet --fold 0 --out data/derived/train_cv/fold0_invalid_parquet_test --train-labels-parquet-dir data/derived/does_not_exist --memory-budget-mb 22000` (falha esperada fail-fast: diretorio parquet ausente, sem fallback)
- Riscos/follow-ups:
  - `fold2` do benchmark local (`runs/20260210_204413_benchmark_safe_v2/fold2`) ainda em execucao longa na data deste registro; consolidar `score.json` final ao concluir.

## 2026-02-10 - marcusvinicius/Codex - PLAN-007

- Resumo:
  - Reduzido pico de RAM no `export_train_solution_for_targets` ao substituir materializacao eager (`collect(...).write_parquet(...)`) por escrita lazy streaming (`sink_parquet(...)`).
  - Corrigida extracao de `target_id` para IDs com underscore via regex de prefixo antes do ultimo `_` (evita truncamento incorreto).
  - Aplicada leitura lazy/projetada de `targets.parquet` e `train_sequences.csv` em `datasets` e `cli` para evitar carregar colunas desnecessarias.
  - Adicionado teste para `target_id` com underscore no export de solution.
- Arquivos principais:
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_labels_parquet.py`
  - `PLANS.md`
- Validacao local executada:
  - `python -m pytest -q tests/test_labels_parquet.py tests/test_memory_guardrails.py tests/test_data_access.py tests/test_scoring.py tests/test_contracts.py` (20 passed)
  - `python -m pytest -q` (24 passed)
  - Benchmark de memoria (antes vs depois) no caso critico:
    - Antes (`runs/optcmp_plan005/export_fold2_csv.time`): max RSS `15714204 kB` (~14.99 GB), elapsed `5.92 s`
    - Depois (`runs/optcmp_plan005_post/export_fold2_csv.time`): max RSS `3862300 kB` (~3.68 GB), elapsed `3.03 s`
    - Antes (`runs/optcmp_plan005/export_fold2_parquet.time`): max RSS `15836008 kB` (~15.10 GB), elapsed `5.63 s`
    - Depois (`runs/optcmp_plan005_post/export_fold2_parquet.time`): max RSS `4688564 kB` (~4.47 GB), elapsed `3.37 s`
  - Validação E2E do builder:
    - `python -m rna3d_local build-train-fold --input input/stanford-rna-3d-folding-2 --targets data/derived/train_cv_targets/targets.parquet --fold 2 --out data/derived/train_cv/fold2_post_parquet_optcmp --train-labels-parquet-dir data/derived/train_labels_parquet --memory-budget-mb 22000` (ok; max RSS `4871392 kB` ~4.65 GB)
- Riscos/follow-ups:
  - O ganho de RAM foi comprovado no export/build de dataset; a etapa de `score` ainda depende de pandas + metric vendorizado e pode continuar exigindo RAM alta em folds muito grandes.

## 2026-02-10 - marcusvinicius/Codex - PLAN-008

- Resumo:
  - Removido codigo legado de labels CSV nos consumidores de pipeline e padronizado contrato unico de labels canonicos em Parquet.
  - Simplificada API central `scan_labels` para aceitar apenas `labels_parquet_dir`, eliminando dual-path e fallback legado.
  - Simplificadas assinaturas de `export_train_solution_for_targets`, `build_train_cv_fold_dataset`, `build_template_db` e `train_rnapro` para consumo exclusivo de parquet.
  - CLI atualizada: removidas flags `--train-labels` e `--train-labels-csv` dos comandos consumidores; `--train-labels-parquet-dir` passou a ser obrigatoria.
  - Testes atualizados para o novo contrato unico e benchmark rapido por fold registrado para validar pico de RAM operacional.
- Arquivos principais:
  - `src/rna3d_local/data_access.py`
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/template_db.py`
  - `src/rna3d_local/rnapro/train.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_data_access.py`
  - `tests/test_labels_parquet.py`
  - `tests/test_template_workflow.py`
  - `README.md`
  - `PLANS.md`
- Validacao local executada:
  - `python -m pytest -q tests/test_data_access.py tests/test_labels_parquet.py tests/test_template_workflow.py` (11 passed)
  - `python -m pytest -q` (24 passed)
  - `python -m rna3d_local build-train-fold --help | rg -n "train-labels|train-labels-parquet-dir"` (ok; apenas `--train-labels-parquet-dir`)
  - `python -m rna3d_local export-train-solution --help | rg -n "train-labels|input\\b"` (ok; apenas `--train-labels-parquet-dir`)
  - Benchmark por fold (`/usr/bin/time -v`, labels parquet):
    - `python -m rna3d_local build-train-fold --input input/stanford-rna-3d-folding-2 --targets data/derived/train_cv_targets/targets.parquet --fold {0..4} --out data/derived/train_cv/plan008_fold{fold} --train-labels-parquet-dir data/derived/train_labels_parquet --memory-budget-mb 22000` (ok)
    - `python -m rna3d_local export-train-solution --targets data/derived/train_cv_targets/targets.parquet --fold 2 --out data/derived/plan008_solution_fold2.parquet --train-labels-parquet-dir data/derived/train_labels_parquet --memory-budget-mb 22000` (ok)
- Riscos/follow-ups:
  - A remocao do legado reduz ambiguidade e risco de fallback; porem score completo de folds muito grandes ainda pode exigir RAM alta por dependencia do metric vendorizado via pandas.

## 2026-02-10 - marcusvinicius/Codex - PLAN-009

- Resumo:
  - Consolidado modulo unico de big data em `src/rna3d_local/bigdata.py` com API canônica para leitura lazy, coleta streaming, escrita parquet particionada e guardrails de memoria.
  - Migrados consumidores do pipeline (`cli`, `datasets`, `template_db`, `retrieval`, `tbm_predictor`, `rnapro`, `ensemble`, `export`, `scoring`) para importar diretamente de `bigdata.py`.
  - `data_access.py` e `memory.py` convertidos para wrappers de compatibilidade (re-export), removendo logica duplicada.
  - README atualizado com a diretriz de reutilizacao do modulo unico em novos experimentos/treinos.
- Arquivos principais:
  - `src/rna3d_local/bigdata.py`
  - `src/rna3d_local/data_access.py`
  - `src/rna3d_local/memory.py`
  - `src/rna3d_local/cli.py`
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/template_db.py`
  - `src/rna3d_local/retrieval.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/rnapro/train.py`
  - `src/rna3d_local/rnapro/infer.py`
  - `src/rna3d_local/ensemble.py`
  - `src/rna3d_local/export.py`
  - `src/rna3d_local/scoring.py`
  - `tests/test_data_access.py`
  - `tests/test_memory_guardrails.py`
  - `README.md`
  - `PLANS.md`
- Validacao local executada:
  - `python -m compileall src` (ok)
  - `python -m pytest -q` (24 passed)
  - `rg -n "from \\.data_access|from \\.memory|from \\.\\.data_access|from \\.\\.memory|from rna3d_local\\.data_access|from rna3d_local\\.memory" src tests` (ok; sem ocorrencias)
  - `rg -n "from \\.bigdata|from \\.\\.bigdata|from rna3d_local\\.bigdata" src tests` (ok; consumidores migrados)
- Riscos/follow-ups:
  - Wrappers de compatibilidade (`data_access.py` e `memory.py`) podem ser removidos em cleanup futuro quando nao houver consumidores externos legados.

## 2026-02-11 - marcusvinicius/Codex - PLAN-010

- Resumo:
  - Refatorado `score_submission` para processamento em lotes por `target_id` (CSV/Parquet) sem carregar a tabela inteira em memoria.
  - Adicionados guardrails de score (`chunk_size`, `max_rows_in_memory`) e expostos na CLI (`rna3d_local score`).
  - Endurecida validacao de contrato para incluir ordem de chaves e duplicatas tambem na solucao.
  - Corrigido `export_train_solution_for_targets` para gerar `solution.parquet` em ordem canonica de fold (`target_id` + `resid`) com fail-fast para targets sem labels.
- Arquivos principais:
  - `src/rna3d_local/scoring.py`
  - `src/rna3d_local/contracts.py`
  - `src/rna3d_local/datasets.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_contracts.py`
  - `tests/test_scoring.py`
  - `README.md`
  - `PLANS.md`
- Validacao local executada:
  - `pytest -q` (28 passed)
  - `pytest -q tests/test_contracts.py tests/test_scoring.py tests/test_data_access.py tests/test_labels_parquet.py tests/test_memory_guardrails.py` (23 passed)
  - Rebuild de folds com budget 8 GB:
    - `python -m rna3d_local build-train-fold ... --fold 0 --out data/derived/train_cv/plan010_fold0 --memory-budget-mb 8192` (max RSS `1095040 kB`, elapsed `0:00.92`)
    - `python -m rna3d_local build-train-fold ... --fold 1 --out data/derived/train_cv/plan010_fold1 --memory-budget-mb 8192` (max RSS `1141152 kB`, elapsed `0:00.98`)
    - `python -m rna3d_local build-train-fold ... --fold 2 --out data/derived/train_cv/plan010_fold2 --memory-budget-mb 8192` (max RSS `5951672 kB`, elapsed `0:18.30`)
    - `python -m rna3d_local build-train-fold ... --fold 3 --out data/derived/train_cv/plan010_fold3 --memory-budget-mb 8192` (max RSS `1091312 kB`, elapsed `0:01.09`)
    - `python -m rna3d_local build-train-fold ... --fold 4 --out data/derived/train_cv/plan010_fold4 --memory-budget-mb 8192` (max RSS `1128764 kB`, elapsed `0:01.11`)
  - Benchmark de score iniciado em `runs/20260211_005143_benchmark_plan010_full` com `--memory-budget-mb 8192 --max-rows-in-memory 500000 --chunk-size 50000`.
- Riscos/follow-ups:
  - O metric vendorizado (`USalign`) ainda pode consumir tempo elevado em folds grandes; benchmark completo deve continuar por fold (serial) para manter previsibilidade operacional.

## 2026-02-11 - marcusvinicius/Codex - PLAN-011

- Resumo:
  - Documentado o Artigo 1 como baseline/benchmark operacional e adicionado runbook reprodutivel dedicado.
  - Melhorado o dataset de CV por fold: `build-train-fold` agora exporta `target_sequences.csv` (alvos do fold) para permitir rodar pipeline de inferencia e export sem scripts auxiliares.
  - Adicionado teste unitario garantindo a presenca e o contrato de `target_sequences.csv` no fold.
- Arquivos principais:
  - `benchmarks/ARTICLE1_BASELINE.md`
  - `README.md`
  - `src/rna3d_local/datasets.py`
  - `tests/test_train_fold_sequences.py`
  - `PLANS.md`
- Validacao local executada:
  - `pytest -q` (28 passed)
- Riscos/follow-ups:
  - O baseline do Artigo 1 depende de uma base externa real (`external_templates.csv`); o score baseline oficial deve ser obtido rodando o runbook e registrando a rodada em `EXPERIMENTS.md` (com hashes/snapshot do input externo).

## 2026-02-11 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Atualizado help da CLI para refletir que `build-train-fold` tambem exporta `target_sequences.csv`.
- Arquivos principais:
  - `src/rna3d_local/cli.py`
- Validacao local executada:
  - `pytest -q` (28 passed)

## 2026-02-11 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Limpeza de versionamento: adicionado `external_templates.csv` ao `.gitignore` para impedir commit acidental de artefato local grande usado em benchmark/template retrieval.
- Arquivos principais:
  - `.gitignore`
- Validacao local executada:
  - `git check-ignore -v external_templates.csv` (ok; regra `.gitignore:21:/external_templates.csv` aplicada)
  - `git status --short` (ok; apenas `.gitignore` modificado)
- Riscos/follow-ups:
  - Se o dataset externo mudar de nome, adicionar nova regra explicita no `.gitignore` para manter a politica de artefatos locais fora de versionamento.

## 2026-02-11 - marcusvinicius/Codex - PLAN-004

- Resumo:
  - Refatorado `build-template-db` para reduzir pico de RAM sem fallback: `templates.parquet` agora armazena apenas coordenadas por residuo (sem repetir `sequence` por linha), enquanto `template_index.parquet` preserva `sequence` + metadados para retrieval.
  - Ajustado `predict-tbm` para consumir `sequence` a partir de `template_index.parquet` e selecionar os primeiros `n_models` candidatos que passam `min_coverage` dentro do pool recuperado (em vez de falhar ao primeiro candidato ruim).
  - Ajustado `predict-rnapro` com a mesma politica operacional de cobertura (pool ampliado + selecao de modelos validos por alvo).
  - Refatorado `ensemble-predict` e `export-submission` para caminho lazy/streaming (`sink_parquet`/`sink_csv`) evitando materializacao total em RAM.
  - Adicionados testes para cobrir o novo contrato de templates/index e a selecao de candidato TBM por cobertura.
- Arquivos principais:
  - `src/rna3d_local/template_db.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/rnapro/infer.py`
  - `src/rna3d_local/ensemble.py`
  - `src/rna3d_local/export.py`
  - `tests/test_template_workflow.py`
  - `tests/test_tbm_coverage_selection.py`
- Validacao local executada:
  - `python -m compileall -q src tests` (ok)
  - `pytest -q` (29 passed)
  - Pipeline real Kaggle (runbook completo local):
    - `python -m rna3d_local prepare-train-labels-clean ...`
    - `python -m rna3d_local build-template-db ...`
    - `python -m rna3d_local retrieve-templates --top-k 200 ...`
    - `python -m rna3d_local predict-tbm --min-coverage 0.01 ...`
    - `python -m rna3d_local train-rnapro --min-coverage 0.01 ...`
    - `python -m rna3d_local predict-rnapro --min-coverage 0.01 ...`
    - `python -m rna3d_local ensemble-predict ...`
    - `python -m rna3d_local export-submission ...`
    - `python -m rna3d_local check-submission ...` (OK)
    - `python -m rna3d_local score --dataset-dir data/derived/public_validation ...` (score `0.05522357142857142`)
- Riscos/follow-ups:
  - O baseline funcional com dados reais foi executado sem OOM, mas o score final permanece baixo; proximas iteracoes devem focar qualidade de retrieval/ranking (nao infraestrutura de memoria).

## 2026-02-11 - marcusvinicius/Codex - PLAN-004

- Resumo:
  - Corrigido bug critico no `export-submission`: a saida estava preservando as colunas de coordenadas do `sample_submission` (zeros) em vez das coordenadas previstas. Agora o export monta a base apenas com `ID,resname,resid` do sample e injeta coordenadas do predito long.
  - Adicionado teste de regressao para garantir que coordenadas previstas sao efetivamente escritas no CSV final.
  - Ajustado `predict-tbm`/`predict-rnapro` para selecionar `n_models` validos por cobertura dentro do pool recuperado, evitando falha por candidatos iniciais ruins quando existem alternativas validas.
- Arquivos principais:
  - `src/rna3d_local/export.py`
  - `tests/test_export_strict.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/rnapro/infer.py`
- Validacao local executada:
  - `python -m compileall -q src tests` (ok)
  - `pytest -q` (30 passed)
  - Re-score pos-correcao (mesmo artefato de ensemble):
    - antes: `0.05522357142857142`
    - depois: `0.13244464285714286`
  - Verificacao objetiva de divergencia vs sample:
    - `mean_abs_diff(sample,submission)=271.034791933347` (antes era `0.0`)
- Riscos/follow-ups:
  - Qualidade ainda depende fortemente de retrieval/ranking; infraestrutura de memoria/contrato ficou estabilizada.

## 2026-02-11 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Atualizadas as regras operacionais de submissao em `AGENTS.md` com o contrato real da competicao para evitar fluxo invalido.
  - Definido explicitamente que esta competicao e notebook-only (code competition), com submit via `-k/-f/-v` e exigencia de `enable_internet=false` no notebook.
  - Documentado fluxo obrigatorio: validacao local estrita (`check-submission`) antes de publicar/submit, mais registro do payload completo em erro de API.
- Arquivos principais:
  - `AGENTS.md`
- Validacao local executada:
  - `rg -n "notebook-only|code competition|enable_internet|CreateSubmission|kaggle competitions submit -c" AGENTS.md` (ok; regras presentes nas linhas 72-78)
  - `git diff -- AGENTS.md` (ok; apenas secao de regras de submissao foi adicionada)
- Riscos/follow-ups:
  - Se a Kaggle alterar novamente a politica de submissao da competicao, atualizar esta secao imediatamente para manter aderencia operacional.

## 2026-02-11 - marcusvinicius/Codex - PLAN-012

- Resumo:
  - Implementado rerank de candidatos template-aware com prioridade por qualidade efetiva em `retrieve/predict-tbm/predict-rnapro`.
  - `retrieve-templates` agora incorpora sinal de compatibilidade de comprimento (`length_weight`) no ranking dos candidatos.
  - `predict-tbm` e `predict-rnapro` passaram de estrategia "primeiro candidato valido" para selecao dos melhores candidatos validos por `coverage` e `similarity` dentro de pool controlado.
  - Expostos novos knobs na CLI para controle do rerank:
    - `retrieve-templates --length-weight`
    - `predict-tbm --rerank-pool-size`
    - `predict-rnapro --rerank-pool-multiplier`
  - Adicionados testes para cobrir rerank por comprimento e por cobertura.
- Arquivos principais:
  - `src/rna3d_local/retrieval.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/rnapro/infer.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_retrieval_rerank.py`
  - `tests/test_tbm_coverage_selection.py`
- Validacao local executada:
  - `python -m compileall -q src tests` (ok)
  - `pytest -q tests/test_retrieval_rerank.py tests/test_tbm_coverage_selection.py tests/test_template_workflow.py` (5 passed)
  - `pytest -q` (32 passed)
- Riscos/follow-ups:
  - O rerank aumenta custo computacional em `predict-tbm/predict-rnapro` devido avaliacao de pool maior; manter monitoramento de tempo em dados maiores.
  - Integrar o novo modelo 512 no notebook de submissao dinamica para validar ganho no hidden LB.

## 2026-02-11 - marcusvinicius/Codex - PLAN-013

- Resumo:
  - Substituido alinhamento heuristico (`difflib.SequenceMatcher`) por alinhamento global com Biopython (`Bio.Align.PairwiseAligner`) em `alignment.py`, com mapeamento deterministico por blocos alinhados.
  - Reescrita da projecao de coordenadas para interpolacao/extrapolacao linear entre ancoras alinhadas (removendo drift fixo por delta).
  - Adicionado modo opcional de ensemble dinamico por cobertura em `blend_predictions` e exposto na CLI (`--dynamic-by-coverage`, `--coverage-power`, `--coverage-floor`).
  - Declarada dependencia `biopython` no `pyproject.toml`.
  - Incluidos testes unitarios novos para alinhamento Biopython e blending dinamico por cobertura.
- Arquivos principais:
  - `src/rna3d_local/alignment.py`
  - `src/rna3d_local/ensemble.py`
  - `src/rna3d_local/cli.py`
  - `pyproject.toml`
  - `tests/test_alignment_biopython.py`
  - `tests/test_ensemble_dynamic.py`
  - `tests/test_tbm_coverage_selection.py`
- Validacao local executada:
  - `python -m compileall -q src tests` (ok)
  - `pytest -q tests/test_alignment_biopython.py tests/test_ensemble_dynamic.py tests/test_tbm_coverage_selection.py tests/test_template_workflow.py` (8 passed)
  - `pytest -q` (36 passed)
  - Experimento completo PLAN-013 (sem OOM): `runs/20260211_173901_plan013_biopython_dynamic`
- Riscos/follow-ups:
  - A troca para alinhamento global Biopython aumentou custo de runtime em `predict-tbm`/`predict-rnapro`.
  - Resultado local desta configuracao regrediu vs baseline atual; manter a feature sob controle de parametros e iterar calibracao de score/alinhamento antes de novo submit.

## 2026-02-11 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Reorganizado `EXPERIMENTS.md` para manter um bloco unico por plano (`## PLAN-###` e `## ADHOC`), com cada registro historico preservado como subentrada cronologica (`### <timestamp> - autor`).
  - Mantidos todos os registros existentes (19 entradas), sem remocao de conteudo tecnico de cada experimento.
- Arquivos principais:
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `rg -n "^## |^### " EXPERIMENTS.md` (ok; 1 secao por plano + subentradas por execucao)
  - `python - <<'PY' ... count(^##), count(^###), count('- Objetivo/hipotese:') ... PY` (ok; `plan_sections=10`, `entries=19`, `objetivo_blocks=19`)
  - `git diff -- EXPERIMENTS.md` (ok; mudanca estrutural de organizacao por plano)
- Riscos/follow-ups:
  - A regra original de log append-only em `EXPERIMENTS.md` foi flexibilizada nesta acao por solicitacao explicita do usuario para reorganizacao estrutural.

## 2026-02-11 - marcusvinicius/Codex - PLAN-015/PLAN-016

- Resumo:
  - Formalizadas novas frentes operacionais em `PLANS.md`:
    - `PLAN-015`: escala RNAPro `feature_dim=768` + calibracao de blend.
    - `PLAN-016`: escala RNAPro `feature_dim=1024` + blend de diversidade + submit notebook.
  - Executada publicacao de nova versao do dataset Kaggle de ativos (`marcux777/stanford-rna3d-infer-assets-v1`) contendo `rnapro_model_768`.
  - Atualizado notebook de submissao Kaggle (`marcux777/stanford-rna3d-submit-prod-v2`, versao `42`) para inferencia dual-model (512 + 768) e blend final (`0.4*linha_768 + 0.6*linha_512`) com validacao estrita interna.
  - Realizado code submit notebook-only conforme contrato da competicao (`ref=50317991`, pendente no momento do registro).
- Arquivos principais:
  - `PLANS.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission /tmp/kaggle_kernel_output_v42_1770840708/submission.csv` (OK)
  - Consulta de estado Kaggle API:
    - `python - <<'PY' ... competition_submissions('stanford-rna-3d-folding-2', page_size=20) ... PY` (confirmado `ref=50317991`, `status=PENDING`, sem erro de formato)
- Riscos/follow-ups:
  - `ref=50317991` ainda aguardando score publico final; registrar resultado assim que sair de `PENDING`.

## 2026-02-11 - marcusvinicius/Codex - PLAN-017

- Resumo:
  - Implementado refinamento de busca em `retrieve-templates` com duas etapas:
    - ranking coarse por `kmer + length_compatibility`;
    - rerank opcional por similaridade de alinhamento global (`alignment_similarity`) em pool limitado.
  - Evoluido `predict-tbm` para gerar variacoes deterministicas por candidato:
    - realinhamento com perfis de `gap penalties` configuraveis;
    - perturbacao pequena deterministica nas coordenadas projetadas;
    - selecao final top-`n_models` com suporte a diversidade mesmo com poucos templates unicos.
  - Expostos novos knobs CLI:
    - `retrieve-templates`: `--refine-pool-size`, `--refine-alignment-weight`, `--refine-open-gap-score`, `--refine-extend-gap-score`
    - `predict-tbm`: `--gap-open-scores`, `--gap-extend-scores`, `--max-variants-per-template`, `--perturbation-scale`
  - Adicionados testes unitarios para:
    - rerank de retrieval por alinhamento;
    - geracao de multiplos modelos TBM via variacoes de gap;
    - determinismo da perturbacao em TBM.
- Arquivos principais:
  - `src/rna3d_local/alignment.py`
  - `src/rna3d_local/retrieval.py`
  - `src/rna3d_local/tbm_predictor.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_retrieval_rerank.py`
  - `tests/test_tbm_coverage_selection.py`
  - `PLANS.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `python -m compileall -q src tests` (ok)
  - `pytest -q tests/test_retrieval_rerank.py tests/test_tbm_coverage_selection.py tests/test_template_workflow.py` (8 passed)
  - `pytest -q` (40 passed)
  - Experimento real PLAN-017: `runs/20260211_202637_plan017_tbm_fast_multitemplate` (pipeline completo sem OOM; score local `0.20890214285714293`)
- Riscos/follow-ups:
  - O baseline TBM multi-template ficou abaixo do melhor local atual; manter como linha de baseline robusta/rápida e seguir com frentes híbridas (TBM + RNAPro + blends) para ganho de score.

## 2026-02-11 - marcusvinicius/Codex - PLAN-018

- Resumo:
  - Implementado fluxo estrito de templates precomputados para RNAPro:
    - novo conversor `convert-templates-to-pt` para transformar submissao wide (`ID,resname,resid,x_i,y_i,z_i`) em `template_features.pt` por alvo;
    - novo modulo `src/rna3d_local/template_pt.py` com validacao fail-fast de contrato (colunas, ordem de residuos, targets faltantes/extras, valores nao-finitos, consistencia de modelos);
    - extensao de `predict-rnapro` para `--use-template ca_precomputed` com `--template-features-dir` e `--template-source`, sem fallback silencioso.
  - Mantido caminho legado de inferencia RNAPro (`--use-template none`) sem regressao.
  - Adicionados testes para:
    - conversao `.csv -> .pt` + inferencia precomputada;
    - falha quando falta target no conversor;
    - falha quando falta `template_features.pt` no infer.
  - Registrado novo plano `PLAN-018` em `PLANS.md`.
- Arquivos principais:
  - `PLANS.md`
  - `src/rna3d_local/template_pt.py`
  - `src/rna3d_local/rnapro/infer.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_template_pt.py`
- Validacao local executada:
  - `python -m compileall -q src tests` (ok)
  - `pytest -q tests/test_template_pt.py tests/test_template_workflow.py` (5 passed)
  - `pytest -q` (43 passed)
  - Smoke CLI end-to-end do novo fluxo:
    - `python -m rna3d_local convert-templates-to-pt --templates-submission runs/20260211_220500_plan018_template_pt_smoke/input/template_submission.csv --targets runs/20260211_220500_plan018_template_pt_smoke/input/targets.csv --out-dir runs/20260211_220500_plan018_template_pt_smoke/template_pt --n-models 2 --template-source tbm` (ok)
    - `python -m rna3d_local predict-rnapro --model-dir runs/20260211_220500_plan018_template_pt_smoke/model --targets runs/20260211_220500_plan018_template_pt_smoke/input/targets.csv --out runs/20260211_220500_plan018_template_pt_smoke/rnapro_precomputed.parquet --n-models 2 --min-coverage 0.5 --use-template ca_precomputed --template-features-dir runs/20260211_220500_plan018_template_pt_smoke/template_pt --template-source tbm` (ok)
    - `python -m rna3d_local export-submission --sample runs/20260211_220500_plan018_template_pt_smoke/input/sample_submission.csv --predictions runs/20260211_220500_plan018_template_pt_smoke/rnapro_precomputed.parquet --out runs/20260211_220500_plan018_template_pt_smoke/submission.csv` (ok)
    - `python -m rna3d_local check-submission --sample runs/20260211_220500_plan018_template_pt_smoke/input/sample_submission.csv --submission runs/20260211_220500_plan018_template_pt_smoke/submission.csv` (OK)
- Riscos/follow-ups:
  - O contrato `.pt` atual e interno do repositorio (pickle + arrays numpy); se houver integracao com stack externa que espere formato Torch especifico, adicionar adaptador de serializacao explicito mantendo fail-fast.
  - Integrar caminho opcional MMseqs2 como gerador nativo de templates precomputados para reduzir dependencia do fluxo TBM quando necessario.

## 2026-02-11 - marcusvinicius/Codex - ADHOC

- Resumo:
  - Atualizada politica de seguranca operacional em `AGENTS.md` para proibir download/copia/incorporacao de solucoes completas de terceiros no projeto.
  - Definida excecao explicita: permitido apenas download de modelos pre-treinados como insumo tecnico, com registro de origem/licenca e validacao local.
- Arquivos principais:
  - `AGENTS.md`
- Validacao local executada:
  - `rg -n "Seguranca operacional|terceiros|pre-treinados|pre-treinado" AGENTS.md` (ok; novas regras localizadas)
- Riscos/follow-ups:
  - Se houver necessidade de detalhar compliance de licencas por fonte/modelo, registrar procedimento operacional em documento dedicado.
