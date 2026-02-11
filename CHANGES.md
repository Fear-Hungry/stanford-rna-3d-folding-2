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
