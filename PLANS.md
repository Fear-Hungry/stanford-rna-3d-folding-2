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

## PLAN-025 - Harness de pesquisa automatizada para Kaggle (loop + gates)

- Objetivo: operacionalizar ciclo de pesquisa/experimentos para competicao Kaggle com verificacao obrigatoria e reproducibilidade.
- Escopo:
  - Estrutura leve de pesquisa (`research/`, `scripts/`, `configs/research/`) sem reorg profundo do core.
  - Novos comandos CLI:
    - `research-sync-literature`
    - `research-run`
    - `research-verify`
    - `research-report`
  - Gate estrito em `research-verify`:
    - solver status valido + checks de saida + reproducao;
    - bloco opcional `kaggle_gate` (contrato sample/submission, score minimo vs baseline, limites de tamanho/tempo).
  - Artefatos estruturados em `runs/research/...` com manifestos JSON e resultados parquet.
- Criterios de aceite:
  - Testes unitarios novos do harness verdes.
  - Smoke local `research-run -> research-verify -> research-report` concluido sem erro.
  - Documentacao no `README.md` e `research/README.md` com runbook minimo.

## PLAN-053 - Limpeza agressiva do core (sem legado/permissivo na CLI competitiva)

- Objetivo: reduzir superficie operacional ao core competitivo estrito, removendo caminhos legados/permissivos e comandos fora do fluxo oficial de competicao.
- Escopo:
  - Remover wrappers legados `data_access.py` e `memory.py`.
  - Endurecer `submit-kaggle`/`evaluate-robust`/`evaluate-submit-readiness`/`calibrate-kaggle-local` removendo flags `allow-*` e bypass de gate.
  - Tornar `--robust-report` e `--readiness-report` obrigatorios no `submit-kaggle`.
  - Remover comandos `research-*` da CLI principal.
  - Atualizar README para refletir API canonica (`bigdata.py`) e CLI competitiva estrita.
- Criterios de aceite:
  - `python -m rna3d_local --help` nao exibe comandos `research-*`.
  - `python -m rna3d_local submit-kaggle --help` nao exibe flags `allow-*`.
  - `src/rna3d_local/data_access.py` e `src/rna3d_local/memory.py` inexistentes.
  - Suite de testes de gate/CLI estrita verde.

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

## PLAN-004 - Otimizacao ampla de memoria RAM (pipeline big data)

- Objetivo: reduzir pico de RAM e evitar OOM no pipeline completo (`template_db`, `retrieval`, `predict-tbm`, `train-rnapro`, `predict-rnapro`, `ensemble`, `export`, `score`) sem alterar contratos funcionais.
- Escopo:
  - Introduzir memory-budget explicito e fail-fast padronizado.
  - Remover estruturas em memoria que escalam mal (`to_dicts` massivo, `set` gigante de chaves, listas globais de rows).
  - Adotar processamento incremental/chunks e escrita incremental para preditos longos.
  - Expor parametros de memoria na CLI (budget/chunk-size).
- Criterios de aceite:
  - Comandos principais aceitam `--memory-budget-mb` e aplicam validacao fail-fast.
  - `ensemble` e `export` validam consistencia de chaves via operacoes tabulares (anti-join), sem construir sets completos.
  - `predict-tbm` e `predict-rnapro` evitam acumular todas as linhas em lista unica global.
  - Testes unitarios existentes continuam passando e novos testes cobrem comportamento de guardrails de memoria.

## PLAN-005 - Labels canonicos em Parquet + leitura lazy/streaming

- Objetivo: reduzir picos de RAM convertendo `train_labels.csv` para Parquet particionado e consumindo esse artefato em `template_db`, `train_rnapro` e `export_train_solution_for_targets` via leitura lazy.
- Escopo:
  - Novo comando CLI para preparar labels canonicos em Parquet particionado.
  - Priorizar leitura `scan_parquet` quando `train_labels_parquet_dir` for fornecido (sem fallback silencioso em caso de erro).
  - Manter compatibilidade explicita com caminho CSV legado.
  - Documentar fluxo recomendado no `README.md`.
- Criterios de aceite:
  - `python -m rna3d_local prepare-labels-parquet` gera `manifest.json` + `part-*.parquet`.
  - `build-template-db`, `train-rnapro` e `export-train-solution` aceitam `--train-labels-parquet-dir`.
  - Em caso de diretorio parquet invalido/sem partes, o fluxo falha com erro acionavel no padrao do repositorio.
  - Suite de testes cobre caminho de sucesso com Parquet e falha sem fallback silencioso.

## PLAN-006 - API unica de acesso a dados grandes para modulos consumidores

- Objetivo: deixar o consumo por outros modulos de treino simples e padronizado, centralizando leitura de labels/tabelas grandes (Parquet particionado, lazy e streaming) em uma API interna unica.
- Escopo:
  - Criar `src/rna3d_local/data_access.py` com helpers reutilizaveis para:
    - resolver labels (`train_labels_parquet_dir` vs CSV explicito) sem fallback silencioso;
    - leitura lazy de tabelas com projection/filtro;
    - coleta streaming padronizada.
  - Refatorar consumidores diretos (`datasets`, `template_db`, `rnapro/train`) para usar essa API.
  - Aplicar a API tambem em caminhos de tabelas grandes (`retrieval`, `tbm_predictor`, `ensemble`, `export`) onde couber sem mudar contratos.
  - Adicionar testes de API central e de no-fallback.
- Criterios de aceite:
  - Existe um ponto unico de resolucao de labels e leitura lazy em `data_access.py`.
  - `build-template-db`, `train-rnapro`, `export-train-solution` e `build-train-fold` usam esse ponto unico.
  - Fluxo com `data/derived/train_labels_parquet/` esta documentado e validado em testes.
  - Suite de testes permanece verde.

## PLAN-007 - Streaming sink no export de solution + projecao lazy em pontos ainda eager

- Objetivo: reduzir pico de RAM no caminho de criacao de `solution.parquet` e eliminar leituras eager remanescentes de tabelas grandes em `datasets`/`cli`.
- Escopo:
  - Trocar `collect(...).write_parquet(...)` por `sink_parquet(...)` no `export_train_solution_for_targets`.
  - Corrigir derivacao de `target_id` a partir de `ID` usando regex de prefixo antes do ultimo `_`.
  - Aplicar leitura lazy/projetada para:
    - `targets.parquet` (selecionar apenas `target_id`, `fold_id`);
    - `train_sequences.csv` (selecionar apenas `target_id`, `sequence`).
  - Cobrir em teste caso de `target_id` contendo underscore.
- Criterios de aceite:
  - `export-train-solution` continua funcional e passa nos testes existentes.
  - Novo teste valida extração correta de `target_id` com underscore.
  - Comparativo com `/usr/bin/time -v` no `fold2` mostra redução significativa de `Maximum resident set size` no `export-train-solution` em relação à versão anterior.
  - Suite de testes permanece verde.

## PLAN-008 - Remocao de codigo legado de labels CSV e contrato unico em Parquet

- Objetivo: eliminar caminhos legados de labels CSV nos consumidores de pipeline (`build-train-fold`, `export-train-solution`, `build-template-db`, `train-rnapro`), mantendo apenas labels canonicos em Parquet com fail-fast.
- Escopo:
  - Remover fallback/dual-path CSV da API central de dados (`scan_labels`).
  - Simplificar assinaturas de `datasets`, `template_db` e `rnapro/train` para aceitar somente `train_labels_parquet_dir`.
  - Remover flags CLI legadas (`--train-labels`, `--train-labels-csv`) dos comandos consumidores e tornar `--train-labels-parquet-dir` obrigatoria.
  - Atualizar testes para o novo contrato unico e validar comportamento fail-fast quando o parquet estiver invalido.
  - Atualizar README com exemplos sem caminhos legados.
- Criterios de aceite:
  - Nenhum consumidor de labels usa CSV como entrada operacional (exceto `prepare-labels-parquet`, que e conversao one-shot).
  - `python -m pytest -q` permanece verde.
  - Help da CLI para `build-train-fold` e `export-train-solution` nao exibe mais flags CSV legadas.
  - Benchmark rapido por fold com `build-train-fold` em labels parquet registra pico de RAM para validacao operacional.

## PLAN-009 - Modulo unico de Big Data reutilizavel

- Objetivo: consolidar as praticas de big data (lazy scan, streaming collect, sink parquet particionado e guardrails de memoria) em um unico modulo reutilizavel para novos experimentos e treinamentos.
- Escopo:
  - Criar `src/rna3d_local/bigdata.py` como API canonica.
  - Migrar consumidores do pipeline para importar diretamente de `bigdata.py`.
  - Manter `data_access.py` e `memory.py` apenas como wrappers de compatibilidade sem logica propria.
  - Atualizar testes para validar o uso do modulo unico.
  - Documentar no `README.md` o ponto unico oficial.
- Criterios de aceite:
  - API de big data centralizada em `rna3d_local.bigdata`.
  - Sem imports diretos de `data_access`/`memory` nos modulos de pipeline.
  - `python -m compileall src` e `python -m pytest -q` verdes.

## PLAN-011 - Baseline/benchmark do Artigo 1 (runbook reprodutivel + comparacao)

- Objetivo: formalizar o **Artigo 1** como baseline e benchmark operacional do repositorio, com um runbook reprodutivel para rodar o pipeline completo (template-aware + RNAPro proxy + ensemble), gerar submissao e medir score local identico ao Kaggle.
- Escopo:
  - Criar arquivo dedicado `benchmarks/ARTICLE1_BASELINE.md` contendo:
    - pre-requisitos (dados oficiais + metric vendorizado);
    - formato do `external_templates.csv` (colunas obrigatorias);
    - configuracao baseline (valores default + seeds fixas);
    - runbook para `public_validation` e para `train_cv` (por fold) usando `rna3d_local score --per-target`;
    - checklist do que registrar em `EXPERIMENTS.md` e artefatos esperados em `runs/`.
  - Atualizar `README.md` para declarar explicitamente Artigo 1 como baseline/benchmark e apontar para o runbook.
- Criterios de aceite:
  - `benchmarks/ARTICLE1_BASELINE.md` existe e permite rodar o baseline de ponta-a-ponta sem decisao aberta.
  - `README.md` referencia o benchmark do Artigo 1 e deixa claro como comparar solucoes novas contra o baseline.

## PLAN-010 - Score streaming por target + ordenacao canonica da solution por fold

- Objetivo: evitar travamento por RAM no score de folds grandes, mantendo contrato estrito e sem fallback silencioso.
- Escopo:
  - Refatorar `score_submission` para leitura em lotes (CSV/Parquet) e processamento por `target_id`, sem carregar tabela inteira em memoria.
  - Adicionar guardrails explicitos de `chunk_size` e `max_rows_in_memory` no caminho de score.
  - Endurecer contratos para validar ordem de chaves e duplicatas tambem na solucao.
  - Garantir que `export_train_solution_for_targets` escreva `solution.parquet` em ordem canonica de fold (mesma ordem de targets + `resid`) para compatibilidade com validacao estrita.
- Criterios de aceite:
  - `python -m pytest -q` verde.
  - `build-train-fold` do fold critico (`fold2`) conclui com budget de memoria limitado (`--memory-budget-mb 8192`) sem OOM.
  - `score` aceita e aplica `--chunk-size` e `--max-rows-in-memory`.
  - Benchmark completo por folds (public + fold0..4) iniciado com evidencias de RAM controlada e logs em `runs/`.

## PLAN-012 - Rerank de templates + treino RNAPro maior (feature_dim=512)

- Objetivo: aumentar qualidade do baseline com duas frentes sequenciais: (A) rerank de candidatos template-aware por sinal estrutural/coverage, (B) treino RNAPro maior mantendo estabilidade operacional de memoria.
- Escopo:
  - Implementar rerank com prioridade para cobertura efetiva (coverage) e similaridade nos caminhos `predict-tbm` e `predict-rnapro`, evitando selecao "primeiro valido".
  - Adicionar sinal de compatibilidade de comprimento no retrieval (`retrieve-templates`) para ranking mais robusto em alvos longos/curtos.
  - Expor knobs de rerank na CLI para experimentacao controlada (sem modo permissivo).
  - Treinar RNAPro com `feature_dim=512` localmente (GPU local quando aplicavel), mantendo `memory_budget_mb` explicito.
  - Medir impacto local com score oficial e registrar comparacao baseline vs novo em `EXPERIMENTS.md`.
- Criterios de aceite:
  - `python -m pytest -q` verde com cobertura de ao menos um cenario de rerank.
  - Pipeline segue fail-fast em contratos e sem fallback silencioso.
  - Existe pelo menos um experimento append-only com `feature_dim=512` e score local comparativo registrado.
  - Nenhuma etapa critica excede budget de memoria configurado no experimento reportado.

## PLAN-013 - Alinhamento Biopython no TBM/RNAPro + ensemble dinamico por cobertura

- Objetivo: elevar qualidade de predicao no hidden/public mantendo estabilidade operacional, trocando alinhamento heuristico por alinhamento global com Biopython e adicionando blending dinamico guiado por cobertura.
- Escopo:
  - Substituir `difflib.SequenceMatcher` por `Bio.Align.PairwiseAligner` em `alignment.py` (modo global, gap affine, deterministico).
  - Melhorar `project_target_coordinates` para interpolacao/extrapolacao linear entre residuos ancora alinhados (removendo drift fixo por delta).
  - Adicionar modo opcional de ensemble dinamico por cobertura (`ensemble-predict`) com validacoes estritas e knobs CLI.
  - Cobrir o novo comportamento com testes unitarios dedicados (alinhamento + ensemble dinamico).
- Criterios de aceite:
  - `python -m pytest -q` verde com novos testes de alinhamento e ensemble.
  - `predict-tbm`/`predict-rnapro` continuam fail-fast em contrato sem fallback silencioso.
  - `ensemble-predict` aceita modo dinamico por cobertura sem quebrar modo estatico legado.
  - Dependencia de alinhamento (`biopython`) declarada no projeto.

## PLAN-014 - Calibracao do alinhamento Biopython (match-only) para recuperar score

- Objetivo: remover inflacao artificial de cobertura causada por mismatch no alinhamento global, usando mapeamento estrito por matches reais para elevar qualidade de TBM/RNAPro.
- Escopo:
  - Ajustar `map_target_to_template_positions` para construir mapping a partir de `alignment.coordinates`, mapeando apenas posicoes com mesma base (`target[pos] == template[pos]`).
  - Manter comportamento fail-fast sem fallback silencioso em todos os consumidores (`predict-tbm`, `predict-rnapro`).
  - Adicionar testes unitarios para garantir que mismatch nao conta como cobertura.
  - Rodar benchmark comparativo local (score estatico) versus baseline local vigente (`0.23726392857142856`).
- Criterios de aceite:
  - `python -m pytest -q` verde.
  - Experimento local append-only em `EXPERIMENTS.md` com score comparativo e custo.
  - Se score local superar baseline vigente, variante fica elegivel para notebook submit; caso contrario, bloqueada por gating.

## PLAN-015 - Escala RNAPro (feature_dim=768) + calibracao de blend no melhor artefato

- Objetivo: buscar novo ganho de score partindo da melhor linha validada (`PLAN-012`, score local `0.23726392857142856`, score publico `0.268`) sem abrir risco operacional de OOM.
- Escopo:
  - Rodar sweep curto de pesos TBM/RNAPro usando os artefatos vencedores ja gerados em `runs/20260211_154539_plan012_rerank_bigmodel` para medir teto rapido de melhoria local.
  - Treinar novo RNAPro com `feature_dim=768` mantendo mesmo perfil operacional estavel (`memory_budget_mb=8192`, `max_rows_in_memory` explicito, chunking controlado).
  - Rodar pipeline completo de inferencia/ensemble/export/check/score com o modelo 768 e comparar contra baseline local vigente.
  - Aplicar gating estrito de submissao: so elegivel se `local_score_novo > 0.23726392857142856`.
- Criterios de aceite:
  - Experimentos append-only em `EXPERIMENTS.md` com comandos, custos e comparacao objetiva baseline vs novo.
  - Nenhuma etapa critica do pipeline excede budget de memoria configurado.
  - Submit para Kaggle somente se houver melhoria local comprovada e validacao local estrita aprovada.

## PLAN-016 - Escala RNAPro para feature_dim=1024 + blend com diversidade de submissao

- Objetivo: buscar ganho adicional de score local sobre o melhor ponto atual via aumento de capacidade do RNAPro e blend entre artefatos de linhas fortes.
- Escopo:
  - Treinar `train-rnapro` com `feature_dim=1024` mantendo guardrails de memoria (`memory_budget_mb=8192`, `max_rows_in_memory` explicito).
  - Rodar `predict-rnapro` com o modelo 1024 e combinar com TBM forte (`PLAN-012`) via sweep curto de pesos.
  - Avaliar blend em nivel de submissao entre as melhores linhas ja validadas para explorar diversidade.
  - Aplicar gating estrito de submit (somente com melhoria local objetiva sobre o melhor score local vigente).
- Criterios de aceite:
  - Score local comparativo registrado com comandos/artefatos em `EXPERIMENTS.md`.
  - Nenhuma regressao de contrato em `check-submission`.
  - Nenhum OOM durante treino/inferencia/score no perfil operacional atual.

## PLAN-017 - TBM baseline rapido multi-template com busca refinada e variacoes deterministicas

- Objetivo: implementar exatamente o baseline TBM rapido solicitado, com 5 candidatos por alvo, fortalecendo a busca por templates com refinamento por alinhamento e diversidade controlada sem fallback silencioso.
- Escopo:
  - Manter `build-template-db` como fonte unica (treino + externos publicos) e validar o fluxo com base real.
  - Evoluir `retrieve-templates` para ranking em duas etapas:
    - etapa 1: k-mer + compatibilidade de comprimento (coarse ranking);
    - etapa 2: refinamento por score de alinhamento global (Biopython) sobre pool limitado.
  - Evoluir `predict-tbm` para gerar diversidade controlada por candidato:
    - variacoes de realinhamento com diferentes `gap penalties`;
    - pequenas perturbacoes deterministicas nas coordenadas projetadas;
    - selecao final top-`n_models` (meta padrao: 5 por alvo) por cobertura/similaridade sem quebrar contrato.
  - Expor knobs na CLI para experimento rapido (`retrieve-templates` e `predict-tbm`) mantendo defaults retrocompativeis.
  - Cobrir com testes unitarios os novos comportamentos (rerank por alinhamento, variantes de gap e determinismo de perturbacao).
  - Rodar experimento local objetivo com score e registrar em `EXPERIMENTS.md`.
- Criterios de aceite:
  - `pytest` verde com novos testes.
  - `predict-tbm --n-models 5` consegue gerar 5 modelos por alvo quando houver candidatos/variantes suficientes, sem fallback.
  - Novo ranking de retrieval reflete refinamento por alinhamento quando habilitado.
  - Experimento local append-only com comandos, score e custos em `EXPERIMENTS.md`.

## PLAN-018 - Templates compativeis com RNAPro (`submission.csv` -> `template_features.pt` -> `predict-rnapro`)

- Objetivo: integrar no repositorio um fluxo estrito para consumir templates precomputados em formato `.pt`, com suporte operacional no notebook Kaggle (notebook-only) e sem fallback silencioso.
- Escopo:
  - Adicionar conversor oficial de templates para `.pt` a partir de `submission.csv`/Parquet no formato canônico.
  - Estender `predict-rnapro` com `--use-template ca_precomputed` e `--template-features-dir`.
  - Definir contrato estrito do artefato `.pt` (shape, chaves, tipos, cobertura minima, target_id).
  - Manter modo legado (`--use-template none`) sem regressao.
  - Cobrir sucesso e falha de contrato com testes unitarios/integração.
  - Rodar validacao local objetiva (pytest + smoke de conversao/inferencia) e registrar em `CHANGES.md`/`EXPERIMENTS.md`.
- Criterios de aceite:
  - Existe comando CLI `convert-templates-to-pt` funcional.
  - `predict-rnapro --use-template ca_precomputed` gera preditos long validos sem usar fallback.
  - Falhas de contrato em `.pt` ou no CSV de entrada encerram com erro no padrao do `AGENTS.md`.
  - `check-submission` e testes locais permanecem verdes apos a integracao.

## PLAN-019 - Gerador alternativo DRfold2 local (segunda opiniao) + blend com gating estrito

- Objetivo: adicionar uma segunda branch de predicao estrutural com distribuicao de erros diferente (DRfold2 local), para testar ganho de ensemble sem depender de treino Kaggle.
- Escopo:
  - Integrar comando local `predict-drfold2` para:
    - executar inferencia DRfold2 por alvo (ate 5 modelos);
    - converter PDBs finais para formato long canônico do pipeline (`ID,resid,resname,model_id,x,y,z` + metadata).
  - Validar contratos de saida em modo estrito:
    - modelos esperados presentes;
    - comprimento e resname compativeis com a sequencia alvo;
    - ausencia de coordenadas nao-finitas.
  - Rodar benchmark local em `public_validation`:
    - score DRfold2 isolado;
    - score em blend com melhor linha local vigente.
  - Aplicar regra de submit:
    - somente elegivel se score local final superar baseline vigente.
  - Relax com OpenMM permanece opcional e fora do caminho padrao inicial (apenas se houver ganho claro antes).
- Criterios de aceite:
  - `python -m rna3d_local predict-drfold2 ...` gera parquet long valido e passa `export-submission` + `check-submission`.
  - Experimento local completo (comandos, custo, score) registrado em `EXPERIMENTS.md`.
  - Decisao de submit documentada com comparacao objetiva contra o melhor score local anterior.

## PLAN-020 - DRfold2 C1' first no extrator de coordenadas (compatibilidade com metrica)

- Objetivo: eliminar risco de incompatibilidade de representacao no DRfold2 garantindo prioridade de extracao no atomo `C1'`.
- Escopo:
  - Alterar `extract_target_coordinates_from_pdb` para priorizar `C1'` antes de `C4'`.
  - Manter fallback atual de atomos para nao ampliar escopo funcional nesta etapa.
  - Adicionar teste unitario cobrindo preferencia explicita por `C1'` quando coexistir com `C4'`.
  - Manter cobertura de regressao para cenario sem `C1'`.
- Criterios de aceite:
  - `pytest -q tests/test_drfold2_parser.py` verde.
  - Caso com `C1'` e `C4'` no mesmo residuo retorna coordenadas de `C1'`.
  - Nao ha regressao no cenario de fallback quando `C1'` estiver ausente.

## PLAN-021 - Mapeamento hibrido + projecao aprimorada + QA treinado leve (TBM/RNAPro)

- Objetivo: melhorar consistencia estrutural e diversidade de candidatos no pipeline local, substituindo filtro estrito por matches exatos por mapeamento hibrido, adicionando selecao final com QA treinado leve e diversidade.
- Escopo:
  - Evoluir `alignment.py` com:
    - `map_target_to_template_alignment` (metadados de match/mismatch e modos `strict_match|hybrid|chemical_class`);
    - `project_target_coordinates` com `projection_mode` (`target_linear|template_warped`).
  - Integrar novos modos em `predict-tbm` e `predict-rnapro`.
  - Implementar modulo `qa_ranker.py` (ridge deterministico + validacao por grupos + score de candidatos + selecao greedy com penalidade de redundancia).
  - Expor novos knobs CLI:
    - `predict-tbm`/`predict-rnapro`: `--mapping-mode`, `--projection-mode`, `--qa-model`, `--qa-top-pool`, `--diversity-lambda`;
    - novo comando `train-qa-ranker`.
  - Atualizar manifests de inferencia com parametros e hash do modelo QA quando aplicavel.
  - Cobrir com testes unitarios de alinhamento/QA e manter regressao verde da suite.
- Criterios de aceite:
  - `pytest -q` verde.
  - `predict-tbm` e `predict-rnapro` funcionam com defaults novos sem fallback silencioso.
  - `train-qa-ranker` gera `qa_model.json` valido e consumivel na inferencia.
  - Manifests registram configuracao de mapeamento/projecao/QA para rastreabilidade.

## PLAN-022 - Ensemble ortogonal por cobertura (TBM/RNAPro + DRfold2) com sweep de substituicao por alvo

- Objetivo: buscar salto de score local combinando a linha baseline vigente com uma branch ortogonal (DRfold2), substituindo apenas alvos de baixa confianca template-aware para reduzir risco de regressao global.
- Escopo:
  - Reusar baseline local vigente (`submission_hybrid_short7.csv`) e artefatos TBM para calcular confianca por alvo (coverage/similarity agregadas).
  - Gerar predicoes DRfold2 para todos os alvos do `test_sequences.csv` com `--reuse-existing-targets` a partir do work-dir ja aquecido do PLAN-019.
  - Construir sweep deterministico de substituicao por alvo (top-N piores por coverage): `N in {7,10,14,18,21,28}`.
  - Em cada variante:
    - patch estrito da submissao wide (5 modelos) somente para IDs dos alvos selecionados;
    - validacao `check-submission` obrigatoria;
    - score local em `data/derived/public_validation`.
  - Selecionar candidato final apenas se superar o melhor local vigente com margem objetiva.
  - Elegibilidade de submit permanece bloqueada sem melhora local e sem validacao estrita.
- Criterios de aceite:
  - `predict-drfold2` completo para os 28 alvos do teste sem erro de contrato.
  - Todas as variantes do sweep passam `check-submission`.
  - `EXPERIMENTS.md` recebe registro append-only com:
    - ranking por variante (`N`, score, delta vs baseline),
    - custo (tempo/maxrss) do DRfold2 e scoring,
    - decisao explicita de submit (sim/nao) pelo gating.

## PLAN-023 - Proxy local robusto anti-leak + selecao de candidatos para alvo >0.30

- Objetivo: reduzir o gap entre score local e Kaggle criando um proxy local mais fiel (anti-leak por fold) e usar esse proxy para selecionar apenas linhas com potencial real de ganho de leaderboard.
- Escopo:
  - Montar validacao anti-leak por fold (inicialmente folds `3` e `4`):
    - excluir targets de holdout do treino de templates e do treino RNAPro;
    - rodar pipeline completo no holdout (`retrieve -> predict-tbm -> train/predict-rnapro -> ensemble -> export -> check -> score`).
  - Comparar configuracoes candidatas no proxy anti-leak:
    - `TBM strict` (baseline de referencia),
    - `TBM strict + RNAPro` (pesos calibrados),
    - `TBM strict + patch DRfold2` em targets de baixa coverage.
  - Definir regra de promocao para submit Kaggle:
    - promover apenas candidatos com melhora consistente no proxy anti-leak (media folds + worst-fold) e melhora estrita no score local oficial.
  - Registrar matriz de resultados (por fold/config) com custo e rastreabilidade em `runs/` + `EXPERIMENTS.md`.
- Criterios de aceite:
  - Pipeline anti-leak concluido sem fallback silencioso em pelo menos 2 folds (`3` e `4`), com `check-submission` OK.
  - Tabela comparativa por configuracao com metricas por fold e agregado (`mean`, `min`, `delta`).
  - Definicao objetiva de candidato promovido/bloqueado para submit conforme gating.
  - Registro append-only completo em `EXPERIMENTS.md` no mesmo dia (UTC).

## PLAN-024 - Sweep TBM estrito no public_validation + patch seletivo por alvo (DRfold2)

- Objetivo: maximizar ganho local imediato com menor risco operacional, partindo da evidencia anti-leak de que `TBM strict` e superior ao blend fixo com RNAPro.
- Escopo:
  - Rodar baseline `TBM strict` no `public_validation` com validacao estrita (`check-submission`) e score local com quebra por alvo (`--per-target`).
  - Comparar contra melhor baseline local vigente (`0.2443675`) para decidir elegibilidade de submit.
  - Se `TBM strict` nao superar o melhor local, executar patch seletivo com DRfold2 apenas nos alvos de pior confianca template-aware (baixa similaridade/cobertura) e testar variantes deterministicas.
  - Registrar ranking por variante e decisao de submit sob gating estrito (somente melhora local estrita).
- Criterios de aceite:
  - Pelo menos 1 candidato novo completo (`predict -> export -> check -> score`) com artefatos em `runs/`.
  - `score.json` e `per_target_scores.csv` gerados para comparacao objetiva.
  - `EXPERIMENTS.md` atualizado com comandos, metricas e conclusao.

## PLAN-025 - Melhoria generica sem patch por target_id (TBM/RNAPro mapping+blend sweep)

- Objetivo: buscar ganho robusto para hidden leaderboard evitando dependencia de patch por IDs especificos do public, avaliando apenas configuracoes genericas reproduziveis no notebook.
- Escopo:
  - Reproduzir pipeline completo local (retrieve -> predict-tbm -> predict-rnapro 512/768 -> ensemble -> export/check/score) sem overlay por `target_id`.
  - Varrer configuracoes de `mapping_mode` e blends:
    - `mapping_mode in {hybrid, strict_match}` para TBM e RNAPro;
    - pesos de ensemble por linha (512/768) e `alpha` de blend final entre as linhas.
  - Aplicar gating estrito:
    - promover somente candidato com melhora local estrita vs melhor vigente (`0.2839053571428572`);
    - bloquear submit se houver regressao ou empate.
  - Se houver candidato promovido, preparar notebook-only submit sem patches por IDs.
- Criterios de aceite:
  - Pelo menos 3 candidatos genericos completos avaliados com `check-submission` + `score`.
  - Tabela comparativa com ranking e delta vs melhor local vigente.
  - Registro append-only em `EXPERIMENTS.md` e `CHANGES.md` com decisao final de submit/bloqueio.

## PLAN-027 - Notebook hidden-safe com patch generico >0.30 + Biopython local por dataset

- Objetivo: superar score local `0.30` com fluxo de submissao notebook-only robusto ao hidden rerun, sem patch hardcoded por IDs do public e sem dependencia implícita de pacote do ambiente.
- Escopo:
  - Derivar candidato local >0.30 a partir de regra generica reproduzivel.
  - Corrigir notebook para tolerar variacoes do hidden dataset e evitar erro de rerun por mismatch de patch.
  - Empacotar `biopython` como wheel em dataset Kaggle e instalar localmente no notebook (`--no-index`).
  - Aplicar gating estrito: `check-submission` + `score` local > melhor score local vigente antes de submit.
- Criterios de aceite:
  - Notebook executa `COMPLETE` sem `error_description` de rerun no submit candidato.
  - Output do notebook passa `check-submission` local.
  - `score` local do output do notebook > `0.2839053571428572`.
  - Submissao Kaggle criada apenas apos validacao local estrita.

## PLAN-028 - Poolscan de candidatos historicos + teto de oracle por alvo

- Objetivo: medir o teto real da familia atual de candidatos (TBM/RNAPro/patches) antes de investir mais compute, evitando novos submits cegos com pouca chance de ganho.
- Escopo:
  - Revalidar candidatos historicos com `check-submission` e `score --per-target` no `public_validation`.
  - Consolidar `score.json` + `per_target.csv` em um run unico para comparacao direta.
  - Calcular oracle por alvo (melhor candidato por target no public) apenas como teto diagnostico, sem uso para submit.
  - Decidir continuidade:
    - se teto ficar proximo do melhor atual (`~0.31`), abrir linha nova de geracao (maior ortogonalidade);
    - se teto subir fortemente, priorizar seletor/reranker.
- Criterios de aceite:
  - Todos os candidatos do scan passam `check-submission`.
  - Tabela final com score e delta vs melhor local vigente.
  - Conclusao objetiva registrada em `EXPERIMENTS.md` com decisao de proxima frente.

## PLAN-029 - DRfold2 full + blend estrito com baseline >0.31 (foco em salto)

- Objetivo: buscar ganho estrutural relevante (meta intermediaria `>0.33`, alvo final `>0.35`) completando DRfold2 em todos os alvos publicos e combinando com a melhor baseline local atual.
- Escopo:
  - Garantir ambiente DRfold2 local valido (pesos + binarios) e completar inferencia para `test_sequences` completo (`n_models=5`, `reuse-existing-targets`).
  - Gerar predicoes long DRfold2 e montar variantes de blend/patch sobre baseline vigente (`PLAN-027 pos_qac`):
    - substituicao total por alvo (`alpha=1.0`);
    - blends parciais (`alpha in {0.25, 0.50, 0.75}`);
    - subset generico por criterio reproduzivel (ex.: comprimento/coverage/confidence), sem hardcode de IDs do public.
  - Validar cada variante com `check-submission` e `score` local.
  - Aplicar gating de submit:
    - so promover candidato com melhora local estrita sobre o melhor oficial em `EXPERIMENTS.md`;
    - bloquear submit em empate/regressao.
- Criterios de aceite:
  - `predict-drfold2` completo no conjunto alvo sem erro de contrato.
  - Pelo menos 4 variantes completas avaliadas (com score local e custo).
  - Registro append-only em `EXPERIMENTS.md` com ranking e decisao final (submeter/bloquear).

## PLAN-030 - QA-ranker supervisionado em escala + pool amplo TBM/RNAPro (local-only)

- Objetivo: aumentar o teto da familia interna (sem dependencias externas novas) treinando um reranker QA mais robusto e ampliando a diversidade do pool de candidatos para selecao das 5 estruturas finais.
- Escopo:
  - Construir dataset QA supervisionado maior (folds locais, sem leak), com mais targets e features completas (`QA_FEATURE_NAMES`).
  - Treinar `qa_model.json` com validacao por grupos e rastreabilidade completa de metricas.
  - Rodar inferencia no `public_validation` com sweep controlado:
    - `mapping_mode in {hybrid, chemical_class}`,
    - `qa_top_pool in {40, 80, 120}`,
    - `diversity_lambda in {0.05, 0.10, 0.15, 0.25}`,
    - opcionalmente combinar com RNAPro (blends fixos e dinamicos por coverage).
  - Validar cada candidato com `check-submission` e `score` local; promover apenas melhora estrita.
- Criterios de aceite:
  - Pelo menos 6 candidatos completos avaliados com score local.
  - `EXPERIMENTS.md` com ranking, custo e analise de ganho/perda por alvo.
  - Candidato promovido somente se `score_local` superar estritamente o melhor oficial vigente.

## PLAN-031 - Expansao do pool com submissões historicas sem score publico + reselecao

- Objetivo: aumentar o teto do ensemble por alvo avaliando submissões globais que ainda nao foram pontuadas no `public_validation`, para tentar elevar a fronteira acima de `0.35`.
- Escopo:
  - Identificar `submission*.csv` globais pendentes de score local publico (excluindo folds/smokes/template-only).
  - Rodar `check-submission` + `score --per-target` para cada candidato valido.
  - Recalcular oracle do pool ampliado e treinar/ajustar seletor generico (regras por features de sequencia).
  - Materializar nova submissao candidata e validar estritamente (`check + score`).
- Criterios de aceite:
  - Pelo menos 8 submissões historicas globais novas pontuadas no `public_validation`.
  - Oracle recalculado e documentado com comparacao contra teto anterior.
  - Se houver melhora local estrita do candidato final, marcar elegibilidade para submit notebook-only.

## PLAN-032 - Calibracao local-vs-Kaggle e gate estrito anti-empate

- Objetivo: reduzir submit cego causado por descolamento entre `score` local e `public_score` Kaggle, e reforcar o contrato de promocao com melhora estrita.
- Escopo:
  - Implementar calibracao baseada em historico real de submissoes Kaggle:
    - extrair pares (`local_score` do texto da submissao, `public_score` Kaggle),
    - gerar relatorio com deltas e estimativas (mediana/p10/pior caso observado).
  - Expor comando CLI para gerar relatorio de calibracao e estimar score publico esperado para um candidato local.
  - Endurecer gating de submit local:
    - baseline passa a exigir melhora estrita (`candidate > baseline + min_improvement`),
    - empate com baseline deve ser bloqueado por padrao.
  - Cobrir com teste de regressao para bloqueio em empate.
- Criterios de aceite:
  - Novo comando `calibrate-kaggle-local` funcional e gerando JSON em `runs/kaggle_calibration/`.
  - `submit-kaggle` passa a aceitar `--min-improvement` e bloquear empate quando baseline informado.
  - Teste automatizado cobrindo bloqueio de empate verde (`pytest`).

## PLAN-033 - Gate calibrado conservador + submit notebook-only por contrato

- Objetivo: alinhar decisao de submit com comportamento real do Kaggle usando historico empirico e garantir contrato notebook-only em producao.
- Escopo:
  - Estender calibracao com diagnosticos adicionais de alinhamento (`pearson`, `spearman`, ajuste linear) e estimativas multiplas (`median`, `p10`, `worst_seen`, `linear_fit`).
  - Implementar decisao calibrada de submit baseada em baseline publico:
    - gate por estimativa conservadora configuravel (default `p10`);
    - exigencia minima de pares historicos (`min_pairs`) para evitar decisao fraca.
  - Integrar gate calibrado no `submit-kaggle` (alem do gate local estrito existente).
  - Forcar submit notebook-only em CLI (`--notebook-ref`, `--notebook-version`, `--notebook-file`) e bloquear caminho `CreateSubmission` local por arquivo.
  - Cobrir decisao calibrada por testes unitarios.
- Criterios de aceite:
  - `calibrate-kaggle-local` gera relatorio com `alignment_decision` quando `--local-score` e `--baseline-public-score` forem informados.
  - `submit-kaggle` usa fluxo notebook-only e aplica gate calibrado quando `--baseline-public-score` for informado.
  - Testes de calibracao/decisao (`pytest`) verdes.

## PLAN-034 - Avaliacao robusta multi-score + gating integrado para promocao

- Objetivo: promover candidatos com criterio mais robusto que um unico `score` local, combinando agregacao conservadora de scores e calibracao local->public.
- Escopo:
  - Adicionar comando de avaliacao robusta que consome multiplos `score.json` nomeados (`name=path`) e produz relatorio unico.
  - Definir `robust_score` conservador usando componentes:
    - `p25` dos scores informados;
    - `public_validation` (quando presente);
    - media de CV (`name` com prefixo `cv:`), quando presente.
  - Aplicar gate de melhora estrita em `robust_score` (`baseline_robust_score + min_robust_improvement`).
  - Aplicar opcionalmente gate calibrado Kaggle (`baseline_public_score`, `method`, `min_pairs`) no mesmo relatorio.
  - Integrar `submit-kaggle` para aceitar `--robust-report` e bloquear submit quando `allowed=false`.
- Criterios de aceite:
  - Novo comando `evaluate-robust` funcional e gerando JSON em `runs/`.
  - `submit-kaggle --robust-report <...>` bloqueia submit quando relatorio robusto reprovar.
  - Testes unitarios cobrindo agregacao e gate robusto verdes.

## PLAN-035 - Pool expansion completo + sintetese de candidato para alvo >0.35

- Objetivo: maximizar chance de atingir `score` local esperado `>0.35` com evidencias, evitando recombinacoes cegas de um pool limitado.
- Escopo:
  - Fase 1 (diagnostico de teto):
    - descobrir e pontuar todas as submissões globais ainda sem `score` em `public_validation` (`check-submission` + `score` estritos);
    - recalcular oracle por alvo no pool ampliado e registrar teto alcançavel com os candidatos existentes.
  - Fase 2 (novo candidato):
    - se teto do pool continuar abaixo de `0.35`, gerar candidato novo ortogonal (frente nova de geracao) e pontuar;
    - comparar contra melhor local atual e contra gate robusto/calibrado.
  - Promocao:
    - candidato so e elegivel se houver melhora estrita em `score` local e `evaluate-robust` aprovado.
- Criterios de aceite:
  - Inventario de candidatos globais atualizado com status `scored/skipped/failed`.
  - `oracle_mean` recalculado no pool ampliado com artefato rastreavel em `runs/`.
  - Pelo menos 1 candidato novo (nao apenas recombinacao de pool antigo) pontuado e comparado.

## PLAN-036 - Gate de entropia (seletores existentes) para subir score com menor complexidade

- Objetivo: melhorar estritamente o melhor score local atual (`0.3620110714285714`) com uma regra simples e mais robusta que seletor profundo, mantendo expectativa local `>0.35`.
- Escopo:
  - Reavaliar candidatos ja pontuados no `public_validation` e treinar seletor guloso de baixa complexidade (profundidade rasa) por features de sequencia.
  - Priorizar regra interpretavel com baixo risco operacional (poucos ramos/fontes), evitando crescimento de overfit por hardcode de target.
  - Materializar nova submissao candidata por merge estrito entre submissões existentes.
  - Validar com `check-submission`, `score --per-target` e `evaluate-robust` contra baseline robusto anterior.
- Criterios de aceite:
  - `score` local estritamente maior que `0.3620110714285714`.
  - `check-submission` estrito aprovado sem mismatch de contrato.
  - `evaluate-robust` com `allowed=true` no gate local e calibrado.

## PLAN-037 - DRfold2 full + selecao robusta multi-candidato (foco >0.40)

- Objetivo: ampliar o teto de score com um gerador ortogonal forte (DRfold2 full em todos os targets) e reotimizar a selecao final com gating estrito, visando candidato com potencial real de ultrapassar `0.40` no Kaggle.
- Escopo:
  - Concluir inferencia DRfold2 local full (`n_models=5`) para todos os alvos de `test_sequences.csv`, com `reuse-existing-targets` e limites de memoria estritos.
  - Exportar candidato DRfold2 em `submission.csv`, validar contrato estrito (`check-submission`) e pontuar no `public_validation` (`score --per-target`).
  - Executar sintese multi-candidato usando pool atualizado (`PLAN-036`, variantes TBM/RNAPro e novo DRfold2 full), com selecao por alvo baseada em criterios reproduziveis (sem hardcode de IDs).
  - Aplicar gate de promocao:
    - melhora estrita de `score` local sobre melhor baseline registrado (`0.3637460714285714`);
    - `evaluate-robust` aprovado (`allowed=true`);
    - submissao Kaggle somente se gates locais/calibrados aprovarem.
- Criterios de aceite:
  - Artefato DRfold2 full concluido e rastreavel em `runs/20260212_plan037_drfold2_full/`.
  - Pelo menos 1 novo candidato completo (nao smoke) validado e scoreado localmente com `check-submission` + `score`.
  - Decisao final de submit registrada com evidencia objetiva de gate (aprovado ou bloqueado).

## PLAN-038 - Patch por alvo sobre PLAN-036 com pool TBM (foco >0.38 local)

- Objetivo: construir candidato estritamente melhor que `PLAN-036` usando selecao por alvo entre candidatos ja validados (sem fallback permissivo), mantendo fluxo notebook-only hidden-safe.
- Escopo:
  - Gerar candidato `submission_plan038_patch.csv` por selecao por alvo entre `plan036` e variantes TBM (`c02`, `c03`, `c04`) usando metricas locais por alvo.
  - Validar contrato estrito (`check-submission`) e pontuar no `public_validation` (`score --per-target`).
  - Avaliar gate robusto (`evaluate-robust`) e promover somente com melhora estrita sobre baseline local `0.3637460714285714`.
  - Publicar dataset estatico atualizado para o notebook de submit e validar equivalencia de saida (hash) no runtime Kaggle.
  - Submeter apenas via notebook (`submit-kaggle` notebook-only) se todos os gates passarem.
- Criterios de aceite:
  - `score` local do candidato > `0.3637460714285714`.
  - `evaluate-robust` com `allowed=true`.
  - Notebook Kaggle concluindo `COMPLETE` com `check-submission` OK no output e hash igual ao candidato promovido.
  - Submissao Kaggle criada com mensagem rastreavel e sem bypass de gate.

## PLAN-039 - GNN reranker supervisionado (montagem de experimento local)

- Objetivo: montar uma frente de rerank com GNN para melhorar a selecao de candidatos por alvo sem alterar o contrato estrito do pipeline de submissao.
- Escopo:
  - Implementar modulo novo de QA GNN (`qa_gnn_ranker`) para treino e inferencia em tabela de candidatos com labels.
  - Treino supervisionado com split por grupo (`target_id`) para evitar vazamento entre train/val.
  - Expor comandos CLI dedicados:
    - `train-qa-gnn-ranker` (gera `qa_gnn_model.json` + pesos `.pt`);
    - `score-qa-gnn-ranker` (gera tabela com `gnn_score` por candidato).
  - Validar em dataset real ja existente no repositorio (`qa_train_fold0_subset.parquet`) e registrar metricas.
  - Manter integracao com pipeline principal opcional nesta fase (sem substituir o ranker linear por padrao).
- Criterios de aceite:
  - Novo modulo + comandos CLI funcionais e cobertos por testes unitarios.
  - Treino local executado em dado real com artefatos rastreaveis em `runs/`.
  - Registro append-only em `CHANGES.md` e `EXPERIMENTS.md` com comandos, metricas e riscos.

## PLAN-040 - Endurecimento do gate de submit (anti-overfit local)

- Objetivo: reduzir regressao no Kaggle quando houver aumento artificial no score local, tornando o gate de promocao mais conservador e alinhado a generalizacao.
- Escopo:
  - Endurecer gate robusto para exigir minimo de scores CV antes de permitir submit competitivo.
  - Bloquear promocao quando calibracao local->public estiver em extrapolacao fora do range historico observado, por padrao.
  - Bloquear candidatos dependentes apenas de `public_validation` sem evidencia CV (padrao anti-target-patch).
  - Integrar as novas travas tanto em `evaluate-robust` quanto em `submit-kaggle`.
  - Cobrir os novos comportamentos com testes unitarios.
- Criterios de aceite:
  - `evaluate-robust` reprova por padrao quando `cv_count` insuficiente.
  - `build_alignment_decision` marca extrapolacao e bloqueia quando nao permitido.
  - `submit-kaggle` bloqueia submit sem `robust_report` (por padrao), com `cv_count` abaixo do minimo, ou com `public_validation` sem CV.
  - Testes unitarios novos/atualizados verdes.

## PLAN-041 - Pool patch global multi-candidato (foco >0.39 local + submit notebook-only)

- Objetivo: ampliar o teto local via selecao por alvo em pool global de candidatos ja scoreados/validados e promover somente se houver melhora estrita sobre `0.38650071428571425`.
- Escopo:
  - Inventariar candidatos compativeis com `public_validation` (mesmo contrato de IDs/linhas) com `score.json + per_target.csv + submission.csv`.
  - Sintetizar submissao por alvo escolhendo a melhor fonte por `target_score` (desempate por `global_score`).
  - Validar contrato estrito (`check-submission`) e pontuar localmente (`score --per-target`).
  - Aplicar gate robusto/calibrado e promover apenas com melhora estrita.
  - Publicar dataset estatico de candidato, executar notebook Kaggle e submeter via referencia do notebook.
- Criterios de aceite:
  - `score` local do novo candidato > `0.38650071428571425`.
  - Notebook Kaggle `COMPLETE` com `submission.csv` hash-identica ao candidato promovido.
  - Nova submissao Kaggle criada com mensagem rastreavel e sem falha de contrato.

## PLAN-042 - DRfold2 full (28 alvos) + reselecao de pool para tentar superar 0.3914 local

- Objetivo: adicionar um gerador ortogonal completo (DRfold2 em todos os alvos) ao pool de candidatos e reexecutar a selecao por alvo para buscar melhora estrita sobre o melhor score local atual (`0.3913999999999999`), mantendo fluxo notebook-only e validacao estrita.
- Escopo:
  - Rodar `predict-drfold2` completo para `test_sequences.csv` com `n_models=5`, `reuse-existing-targets` e perfil operacional de memoria.
  - Exportar submissao DRfold2 estrita (`export-submission`), validar contrato (`check-submission`) e pontuar `public_validation` (`score --per-target`).
  - Recalcular pool compativel e sintetizar novo candidato por alvo incluindo a linha DRfold2 full.
  - Aplicar gate robusto/calibrado e promover apenas se houver melhora local estrita sobre `0.3913999999999999`.
  - Se promovido, atualizar dataset estatico + notebook de submit e submeter apenas via referencia do notebook.
- Criterios de aceite:
  - `predict-drfold2` full concluido sem erro de contrato e com artefato long rastreavel.
  - `submission` DRfold2 full passa `check-submission` estrito e possui `score` local registrado.
  - Novo candidato agregado so e elegivel se `score` local > `0.3913999999999999`.
  - Decisao final (submit/bloqueio) registrada em `EXPERIMENTS.md` e `CHANGES.md` com evidencias objetivas.

## PLAN-043 - Sweep TBM ortogonal + patch incremental sobre melhor pool (foco >0.3914 local)

- Objetivo: gerar candidatos TBM ortogonais de baixo custo e usar patch incremental por alvo sobre o melhor candidato atual para obter melhora estrita local sem quebrar contrato notebook-only.
- Escopo:
  - Rodar variantes novas de `predict-tbm` em `public_validation` com parametros mais agressivos de alinhamento/diversidade.
  - Validar cada variante com `export-submission -> check-submission -> score --per-target`.
  - Sintetizar candidato incremental (`best + variante`) por selecao por alvo (`max target_score`) e confirmar `score` local.
  - Aplicar gate robusto/calibrado e promover apenas se `score` local superar estritamente `0.3913999999999999`.
  - Publicar dataset estatico, atualizar notebook de submit, validar hash de saida do notebook e submeter via `submit-kaggle`.
- Criterios de aceite:
  - Pelo menos uma variante nova scoreada com artefatos completos em `runs/`.
  - Candidato incremental final com `score` local > `0.3913999999999999`.
  - `evaluate-robust` aprovado e submissao notebook-only criada com mensagem rastreavel.

## PLAN-044 - CV-first blend strict+chemical (sem bypass) para recuperar leaderboard >0.268

- Objetivo: substituir o criterio de promocao dominado por `public_validation` por um criterio CV-first, usando uma familia de modelos ja estavel (`strict` + `chemical`) e promovendo apenas candidatos com evidencia em folds.
- Escopo:
  - Construir candidatos por blend deterministico de submissao wide entre `strict` e `chemical` para `alpha in {0.25, 0.50, 0.75}`.
  - Avaliar os mesmos alphas em `fold3` e `fold4` (datasets de CV ja prontos), com `check-submission` e `score` estritos.
  - Aplicar o alpha vencedor no `public_validation` e registrar score local apenas como diagnostico secundario.
  - Gerar `robust_report` com `cv:fold3`, `cv:fold4` e `public_validation`, mantendo regras default (sem `allow-public-validation-without-cv`, sem `allow-calibration-extrapolation`, sem bypass de `min_cv_count`).
  - Promover para submit notebook-only somente se:
    - robust gate `allowed=true`;
    - calibracao conservadora nao bloquear;
    - candidato nao depender de patch por `target_id`.
- Criterios de aceite:
  - Tabela comparativa por alpha com `fold3`, `fold4`, `mean_cv`, `min_cv` e `public_validation`.
  - Artefatos completos em `runs/` para reproduzir blend e scores.
  - Decisao final de promocao (ou bloqueio) registrada em `EXPERIMENTS.md` com justificativa objetiva.

## PLAN-045 - Gate anti-overfitting de treino (QA/QA-GNN) integrado ao pipeline

- Objetivo: bloquear promocao de modelos QA com sinais claros de overfitting (gap train/val), reduzindo risco de regressao nas submissões.
- Escopo:
  - Implementar modulo dedicado de gate de treino que compara metricas `train_metrics` vs `val_metrics` e gera relatorio estruturado.
  - Regras minimas de bloqueio:
    - limite de `n_samples` de validacao;
    - limites de gap para losses (`mae`, `rmse`);
    - limites de queda para correlacoes/qualidade (`r2`, `spearman`, `pearson`).
  - Integrar gate automaticamente em `train-qa-ranker` e `train-qa-gnn-ranker` (fail-fast por padrao; bypass apenas explicito por flag).
  - Expor comando CLI para avaliacao offline de qualquer `qa_model.json`/`qa_gnn_model.json` antes de promocao.
  - Cobrir o comportamento com testes unitarios.
- Criterios de aceite:
  - Comando `evaluate-train-gate` funcional, gerando JSON em `runs/`.
  - Treino QA/QA-GNN passa a escrever `train_gate_report.json` e bloquear por default quando gate reprovar.
  - Testes unitarios do novo gate verdes sem regressao nos testes de gating existentes.

## PLAN-046 - Sweep CV dos configs TBM c01..c04 para selecao robusta sem patch

- Objetivo: escolher o melhor candidato de submissao entre configs TBM genericos (`c01..c04`) com base em evidencias de CV (`fold3/fold4`), reduzindo dependencia de ajuste em `public_validation`.
- Escopo:
  - Reexecutar `predict-tbm -> export-submission -> check-submission -> score --per-target` para `c01..c04` em `fold3` e `fold4`, com perfil operacional conservador de memoria.
  - Consolidar ranking por `mean_cv` e `min_cv` para decidir candidato robusto.
  - Avaliar candidato vencedor no `public_validation` apenas como diagnostico secundario (sem treinar/ajustar regra no proprio public).
  - Gerar `evaluate-robust` com `cv:fold3`, `cv:fold4` e `public_validation` em modo estrito (sem bypass).
  - Promover para submissao somente se gates vigentes aprovarem e houver justificativa objetiva no ranking CV.
- Criterios de aceite:
  - Matriz de resultados `fold3/fold4/public` para `c01..c04` registrada em `runs/` com artefatos reproduziveis.
  - Relatorio robusto CV-first gerado e rastreavel.
  - Decisao final (promover ou bloquear) registrada em `EXPERIMENTS.md` com evidencias de score.

## PLAN-047 - Extensao CV (3o fold) para desempate c02 vs c04 e decisao de promocao

- Objetivo: reduzir incerteza de generalizacao entre `c02` e `c04` adicionando um terceiro fold (`fold0`) ao criterio CV-first antes de qualquer nova promocao competitiva.
- Escopo:
  - Rodar `predict-tbm -> export-submission -> check-submission -> score --per-target` para `c02` e `c04` em `fold0` (dataset ja existente em `PLAN-021`).
  - Consolidar ranking `mean_cv/min_cv` com 3 folds (`fold0/fold3/fold4`).
  - Recalcular `evaluate-robust` para `c02` e `c04` com `cv:fold0,cv:fold3,cv:fold4,public_validation`.
  - Promocao somente se gate estrito/calibrado permitir, sem bypass.
- Criterios de aceite:
  - `score.json` e `per_target.csv` gerados para `c02` e `c04` em `fold0`.
  - Tabela comparativa 3-fold registrada em `runs/`.
  - Decisao objetiva de promocao (ou bloqueio) registrada em `EXPERIMENTS.md`.

## PLAN-048 - Correcoes criticas de selecao precomputed + gate de submit estrito

- Objetivo: remover dois caminhos de regressao de corretude que podem degradar score no Kaggle por falhas de selecao/gating.
- Escopo:
  - `rnapro/infer` (`use_template=ca_precomputed`): excluir modelos incompletos (`mask` com residuos faltantes) antes do rerank QA/diversidade e garantir selecao apenas de modelos completos.
  - `gating.assert_submission_allowed`: bloquear `min_improvement` negativo com erro fail-fast explicito.
  - Cobrir os dois casos com testes unitarios/reprodutiveis.
- Criterios de aceite:
  - Inferencia precomputed nao seleciona candidato incompleto quando ha modelos completos suficientes.
  - Gate local rejeita `min_improvement < 0` mesmo com `allow_regression=false`.
  - Testes direcionados passam localmente e sem bypass permissivo.

## PLAN-049 - QA de ranking forte estilo RNArank para selecao global dos 5 candidatos

- Objetivo: aumentar robustez de `best-of-5` trocando o seletor final por um reranker supervisionado mais forte (hibrido regressao+ranking), mantendo os geradores existentes e selecionando globalmente em pool combinado.
- Escopo:
  - Adicionar construcao de pool global de candidatos a partir de predicoes long (`TBM`, `RNAPro`, `DRfold2`) com validacao estrita de contrato e sem fallback silencioso.
  - Implementar modulo `qa_rnrank.py` com:
    - treino supervisionado hibrido (`MSE + pairwise ranking loss`);
    - inferencia/score de candidatos;
    - selecao final de 5 modelos por alvo com diversidade.
  - Integrar novos comandos CLI:
    - `build-candidate-pool`
    - `train-qa-rnrank`
    - `score-qa-rnrank`
    - `select-top5-global`
  - Integrar gate anti-overfitting do treino (`train_gate_report.json`) no novo comando de treino.
  - Cobrir com testes unitarios e de integracao leve (pool, treino/score, selecao top-5).
- Criterios de aceite:
  - `build-candidate-pool` gera parquet com schema estavel e falha cedo em duplicatas/contiguidade invalida por candidato.
  - `train-qa-rnrank` gera `qa_rnrank_model.json` + `.pt` + `train_gate_report.json` e bloqueia overfitting por padrao.
  - `select-top5-global` gera predições long com exatamente 5 modelos por alvo e compativeis com `export-submission`.
  - Testes novos passam localmente sem quebrar testes QA existentes.

## PLAN-050 - Metodo completo de validacao pre-submit (CV + estabilidade + calibracao)

- Objetivo: reduzir divergencia entre score local e Kaggle com um gate unico de prontidao de submissao que bloqueia promocoes sem evidencia robusta de melhoria.
- Escopo:
  - Implementar avaliador de prontidao pre-submit que consolida:
    - melhoria robusta contra baseline;
    - melhoria por fold CV (contagem minima de folds melhorados);
    - limite de regressao por fold e estabilidade CV;
    - gate de calibracao local->Kaggle com verificacao de saude da calibracao (correlacao minima + pares minimos).
  - Expor novo comando CLI `evaluate-submit-readiness` para gerar relatorio unico em `runs/`.
  - Integrar `submit-kaggle` para exigir `--readiness-report` por padrao (com bypass explicito), alem do `robust_report`.
  - Cobrir com testes unitarios para cenarios de aprovacao e bloqueio (sem CV, regressao em fold, correlacao negativa de calibracao e submit sem readiness).
- Criterios de aceite:
  - `evaluate-submit-readiness` gera JSON estruturado com `allowed` e `reasons` acionaveis.
  - `submit-kaggle` bloqueia por padrao quando `readiness_report` estiver ausente ou reprovado.
  - Suite de testes de robustez/gate permanece verde com novos casos de readiness.

## PLAN-051 - Blend adaptativo TBM+RNAPro orientado por confianca de template

- Objetivo: superar o baseline `ensemble 0.99` sem trocar geradores, usando ponderacao por alvo/residuo baseada em sinais de confianca (`coverage`, `similarity`) do TBM.
- Escopo:
  - Construir candidatos de blend adaptativo com pesos por linha (`w_tbm`) derivados de `coverage/similarity` (regra deterministica, sem fallback silencioso).
  - Rodar `export-submission -> check-submission -> score --per-target` em `fold0` para triagem inicial.
  - Promover apenas candidatos que melhorem estritamente o baseline local no fold de triagem.
  - Validar candidatos promovidos em `fold3` e `fold4` e consolidar `evaluate-robust`.
  - Registrar decisao final de submit apenas se o gate robusto aprovar.
- Criterios de aceite:
  - Pelo menos um candidato adaptativo avaliado de ponta a ponta com artefatos rastreaveis em `runs/`.
  - Relatorio comparativo com baseline (`delta` por fold quando aplicavel).
  - Sem submissao Kaggle quando nao houver melhora robusta estrita.

## PLAN-052 - Wrapper operacional "main GPU" para pipeline end-to-end

- Objetivo: padronizar execucao em modo GPU-first para todas as etapas elegiveis do pipeline, com fail-fast explicito quando CUDA nao estiver disponivel.
- Escopo:
  - Criar wrapper de execucao (`scripts/rna3d_main_gpu.sh`) para comandos da CLI.
  - Forcar automaticamente:
    - `--qa-device cuda` em `predict-tbm` e `predict-rnapro`;
    - `--device cuda` em `train-qa-rnrank`, `score-qa-rnrank`, `select-top5-global`, `train-qa-gnn-ranker`, `score-qa-gnn-ranker`.
  - Validar disponibilidade de CUDA antes de comandos GPU-capable (com erro acionavel).
  - Preservar comandos CPU-only (`score`, `check-submission`, `export-submission`, `ensemble-predict`, etc.) sem tentativa de forcar GPU.
  - Documentar uso no `README.md` com exemplos de pipeline completo em modo GPU.
- Criterios de aceite:
  - Wrapper executa subcomandos existentes sem quebrar contrato de argumentos.
  - Comando GPU-capable falha cedo com mensagem clara se CUDA indisponivel.
  - `bash -n scripts/rna3d_main_gpu.sh` e `scripts/rna3d_main_gpu.sh <cmd> --help` funcionam localmente.

## PLAN-054 - Seletor por alvo c02 vs c04 (CV-first) para recuperar generalizacao

- Objetivo: explorar ganho real sem patch permissivo combinando `c02` (melhor CV) e `c04` (melhor public local) via seletor por alvo treinado em folds, com regra deterministica e auditavel.
- Escopo:
  - Medir teto de ganho (`oracle`) entre `c02` e `c04` no `public_validation` e em `fold3/fold4`.
  - Construir dataset por alvo com features de sequencia (`len`, `gc_ratio`, `au_ratio`) e label de escolha (`c02` vs `c04`) a partir de `fold3/fold4`.
  - Fazer sweep de regras deterministicas simples (thresholds) e selecionar regra por validacao cruzada entre folds.
  - Aplicar regra vencedora no `public_validation`, gerar submissao blended por alvo, validar contrato estrito e calcular score local oficial.
  - Comparar contra baselines `c02` e `c04` sem bypass.
- Criterios de aceite:
  - Artefatos em `runs/<plan054>/` contendo regras testadas, escolhas por alvo, submissao final e `score.json`.
  - `check-submission` aprovado para submissao blended.
  - Resultado comparativo registrado em `EXPERIMENTS.md` com conclusao objetiva (melhorou ou nao).

## PLAN-055 - Seletor multi-config c01..c04 por confianca TBM (CV-first)

- Objetivo: aumentar robustez de selecao por alvo sem patch permissivo escolhendo entre `c01..c04` via sinais de confianca TBM agregados por alvo.
- Escopo:
  - Construir features por alvo para cada config (`mean/std` de `similarity`, `coverage`, `qa_score`, `template_rank`, diversidade de templates/modelos).
  - Avaliar seletores deterministas multi-config (`argmax` de sinais e combinacoes lineares) em `fold3/fold4`.
  - Selecionar regra por `mean_cv` (com rastreio de `fold3` e `fold4`) e aplicar no `public_validation`.
  - Gerar submissao blended por alvo entre `c01..c04`, validar em modo estrito e calcular score local oficial.
- Criterios de aceite:
  - Tabela de comparacao de seletores em `fold3/fold4` registrada em `runs/<plan055>/`.
  - `check-submission` e `score` concluidos para a submissao selecionada.
  - Registro append-only em `EXPERIMENTS.md` com decisao de promover ou bloquear.

## PLAN-056 - Meta-reranker supervisionado por alvo/config (c01..c04)

- Objetivo: substituir regras fixas por um meta-modelo supervisionado que aprende, por alvo, qual config (`c01..c04`) tende a maximizar score local.
- Escopo:
  - Montar dataset supervisionado por alvo/config em `fold3/fold4` com:
    - features agregadas dos TBMs (`similarity`, `coverage`, `qa_score`, `template_rank`, diversidade),
    - features de sequencia (`len`, `gc`, `au`),
    - label real = score por alvo da config no fold.
  - Treinar/avaliar meta-reranker com validação cruzada entre folds (`train fold3 -> eval fold4`, `train fold4 -> eval fold3`).
  - Se CV superar baseline `c04` ou `c02` no critério definido, treinar modelo final em `fold3+fold4`, aplicar no `public_validation`, gerar submissao blended por alvo e medir score oficial.
- Criterios de aceite:
  - Artefatos de treino/score por fold em `runs/<plan056>/` com métricas e escolhas por alvo.
  - Comparação explícita contra baselines `c02` e `c04`.
  - `check-submission` + `score` concluídos para candidato final de `public_validation`.

## PLAN-057 - Pool expandido + selector CV-first (TBM multi-fonte) com teto oracle

- Objetivo: aumentar teto de seleção por alvo além de `c01..c04`, combinando fontes TBM adicionais já existentes no repositório e promovendo apenas se houver ganho robusto em CV.
- Escopo:
  - Consolidar candidatos por alvo de múltiplas fontes TBM já geradas (`c01..c06`, `strict/hybrid/chemical/hybrid_qa` quando disponíveis por fold).
  - Medir teto oracle por fold/public (melhor fonte por `target_id`) para validar potencial real antes de novo treino.
  - Treinar seletor supervisionado por alvo (modelo leve + regra determinística auditável) em `fold3/fold4` com validação cruzada.
  - Aplicar seletor vencedor em `public_validation`, gerar submissão estrita e calcular score oficial local.
  - Bloquear submit se não houver melhora local estrita sobre baseline oficial vigente em `EXPERIMENTS.md`.
- Criterios de aceite:
  - Tabela de oracle e baseline por fold/public registrada em `runs/<plan057>/`.
  - `check-submission` + `score --per-target` concluídos para candidato final.
  - Registro append-only em `EXPERIMENTS.md` com decisão objetiva de promoção/bloqueio.

## PLAN-058 - Recalibracao por regime e gate de submit competitivo (CV2 comparavel)

- Objetivo: reduzir bloqueios indevidos por calibracao global historica (mistura de regimes) e alinhar decisao de submit ao regime competitivo atual, preservando fail-fast e rastreabilidade.
- Escopo:
  - Construir relatorio de calibracao segmentado por regime de score local (ex.: `<0.30` vs `>=0.30`) e por janela temporal recente.
  - Comparar correlacoes (`pearson/spearman`) e erro de estimativa por regime para justificar regra operacional de calibracao competitiva.
  - Integrar avaliacao de readiness focada em folds comparaveis (`fold3/fold4`) com limiares explicitos de estabilidade CV2.
  - Definir decisao final de submit notebook-only para o candidato `best+vA+vC+vD` usando:
    - validacao local estrita ja concluida;
    - robust/readiness aprovados no regime selecionado;
    - registro explicito de risco residual.
- Criterios de aceite:
  - Artefato de calibracao segmentada em `runs/<plan058>/` com tabelas e conclusao objetiva.
  - Relatorio readiness final (`allowed=true` ou `allowed=false`) com racional reproduzivel.
  - Decisao de submit/bloqueio registrada em `EXPERIMENTS.md` sem tentativa cega.

## PLAN-059 - Limpeza de calibracao por `submission ref` + rescore de submetidos

- Objetivo: eliminar bloqueios indevidos no gate calibrado causados por `local_score` historico desatualizado no texto da submissao Kaggle.
- Escopo:
  - Recalcular localmente (scorer atual) os artefatos ja submetidos e consolidar tabela auditavel de `local_score` por submissao.
  - Introduzir suporte a overrides de calibracao por `submission ref` (JSON) em todos os fluxos de gate:
    - `calibrate-kaggle-local`
    - `evaluate-robust`
    - `evaluate-submit-readiness`
    - `submit-kaggle`
  - Permitir modo estrito `only_override_refs=true` para calibrar apenas com pares explicitamente recalculados.
  - Gerar artefatos de comparacao (`full_history` vs `clean_submitted_only`) para quantificar impacto no gate.
- Criterios de aceite:
  - 5 submissões historicas recalculadas com `check-submission` + `score` sem OOM.
  - Arquivo de overrides gerado em `runs/.../calibration_overrides*.json` com `by_ref` valido.
  - Testes de unidade dos modulos alterados verdes.
  - Resultado da calibracao limpa registrado em `EXPERIMENTS.md` com decisao operacional objetiva.

## PLAN-060 - Pool hibrido + reranker Top-5 CV-first para submit competitivo

- Objetivo: maximizar chance de novo melhor score Kaggle com estrategia de selecao forte (Top-5 por alvo) sobre pool hibrido, sem fallback permissivo e com gating estrito de promocao.
- Escopo:
  - Consolidar pool de candidatos por alvo com diversidade real de fontes ja disponiveis no repositorio:
    - TBM multi-config (`c01..c06`, quando existir por fold);
    - ao menos uma fonte nao-template/e2e ja integrada ao pipeline.
  - Treinar/aplicar reranker QA supervisionado para selecionar Top-5 por alvo no pool combinado, com modo estrito por padrao (fail-fast em contrato invalido, faltantes, duplicatas ou inconsistencias de contiguidade).
  - Validar de ponta a ponta em regime CV-first comparavel (`fold0`, `fold3`, `fold4`):
    - `predict/build pool -> select Top-5 -> export-submission -> check-submission -> score --per-target`.
  - Comparar explicitamente contra baseline oficial vigente em `EXPERIMENTS.md` e contra melhor candidato local atual antes de qualquer promocao.
  - Rodar `evaluate-robust` e `evaluate-submit-readiness` com calibracao por `submission ref` recalculada; bloquear promocao quando `allowed=false`.
  - Se (e somente se) houver melhora local estrita e gates aprovados, preparar fluxo de submissao notebook-only:
    - validar localmente com `check-submission`;
    - gerar `submission.csv` no notebook Kaggle (`/kaggle/working`);
    - submeter por referencia de notebook (`kaggle competitions submit -k <owner/notebook> -f submission.csv ...`), sem internet no notebook.
- Criterios de aceite:
  - Artefatos completos e reproduziveis em `runs/<plan060>/` (pool, modelo de reranker, escolhas Top-5 por alvo, `submission.csv`, `score.json`, `per_target.csv`, relatorios de gate).
  - `check-submission` estrito aprovado para todos os candidatos promovidos.
  - Tabela comparativa com `fold0/fold3/fold4`, `mean_cv`, `min_cv`, `public_validation` e deltas vs baseline.
  - `evaluate-robust` e `evaluate-submit-readiness` com resultado final registrado (`allowed=true/false`) e razoes acionaveis.
  - Decisao final (promover/bloquear submit) registrada em `EXPERIMENTS.md` com comandos, configuracao efetiva, custos e riscos residuais.

## PLAN-061 - Sprint CV-first: Top-5 hardening + template audit + retrieval reforcado

- Objetivo: aumentar robustez do caminho competitivo `best-of-5` sem regressao CV, mantendo fail-fast estrito e bloqueio de submit sem evidencia robusta.
- Escopo:
  - Tratar migracao `blend -> Top-5 por uniao de candidatos` como concluida e reforcar hardening operacional (sem promover `branch=ensemble` no caminho competitivo).
  - Adicionar auditoria automatica de templates externos (datas, coordenadas, consistencia estrutural) com bloqueio fail-fast.
  - Fortalecer retrieval com pool/rerank global maior e cache deterministico de candidatos por hash de entradas+parametros.
  - Evoluir rotulagem supervisionada do candidate pool para metodo alinhado ao objetivo competitivo (`TM-score/USalign`) com opcao explicita `RMSD+Kabsch` para ablacao.
  - Adicionar modo seletivo no `predict-drfold2` para executar apenas subconjunto de `target_id` de baixa confianca.
  - Endurecer gate de submit para exigir mais evidencia CV por padrao (`require_min_cv_count=3`).
  - Atualizar documentacao operacional com defaults da sprint e caminho competitivo oficial.
- Criterios de aceite:
  - `build-template-db` falha cedo quando `external_templates` viola auditoria obrigatoria.
  - `retrieve-templates` registra `cache_key/cache_hit` no manifest e reutiliza cache valido; cache inconsistente falha explicitamente.
  - `add-labels-candidate-pool --label-method tm_score_usalign` gera labels finitos em `[0,1]` e manifesta metodo usado.
  - `predict-drfold2 --target-ids-file <...>` filtra corretamente alvos e falha cedo em IDs invalidos/duplicados.
  - `submit-kaggle --help` exibe `--require-min-cv-count` com default `3`.
  - Suite de testes direcionados para auditoria/cache/labels/DRfold2 seletivo permanece verde.

## PLAN-062 - Refatoracao de arquivos grandes (modularizacao sem mudar contrato)

- Objetivo: reduzir acoplamento e facilitar manutencao separando arquivos grandes em modulos menores, preservando comportamento/CLI e validacao estrita.
- Escopo:
  - Extrair `src/rna3d_local/candidate_pool.py` em modulos organizados:
    - build/agregacao de candidatos;
    - rotulagem supervisionada (`tm_score_usalign` e `rmsd_kabsch`);
    - fachada de compatibilidade para imports existentes.
  - Extrair logica de cache de retrieval para modulo dedicado, mantendo contrato de `retrieve-templates` e manifest.
  - Manter API publica e nomes de comandos intactos (sem modo permissivo novo, sem fallback silencioso).
  - Atualizar testes/coverage para garantir equivalencia funcional.
- Criterios de aceite:
  - `pytest` direcionado dos modulos alterados permanece verde.
  - `python -m rna3d_local --help` e subcomandos impactados seguem com mesma superficie publica.
  - Nenhuma regressao de contrato estrito nos fluxos de build pool/labels/retrieval.

## PLAN-063 - Modularizacao da CLI (parser + handlers) com compatibilidade total

- Objetivo: reduzir acoplamento e tamanho de `src/rna3d_local/cli.py` separando parser e handlers em modulos dedicados, sem alterar contrato de comandos.
- Escopo:
  - Extrair handlers/utilitarios de comando para `src/rna3d_local/cli_commands.py`.
  - Extrair construcao do parser para `src/rna3d_local/cli_parser.py`, com organizacao por grupos de comandos.
  - Manter `src/rna3d_local/cli.py` como fachada de compatibilidade, preservando:
    - `build_parser`;
    - `main`;
    - helpers importados por testes (`_enforce_non_ensemble_predictions`, `_enforce_submit_hardening`).
  - Garantir ausencia de fallback permissivo e manter fail-fast nos fluxos ja existentes.
- Criterios de aceite:
  - `python -m rna3d_local --help` sem regressao de superficie.
  - `pytest -q tests/test_cli_strict_surface.py tests/test_submit_gate_hardening.py` verde.
  - `pytest` direcionado dos comandos tocados permanece verde.

## PLAN-064 - Quebra de `cli_commands.py` por dominio (fase 2)

- Objetivo: reduzir acoplamento operacional da camada de comandos, separando `cli_commands.py` em modulos menores por dominio sem alterar superficie do parser/CLI.
- Escopo:
  - Extrair helpers comuns para `cli_commands_common.py`.
  - Extrair comandos em modulos dedicados:
    - `cli_commands_data.py`;
    - `cli_commands_templates.py`;
    - `cli_commands_qa.py`;
    - `cli_commands_gating.py`.
  - Manter `cli_commands.py` como fachada de compatibilidade para imports existentes do parser e da fachada `cli.py`.
  - Preservar fail-fast, mensagens de erro e contratos atuais de comandos/flags.
- Criterios de aceite:
  - `python -m rna3d_local --help` sem regressao de comandos e flags.
  - `pytest -q tests/test_cli_strict_surface.py tests/test_submit_gate_hardening.py` verde.
  - Testes direcionados de fluxos tocados (`candidate_pool`, `retrieval`, `template_workflow`) verdes.

## PLAN-065 - Quebra de `cli_parser.py` por dominio (fase 3)

- Objetivo: reduzir tamanho e churn do parser principal, separando o registro de subcomandos por domínio sem alterar interface pública da CLI.
- Escopo:
  - Extrair registradores de parser para módulos dedicados:
    - `cli_parser_data.py`;
    - `cli_parser_templates.py`;
    - `cli_parser_qa.py`;
    - `cli_parser_gating.py`.
  - Manter `cli_parser.py` como fachada de montagem (`build_parser`) chamando registradores em ordem estável.
  - Preservar todos os nomes de subcomandos, flags, defaults e handlers.
  - Não alterar regras de fail-fast/strict já implementadas nos handlers.
- Criterios de aceite:
  - `python -m rna3d_local --help` sem regressão da superfície.
  - `pytest -q tests/test_cli_strict_surface.py tests/test_submit_gate_hardening.py` verde.
  - Testes direcionados de fluxos impactados (`candidate_pool`, `retrieval`, `template_workflow`, `kaggle_submissions`) verdes.

## PLAN-066 - Factories de argumentos compartilhados no parser (fase 4)

- Objetivo: reduzir duplicação em definição de argumentos de CLI sem alterar contratos (nomes, defaults, tipos, help e handlers).
- Escopo:
  - Introduzir `cli_parser_args.py` com helpers reutilizáveis para blocos recorrentes:
    - memória (`memory-budget`/`max-rows`);
    - backend de compute (`compute-backend`/GPU);
    - calibração (`calibration-*` e overrides);
    - guard anti-overfit de treino.
  - Aplicar helpers em:
    - `cli_parser_data.py`;
    - `cli_parser_templates.py`;
    - `cli_parser_qa.py`;
    - `cli_parser_gating.py`.
  - Manter ordem estável dos subcomandos e compatibilidade da superfície no `--help`.
- Criterios de aceite:
  - `python -m rna3d_local --help` sem regressão de comandos/opções.
  - `pytest -q tests/test_cli_strict_surface.py tests/test_submit_gate_hardening.py` verde.
  - Testes direcionados de pipeline tocado (`candidate_pool`, `retrieval`, `template_workflow`, `kaggle_submissions`) verdes.

## PLAN-067 - Hardening Top-5 competitivo (sem blend + diversidade invariante + C1' estrito)

- Objetivo: remover perda sistematica de score por blend de coordenadas e por diversidade dependente de frame, alinhando o pipeline ao contrato competitivo `best-of-5`.
- Escopo:
  - Bloquear `ensemble-predict` na CLI com erro fail-fast acionavel e orientacao para `build-candidate-pool` + `select-top5-global`.
  - Substituir `_pair_similarity` em `qa_ranker.py` por similaridade invariante a rotacao/translacao:
    - alinhamento rigido via Kabsch;
    - `RMSD`;
    - conversao para similaridade `exp(-rmsd/10.0)`.
  - Endurecer parser DRfold2 para exigir atomo `C1'` em todos os residuos do alvo (sem fallback para outros atomos).
  - Atualizar testes e documentacao operacional para o caminho competitivo sem blend.
- Criterios de aceite:
  - `python -m rna3d_local ensemble-predict ...` falha sempre com mensagem de bloqueio explicita.
  - Testes de QA/diversidade validam invariancia a rotacao e penalizacao de duplicatas com a nova metrica.
  - DRfold2 falha cedo quando `C1'` estiver ausente em qualquer residuo.
  - `pytest` direcionado dos modulos impactados permanece verde.

## PLAN-068 - Gate competitivo por comparabilidade de score (schema/hash/regime)

- Objetivo: impedir promocao de candidatos com evidencia de score nao comparavel ao regime oficial Kaggle, mantendo suporte diagnostico para regimes alternativos.
- Escopo:
  - Tornar `score.json` auto-descritivo no comando `score` com metadados obrigatorios:
    - `dataset_type`, `sample_columns`, `sample_schema_sha`, `n_models`, `metric_sha256`, `usalign_sha256`, `regime_id`.
  - Exigir metadados obrigatorios em leitura de score para gates competitivos (`evaluate-robust`, `evaluate-submit-readiness`, `submit-kaggle` quando usar `--score-json`).
  - Separar regimes em `evaluate-robust`/`evaluate-submit-readiness`:
    - incluir apenas scores do regime competitivo selecionado;
    - excluir e reportar scores de outros regimes em `excluded_by_regime`;
    - bloquear fingerprint divergente dentro do regime (`sample_schema_sha`, `n_models`, `metric_sha256`, `usalign_sha256`).
  - Endurecer `submit` para aceitar apenas reports com:
    - `compatibility_checks.allowed=true`;
    - `regime_summary.competitive_regime_id == kaggle_official_5model`.
  - Endurecer contrato de submissao em `check-submission`:
    - coordenadas `x_k/y_k/z_k` devem ser numericas, finitas e em faixa plausivel (`abs <= 1e6`).
  - Expandir testes para cobrirem metadados de score, separacao de regime e hardening de submit.
- Criterios de aceite:
  - `evaluate-robust` falha cedo para `score.json` legado sem metadados obrigatorios.
  - `evaluate-robust`/`evaluate-submit-readiness` reportam exclusao de scores fora do regime competitivo sem contaminar score robusto/readiness.
  - `submit-kaggle` bloqueia reports sem `compatibility_checks` ou fora do regime oficial.
  - `check-submission` bloqueia coordenadas nao numericas, nao-finitas e fora de faixa.
  - Suite direcionada de testes de `contracts/scoring/robust/readiness/submit_hardening` permanece verde.

## PLAN-069 - Remocao de legado competitivo (sem blend + sem QA linear)

- Objetivo: remover caminhos legados nao competitivos (blend de coordenadas e QA linear) para reduzir acoplamento e evitar uso acidental de estrategias regressivas.
- Escopo:
  - Remover completamente a superficie de CLI de `ensemble-predict` (parser + handler + re-exports) e excluir `src/rna3d_local/ensemble.py`.
  - Atualizar testes de workflow para fluxo sem blend (`predict-tbm`/`predict-rnapro` -> `export-submission`) e eliminar testes dedicados ao modulo de ensemble.
  - Remover comando `train-qa-ranker` da CLI (parser + handler + re-exports), mantendo apenas QA_GNN/RNArank para treino/rerank.
  - Endurecer `predict_tbm` e `infer_rnapro` para aceitar `--qa-model` apenas quando `model_type=qa_gnn`; qualquer outro JSON deve falhar cedo com erro acionavel.
  - Remover API linear ociosa de `qa_ranker.py` (`QA_FEATURE_NAMES`, `QaMetrics`, `QaModel`, `load_qa_model`, `score_candidate_with_model`, `train_qa_ranker`) e ajustar testes para cobrir apenas features/diversidade utilizadas no pipeline.
  - Atualizar README para refletir remocao dos comandos legados.
- Criterios de aceite:
  - `python -m rna3d_local --help` nao lista `ensemble-predict` nem `train-qa-ranker`.
  - `predict-tbm` e `predict-rnapro` falham com mensagem explicita ao receber `--qa-model` nao-GNN.
  - Suite direcionada (`cli_strict_surface`, `qa_ranker`, `qa_gnn_ranker`, `template_workflow`) permanece verde.

## PLAN-071 - Cobertura de teste para ausencia de runtime DRfold2

- Objetivo: garantir que a falta de instalacao DRfold2 seja capturada automaticamente pelos testes com fail-fast explicito.
- Escopo:
  - Adicionar teste unitario chamando `predict_drfold2` com `drfold2_root` inexistente.
  - Validar que o erro retornado seja acionavel e no formato de contrato (`drfold2_root nao encontrado`).
- Criterios de aceite:
  - `pytest -q tests/test_drfold2_parser.py` verde.
  - Novo teste falha se o fail-fast de readiness de DRfold2 for removido/regredido.

## PLAN-070 - Instalacao persistente do DRfold2 oficial + validacao end-to-end real

- Objetivo: garantir que o branch ab initio DRfold2 esteja instalado localmente de forma persistente (fora de `/tmp`) e validado com inferencia real (sem stub), eliminando falhas por pesos corrompidos.
- Escopo:
  - Clonar repositorio oficial DRfold2 em `third_party/drfold2_official`.
  - Resolver dependencia de build do `Arena` no ambiente local (compilar binarios requeridos).
  - Baixar `model_hub.tar.gz` com estrategia resiliente de resume e checagem de integridade por tamanho + `tar`.
  - Validar integridade funcional dos pesos carregando todos os checkpoints (`torch.load`) com fail-fast.
  - Executar `predict-drfold2` real em GPU e confirmar geracao de parquet com `target_id`/`model_id` esperados e coordenadas sem nulos.
- Criterios de aceite:
  - `third_party/drfold2_official/model_hub` presente e consistente (checagem `tar` OK).
  - `Arena/Arena` e `Arena/Arena_counter` executaveis presentes.
  - Validacao de checkpoints sem corrupcao (`ok=81 fail=0`).
  - Smoke real `predict-drfold2` conclui com parquet valido (`n_models=5`, sem `null` em `x/y/z`).

## PLAN-072 - Preflight do output do notebook no submit-kaggle (arquivo + ordem + hash)

- Objetivo: impedir submissao no Kaggle com arquivo errado/stale (notebook-only), garantindo que o arquivo realmente submetido bate com o `sample_submission` (colunas + ordem) e com o `--submission` validado localmente.
- Escopo:
  - No `submit-kaggle`, antes de `kaggle competitions submit`, baixar o arquivo `--notebook-file` do output do notebook `--notebook-ref` via Kaggle API com suporte a paginacao (page_token).
  - Validar estritamente o arquivo baixado contra `--sample` (colunas e ordem; chaves e ordem; sem nulos; coordenadas numericas/finito/faixa).
  - Calcular `sha256` do `--submission` local e do arquivo baixado; bloquear submissao se divergirem (modo estrito por padrao, sem fallback silencioso).
  - Bloquear quando `--notebook-version` nao for o `current_version_number` do notebook (nao ha API confiavel para baixar outputs de versoes antigas).
- Criterios de aceite:
  - `submit-kaggle` falha cedo quando `--notebook-file` nao existir no output do notebook (mesmo quando estiver em paginas posteriores).
  - `submit-kaggle` falha cedo quando o arquivo do notebook violar o contrato do `sample_submission` (incluindo ordem).
  - `submit-kaggle` falha cedo quando `sha256(local) != sha256(notebook_output)` com erro acionavel.
  - Testes unitarios cobrem: paginacao do output, mismatch de hash e mismatch de versao.

## PLAN-073 - Fase 1 Template Oracle (greenfield)

- Objetivo: reconstruir pipeline minimo competitivo para reduzir gap de retrieval TBM com fusao de sinais latentes, semanticos e quimicos.
- Escopo:
  - Recriar pacote `rna3d_local` com comandos essenciais da Fase 1.
  - Implementar retrieval por embeddings com encoder fixo (`ribonanzanet2`) e ANN (`faiss_ivfpq`), com modo explicito de teste (`mock`/`numpy_bruteforce`).
  - Implementar mineracao de `description` offline (backend `llama_cpp` obrigatorio por padrao, `rules` explicito para testes).
  - Implementar preparacao de features quimicas DMS/2A3 e reranker em PyTorch com bias quimico agregado.
  - Implementar `predict-tbm`, `export-submission`, `check-submission` e `submit-kaggle-notebook` com contratos estritos.
- Criterios de aceite:
  - `pytest -q` verde para testes de contrato/fail-fast e fluxo minimo.
  - `check-submission` bloqueia colunas/chaves/ordem/coordenadas invalidas.
  - `retrieve-templates-latent` aplica filtro temporal estrito e gera manifest com scores de fusao.
  - `submit-kaggle-notebook` bloqueia quando score local nao supera baseline e quando hash local diverge do arquivo de notebook.

## PLAN-074 - Fase 2 Arsenal Hibrido 3D (RNAPro + Boltz-1 + Chai-1)

- Objetivo: habilitar inferencia offline multimodelo para targets orfaos/multicomponente, com roteamento deterministico e gate competitivo estrito antes de submit notebook-only.
- Escopo:
  - Criar manifest de assets offline fase 2 (`models/rnapro`, `models/boltz1`, `models/chai1`, `wheels`) com hash e fail-fast.
  - Adicionar inferencias offline:
    - `predict-rnapro-offline`
    - `predict-chai1-offline`
    - `predict-boltz1-offline`
  - Implementar roteador hibrido por alvo:
    - template forte -> TBM;
    - orfao sem ligante -> ensemble Chai+Boltz;
    - alvo com `ligand_SMILES` -> Boltz primario;
    - RNAPro como candidato suplementar.
  - Implementar selecao `Top-5` por alvo no pool hibrido e gate `evaluate-submit-readiness` com melhoria estrita local.
  - Cobrir com testes de contrato/integração sem fallback silencioso.
- Criterios de aceite:
  - `python -m pytest -q` verde com cobertura dos novos fluxos.
  - `python -m rna3d_local --help` lista novos comandos de fase 2.
  - `build-hybrid-candidates` gera `routing.parquet` com regra aplicada por alvo.
  - `select-top5-hybrid` entrega exatamente 5 modelos por alvo ou falha cedo.
  - `evaluate-submit-readiness` bloqueia candidatos sem melhoria estrita vs baseline.

## PLAN-075 - Notebook de submissao com pipeline completo Fase 1 + Fase 2

- Objetivo: atualizar o notebook Kaggle de submissao para executar o fluxo completo (Fase 1 + Fase 2) em modo estrito/fail-fast, sem fallback silencioso.
- Escopo:
  - Substituir o notebook `stanford-rna3d-submit-prod-v2.ipynb` por versao que roda:
    - Fase 1: `infer-description-family`, `prepare-chemical-features`, `retrieve-templates-latent`, `score-template-reranker`, `predict-tbm`.
    - Fase 2: `build-phase2-assets`, `predict-rnapro-offline`, `predict-chai1-offline`, `predict-boltz1-offline`, `build-hybrid-candidates`, `select-top5-hybrid`.
    - Export/contrato: `export-submission` + `check-submission`.
  - Implementar descoberta e validacao estrita de ativos em `/kaggle/input` (src, embeddings, FAISS, modelo LLM GGUF, modelo Ribonanza, mapa de familias, quickstart quimico, modelos phase2, reranker pretreinado).
  - Garantir bloqueio explicito quando houver ativo ausente/ambiguo ou comando CLI ausente.
  - Preservar contrato notebook-only (`submission.csv` final em `/kaggle/working`).
- Criterios de aceite:
  - Notebook JSON valido e codigo compilavel localmente.
  - Notebook contem todos os comandos de pipeline Fase 1 + Fase 2 listados no escopo.
  - Validacao local confirma superficie CLI exigida por esse notebook.

## PLAN-076 - Hardening do pipeline completo no Kaggle (quickstart + TBM coverage)

- Objetivo: remover falhas de execucao do notebook completo Fase 1+2 no Kaggle, preservando contrato estrito e garantindo geracao valida de `submission.csv`.
- Escopo:
  - Adaptar `prepare-chemical-features` para schemas QUICK_START de templates (`ID,resid,x_i,y_i,z_i`), incluindo caso com apenas um triplet (`x_1,y_1,z_1`), sem fallback silencioso.
  - Endurecer `predict-tbm` para selecionar os primeiros templates com cobertura completa de residuos do alvo (pular candidatos invalidos e falhar cedo apenas se cobertura valida insuficiente).
  - Cobrir os novos contratos com testes unitarios dedicados.
  - Publicar nova versao do dataset de `src` no Kaggle e reexecutar notebook de submissao.
  - Validar estritamente o `submission.csv` do output remoto contra `sample_submission`.
- Criterios de aceite:
  - `pytest -q` verde com testes novos para `chemical_features` e `tbm`.
  - Notebook Kaggle `marcux777/stanford-rna3d-submit-prod-v2` conclui (`COMPLETE`) com pipeline completo.
  - Notebook gera `submission.csv` e `check-submission` retorna `ok=true`.
  - Validacao local adicional `python -m rna3d_local check-submission --sample ... --submission ...` retorna `ok=true`.

## PLAN-077 - Robustez para rerun oculto do Kaggle

- Objetivo: eliminar erro de rerun em hidden dataset ("larger/smaller/different"), mantendo contrato estrito de submissao e sem treino no notebook Kaggle.
- Escopo:
  - Remover do notebook de submissao o treino online do reranker (`train-template-reranker`/`score-template-reranker`) para cumprir politica de "no train on Kaggle".
  - Tornar retrieval/TBM robustos a alvos sem candidatos ou com cobertura parcial de template, registrando diagnostico em manifest em vez de abortar pipeline completo.
  - Ajustar roteamento hibrido para desviar explicitamente para fase2 quando `template_strong=true` mas nao houver cobertura TBM para o alvo.
  - Cobrir novos cenarios em testes unitarios.
  - Publicar nova versao de `src` no dataset Kaggle, reexecutar notebook e validar `submission.csv`.
  - Submeter nova versao do notebook na competicao para verificar se o erro de rerun oculto desaparece.
- Criterios de aceite:
  - `pytest -q` verde com testes atualizados.
  - Notebook `version 84` executa `COMPLETE` com `check-submission` `ok=true`.
  - `submission.csv` validado localmente contra `sample_submission`.
  - Nova submissao Kaggle criada com notebook `v84` sem alterar contrato notebook-only.

## PLAN-078 - Branch SE(3) Generativo para Best-of-5

- Objetivo: adicionar um novo branch experimental integrado para gerar ensembles estruturais diversos (Best-of-5) com arquitetura equivariante e modelagem generativa, sem substituir o pipeline atual.
- Escopo:
  - Criar pacotes novos `se3/`, `generative/`, `ensemble/`, `training/` com:
    - backbone EGNN + IPA adaptado;
    - geracao por Diffusion SE(3) + Flow Matching;
    - rank de qualidade + penalizacao de diversidade;
    - selecao top-5 por alvo.
  - Expor novos comandos CLI:
    - `train-se3-generator`
    - `sample-se3-ensemble`
    - `rank-se3-ensemble`
    - `select-top5-se3`
  - Integrar branch opcional ao roteador hibrido com flag/entrada explicita (default desativado).
  - Preservar contratos estritos/fail-fast e formato final compativel com `export-submission`.
  - Cobrir com testes de fluxo e contratos novos.
- Criterios de aceite:
  - `pytest -q` verde com testes para treino/amostragem/ranking/selecao SE(3).
  - Novos comandos aparecem em `python -m rna3d_local --help`.
  - `select-top5-se3` gera exatamente 5 modelos por alvo (ou falha cedo com erro acionavel).
  - `build-hybrid-candidates` aceita opcionalmente predicoes SE(3) sem regressao dos fluxos atuais.

## PLAN-079 - Escalabilidade de memoria para alvos longos (L<=5500)

- Objetivo: reduzir consumo de memoria e custo computacional do branch SE(3) para suportar sequencias longas sem violar contratos estritos.
- Escopo:
  - Substituir o bloco de atencao quadratica da torre 1D por opcoes de memoria eficiente:
    - `flash` com `scaled_dot_product_attention` + checkpointing;
    - `mamba_like` (SSM causal simplificado) com custo linear em memoria.
  - Introduzir construcao de grafo 3D esparso por raio fisico (`radius_angstrom`) e limite de vizinhos (`max_neighbors`) usando tensores `torch.sparse` com opcao explicita de backend (`torch_sparse`/`torch_geometric`).
  - Integrar o grafo esparso no backbone EGNN/IPA para eliminar materializacao densa NxN.
  - Expor parametros de escalabilidade no treino/amostragem via config (`sequence_tower`, `use_gradient_checkpointing`, `radius_angstrom`, `max_neighbors`, `graph_backend`).
  - Preservar fail-fast para contratos invalidos (config, backend indisponivel, grafo sem arestas, limites invalidos).
  - Cobrir com testes de contrato/escala para garantir ausencia de regressao no fluxo.
- Criterios de aceite:
  - `pytest -q` verde com testes novos de backend esparso e fail-fast.
  - `train-se3-generator` e `sample-se3-ensemble` funcionam com `sequence_tower=mamba_like` e `graph_backend=torch_sparse`.
  - Execucao com `graph_backend=torch_geometric` sem dependencia instalada falha cedo com erro acionavel.
  - Manifests de treino/amostragem registram parametros efetivos de escalabilidade.

## PLAN-080 - Matriz termodinamica 2D (BPP) no DataLoader

- Objetivo: incorporar probabilidades de pareamento de bases do ensemble de Boltzmann (BPP) ao fluxo SE(3), como viés continuo na representacao 2D.
- Escopo:
  - Implementar extracao BPP por backend explicito (`rnafold`, `linearfold`) com modo de teste explicito (`mock`) e sem fallback silencioso.
  - Integrar a extracao no DataLoader SE(3) para construir, por alvo:
    - marginais por residuo (probabilidade de estar pareado);
    - mapa esparso de pares (`i,j,p_ij`) para lookup em arestas dinamicas.
  - Estender `TargetGraph` para transportar sinais BPP e atualizar treino/inferencia para consumi-los.
  - Injetar BPP como viés continuo no backbone 2D (atenção/mensageria de arestas).
  - Expor configuracoes de backend/binarios/cache em `config_se3` e manifests efetivos.
  - Cobrir com testes de contrato (backend indisponivel, shape/consistencia) e smoke de integracao.
- Criterios de aceite:
  - `pytest -q` verde com testes novos para BPP.
  - Treino/amostragem SE(3) operam com `thermo_backend=mock` em testes e registram backend no manifest.
  - `thermo_backend=rnafold|linearfold` sem binario disponivel falha cedo com erro acionavel.
  - Bias BPP e efetivamente aplicado no caminho 2D sem quebrar contratos existentes.

## PLAN-081 - MSA covariancia + awareness multicadeia

- Objetivo: incorporar sinal evolutivo (MSA/covariancia) e awareness de multiplas cadeias no branch SE(3), preservando fail-fast.
- Escopo:
  - Implementar extracao de MSA por backend explicito (`mmseqs2`) com modo de teste (`mock`) e erro acionavel quando binario/DB nao estiverem disponiveis.
  - Derivar recursos de covariancia por residuo/par:
    - marginais por residuo (sinal de compensatory mutations);
    - mapa esparso por par (`i,j,cov_prob`) para injetar como bias continuo em arestas.
  - Adicionar parser de sequencias multicadeia (`chain separator`) no DataLoader e gerar IDs de cadeia por residuo.
  - Implementar Relative Positional Encoding 2D com offset artificial massivo em chain breaks (`chain_break_offset`, ex: +1000), bloqueando conectividade covalente entre cadeias.
  - Injetar bias combinado (BPP + covariancia + chain-break RPE) no EGNN/IPA.
  - Expor configuracao no `config_se3` e registrar parametros efetivos em manifests.
  - Cobrir com testes de contrato (backend indisponivel, parser multicadeia, offsets, consistencia de shapes) e smoke de integracao.
- Criterios de aceite:
  - `pytest -q` verde com testes novos de MSA e multicadeia.
  - Treino/amostragem SE(3) operam com `msa_backend=mock` em testes.
  - `msa_backend=mmseqs2` sem binario/db disponivel falha cedo com erro acionavel.
  - Bias de chain-break com offset massivo e aplicado no caminho 2D sem regressao dos fluxos existentes.

## PLAN-082 - Sondagem quimica PDB x QUICK_START

- Objetivo: cruzar reatividade quimica (DMS/2A3) com coordenadas PDB para estimar exposicao ao solvente por nucleotideo e injetar esse sinal no branch SE(3).
- Escopo:
  - Implementar modulo de `chemical mapping` que combina:
    - reatividade normalizada por alvo (QUICK_START);
    - exposicao geometrica por distancia ao centroide (PDB/labels);
    - exposicao quimica fusionada por residuo.
  - Integrar o mapping no DataLoader SE(3) para treinos (com PDB) e inferencia (sem PDB, modo explicito `quickstart_only`).
  - Estender `TargetGraph` com sinal de exposicao quimica e validacoes de cobertura/consistencia.
  - Injetar bias continuo de exposicao quimica nas arestas dinamicas EGNN/IPA.
  - Registrar origem do sinal (`quickstart_pdb_cross` vs `quickstart_only`) em manifest.
  - Cobrir com testes de contrato (cobertura incompleta, duplicatas, shape) e integracao.
- Criterios de aceite:
  - `pytest -q` verde com testes novos de `chemical mapping`.
  - Treino SE(3) usa sinal cruzado PDB+QUICK_START e falha cedo em cobertura inconsistente.
  - Inferencia SE(3) opera com modo explicito sem PDB (`quickstart_only`) sem fallback silencioso.
  - Bias quimico continuo aplicado no caminho 2D sem regressao dos fluxos existentes.
