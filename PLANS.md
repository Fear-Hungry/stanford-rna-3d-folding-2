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
