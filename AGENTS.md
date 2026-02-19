# AGENTS - Regras Operacionais do Repositorio

Estas regras sao obrigatorias para qualquer alteracao de pipeline, treino, validacao e submissao.

## Regras Criticas

1. Nao usar fallback silencioso.
- Proibido preencher saida com valores padrao (zeros, heuristica, baseline sintetico) quando houver falha de dados, modelo, chaves ou contrato.
- Se faltar dado obrigatorio, o fluxo deve falhar imediatamente.

2. Sempre falhar cedo (fail-fast) em erro de contrato.
- Qualquer divergencia entre submissao e `sample_submission` (colunas, tipos esperados, chaves, duplicatas, faltantes, extras) deve encerrar execucao com erro.

3. Erro deve ser explicito e acionavel.
- Toda excecao precisa informar:
- etapa (ex.: treino, export, validacao, submissao);
- arquivo/funcao quando aplicavel;
- causa tecnica;
- quantidade afetada e exemplos concretos (ex.: 5 chaves faltantes, exemplos `target:resid`).

4. Nunca "mascarar" erro para continuar pipeline.
- Nao fazer `try/except` que apenas loga aviso e segue fluxo em etapas criticas.
- Se o erro compromete corretude da saida, deve interromper.

5. Validacao local obrigatoria antes de submissao Kaggle.
- Toda submissao deve passar por validacao local estrita.
- Em caso de falha na validacao, submissao deve ser bloqueada.

## Padrao de Mensagem de Erro

Use o formato:

`[ETAPA] [ARQUIVO/FUNCAO] <causa> | impacto=<qtd> | exemplos=<itens>`

Exemplo:

`[EXPORT] [src/pipeline/export.py:reconcile_submission_to_sample] chaves da submissao nao batem com sample | missing=12 extra=3 | exemplos=R1107:45,R1107:46`

## Escopo Minimo Obrigatorio em Novas Mudancas

- Submissao e export devem operar em modo estrito por padrao.
- Qualquer modo permissivo, se existir, deve ser explicitamente desativado por padrao e nao pode ser usado em producao.
- Testes devem cobrir cenarios de erro de contrato (duplicata, faltante, extra, ordem invalida quando aplicavel).

## Boas Praticas para IA (Obrigatorias)

1. Disciplina de execucao
- Antes de alterar codigo, mapear arquivos impactados e validar dependencias.
- Toda mudanca deve vir com validacao local objetiva (teste, dry-run ou checagem equivalente).
- Nunca afirmar sucesso sem evidencias de execucao local.

2. Reprodutibilidade de treino/avaliacao
- Fixar seed em todos os experimentos quando aplicavel.
- Persistir artefatos em `runs/` com configuracao, metricas e timestamp.
- Registrar parametros efetivos usados no treino e na inferencia.

3. Eficiencia e estabilidade (big data)
- Priorizar leitura colunar, filtros por `target_id` e processamento em lotes.
- Evitar carregar dataset completo em memoria quando houver alternativa incremental.
- Em risco de OOM, reduzir budget explicitamente e reportar o motivo da reducao.
- Todo codigo implementado deve ser otimizado para a maquina atual, com foco em evitar OOM de RAM (uso de memoria deve ser controlado e previsivel).

4. Padrao para treino longo
- Executar smoke test apenas para validar integracao tecnica.
- Nao usar resultado de smoke test como base para decisao de qualidade final.
- Rodar treino completo somente apos smoke test passar sem erros.

5. Gating de submissao Kaggle
- Submissao so pode ocorrer apos validacao local estrita passar.
- Submeter no Kaggle apenas quando o `score` local do artefato candidato for estritamente maior que o melhor `score` local ja registrado no repositorio (baseline oficial em `EXPERIMENTS.md`).
- Nao submeter quando o artefato for de smoke test, execucao parcial ou com metrica regressiva sem justificativa explicita.
- Em falha de submissao (quota, notebook exception, scoring error), registrar causa exata e bloquear novas submissoes cegas.
- Regras obrigatorias desta competicao (notebook-only):
  - A competicao aceita apenas submissao via notebook (code competition). Submissao por arquivo local (`CreateSubmission`) deve ser considerada bloqueada por contrato.
  - Fluxo obrigatorio de submit:
    1) validar localmente com `python -m rna3d_local check-submission --sample <sample_submission.csv> --submission <submission.csv>`;
    2) executar notebook Kaggle de submissao para gerar `submission.csv` em `/kaggle/working`;
    3) submeter via referencia do notebook: `kaggle competitions submit -c <competicao> -k <owner/notebook> -f submission.csv -v <versao> -m <mensagem>`.
  - O notebook de submissao deve estar com internet desativada (`enable_internet=false`), senao a API pode retornar `FAILED_PRECONDITION`.
  - Em erro da API de submissao, registrar payload completo da resposta (status HTTP + body) e interromper novas tentativas cegas.

6. Politica de treino e Kaggle (Obrigatoria)
- Treinos (curtos ou longos) devem ser executados localmente com GPU.
- No Kaggle deve existir apenas notebook de submissao (inferencia/export/submission); proibido treinar no Kaggle.

7. Qualidade de codigo
- Preferir mudancas pequenas, rastreaveis e com baixo acoplamento.
- Remover codigo morto e evitar duplicacao de logica.
- Evitar caminhos absolutos locais fora do contexto Kaggle notebook.

8. Observabilidade e diagnostico
- Mensagens de erro devem seguir o padrao definido neste arquivo.
- Logs devem incluir etapa atual e contadores relevantes (linhas, targets, candidatos, tempo).
- Quando houver mismatch de dados, mostrar contagem e exemplos concretos.

9. Seguranca operacional
- Nao executar acoes destrutivas em dados/artefatos sem necessidade explicita.
- Nao sobrescrever artefatos criticos sem gerar versao rastreavel.
- Em caso de interrupcao/abort, verificar estado parcial antes de retomar pipeline.
- Proibido baixar, copiar ou incorporar solucoes completas de terceiros (repositorios, notebooks, submissions ou pipelines prontos) para uso direto no projeto.
- E permitido apenas baixar modelos pre-treinados como insumo tecnico, desde que a origem/licenca seja registrada e o uso seja validado localmente.

10. Transparencia com o usuario
- Informar claramente o que foi executado, o que falhou e o que ficou pendente.
- Separar fatos observados de inferencias.
- Nunca prometer meta de leaderboard sem evidencia empirica.

## Referencias Cientificas de Desenvolvimento (Guia)

Estas referencias devem orientar a evolucao tecnica do repositorio (pipeline, treino, validacao e submissao) como base de decisao e comparacao de abordagens. O uso e de guia tecnico: novas mudancas podem divergir desses trabalhos quando houver evidencia empirica e validacao local rigorosa.

Detalhamento comparativo (contribuicao, mecanismo, limitacoes, prioridade e aplicabilidade pratica): ver `README.md`.

- Template-based RNA structure prediction advanced through a blind code competition - https://doi.org/10.64898/2025.12.30.696949
- Assessment of Nucleic Acid Structure Prediction in CASP16 - https://doi.org/10.1002/prot.70072
- The RNA -Puzzles Assessments of RNA -Only Targets in CASP16 - https://doi.org/10.1002/prot.70052
- Accurate RNA 3D structure prediction using a language model-based deep learning approach - https://doi.org/10.1038/s41592-024-02487-0
- trRosettaRNA: automated prediction of RNA 3D structure with transformer network - https://doi.org/10.1038/s41467-023-42528-4
- Accurate Biomolecular Structure Prediction in CASP16 With Optimized Inputs to State-Of-The-Art Predictors - https://doi.org/10.1002/prot.70030
- Integrating end-to-end learning with deep geometrical potentials for ab initio RNA structure prediction - https://doi.org/10.1038/s41467-023-41303-9
- NuFold: end-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation - https://doi.org/10.1038/s41467-025-56261-7
- Geometric deep learning of RNA structure - https://doi.org/10.1126/science.abe5650
- Quality assessment of RNA 3D structure models using deep learning and intermediate 2D maps - https://doi.org/10.1101/2025.07.25.666746
- RNA3DB: A structurally-dissimilar dataset split for training and benchmarking deep learning models for RNA structure prediction - https://doi.org/10.1016/j.jmb.2024.168552
- Systematic benchmarking of deep-learning methods for tertiary RNA structure prediction - https://doi.org/10.1371/journal.pcbi.1012715

## Registros Obrigatorios (Planos e Mudancas)

Este arquivo (`AGENTS.md`) contem apenas regras operacionais. Planejamento e historico de mudancas DEVEM ficar em arquivos separados.

Arquivos oficiais:
- `PLANS.md`: backlog e planos ativos (com IDs `PLAN-###`).
- `CHANGES.md`: log append-only das mudancas implementadas.
- `EXPERIMENTS.md`: log append-only de experimentos executados (treino/inferencia/validacao/score).

Regras:
1. Proibido escrever planos, roadmaps ou "ultimas alteracoes" dentro de `AGENTS.md`.
2. Todo trabalho novo relevante para pipeline/treino/validacao/submissao deve estar descrito em `PLANS.md` antes de implementacao (com escopo e criterios de aceitacao).
3. Toda mudanca implementada deve gerar uma entrada em `CHANGES.md` (append-only) no mesmo dia em que foi finalizada.
4. Em `CHANGES.md`, e obrigatorio registrar:
- data (UTC) e autor;
- ID do plano (`PLAN-###`) ou `ADHOC`;
- resumo do que mudou;
- arquivos principais tocados;
- validacao local executada (comandos) e resultado;
- riscos conhecidos / follow-ups (se houver).
5. Nunca registrar "mudanca concluida" sem evidencias objetivas (comandos/testes/artefatos) conforme as regras deste repositorio.
6. Todo experimento executado deve gerar uma entrada em `EXPERIMENTS.md` (append-only) no mesmo dia (UTC) em que foi rodado.
7. Em `EXPERIMENTS.md`, e obrigatorio registrar (minimo):
- data (UTC) e autor;
- ID do plano (`PLAN-###`) ou `ADHOC`;
- objetivo/hipotese e comparacao (baseline vs novo);
- comandos executados + configuracao efetiva (config + overrides);
- parametros e hiperparametros efetivos (com valores);
- seeds usadas (quando aplicavel);
- versao do codigo (git commit) e dados (snapshot/caminhos);
- artefatos gerados em `runs/` (paths) + logs;
- metricas/score obtidos e custo (tempo, GPU/CPU, RAM quando relevante);
- conclusao + proximos passos (referenciar `PLANS.md` quando virar trabalho).
8. Escopo permitido de `PLANS.md`, `CHANGES.md` e `EXPERIMENTS.md`:
- Registrar somente planos, mudancas e experimentos com impacto direto em:
  - acuracia/poder preditivo dos modelos;
  - pontuacao no Kaggle (local ou leaderboard);
  - robustez competitiva relacionada ao score (ex.: evitar OOM/timeout que gera score 0).
- Nao usar esses arquivos para registrar engenharia de ML geral sem impacto competitivo claro.
