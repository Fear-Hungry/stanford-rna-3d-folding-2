# EXPERIMENTS.md

Log append-only de experimentos executados.

## PLAN-073

### 2026-02-16T01:37:02Z - marcusvinicius/Codex (smoke de integracao pos-reboot)

- Objetivo/hipotese:
  - Confirmar que o reboot greenfield da Fase 1 compila e executa validacoes essenciais com contratos estritos ativos.
- Comparacao:
  - Baseline: repositorio sem codigo executavel (`src/` removido).
  - Novo: pacote funcional com CLI + testes de contrato.
- Comandos executados:
  - `python -m pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Testes locais com fixtures sinteticas em `tmp_path`.
  - Sem uso de internet; sem treino longo.
- Parametros/hiperparametros:
  - N/A para treino longo (apenas smoke e unit tests).
- Seeds:
  - `seed=42` no teste de treino do reranker.
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dados: `input/stanford-rna-3d-folding-2/*`, `external_templates.csv` (nao mutados pelo experimento).
- Artefatos:
  - N/A (artefatos temporarios em `tmp_path` dos testes).
- Metricas/score/custo:
  - `pytest`: `7 passed in 1.22s`
  - CLI help: sucesso (`exit=0`) com superficie esperada.
- Conclusao:
  - Integracao minima da Fase 1 funcional e validada localmente.
- Proximos passos:
  - Rodar experimento real com artefatos locais de modelo GGUF/Ribonanza e registrar score local comparativo antes de submit.

## PLAN-074

### 2026-02-16T01:57:03Z - marcusvinicius/Codex (smoke de integracao Fase 2 hibrida)

- Objetivo/hipotese:
  - Validar integracao tecnica dos novos blocos da Fase 2 (assets offline, preditores RNAPro/Chai/Boltz, roteador hibrido, seletor top-5 e readiness gate) sem fallback silencioso.
- Comparacao:
  - Baseline: Fase 1 apenas (`PLAN-073`), sem comando/modelo da Fase 2.
  - Novo: CLI e testes cobrindo fluxo hibrido completo da Fase 2.
- Comandos executados:
  - `python -m rna3d_local --help`
  - `python -m pytest -q`
- Configuracao efetiva:
  - Testes sinteticos com fixtures locais:
    - 3 alvos (template forte, orfao, ligante);
    - roteamento threshold `template_score_threshold=0.65`;
    - selecao final `n_models=5`.
- Parametros/hiperparametros:
  - `n_models=5` em preditores offline e selecao top-5.
  - `baseline_score` no teste readiness: `0.20` com score candidato `0.10` (cenario de bloqueio esperado).
- Seeds:
  - N/A (sem treino estocastico nesta fase de smoke).
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dados locais: `input/stanford-rna-3d-folding-2/*`, `external_templates.csv`.
- Artefatos:
  - Artefatos sinteticos em `tmp_path` dos testes (`assets`, `routing.parquet`, `hybrid_top5.parquet`, `readiness.json`).
- Metricas/score/custo:
  - `pytest`: `11 passed in 1.20s`.
  - Cobertura efetiva de cenarios criticos:
    - manifest de assets;
    - rotas `template->tbm`, `orphan->chai1+boltz1`, `ligand->boltz1`;
    - gate de melhoria estrita bloqueando regressao.
- Conclusao:
  - Fase 2 implementada e tecnicamente funcional no ambiente local com contratos estritos preservados.
- Proximos passos:
  - Conectar runtimes reais de RNAPro/Boltz/Chai aos wrappers offline, medir custo real (tempo/GPU/RAM) e registrar comparativo local antes de novo submit.

## ADHOC

### 2026-02-16T02:09:31Z - marcusvinicius/Codex (preflight de submissao Kaggle apos Fase 2)

- Objetivo/hipotese:
  - Verificar se o candidato atual de notebook e submissao atende contrato estrito e se vale submit competitivo.
- Comparacao:
  - Baseline: ultimo score publico completo observado para o mesmo notebook (`0.261`, ref `50369108`, `scriptVersionId=297764190`).
  - Novo: output atual do mesmo notebook (`submission.csv` baixado do output de kernel).
- Comandos executados:
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_submit_attempt/notebook_output/submission.csv`
  - `kaggle kernels pull marcux777/stanford-rna3d-submit-prod-v2 -p runs/tmp_kernel_meta -m`
  - Paginar outputs via Kaggle API (`ApiListKernelSessionOutputRequest`) e baixar:
    - `submission.csv`
    - `run_dynamic_submit_v77/submission_patched.csv`
    - `run_dynamic_submit_v77/submission_tbm.csv`
  - Listar submissions via Kaggle API (`competition_submissions`) para status/score recentes.
  - `sha256sum runs/20260216_submit_attempt/notebook_output/submission.csv`
- Configuracao efetiva:
  - Competicao: `stanford-rna-3d-folding-2`.
  - Notebook: `marcux777/stanford-rna3d-submit-prod-v2`.
  - `kernel-metadata.json` com `enable_internet=false`.
- Parametros/hiperparametros:
  - N/A (apenas preflight de submissao e validacao contratual).
- Seeds:
  - N/A.
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dados: `input/stanford-rna-3d-folding-2/sample_submission.csv` + output remoto do kernel.
- Artefatos:
  - `runs/20260216_submit_attempt/notebook_output/submission.csv`
  - `runs/20260216_submit_attempt/notebook_output/submission_patched.csv`
  - `runs/20260216_submit_attempt/notebook_output/submission_tbm.csv`
  - `runs/tmp_kernel_meta/kernel-metadata.json`
- Metricas/score/custo:
  - `check-submission`: `ok=true`.
  - `sha256(submission.csv)`: `392e12d1a957af66cfa382d5e64f3b3cb652bbbb2146f089fbf3611ee430b583`.
  - Hash identico ao log do notebook da versao atual.
  - Ultimo score publico completo observado para este notebook/versao: `0.261` (ref `50369108`).
- Conclusao:
  - Candidato atual e formatado corretamente, mas nao traz evidencia de melhoria (mesmo notebook/mesmo hash ja pontuado).
  - Sem score local novo do candidato, gate estrito de submit competitivo nao pode aprovar envio.
- Proximos passos:
  - Gerar candidato com mudancas reais de Fase 2 no notebook de submit, calcular score local novo e somente entao submeter.

## PLAN-075

### 2026-02-16T02:18:51Z - marcusvinicius/Codex (integracao do notebook de submissao com Fase 1 + Fase 2)

- Objetivo/hipotese:
  - Garantir que o notebook de submissao execute o pipeline completo Fase 1 + Fase 2 com contratos estritos e sem fallback silencioso.
- Comparacao:
  - Baseline: notebook anterior focado em `retrieve-templates` + `predict-tbm` + patch DRfold2.
  - Novo: notebook com cadeia completa `description + chemical + latent retrieval + reranker + TBM + modelos fase2 + roteamento hibrido + top5 + export/check`.
- Comandos executados:
  - Validador estrutural/sintatico do notebook:
    - `python - <<'PY' ... json.loads(...); compile(code, ...); assert tokens ... ; print('NOTEBOOK_OK') ... PY`
  - Validador da superficie CLI usada no notebook:
    - `python - <<'PY' ... subprocess.run([python -m rna3d_local <cmd> --help]) ... ; print('CLI_SURFACE_OK') ... PY`
  - Verificacao de metadado do kernel:
    - `python - <<'PY' ... print(code_file, enable_internet) ... PY`
- Configuracao efetiva:
  - Notebook-alvo: `runs/20260216_plan073_submit_preflight/kernel_source/stanford-rna3d-submit-prod-v2.ipynb`.
  - Pipeline definido para gerar `submission.csv` em `/kaggle/working`.
  - `enable_internet=false` preservado em `kernel-metadata.json`.
- Parametros/hiperparametros:
  - `N_MODELS=5`
  - `TOP_K=128`
  - `TEMPLATE_SCORE_THRESHOLD=0.65`
  - Pesos de fusao latent retrieval: `embed=0.70`, `llm=0.20`, `seq=0.10`.
- Seeds:
  - N/A (nao houve treino local nesta validacao; notebook usa reranker pretreinado por contrato).
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dados: validacao local apenas (sem execucao full do notebook no Kaggle nesta etapa).
- Artefatos:
  - Notebook atualizado: `runs/20260216_plan073_submit_preflight/kernel_source/stanford-rna3d-submit-prod-v2.ipynb`.
- Metricas/score/custo:
  - `NOTEBOOK_OK`
  - `CLI_SURFACE_OK`
  - Sem score local nesta etapa (mudanca estrutural de notebook).
- Conclusao:
  - Notebook migrado para fluxo completo Fase 1 + Fase 2 com fail-fast e validacoes estritas de ativos/CLI.
- Proximos passos:
  - Executar o notebook no Kaggle, coletar `submission.csv` e comparar score local/competitivo antes de submit.

## PLAN-076

### 2026-02-16T03:10:14Z - marcusvinicius/Codex (hardening de execucao Kaggle do pipeline completo)

- Objetivo/hipotese:
  - Eliminar falhas contratuais no notebook completo Fase 1+2 e validar geracao de submissao final em modo estrito.
- Comparacao:
  - Baseline imediato: execucoes Kaggle `v80-v82` com abortos em `prepare-chemical-features` (schema quickstart) e `predict-tbm` (template incompleto).
  - Novo: schema adapter estrito para QUICK_START de templates + selecao TBM por cobertura completa de residuos.
- Comandos executados:
  - Validacao local:
    - `pytest -q tests/test_chemical_features.py tests/test_tbm.py tests/test_phase2_hybrid.py tests/test_reranker.py`
    - `pytest -q`
  - Publicacao de src Kaggle:
    - `kaggle datasets version -p runs/20260216_plan075_src_dataset_v2 -m "PLAN-075: chemical quickstart schema adapter strict" -r zip`
    - `kaggle datasets version -p runs/20260216_plan075_src_dataset_v2 -m "PLAN-075: support template quickstart single xyz triplet" -r zip`
    - `kaggle datasets version -p runs/20260216_plan075_src_dataset_v2 -m "PLAN-075: TBM chooses first valid full-coverage templates" -r zip`
  - Publicacao e execucao do notebook:
    - `kernel_push` -> `marcux777/stanford-rna3d-submit-prod-v2` versoes `81`, `82`, `83`.
    - `kaggle kernels status marcux777/stanford-rna3d-submit-prod-v2` (polling ate estado final).
    - `kernel_output` para `runs/20260216_plan075_kernel_output_v81`, `..._v82`, `..._v83`.
  - Validacao final de submissao:
    - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan075_kernel_output_v83/submission.csv`
- Configuracao efetiva:
  - Notebook: `runs/20260216_plan073_submit_preflight/kernel_source/stanford-rna3d-submit-prod-v2.ipynb`
  - Dataset src runtime: `marcux777/stanford-rna3d-reboot-src-v2` (ultima versao)
  - `enable_internet=false` (contrato notebook-only preservado)
  - Parametros notebook: `TOP_K=128`, `N_MODELS=5`, `TEMPLATE_SCORE_THRESHOLD=0.65`
- Parametros/hiperparametros:
  - `prepare-chemical-features`: suporte dual de schema (`target_id/resid/dms/2a3` ou `ID/resid/x_i,y_i,z_i`).
  - `predict-tbm`: selecao dos primeiros `n_models` templates com cobertura completa por alvo.
- Seeds:
  - `seed=42` no treino do reranker dentro do notebook.
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Competicao: `stanford-rna-3d-folding-2`
  - Inputs Kaggle: datasets declarados em `kernel-metadata.json` (incluindo `stanford-rna3d-reboot-src-v2` e `ribonanza-quickstart-3d-templates`).
- Artefatos:
  - Logs de execucao:
    - `runs/20260216_plan075_kernel_output_v81/stanford-rna3d-submit-prod-v2.log`
    - `runs/20260216_plan075_kernel_output_v82/stanford-rna3d-submit-prod-v2.log`
    - `runs/20260216_plan075_kernel_output_v83/stanford-rna3d-submit-prod-v2.log`
  - Saida final valida:
    - `runs/20260216_plan075_kernel_output_v83/submission.csv`
    - `runs/20260216_plan075_kernel_output_v83/run_phase1_phase2_full_v2/*` (artefatos completos da Fase 1+2)
- Metricas/score/custo:
  - Local: `pytest -q` -> `17 passed in 1.19s`
  - Kaggle:
    - `v81`: fail em `CHEMICAL_FEATURES` (tripletos insuficientes na regra antiga)
    - `v82`: fail em `PREDICT_TBM` (template incompleto selecionado)
    - `v83`: `COMPLETE`, `check-submission` no notebook -> `ok=true`
  - Validacao local do `submission.csv` remoto -> `ok=true`
  - `submission_sha256` reportado pelo notebook: `6e2ab384d93b5afe65489b592814855e251d8fdd7f099c1557f56ce036296ebc`
- Conclusao:
  - O notebook de submissao agora executa o pipeline completo Fase 1+2 fim-a-fim no Kaggle com contratos estritos preservados.
  - Formato de entrada/saida validado; `submission.csv` pronto do ponto de vista contratual.
- Proximos passos:
  - Rodar gate competitivo com score local novo do candidato (melhoria estrita) antes de qualquer submissao oficial ao leaderboard.

## PLAN-077

### 2026-02-16T12:23:41Z - marcusvinicius/Codex (fix de robustez para hidden rerun)

- Objetivo/hipotese:
  - Eliminar erro de rerun em hidden dataset causado por premissas fortes de tamanho/cobertura no notebook de submissao.
- Comparacao:
  - Baseline: submissao `50393535` com erro `Your notebook hit an unhandled error while rerunning your code... hidden dataset can be larger/smaller/different`.
  - Novo: notebook `v84` sem treino no Kaggle + pipeline tolerante a cobertura parcial de retrieval/TBM.
- Comandos executados:
  - Analise de falha:
    - consulta de metadata da submissao via `competition_submissions` (incluindo `error_description`).
  - Validacao local:
    - `pytest -q`
  - Publicacao de artefatos:
    - `kaggle datasets version -p runs/20260216_plan075_src_dataset_v2 -m "PLAN-077: hidden rerun robustness (no kaggle reranker train; tbm/retrieval tolerant)" -r zip`
    - `kaggle kernels push -p runs/20260216_plan073_submit_preflight/kernel_source` -> notebook `v84`
  - Execucao de notebook:
    - `kaggle kernels status marcux777/stanford-rna3d-submit-prod-v2` ate `COMPLETE`
    - `kaggle kernels output marcux777/stanford-rna3d-submit-prod-v2 -p runs/20260216_plan077_kernel_output_v84 -o -q`
  - Validacao de submissao:
    - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan077_kernel_output_v84/submission.csv`
  - Novo envio de competicao:
    - `kaggle competitions submit -c stanford-rna-3d-folding-2 -k marcux777/stanford-rna3d-submit-prod-v2 -f submission.csv -v 84 -m "PLAN-077: hidden-rerun robustness (remove kaggle reranker train)"`
- Configuracao efetiva:
  - Notebook: `marcux777/stanford-rna3d-submit-prod-v2` `version=84`.
  - `enable_internet=false` preservado.
  - Fase 1+2 ativa, com ranking TBM vindo direto de `retrieve-templates-latent`.
- Parametros/hiperparametros:
  - `TOP_K=128`, `N_MODELS=5`, `TEMPLATE_SCORE_THRESHOLD=0.65`.
  - Reranker de treino no notebook removido (politica no-train-on-Kaggle).
- Seeds:
  - N/A (sem treino no notebook).
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dataset src: `marcux777/stanford-rna3d-reboot-src-v2` (nova versao publicada neste experimento).
- Artefatos:
  - `runs/20260216_plan077_kernel_output_v84/stanford-rna3d-submit-prod-v2.log`
  - `runs/20260216_plan077_kernel_output_v84/submission.csv`
- Metricas/score/custo:
  - `pytest -q`: `18 passed in 1.32s`.
  - Notebook `v84`: `COMPLETE`, `check-submission` interno `ok=true`.
  - `submission_sha256` do notebook: `657454b44169922f7c7cf3bee1bf38043e2085653b19a5873e8ab283617aa822`.
  - Nova submissao criada: `ref=50393739` (status inicial `PENDING`).
- Conclusao:
  - Regressao de robustez para hidden rerun foi enderecada tecnicamente sem quebrar contrato de export/submission.
  - Aguarda resultado final do backend da competicao para confirmar fim do erro de rerun.
- Proximos passos:
  - Monitorar `ref=50393739` ate `COMPLETE` e registrar `score`/`error_description`.
  - Se persistir erro oculto, coletar nova assinatura de falha e hardenizar o ponto especifico.

## PLAN-078

### 2026-02-16T12:49:22Z - marcusvinicius/Codex (integracao do branch SE(3) generativo)

- Objetivo/hipotese:
  - Viabilizar no repositorio um caminho experimental para Best-of-5 com diversidade estrutural real (EGNN+IPA + Diffusion/Flow), mantendo contratos estritos.
- Comparacao:
  - Baseline: pipeline atual sem comandos/modelos SE(3) generativos.
  - Novo: comandos completos de treino/amostragem/ranking/selecao Top-5 SE(3) e integracao opcional no roteador hibrido.
- Comandos executados:
  - `python -m rna3d_local --help`
  - `pytest -q`
- Configuracao efetiva:
  - Novos comandos habilitados no parser/CLI.
  - Roteador hibrido com suporte opcional `--se3` (desativado por padrao quando ausente).
- Parametros/hiperparametros:
  - Defaults do branch SE(3):
    - `method=both`
    - `diversity_lambda=0.35`
    - `n_models=5` na selecao final.
- Seeds:
  - `seed` exposto e validado em `train-se3-generator` e `sample-se3-ensemble`.
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dados de teste sinteticos via fixtures em `tests/test_se3_pipeline.py`.
- Artefatos:
  - Artefatos temporarios criados em `tmp_path` pelos testes de pipeline SE(3).
- Metricas/score/custo:
  - `pytest -q`: `21 passed in 1.41s`.
  - `--help`: superficie CLI contem os quatro comandos novos do branch SE(3).
- Conclusao:
  - Branch SE(3) generativo integrado com contratos fail-fast e cobertura de teste.
  - Integracao no roteador hibrido implementada como opcional sem regressao do fluxo existente.
- Proximos passos:
  - Rodar benchmark local com dados reais (tempo/GPU/RAM + score local) para calibrar Diffusion/Flow e criterio de diversidade antes de promover para uso competitivo.

## PLAN-079

### 2026-02-16T12:59:00Z - marcusvinicius/Codex (escala de memoria L<=5500 no branch SE(3))

- Objetivo/hipotese:
  - Reduzir custo de memoria do branch SE(3) substituindo componentes densos NxN por mecanismos lineares/esparsos sem quebrar contrato fail-fast.
- Comparacao:
  - Baseline: torre sem mecanismo linear dedicado e backbones EGNN/IPA com materializacao densa de pares.
  - Novo: torre `flash|mamba_like` + grafo dinamico esparso por raio fisico (`torch_sparse`/`torch_geometric`).
- Comandos executados:
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Novos campos de treino SE(3): `sequence_tower`, `sequence_heads`, `use_gradient_checkpointing`, `graph_backend`, `radius_angstrom`, `max_neighbors`, `graph_chunk_size`.
  - Defaults: `sequence_tower=mamba_like`, `graph_backend=torch_sparse`, `radius_angstrom=14.0`, `max_neighbors=64`, `graph_chunk_size=512`.
- Parametros/hiperparametros:
  - Teste de smoke linear:
    - `method=diffusion`, `hidden_dim=16`, `num_layers=1`, `sequence_tower=mamba_like`, `use_gradient_checkpointing=true`,
    - `graph_backend=torch_sparse`, `radius_angstrom=14.0`, `max_neighbors=12`, `graph_chunk_size=64`.
- Seeds:
  - `seed=17` no smoke de treino/amostragem em `tests/test_se3_memory.py`.
- Versao de codigo/dados:
  - `git commit`: `8b9f720`
  - Dados sinteticos gerados em `tmp_path` pelos testes (`LONG1`, comprimento 96 para smoke).
- Artefatos:
  - `tests/test_se3_memory.py` valida contratos de:
    - limite de grau no grafo esparso;
    - indisponibilidade de `torch_geometric`/`radius_graph`;
    - treino+amostragem com configuracao linear e persistencia de parametros no manifest.
- Metricas/score/custo:
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `9 passed in 1.91s`
  - `pytest -q` -> `24 passed in 2.04s`
  - `python -m rna3d_local --help` -> comandos CLI esperados visiveis.
- Conclusao:
  - O branch SE(3) passa a operar com mecanismos de memoria linear/esparsa e contratos de erro acionaveis para backends/parametros invalidos.
- Proximos passos:
  - Rodar benchmark dedicado com GPU e comprimentos altos (>=3000 e >=5500) para medir VRAM, throughput e latencia por backend/torre.

## PLAN-080

### 2026-02-16T13:07:32Z - marcusvinicius/Codex (BPP termodinamica 2D no DataLoader)

- Objetivo/hipotese:
  - Injetar sinal termodinamico de estrutura secundaria (BPP) no branch SE(3) para guiar empacotamento terciario com viés continuo 2D.
- Comparacao:
  - Baseline: branch SE(3) sem BPP explicita; uso apenas de `pair_prob` e quimica por residuo.
  - Novo: BPP por alvo com marginais + pares esparsos e bias em arestas EGNN/IPA.
- Comandos executados:
  - `pytest -q tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Novos campos em `config_se3`:
    - `thermo_backend` (`rnafold|linearfold|mock`)
    - `rnafold_bin`
    - `linearfold_bin`
    - `thermo_cache_dir`
  - Em testes de integracao: `thermo_backend=mock`.
- Parametros/hiperparametros:
  - Mantidos defaults de escala SE(3) da PLAN-079.
  - BPP em testes:
    - backend `mock` para smoke deterministico;
    - validacao de erro para `rnafold_bin` inexistente.
- Seeds:
  - Herdadas dos testes SE(3): `seed=123` e `seed=17`.
- Versao de codigo/dados:
  - `git commit`: `311512e`
  - Dados sinteticos em `tmp_path` nos testes.
- Artefatos:
  - Novo modulo `src/rna3d_local/training/thermo_2d.py`.
  - Novos testes `tests/test_thermo_2d.py`.
  - Manifests de treino/amostragem passam a registrar configuracao termodinamica.
- Metricas/score/custo:
  - `pytest -q tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `12 passed in 2.00s`
  - `pytest -q` -> `27 passed in 2.00s`
  - CLI: help sem regressao.
- Conclusao:
  - BPP termodinamica 2D integrada no pipeline SE(3) com fail-fast para binarios/contratos invalidos e com bias continuo aplicado no backbone.
- Proximos passos:
  - Medir custo de inferencia com `thermo_backend=rnafold` e cache habilitado em comprimentos longos para calibrar budget Kaggle.

## PLAN-081

### 2026-02-16T13:15:45Z - marcusvinicius/Codex (MSA covariancia + awareness multicadeia)

- Objetivo/hipotese:
  - Acrescentar sinal evolutivo de covariancia e noção explícita de chain breaks no branch SE(3) para melhorar contatos terciarios em casos multicadeia.
- Comparacao:
  - Baseline: branch com bias BPP, sem covariancia MSA e sem RPE multicadeia com offset massivo.
  - Novo: bias combinado BPP + MSA/cov + relative position com `chain_break_offset`.
- Comandos executados:
  - `pytest -q tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Novos campos em `config_se3`:
    - `msa_backend` (`mmseqs2|mock`)
    - `mmseqs_bin`, `mmseqs_db`, `msa_cache_dir`
    - `chain_separator`, `chain_break_offset`
    - `max_msa_sequences`, `max_cov_positions`, `max_cov_pairs`
  - Em testes de integracao: `msa_backend=mock` e `thermo_backend=mock`.
- Parametros/hiperparametros:
  - Defaults adicionados:
    - `chain_break_offset=1000`
    - `max_msa_sequences=96`
    - `max_cov_positions=256`
    - `max_cov_pairs=8192`
- Seeds:
  - Herdadas dos testes SE(3): `seed=123`, `seed=17`, `seed=7`.
- Versao de codigo/dados:
  - `git commit`: `311512e`
  - Dados sinteticos em `tmp_path` nos testes.
- Artefatos:
  - Novo modulo `src/rna3d_local/training/msa_covariance.py`.
  - Novo parser multicadeia `src/rna3d_local/se3/sequence_parser.py`.
  - Novo teste `tests/test_msa_covariance.py`.
  - Testes atualizados para fluxo multicadeia/BPP integrado.
- Metricas/score/custo:
  - `pytest -q tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `17 passed in 1.96s`
  - `pytest -q` -> `32 passed in 1.98s`
  - CLI sem regressao de comandos.
- Conclusao:
  - O branch SE(3) agora injeta covariancia MSA e encoding multicadeia com offset massivo no caminho 2D, mantendo contratos fail-fast.
- Proximos passos:
  - Rodar benchmark local com `msa_backend=mmseqs2` e DB real para calibrar cobertura/custo em alvos longos e multicadeia.

## PLAN-082

### 2026-02-16T13:20:50Z - marcusvinicius/Codex (sondagem quimica PDB x QUICK_START)

- Objetivo/hipotese:
  - Reduzir ambiguidade estrutural (incluindo pseudoknots) adicionando exposicao ao solvente por residuo obtida via cruzamento de reatividade quimica com geometria PDB.
- Comparacao:
  - Baseline: branch com BPP+MSA+multicadeia, sem sinal dedicado de `chemical mapping` cruzado com PDB.
  - Novo: `chemical mapping` continuo em nos/arestas com fonte explicita (`quickstart_pdb_cross`/`quickstart_only`).
- Comandos executados:
  - `pytest -q tests/test_chemical_mapping.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Sem novo comando CLI; integracao interna no DataLoader SE(3).
  - Treino: cruzamento com `labels` (PDB) -> `quickstart_pdb_cross`.
  - Inferencia: modo explicito sem labels -> `quickstart_only`.
- Parametros/hiperparametros:
  - Exposicao quimica por alvo:
    - normalizacao min-max de `reactivity_dms` e `reactivity_2a3`;
    - fusao com exposicao geometrica (distancia ao centroide) quando PDB disponivel.
  - Bias de aresta: media da exposicao quimica dos dois residuos (`chem_edge_bias`).
- Seeds:
  - Herdadas dos testes SE(3): `seed=123`, `seed=17`, `seed=7`.
- Versao de codigo/dados:
  - `git commit`: `311512e`
  - Dados sinteticos em `tmp_path` para testes unitarios.
- Artefatos:
  - Novo modulo: `src/rna3d_local/training/chemical_mapping.py`.
  - Novo teste: `tests/test_chemical_mapping.py`.
  - Manifests de treino/amostragem com `chemical_mapping_source_counts`.
- Metricas/score/custo:
  - `pytest -q tests/test_chemical_mapping.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `20 passed in 1.99s`
  - `pytest -q` -> `35 passed in 2.08s`
  - CLI sem regressao de superficie.
- Conclusao:
  - Sondagem quimica integrada ao branch SE(3) com fail-fast de cobertura e origem de sinal rastreavel por manifest.
- Proximos passos:
  - Rodar benchmark local com alvos reais ricos em pseudoknot para medir ganho de score e custo computacional.

## PLAN-083

### 2026-02-16T13:28:01Z - marcusvinicius/Codex (folds por homologia com clustering restrito)

- Objetivo/hipotese:
  - Evitar falsa generalizacao de Random K-Folds removendo vazamento de homologia estrutural entre folds.
- Comparacao:
  - Baseline: splits sem garantia de isolamento por cluster homologo.
  - Novo: splits por cluster com identidade/cobertura restritas e checagem explicita de leakage.
- Comandos executados:
  - `pytest -q tests/test_homology_folds.py tests/test_chemical_mapping.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Novo comando: `build-homology-folds`.
  - Backends: `mmseqs2`, `cdhit_est`, `mock`.
  - Defaults competitivos:
    - `identity_threshold=0.40`
    - `coverage_threshold=0.80`
    - `n_folds=5`
- Parametros/hiperparametros:
  - Teste de integracao com `backend=mock`, `identity_threshold=0.85`, `coverage_threshold=0.80`, `n_folds=2`.
  - Teste de contrato com backend ausente:
    - `backend=mmseqs2`, `mmseqs_bin=/bin/nao_existe_mmseqs`.
- Seeds:
  - N/A (split deterministico sem treino estocastico).
- Versao de codigo/dados:
  - `git commit`: `da4391c`
  - Dados sinteticos para testes em `tmp_path`.
- Artefatos:
  - `clusters.parquet`, `train_folds.parquet`, `homology_folds_manifest.json`.
  - `manifest.stats.max_folds_per_cluster_train` como criterio de leakage.
- Metricas/score/custo:
  - `pytest -q tests/test_homology_folds.py ...` -> `23 passed in 2.09s`
  - `pytest -q` -> `38 passed in 2.08s`
  - CLI sem regressao.
- Conclusao:
  - Folds homologo-estritos implementados com fail-fast para contratos/binarios e bloqueio de leakage por cluster.
- Proximos passos:
  - Rodar experimento comparativo oficial (Random K-Folds vs homology folds) com score local para quantificar risco de private-LB drop.

## PLAN-084

### 2026-02-16T13:36:32Z - marcusvinicius/Codex (estratificacao de dominio + prioridade orphan)

- Objetivo/hipotese:
  - Reduzir risco de private-LB drop garantindo representatividade topologica por dominio nos folds e priorizando explicitamente desempenho em alvos orphan.
- Comparacao:
  - Baseline: folds por homologia sem estratificacao de dominio e sem relatorio orphan-prioritario.
  - Novo: folds por cluster + dominio (estrito) e avaliacao com `priority_score` ponderado por orphan.
- Comandos executados:
  - `pytest -q tests/test_homology_folds.py tests/test_homology_eval.py tests/test_phase2_hybrid.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - `build-homology-folds`:
    - `strict_domain_stratification=true` por padrao;
    - fontes de dominio: `--domain-labels` ou `--domain-column` ou `--description-column`.
  - `evaluate-homology-folds`:
    - entrada de metricas por alvo;
    - classificacao orphan por `--orphan-labels` ou `--retrieval`;
    - defaults: `orphan_score_threshold=0.65`, `orphan_weight=0.70`.
- Parametros/hiperparametros:
  - Teste de estratificacao: `identity_threshold=0.95`, `coverage_threshold=0.80`, `n_folds=2`, dominio inferido por descricao.
  - Teste de prioridade orphan: score por alvo com classificador orphan por `final_score < 0.65`.
- Seeds:
  - N/A (split/relatorio deterministico, sem treino estocastico).
- Versao de codigo/dados:
  - `git commit`: `da4391c`
  - Dados sinteticos de teste em `tmp_path`.
- Artefatos:
  - Novo modulo `src/rna3d_local/homology_eval.py`.
  - `train_folds.parquet` com `domain_label`.
  - `homology_folds_manifest.json` com diagnosticos por dominio.
  - Relatorio JSON de `evaluate-homology-folds` com `priority_score`.
- Metricas/score/custo:
  - `pytest -q tests/test_homology_folds.py tests/test_homology_eval.py tests/test_phase2_hybrid.py` -> `11 passed in 0.87s`
  - `pytest -q` -> `42 passed in 2.12s`
  - CLI sem regressao de superficie (`build-homology-folds`, `evaluate-homology-folds` visiveis).
- Conclusao:
  - Estratificacao de dominio e avaliacao orphan-prioritaria foram integradas com fail-fast, mantendo isolamento por homologia.
- Proximos passos:
  - Rodar benchmark local real com metricas por alvo (baseline vs novo) para calibrar `orphan_weight` e thresholds por dominio.

## PLAN-085

### 2026-02-16T13:41:30Z - marcusvinicius/Codex (loss FAPE + TM-core + Clash no treino SE(3))

- Objetivo/hipotese:
  - Melhorar alinhamento topologico ao scorer US-align/TM-score reduzindo dependencia de RMSD/MSE puro e penalizando colisao atomica nao-covalente.
- Comparacao:
  - Baseline: loss principal `MSE(coords)` + perdas gerativas (diffusion/flow).
  - Novo: loss estrutural composta `w_mse*MSE + w_fape*FAPE + w_tm*(1-TM_core) + w_clash*Clash` + perdas gerativas.
- Comandos executados:
  - `pytest -q tests/test_se3_losses.py tests/test_se3_pipeline.py tests/test_se3_memory.py`
  - `pytest -q`
  - `python -m rna3d_local --help`
- Configuracao efetiva:
  - Novos campos em `config_se3`:
    - `loss_weight_mse`, `loss_weight_fape`, `loss_weight_tm`, `loss_weight_clash`
    - `fape_clamp_distance`, `fape_length_scale`
    - `vdw_min_distance`, `vdw_repulsion_power`
    - `loss_chunk_size`
  - Defaults adicionados:
    - `loss_weight_mse=0.0`
    - `loss_weight_fape=1.0`
    - `loss_weight_tm=1.0`
    - `loss_weight_clash=5.0`
    - `fape_clamp_distance=10.0`
    - `fape_length_scale=10.0`
    - `vdw_min_distance=2.1`
    - `vdw_repulsion_power=4`
    - `loss_chunk_size=256`
- Parametros/hiperparametros:
  - FAPE calculado em blocos (`loss_chunk_size`) para limitar pico de RAM.
  - Clash loss ignora pares covalentes (mesma cadeia com `|i-j| <= 1`) e aplica repulsao por potencia configuravel.
  - TM-core usa alinhamento diferenciavel por Kabsch + score robusto por residuo.
- Seeds:
  - Testes de treino/inferencia mantidos com seeds existentes (`123`, `17`, `7`).
- Versao de codigo/dados:
  - `git commit`: `da4391c`
  - Dados sinteticos de teste em `tmp_path`.
- Artefatos:
  - Novo modulo `src/rna3d_local/training/losses_se3.py`.
  - `metrics.json` de treino agora inclui traces e valores finais por componente de loss.
- Metricas/score/custo:
  - `pytest -q tests/test_se3_losses.py tests/test_se3_pipeline.py tests/test_se3_memory.py` -> `10 passed in 1.97s`
  - `pytest -q` -> `46 passed in 2.17s`
  - CLI sem regressao de comandos.
- Conclusao:
  - Loss estrutural alinhada ao core topologico e fisica estereoquimica integrada com contratos estritos.
- Proximos passos:
  - Calibrar pesos de loss por benchmark local de score por alvo (especialmente orphans e alvos flexiveis longos).

## PLAN-086

### 2026-02-16T14:00:43Z - marcusvinicius/Codex (minimizacao energetica pos-inferencia)

- Objetivo/hipotese:
  - Elevar robustez de TM-score pre-submit reduzindo clashes e distorcoes locais residuais via minimizacao geometrica rapida nos 5 modelos finais.
- Comparacao:
  - Baseline: export/submissao direta a partir de `top5` sem refinamento fisico.
  - Novo: passo intermediario `minimize-ensemble` antes de `export-submission`.
- Comandos executados:
  - `pytest -q tests/test_minimization.py tests/test_phase2_hybrid.py tests/test_description_and_submission.py`
  - `python -m rna3d_local --help`
  - `python -m rna3d_local minimize-ensemble --help`
  - `pytest -q`
- Configuracao efetiva:
  - Novo comando `minimize-ensemble` com backend:
    - `openmm` (default)
    - `pyrosetta`
    - `mock` (teste local)
  - Parametros default:
    - `max_iterations=120`
    - `bond_length_angstrom=5.9`
    - `bond_force_k=60.0`
    - `angle_force_k=8.0`
    - `angle_target_deg=120.0`
    - `vdw_min_distance_angstrom=2.1`
    - `vdw_epsilon=0.20`
- Parametros/hiperparametros:
  - OpenMM: bond stretching + angle regularization + repulsao curta distancia (`step(sigma-r)*(sigma/r)^12`), com exclusao de pares covalentes.
  - Mock: suavizacao deterministica + ajuste de bonds + empurrao repulsivo para pares nao-covalentes.
- Seeds:
  - N/A (minimizacao deterministica para mesmo input/config; sem treino estocastico).
- Versao de codigo/dados:
  - `git commit`: `da4391c`
  - Dados sinteticos em `tmp_path` para testes de contrato.
- Artefatos:
  - Novo modulo `src/rna3d_local/minimization.py`.
  - Manifest de refinamento: `minimize_ensemble_manifest.json` com `shift_max_angstrom` e `shift_mean_angstrom`.
- Metricas/score/custo:
  - `pytest -q tests/test_minimization.py tests/test_phase2_hybrid.py tests/test_description_and_submission.py` -> `9 passed in 0.87s`
  - `pytest -q` -> `49 passed in 2.36s`
  - CLI sem regressao e com novo comando de minimizacao.
- Conclusao:
  - Minimizacao pos-inferencia integrada com contrato estrito para uso no fluxo pre-submit.
- Proximos passos:
  - Rodar benchmark local com scorer por alvo (antes/depois da minimizacao) para calibrar `max_iterations` e forcas no budget de tempo Kaggle.
