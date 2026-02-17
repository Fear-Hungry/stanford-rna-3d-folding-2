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

## PLAN-087

### 2026-02-16T14:21:25Z - marcusvinicius/Codex (roteamento anti-OOM + grafo sem distancia densa)

- Objetivo/hipotese:
  - Evitar OOM de VRAM em alvos longos forçando roteamento para SE(3) e removendo caminho de matriz de distancias densa no backend esparso.
- Comparacao:
  - Baseline: roteador podia enviar alvos longos para fontes fundacionais e backend `torch_sparse` usava `torch.cdist` chunk-vs-all.
  - Novo: fallback ultralongo obrigatorio para `generative_se3` + busca de vizinhos espacial sem NxN denso.
- Comandos executados:
  - `pytest -q tests/test_phase2_hybrid.py tests/test_se3_memory.py tests/test_se3_pipeline.py`
  - `pytest -q`
  - `python -m rna3d_local build-hybrid-candidates --help`
- Configuracao efetiva:
  - Novo parametro de roteamento:
    - `ultra_long_seq_threshold` (default `1500`)
  - `hybrid_router` marca no routing:
    - `target_length`
    - `ultralong_fallback`
  - `torch_sparse`:
    - prioriza `torch_cluster.radius_graph` quando disponivel;
    - fallback explicito para particionamento espacial por celulas sem matriz densa.
- Parametros/hiperparametros:
  - Testes de roteamento ultralongo com `ultra_long_seq_threshold=1500`.
  - Teste de contrato anti-denso com monkeypatch em `torch.cdist` para garantir que backend `torch_sparse` nao invoca `cdist`.
- Seeds:
  - N/A (roteamento/grafo deterministico para mesmo input).
- Versao de codigo/dados:
  - `git commit`: `efbcb7c`
  - Dados sinteticos em `tmp_path`.
- Artefatos:
  - Manifest de roteamento agora registra `ultra_long_seq_threshold`.
  - README com comando de roteamento atualizado (`--ultra-long-seq-threshold`).
- Metricas/score/custo:
  - `pytest -q tests/test_phase2_hybrid.py tests/test_se3_memory.py tests/test_se3_pipeline.py` -> `13 passed in 2.67s`
  - `pytest -q` -> `52 passed in 2.80s`
  - CLI com nova flag validada em help.
- Conclusao:
  - Fluxo anti-OOM implementado no roteador e no grafo esparso, com fail-fast quando SE(3) ultralongo nao estiver disponivel.
- Proximos passos:
  - Medir tempo total de inferencia em lote de alvos longos (L~5500) para calibrar `radius_angstrom`, `max_neighbors` e threshold ultralongo sob budget Kaggle.

## PLAN-088

### 2026-02-16T14:30:51Z - marcusvinicius/Codex (IPA com frame local RNA + estabilizacao de gradiente TM-core)

- Objetivo/hipotese:
  - Alinhar o IPA a um referencial local de RNA (proxy `P/C4'/N1/N9`) e remover instabilidade de gradiente observada no treino curto do pipeline SE(3).
- Comparacao:
  - Baseline: IPA sem frame local RNA explicito e perda TM-core com gradiente direto via `svd`.
  - Novo: IPA com frame local por camada + viés orientacional local + TM-core com rotação Kabsch calculada em `no_grad`.
- Comandos executados:
  - `pytest -q tests/test_ipa_geometry.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py`
  - `pytest -q`
- Configuracao efetiva:
  - `IpaBackbone` com `base_features` obrigatorio (A/C/G/U) para inferir purina/pirimidina.
  - Frame local recalculado por camada em `build_rna_local_frames`.
  - Viés orientacional limitado por `0.1 * tanh(...)` para estabilidade numerica.
- Parametros/hiperparametros:
  - Proxies de frame local: `p_distance=2.2`, `c4_distance=1.5`, `n_distance=1.4`.
  - Mistura da atualização coordenada no IPA:
    - `0.5 * displacement_frame_local + 0.5 * displacement_global`.
- Seeds:
  - Testes de pipeline reutilizam seeds de suite (`123`, `17`, `7`).
- Versao de codigo/dados:
  - `git commit`: `efbcb7c` + modificacoes locais nao commitadas do `PLAN-088`.
  - Dados sinteticos dos testes em `tmp_path`.
- Artefatos:
  - Novo teste: `tests/test_ipa_geometry.py`.
  - Sem novos artefatos em `runs/` (execucao de validacao tecnica, nao treino longo).
- Metricas/score/custo:
  - `pytest -q tests/test_ipa_geometry.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py` -> `14 passed in 1.86s`
  - `pytest -q` -> `55 passed in 2.58s`
- Conclusao:
  - IPA passou a usar frame local RNA de forma explicita sem regressao na suite local, e a instabilidade de gradiente nao-finito em treino curto foi removida.
- Proximos passos:
  - Rodar treino local completo com budget real para medir impacto em score local por alvo orphan vs baseline.

## PLAN-089

### 2026-02-16T14:36:41Z - marcusvinicius/Codex (best-of-5 com pre-filtro + clustering max-min)

- Objetivo/hipotese:
  - Aumentar chance de acerto Best-of-5 evitando envio de decoys redundantes da mesma bacia e reforcando diversidade estrutural apos poda dos piores candidatos.
- Comparacao:
  - Baseline: selecao Top-5 por score com penalidade gulosa simples de similaridade.
  - Novo: pre-filtro 50% (score ajustado por clash) + clustering Max-Min + medoides por cluster distinto.
- Comandos executados:
  - `pytest -q tests/test_best_of5_strategy.py tests/test_se3_pipeline.py`
  - `pytest -q tests/test_best_of5_strategy.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_ipa_geometry.py`
  - `pytest -q`
- Configuracao efetiva:
  - `sample-se3-ensemble --n-samples` default alterado para `24`.
  - `sampler.py`:
    - diffusion: rotina DPM-like com malha reduzida adaptada ao budget;
    - flow: integrador Heun com passos reduzidos.
  - `select-top5-se3`:
    - `keep_fraction=0.50`;
    - penalizacao de clash no pre-filtro (`adjusted_score = final_score - 0.40 * clash_ratio`).
- Parametros/hiperparametros:
  - `lambda_diversity=0.35` nos testes de estrategia.
  - `n_models=5`.
  - `n_samples=24` para validacao da amostragem mista diffusion/flow.
- Seeds:
  - `base_seed=123` no teste do sampler.
  - `torch.manual_seed(7)` no setup do teste do sampler.
- Versao de codigo/dados:
  - `git commit`: `efbcb7c` + modificacoes locais nao commitadas do `PLAN-089`.
  - Dados sinteticos em `tmp_path` para os testes de selecao.
- Artefatos:
  - Novo teste: `tests/test_best_of5_strategy.py`.
  - Manifest de selecao (`select_top5_se3_manifest.json`) agora inclui estatisticas de pre-filtro e clustering.
- Metricas/score/custo:
  - `pytest -q tests/test_best_of5_strategy.py tests/test_se3_pipeline.py` -> `6 passed in 1.40s`
  - `pytest -q tests/test_best_of5_strategy.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_ipa_geometry.py` -> `13 passed in 2.05s`
  - `pytest -q` -> `58 passed in 2.78s`
- Conclusao:
  - Fluxo Best-of-5 passou a forcar diversidade estrutural por medoides de clusters distintos apos poda de baixa qualidade, com regressao local verde.
- Proximos passos:
  - Rodar benchmark local com scorer por alvo para medir ganho de diversidade no Top-5 vs baseline guloso.

## PLAN-090

### 2026-02-16T14:40:38Z - marcusvinicius/Codex (clash exponencial + minimizacao curta com restraints)

- Objetivo/hipotese:
  - Reforcar consistencia fisica local sem destruir a topologia global, punindo contatos sub-2.0A com gradiente mais agressivo e restringindo relaxacao pos-inferencia.
- Comparacao:
  - Baseline: clash loss por potencia simples e minimizacao sem restraints explicitas.
  - Novo: clash loss exponencial + termo critico sub-2.0A, e minimizacao com restraints harmonicas fortes no backbone C1'.
- Comandos executados:
  - `pytest -q tests/test_se3_losses.py tests/test_minimization.py`
  - `pytest -q`
- Configuracao efetiva:
  - `losses_se3`:
    - `penalty_main = expm1(alpha * penetration)`
    - `penalty_critical = expm1((2.5*alpha) * penetration_sub2A)`
  - `minimize-ensemble`:
    - contrato de budget: `max_iterations <= 100`;
    - parametro novo: `position_restraint_k` (default CLI `800.0`);
    - default CLI de `max_iterations` ajustado para `80`.
- Parametros/hiperparametros:
  - Testes de loss usam `vdw_min_distance=2.1`, `vdw_repulsion_power=4`.
  - Testes de minimizacao usam `position_restraint_k=800.0`.
- Seeds:
  - N/A (validacao funcional deterministica de contrato; sem treino estocastico neste ciclo).
- Versao de codigo/dados:
  - `git commit`: `efbcb7c` + modificacoes locais nao commitadas do `PLAN-090`.
  - Dados sinteticos em `tmp_path` nos testes.
- Artefatos:
  - `minimize_ensemble_manifest.json` agora inclui `position_restraint_k`.
  - `predictions` minimizadas agora incluem coluna `refinement_position_restraint_k`.
- Metricas/score/custo:
  - `pytest -q tests/test_se3_losses.py tests/test_minimization.py` -> `9 passed in 0.81s`
  - `pytest -q` -> `60 passed in 2.82s`
- Conclusao:
  - Penalizacao de clash ficou mais severa para choques esterequimicos graves e a minimizacao passou a operar em modo conservador de topologia com restraints explicitas.
- Proximos passos:
  - Rodar benchmark local de score por alvo para calibrar `position_restraint_k` e a severidade efetiva do termo critico sub-2.0A.

## PLAN-091

### 2026-02-16T14:43:23Z - marcusvinicius/Codex (multichain com salto absoluto no encoding 1D)

- Objetivo/hipotese:
  - Tornar a quebra de cadeia um separador absoluto no eixo posicional 1D para evitar que atencao/modelo trate cadeias independentes como continuidade fosfodiester.
- Comparacao:
  - Baseline: `residue_index` contiguo (`arange`) no `TargetGraph`.
  - Novo: `residue_index` derivado do parser com salto `+1000` entre cadeias.
- Comandos executados:
  - `pytest -q tests/test_sequence_parser.py tests/test_se3_pipeline.py tests/test_msa_covariance.py`
  - `pytest -q`
- Configuracao efetiva:
  - `parse_sequence_with_chains(..., chain_break_offset_1d=1000)` default.
  - `graph_builder` passou a consumir `parsed.residue_position_index_1d`.
- Parametros/hiperparametros:
  - Salto absoluto entre cadeias: `+1000`.
- Seeds:
  - N/A (mudanca deterministica de parsing/indice).
- Versao de codigo/dados:
  - `git commit`: `efbcb7c` + modificacoes locais nao commitadas do `PLAN-091`.
  - Dados sinteticos de teste em `tmp_path`.
- Artefatos:
  - Novo teste: `tests/test_sequence_parser.py`.
- Metricas/score/custo:
  - `pytest -q tests/test_sequence_parser.py tests/test_se3_pipeline.py tests/test_msa_covariance.py` -> `9 passed in 1.29s`
  - `pytest -q` -> `63 passed in 2.53s`
- Conclusao:
  - O pipeline passa a carregar encoding 1D multicadeia com separacao absoluta desde o parse, mantendo regressao local verde.
- Proximos passos:
  - Medir impacto no score local de alvos multicadeia para ajustar opcionalmente o tamanho do salto (se necessario).

## PLAN-092

### 2026-02-16T14:52:39Z - marcusvinicius/Codex (protocolo de treino 16GB + fix BF16 no TM-core)

- Objetivo/hipotese:
  - Viabilizar treino local SE(3) em 16GB VRAM com crop dinamico + BF16 + checkpointing + accumulation sem quebrar perda estrutural.
- Comparacao:
  - Baseline: treino com autocast BF16 acionado, mas falha no TM-core por `torch.linalg.svd` em BF16 CUDA.
  - Novo: Kabsch/TM-core forcado para `float32` com autocast desabilitado localmente no trecho SVD.
- Comandos executados:
  - `pytest -q tests/test_se3_pipeline.py::test_train_sample_rank_select_se3_pipeline tests/test_se3_pipeline.py::test_train_sample_se3_with_multichain_sequence tests/test_se3_memory.py::test_train_and_sample_se3_with_linear_memory_config`
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py tests/test_sequence_parser.py`
  - `pytest -q`
- Configuracao efetiva:
  - Treino SE(3) com:
    - `dynamic_cropping=true`, `crop_min_length=256`, `crop_max_length=384`;
    - `autocast_bfloat16=true`;
    - `use_gradient_checkpointing=true`;
    - `gradient_accumulation_steps=16`.
  - Perda estrutural:
    - Kabsch/TM-core em `float32` no bloco SVD.
- Parametros/hiperparametros:
  - `crop_sequence_fraction=0.60`
  - `loss_chunk_size` conforme config de treino ativa.
- Seeds:
  - Seeds de teste da suite (`123`, `17`, `7`) nos cenarios de pipeline/treino curto.
- Versao de codigo/dados:
  - `git commit`: `efbcb7c` + modificacoes locais nao commitadas do `PLAN-092`.
  - Dados sinteticos de teste em `tmp_path`.
- Artefatos:
  - Nao houve treino completo com persistencia em `runs/` neste ciclo (apenas validacao tecnica).
- Metricas/score/custo:
  - Suite alvo de regressao: `3 passed in 2.34s`.
  - Suite SE(3) ampliada: `15 passed in 2.71s`.
  - Suite completa: `63 passed in 3.51s`.
- Conclusao:
  - O protocolo de treino para 16GB permanece ativo e o caminho BF16 deixou de quebrar no TM-core.
- Proximos passos:
  - Executar treino completo local em GPU com persistencia em `runs/` e comparar score local vs baseline antes de qualquer submissao Kaggle.

## PLAN-093

### 2026-02-16T15:16:34Z - marcusvinicius/Codex (fase 1 data lab + store lazy zarr)

- Objetivo/hipotese:
  - Estruturar pre-processamento local (termodinamica + MSA) em CPU multithread e empacotar treino em store lazy para reduzir consumo de RAM no carregamento de dados.
- Comparacao:
  - Baseline: extração BPP/MSA sequencial e grafo de treino carregado integralmente em memoria.
  - Novo: BPP/MSA com `num_workers` + cache atomico, store `training_store.zarr` e treino opcional com `--training-store` (1 target por vez).
- Comandos executados:
  - `pytest -q tests/test_thermo_2d.py tests/test_msa_covariance.py tests/test_phase1_data_lab.py`
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_sequence_parser.py tests/test_se3_losses.py`
  - `pytest -q`
- Configuracao efetiva:
  - Novo comando:
    - `prepare-phase1-data-lab --workers 16`
  - Novo caminho de treino:
    - `train-se3-generator --training-store <...>/training_store.zarr`
  - Dependencia opcional:
    - `zarr` (falha cedo quando ausente).
- Parametros/hiperparametros:
  - Testes de paralelismo mock rodados com `num_workers=4`.
  - Data lab de teste:
    - `thermo_backend=mock`
    - `msa_backend=mock`
    - `max_msa_sequences=16`
    - `max_cov_positions=64`
    - `max_cov_pairs=512`
- Seeds:
  - N/A (validacao funcional de pipeline/cache/store; sem treino estocastico completo neste ciclo).
- Versao de codigo/dados:
  - `git commit`: `1f77f26` + modificacoes locais nao commitadas do `PLAN-093`.
  - Dados sinteticos em `tmp_path` para os testes.
- Artefatos:
  - Novo modulo: `src/rna3d_local/training/data_lab.py`.
  - Novo modulo: `src/rna3d_local/training/store_zarr.py`.
  - Novo teste: `tests/test_phase1_data_lab.py`.
- Metricas/score/custo:
  - `pytest -q tests/test_thermo_2d.py tests/test_msa_covariance.py tests/test_phase1_data_lab.py` -> `10 passed in 0.79s`
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_sequence_parser.py tests/test_se3_losses.py` -> `15 passed in 2.68s`
  - `pytest -q` -> `66 passed in 3.56s`
- Conclusao:
  - FASE 1 local ficou operacional em contrato estrito para precompute paralelo e empacotamento lazy com zarr.
- Proximos passos:
  - Instalar `zarr` no ambiente de treino e executar `prepare-phase1-data-lab` em dataset real para medir throughput de CPU/I/O com `workers=16`.

## PLAN-094

### 2026-02-16T15:20:23Z - marcusvinicius/Codex (protocolo local_16gb estrito no treino FASE 2)

- Objetivo/hipotese:
  - Formalizar o protocolo de treino local da FASE 2 para 16GB VRAM e remover fallback silencioso em BF16 runtime.
- Comparacao:
  - Baseline: knobs de treino existiam, mas sem um protocolo unico validado e sem bloqueio explicito de BF16 sem suporte.
  - Novo: `training_protocol=local_16gb` com envelope de parametros estrito + fail-fast de BF16 no runtime.
- Comandos executados:
  - `pytest -q tests/test_se3_losses.py tests/test_se3_pipeline.py tests/test_se3_memory.py`
  - `pytest -q`
- Configuracao efetiva:
  - Novo campo de config: `training_protocol` (`custom` | `local_16gb`).
  - `local_16gb` exige:
    - `dynamic_cropping=true`
    - `crop_min_length/crop_max_length` em `[256,384]`
    - `use_gradient_checkpointing=true`
    - `autocast_bfloat16=true`
    - `gradient_accumulation_steps` em `[16,32]`.
- Parametros/hiperparametros:
  - Teste negativo: `gradient_accumulation_steps=8` em `local_16gb` (deve falhar).
  - Teste positivo: `gradient_accumulation_steps=16` em `local_16gb`.
- Seeds:
  - Seeds de suite padrao dos testes de pipeline (`123`, `17`, `7`) nos cenarios de treino curto.
- Versao de codigo/dados:
  - `git commit`: `1f77f26` + modificacoes locais nao commitadas do `PLAN-094`.
  - Dados sinteticos em `tmp_path` para os testes.
- Artefatos:
  - Nao houve treino completo em `runs/` neste ciclo (validacao de contrato/integração).
- Metricas/score/custo:
  - `pytest -q tests/test_se3_losses.py tests/test_se3_pipeline.py tests/test_se3_memory.py` -> `14 passed in 2.83s`
  - `pytest -q` -> `68 passed in 3.65s`
- Conclusao:
  - O protocolo FASE 2 para 16GB ficou codificado e auditavel, com falha cedo para cenarios BF16 sem suporte.
- Proximos passos:
  - Rodar treino completo local com `training_protocol=local_16gb` e store lazy (`--training-store`) para medir estabilidade/throughput em workload real.

## PLAN-095

### 2026-02-16T15:37:52Z - marcusvinicius/Codex (oraculo local Best-of-5 com USalign)

- Objetivo/hipotese:
  - Reproduzir localmente a metrica Best-of-5 do Kaggle com USalign para gate estrito de submissao.
- Comparacao:
  - Baseline: repositorio sem scorer local oficial do Best-of-5 baseado em USalign.
  - Novo: comando dedicado `score-local-bestof5` gerando `score.json` + relatorio por alvo.
- Comandos executados:
  - `pytest -q tests/test_usalign_scorer.py`
  - `pytest -q`
- Configuracao efetiva:
  - `score-local-bestof5` requer:
    - `--ground-truth`
    - `--submission`
    - `--usalign-bin`
    - `--score-json`
    - `--report` opcional.
  - Contratos aceitos no ground truth:
    - `ID + x/y/z` ou
    - `target_id+resid + x_1/y_1/z_1`.
- Parametros/hiperparametros:
  - `timeout_seconds` default do scorer: `120`.
  - Avaliacao fixa em 5 modelos (`x_1..z_5`) por contrato Best-of-5.
- Seeds:
  - N/A (avaliacao deterministica com binario externo + fixture de teste).
- Versao de codigo/dados:
  - `git commit`: `1f77f26` + modificacoes locais nao commitadas do `PLAN-095`.
  - Dados sinteticos em `tmp_path` nos testes.
- Artefatos:
  - Novo modulo: `src/rna3d_local/evaluation/usalign_scorer.py`.
  - Novo teste: `tests/test_usalign_scorer.py`.
- Metricas/score/custo:
  - `pytest -q tests/test_usalign_scorer.py` -> `3 passed in 1.02s`
  - `pytest -q` -> `71 passed in 3.83s`
- Conclusao:
  - O gate de score local Best-of-5 com USalign ficou disponivel em modo estrito e auditavel.
- Proximos passos:
  - Executar `score-local-bestof5` com binario USalign oficial compilado no host e integrar o `score.json` ao gate `evaluate-submit-readiness` antes de submeter notebook.

## PLAN-096

### 2026-02-16T16:25:07Z - marcusvinicius/Codex (execucao local treino+inferencia+score)

- Objetivo/hipotese:
  - Rodar pipeline local completo com treino SE(3) e medir score local do candidato exportado.
- Comparacao:
  - Baseline: sem score local reproduzivel no ambiente atual apos reboot.
  - Novo: execucao controlada com artefatos em `runs/20260216_full_local_run/`.
- Comandos executados:
  - Preparacao de dados:
    - geracao de `train_targets_256.csv`, `pairings_train_256.parquet`, `chemical_train_256.parquet`, `train_labels_256.parquet`, `pairings_test.parquet`, `chemical_test.parquet`.
  - Pipeline:
    - `python -m rna3d_local prepare-phase1-data-lab --targets runs/20260216_full_local_run/train_targets_256.csv --pairings runs/20260216_full_local_run/pairings_train_256.parquet --chemical-features runs/20260216_full_local_run/chemical_train_256.parquet --labels runs/20260216_full_local_run/train_labels_256.parquet --out-dir runs/20260216_full_local_run/phase1_data_lab_256 --thermo-backend mock --msa-backend mock --workers 16`
    - `python -m rna3d_local train-se3-generator --targets runs/20260216_full_local_run/train_targets_256.csv --pairings runs/20260216_full_local_run/pairings_train_256.parquet --chemical-features runs/20260216_full_local_run/chemical_train_256.parquet --labels runs/20260216_full_local_run/train_labels_256.parquet --config runs/20260216_full_local_run/config_train_256.json --out-dir runs/20260216_full_local_run/se3_model_256 --seed 123 --training-store runs/20260216_full_local_run/phase1_data_lab_256/training_store.zarr`
    - `python -m rna3d_local sample-se3-ensemble --model-dir runs/20260216_full_local_run/se3_model_256 --targets runs/20260216_full_local_run/test_targets.csv --pairings runs/20260216_full_local_run/pairings_test.parquet --chemical-features runs/20260216_full_local_run/chemical_test.parquet --out runs/20260216_full_local_run/se3_candidates.parquet --method diffusion --n-samples 20 --seed 123`
    - `python -m rna3d_local rank-se3-ensemble --candidates runs/20260216_full_local_run/se3_candidates.parquet --out runs/20260216_full_local_run/se3_ranked.parquet --diversity-lambda 0.35`
    - `python -m rna3d_local select-top5-se3 --ranked runs/20260216_full_local_run/se3_ranked.parquet --out runs/20260216_full_local_run/se3_top5.parquet --n-models 5 --diversity-lambda 0.35`
    - `python -m rna3d_local export-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --predictions runs/20260216_full_local_run/se3_top5.parquet --out runs/20260216_full_local_run/submission.csv`
    - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_full_local_run/submission.csv`
  - Score local:
    - tentativa integral (28 alvos) interrompida por custo/timeout de USalign em alvos especificos;
    - score parcial executado com sucesso em 10 alvos curtos:
      - `python -m rna3d_local score-local-bestof5 --ground-truth runs/20260216_full_local_run/validation_labels_short10.csv --submission runs/20260216_full_local_run/submission_short10.csv --usalign-bin src/rna3d_local/evaluation/USalign --score-json runs/20260216_full_local_run/score_short10.json --report runs/20260216_full_local_run/score_report_short10.json`
- Configuracao efetiva:
  - `config_train_256.json`:
    - `training_protocol=local_16gb`
    - `autocast_bfloat16=true`
    - `use_gradient_checkpointing=true`
    - `gradient_accumulation_steps=16`
    - `epochs=1`
    - `radius_angstrom=1000.0`
    - `max_neighbors=32`
    - `thermo_backend=mock`
    - `msa_backend=mock`.
- Parametros/hiperparametros:
  - treino em subset limpo de `256` targets (`11162` residuos);
  - inferencia em `28` targets de validacao.
- Seeds:
  - `seed=123` (treino e amostragem).
- Versao de codigo/dados:
  - `git commit`: `1f77f26` + modificacoes locais nao commitadas do `PLAN-096`.
  - dados: `input/stanford-rna-3d-folding-2/*`.
- Artefatos:
  - `runs/20260216_full_local_run/se3_model_256/metrics.json`
  - `runs/20260216_full_local_run/se3_top5.parquet`
  - `runs/20260216_full_local_run/submission.csv`
  - `runs/20260216_full_local_run/score_short10.json`
  - `runs/20260216_full_local_run/score_report_short10.json`.
- Metricas/score/custo:
  - treino: `ELAPSED=0:37.36`, `RAM_KB=2364760` (comando cronometrado).
  - amostragem: `ELAPSED=0:07.34`.
  - score local parcial (10 alvos curtos): `0.04344`.
  - score local integral (28 alvos): bloqueado por custo de USalign em alvos especificos na janela operacional.
- Conclusao:
  - Pipeline tecnico executa ponta-a-ponta e gera submissao valida, mas a qualidade atual e baixa no recorte pontuado (`0.04344`) e nao caracteriza desempenho competitivo.
- Proximos passos:
  - substituir backends `mock` por sinais reais (BPP/MSA/quimica) e retreinar com budget maior;
  - paralelizar scoring USalign por alvo para viabilizar score integral em prazo pratico.

### 2026-02-16T17:27:40Z - marcusvinicius/Codex (retreino VRNA 1200 + gate de submit)

- Objetivo/hipotese:
  - Reexecutar treino/inferencia com `thermo_backend=viennarna` e medir se o score local sobe no recorte curto.
- Comparacao:
  - Baseline: `score_short10=0.04330` (modelo `se3_model_1200_e4`, backend mock).
  - Novo: `score_short10=0.04543` (modelo `se3_model_1200_vrna_e4`, backend ViennaRNA).
- Comandos executados:
  - Treino (VRNA):
    - `python -m rna3d_local train-se3-generator --targets runs/20260216_full_local_run/train_targets_1200.csv --pairings runs/20260216_full_local_run/pairings_train_1200_vrna.parquet --chemical-features runs/20260216_full_local_run/chemical_train_1200_vrna.parquet --labels runs/20260216_full_local_run/train_labels_1200.parquet --config runs/20260216_full_local_run/config_train_1200_vrna_e4.json --out-dir runs/20260216_full_local_run/se3_model_1200_vrna_e4 --seed 123 --training-store runs/20260216_full_local_run/phase1_data_lab_1200_vrna/training_store.zarr`
  - Pipeline:
    - `python -m rna3d_local sample-se3-ensemble --model-dir runs/20260216_full_local_run/se3_model_1200_vrna_e4 --targets runs/20260216_full_local_run/test_targets.csv --pairings runs/20260216_full_local_run/pairings_test_vrna.parquet --chemical-features runs/20260216_full_local_run/chemical_test_vrna.parquet --out runs/20260216_full_local_run/se3_candidates_1200_vrna_e4.parquet --method diffusion --n-samples 20 --seed 123`
    - `python -m rna3d_local rank-se3-ensemble --candidates runs/20260216_full_local_run/se3_candidates_1200_vrna_e4.parquet --out runs/20260216_full_local_run/se3_ranked_1200_vrna_e4.parquet --diversity-lambda 0.35`
    - `python -m rna3d_local select-top5-se3 --ranked runs/20260216_full_local_run/se3_ranked_1200_vrna_e4.parquet --out runs/20260216_full_local_run/se3_top5_1200_vrna_e4.parquet --n-models 5 --diversity-lambda 0.35`
    - `python -m rna3d_local export-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --predictions runs/20260216_full_local_run/se3_top5_1200_vrna_e4.parquet --out runs/20260216_full_local_run/submission_1200_vrna_e4.csv`
    - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_full_local_run/submission_1200_vrna_e4.csv`
  - Score local:
    - tentativa integral 28 alvos iniciada e abortada por custo alto em alvos longos (USalign em `9J09`).
    - recorte curto:
      - `python -m rna3d_local score-local-bestof5 --ground-truth runs/20260216_full_local_run/validation_labels_short10.csv --submission runs/20260216_full_local_run/submission_1200_vrna_e4_short10.csv --usalign-bin src/rna3d_local/evaluation/USalign --score-json runs/20260216_full_local_run/score_1200_vrna_e4_short10.json --report runs/20260216_full_local_run/score_1200_vrna_e4_short10_report.json`
  - Gate:
    - `python -m rna3d_local evaluate-submit-readiness --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_full_local_run/submission_1200_vrna_e4.csv --score-json runs/20260216_full_local_run/score_1200_vrna_e4_short10.json --baseline-score 0.23171 --report runs/20260216_full_local_run/readiness_1200_vrna_e4_short10_vs_023171.json --allow-disallow`
- Configuracao efetiva:
  - `config_train_1200_vrna_e4.json` com `epochs=4`, `sequence_tower=mamba_like`, `autocast_bfloat16=true`, `use_gradient_checkpointing=true`, `gradient_accumulation_steps=16`, `thermo_backend=viennarna`, `msa_backend=mock`.
- Parametros/hiperparametros:
  - treino em `1200` targets; inferencia em `28` targets.
  - amostragem `n_samples=20`; selecao `n_models=5`.
- Seeds:
  - `seed=123`.
- Versao de codigo/dados:
  - `git commit`: `1f77f26` + modificacoes locais nao commitadas do `PLAN-096`.
  - dados: `input/stanford-rna-3d-folding-2/*`.
- Artefatos:
  - `runs/20260216_full_local_run/se3_model_1200_vrna_e4/metrics.json`
  - `runs/20260216_full_local_run/submission_1200_vrna_e4.csv`
  - `runs/20260216_full_local_run/score_1200_vrna_e4_short10.json`
  - `runs/20260216_full_local_run/readiness_1200_vrna_e4_short10_vs_023171.json`.
- Metricas/score/custo:
  - treino VRNA e4: finalizado em ~`17m` (monitorado por processo local).
  - amostragem: `ELAPSED=0:24.23`, `RAM_KB=1931632`.
  - score local curto (10 alvos): `0.04543` (vs baseline curto `0.23171`).
  - gate de submit: `allowed=false`.
- Conclusao:
  - Houve ganho marginal sobre mock (`0.04330 -> 0.04543`), mas ainda muito abaixo do melhor candidato local curto; submissao bloqueada por regra de melhoria estrita.
- Proximos passos:
  - concluir treino longo (`epochs=30`) ja iniciado em background e repetir pipeline+score curto+gate;
  - para score integral (28 alvos), reduzir custo do scorer (timeout/parallelismo por alvo) antes de novo ciclo.

## PLAN-097

### 2026-02-16T17:42:20Z - marcusvinicius/Codex (bloqueio estrito de mock no runtime competitivo)

- Objetivo/hipotese:
  - Eliminar uso acidental de backends `mock` no fluxo competitivo, mantendo apenas permissivo explicito para ambiente de teste.
- Comparacao:
  - Baseline: `mock` aceito em runtime por padrao em varios modulos criticos.
  - Novo: `mock` bloqueado por contrato fora de `TEST` e sem `RNA3D_ALLOW_MOCK_BACKENDS=1`.
- Comandos executados:
  - `pytest -q tests/test_mock_policy.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_homology_folds.py tests/test_minimization.py tests/test_retrieval_latent.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py`
  - `pytest -q`
- Configuracao efetiva:
  - politica central em `mock_policy.py`:
    - bloqueia `mock` por padrao;
    - libera em `stage` iniciado por `TEST` ou com `RNA3D_ALLOW_MOCK_BACKENDS=1`.
  - `tests/conftest.py` define `RNA3D_ALLOW_MOCK_BACKENDS=1` para manter suite local reprodutivel.
- Parametros/hiperparametros:
  - N/A (mudanca de contrato/runtime; sem treino/inferencia competitiva neste ciclo).
- Seeds:
  - N/A.
- Versao de codigo/dados:
  - `git commit`: `4d99edd` + modificacoes locais nao commitadas do `PLAN-097`.
  - dados de teste sinteticos em `tmp_path` e fixtures existentes.
- Artefatos:
  - Novo modulo: `src/rna3d_local/mock_policy.py`.
  - Novo teste: `tests/test_mock_policy.py`.
- Metricas/score/custo:
  - Suite focada: `40 passed in 2.84s`.
  - Suite completa: `77 passed in 4.02s`.
- Conclusao:
  - O runtime competitivo agora falha cedo quando `mock` e informado, com mensagem acionavel e contrato estrito.
- Proximos passos:
  - manter execucoes competitivas sem `RNA3D_ALLOW_MOCK_BACKENDS`;
  - remover gradualmente fixtures `mock` antigas quando houver cobertura equivalente com backends reais/stubs controlados.

## PLAN-098

### 2026-02-16T18:06:07Z - marcusvinicius/Codex (runners reais Phase 2 + encoder TorchScript)

- Objetivo/hipotese:
  - Remover stubs sinteticos disfarçados de modelos (preditores offline) e substituir `encoder=ribonanzanet2` fake por TorchScript real, garantindo contrato estrito e fail-fast.
- Comparacao:
  - Baseline: preditores offline geravam coordenadas sinteticas quando faltavam modelos/artefatos; embeddings "ribonanzanet2" usavam hashing.
  - Novo: preditores offline exigem runner externo via `entrypoint` e validam o output; embeddings via `torch.jit.load` com validacao de dimensao.
- Comandos executados:
  - `pytest -q`
- Configuracao efetiva:
  - Contrato de `entrypoint`:
    - lista de args com placeholders `{model_dir}`, `{targets}`, `{out}`, `{n_models}`.
  - Validacao de output: parquet com colunas/keys estritas e checagem de finitude/cobertura por alvo.
  - Encoder TorchScript:
    - entrada `tokens: LongTensor[B,L]`;
    - saida `[B,D]` ou `[B,L,D]` com pooling mean e `D == embedding_dim`.
- Parametros/hiperparametros:
  - N/A (mudanca de contrato/runtime; sem treino/inferencia competitiva neste ciclo).
- Seeds:
  - N/A.
- Versao de codigo/dados:
  - `git commit`: `4d99edd` + modificacoes locais nao commitadas do `PLAN-098`.
  - dados: N/A.
- Artefatos:
  - Suite de testes ampliada:
    - runner stub via `entrypoint` em `tests/test_phase2_hybrid.py`;
    - TorchScript toy model em `tests/test_encoder_torchscript.py`.
- Metricas/score/custo:
  - Suite completa: `78 passed in 5.13s`.
- Conclusao:
  - O pipeline agora falha cedo quando modelos offline reais nao estao presentes (sem gerar saidas falsas) e usa embeddings reais quando fornecido um TorchScript compatível.
- Proximos passos:
  - empacotar/registrar os assets reais (weights + runners offline) em Kaggle Datasets privados e validar o notebook de submissao offline com `check-submission` antes de novo submit.

## PLAN-099

### 2026-02-16T18:39:26Z - marcusvinicius/Codex (remocao total de `mock` do runtime/CLI/config)

- Objetivo/hipotese:
  - Remover completamente backends `mock` do runtime para impedir qualquer execucao com saidas sinteticas; testes passam a usar stubs via `monkeypatch` quando precisarem isolar dependencias externas.
- Comparacao:
  - Baseline: suporte a `mock` existia em varios backends (thermo/MSA/encoder/minimizacao/clustering) e em configs/CLI.
  - Novo: `mock` nao e mais um valor valido; ausencias de binarios/dependencias falham cedo com erro acionavel.
- Comandos executados:
  - `pytest -q`
- Configuracao efetiva:
  - `homology_folds backend`: `python|mmseqs2|cdhit_est` (sem `mock`).
  - `thermo_backend`: `rnafold|linearfold|viennarna` (sem `mock`).
  - `msa_backend`: `mmseqs2` (sem `mock`).
  - `encoder`: `ribonanzanet2` (TorchScript) (sem `mock`).
  - `minimize-ensemble`: `openmm|pyrosetta` (sem `mock`).
- Parametros/hiperparametros:
  - N/A (mudanca de contrato/runtime; sem treino/inferencia competitiva neste ciclo).
- Seeds:
  - N/A.
- Versao de codigo/dados:
  - `git commit`: `3ecbe70` + modificacoes locais nao commitadas do `PLAN-099`.
  - dados: N/A.
- Artefatos:
  - Suite de testes atualizada para usar stubs por `monkeypatch` em pontos externos (RNAfold/MMseqs2/OpenMM) quando necessario.
- Metricas/score/custo:
  - Suite completa: `73 passed in 5.35s`.
- Conclusao:
  - O repositorio nao aceita mais `mock` em runtime/CLI/config; para rodar o pipeline e necessario fornecer dependencias/modelos reais (ou stubs apenas no escopo de testes).

## PLAN-100

### 2026-02-16T18:46:49Z - marcusvinicius/Codex (pytest sem warnings)

- Objetivo/hipotese:
  - Remover warnings que poluem o output do `pytest` para tornar regressões mais claras e evitar que warnings relevantes fiquem escondidos.
- Comparacao:
  - Baseline: `pytest -q` emitia warnings (ZarrDeprecationWarning, RuntimeWarning em entropia MSA, DeprecationWarning de torch_geometric).
  - Novo: `pytest -q` roda limpo (0 warnings), com filtro apenas para o warning externo conhecido do torch_geometric.
- Comandos executados:
  - `pytest -q`
- Configuracao efetiva:
  - `pytest.ini` filtra apenas:
    - `DeprecationWarning` com mensagem contendo `torch_geometric.distributed ... deprecated`.
  - `training/msa_covariance.py` calcula log apenas onde `p>0` (sem `log(0)`).
  - `training/store_zarr.py` usa `create_array` quando disponivel (evita deprecacao do zarr v3).
- Parametros/hiperparametros:
  - N/A.
- Seeds:
  - N/A.
- Versao de codigo/dados:
  - `git commit`: `cb2fddf` + modificacoes locais nao commitadas do `PLAN-100`.
  - dados: N/A.
- Metricas/score/custo:
  - Suite completa: `73 passed in 5.33s` (0 warnings).
- Conclusao:
  - A suite passou a ser um sinal mais limpo: warnings remanescentes devem ser tratados (ou filtrados explicitamente se forem de terceiros e conhecidos).

## ADHOC

### 2026-02-17T00:35:07Z - marcusvinicius/Codex (validacao local real do RNAPro + export strict)

- Objetivo/hipotese:
  - Validar que o runtime real do RNAPro roda localmente e gera `predictions_long.parquet` exportavel sob contrato estrito (sem fallback silencioso).
- Comparacao:
  - Baseline: `predict-rnapro-offline` falhava localmente quando o subprocess usava Python fora do venv e/ou torch sem suporte a `sm_120` (RTX 5060 Ti).
  - Novo: venv `py312` + `torch==2.9.1+cu128` executa o runner (GPU) e exporta CSV valido.
- Comandos executados:
  - Preparar suporte CCD/templates (artefatos em `assets/models/rnapro/`):
    - `.venv312/bin/python -m rna3d_local prepare-rnapro-support-files --model-dir assets/models/rnapro --timeout-seconds 1200`
  - Criar targets/sample subset (1 alvo curto):
    - `runs/20260217_rnapro_smoke_9QZJ/targets.csv` a partir de `input/stanford-rna-3d-folding-2/test_sequences.csv` (`target_id=9QZJ`)
    - `runs/20260217_rnapro_smoke_9QZJ/sample_submission_9QZJ.csv` a partir de `input/stanford-rna-3d-folding-2/sample_submission.csv`
  - Instalar venv com torch cu128 (necessario para `sm_120`):
    - `uv venv --python 3.12 .venv312cu128`
    - `uv pip install -p .venv312cu128/bin/python --index https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128`
    - `uv pip install -p .venv312cu128/bin/python -e .`
    - `uv pip install -p .venv312cu128/bin/python numpy==1.26.4 gemmi==0.6.7 rdkit==2023.9.6 pdbeccdutils==0.8.6 ml-collections==1.1.0 biotite==1.4.0 dm-tree==0.1.9 pyyaml==6.0.2 scipy scikit-learn optree einops`
    - `uv pip install -p .venv312cu128/bin/python --no-deps "rnapro @ git+https://github.com/NVIDIA-Digital-Bio/RNAPro.git@ca582630bb2f79193853ecfe859f88a5650cb295"`
  - Rodar RNAPro offline (1 alvo, `n_models=5`):
    - `bash -lc 'source .venv312cu128/bin/activate && python -m rna3d_local predict-rnapro-offline --model-dir assets/models/rnapro --targets runs/20260217_rnapro_smoke_9QZJ/targets.csv --out runs/20260217_rnapro_smoke_9QZJ/rnapro_pred5_cu128.parquet --n-models 5'`
  - Export + validacao strict (contra sample subset):
    - `bash -lc 'source .venv312cu128/bin/activate && python -m rna3d_local export-submission --sample runs/20260217_rnapro_smoke_9QZJ/sample_submission_9QZJ.csv --predictions runs/20260217_rnapro_smoke_9QZJ/rnapro_pred5_cu128.parquet --out runs/20260217_rnapro_smoke_9QZJ/submission_9QZJ.csv'`
    - `bash -lc 'source .venv312cu128/bin/activate && python -m rna3d_local check-submission --sample runs/20260217_rnapro_smoke_9QZJ/sample_submission_9QZJ.csv --submission runs/20260217_rnapro_smoke_9QZJ/submission_9QZJ.csv'`
  - Validar manifest de assets phase2:
    - `bash -lc 'source .venv312cu128/bin/activate && python -m rna3d_local build-phase2-assets --assets-dir assets'`
- Configuracao efetiva:
  - Target smoke: `9QZJ` (len=19).
  - `n_models=5` (contrato Best-of-5).
- Seeds:
  - Runner RNAPro usa `--seed 101` (default do runner).
- Versao de codigo/dados:
  - `git commit`: `707d4d1`
  - Dados: `input/stanford-rna-3d-folding-2/*`.
- Artefatos:
  - `runs/20260217_rnapro_smoke_9QZJ/rnapro_pred5_cu128.parquet`
  - `runs/20260217_rnapro_smoke_9QZJ/submission_9QZJ.csv`
  - `assets/runtime/manifest.json`
- Metricas/score/custo:
  - RNAPro (1 alvo, `n_models=5`): ~60s.
  - `check-submission`: `ok=true` (contra sample subset).
- Conclusao:
  - Pipeline real do RNAPro (offline) executa e exporta em formato estrito; para GPU `sm_120`, requer `torch==2.9.1+cu128` (o pin upstream `torch==2.7.1+cu126` nao roda nesta GPU).

## 2026-02-17 - marcusvinicius/Codex - PLAN-118 (SE(3) smoke train + métricas)

- Data UTC: `2026-02-17T14:07:46Z`
- Plano: `PLAN-118`
- Objetivo:
  - Rodar um treino curto do gerador SE(3) para validar integração (após correções de equivariância + `mamba_like` bidirecional) e coletar métricas locais.
- Comandos executados:
  - Treino (usa `training_store.zarr` lazy; sem extração MSA/termo em runtime):
    - `python -m rna3d_local train-se3-generator --targets runs/20260216_full_local_run/train_targets_256.csv --pairings runs/20260216_full_local_run/pairings_train_256.parquet --chemical-features runs/20260216_full_local_run/chemical_train_256.parquet --labels runs/20260216_full_local_run/train_labels_256.parquet --config runs/20260217_plan118_se3_smoke_20260217T140224Z/config.json --out-dir runs/20260217_plan118_se3_smoke_20260217T140224Z/se3_model --seed 123 --training-store runs/20260216_full_local_run/phase1_data_lab_256/training_store.zarr`
- Configuracao efetiva:
  - `runs/20260217_plan118_se3_smoke_20260217T140224Z/se3_model/config_effective.json`
  - Observacao: `mmseqs_db` foi preenchido com um path local apenas para satisfazer a validação da config; como o treino usou `training_store.zarr`, não houve chamada ao `mmseqs2` neste experimento.
- Seeds:
  - `seed=123`
- Versao de codigo/dados:
  - `git commit`: `9077e94`
  - Dados/artefatos de entrada: `runs/20260216_full_local_run/phase1_data_lab_256/training_store.zarr`
- Artefatos:
  - Modelo: `runs/20260217_plan118_se3_smoke_20260217T140224Z/se3_model/`
  - Métricas: `runs/20260217_plan118_se3_smoke_20260217T140224Z/se3_model/metrics.json`
  - Manifest: `runs/20260217_plan118_se3_smoke_20260217T140224Z/se3_model/train_manifest.json`
- Metricas/score/custo:
  - `n_targets=256`
  - `loss_final=193320.194946`
  - `loss_fape_final=0.898596`
  - `loss_tm_final=0.992462`
  - `loss_clash_final=38663.165365`
  - `loss_generative_final=2.477586`
  - `crop_length_mean_final=41.300781`
- Conclusao:
  - Treino concluiu em CUDA e gerou métricas finitas (sem NaN/inf), validando o caminho de treino SE(3) neste snapshot.

## 2026-02-17 - marcusvinicius/Codex - PLAN-119 (Score local USalign no Public Validation)

- Data UTC: `2026-02-17T14:23:08Z`
- Plano: `PLAN-119`
- Objetivo:
  - Medir “score de verdade” via USalign Best-of-5 contra `validation_labels`, com relatórios por alvo.
- Comandos executados:
  - Subset curto (10 targets, `ground_truth_mode=best_of_gt_copies`):
    - `python -m rna3d_local score-local-bestof5 --ground-truth runs/20260216_full_local_run/validation_labels_short10.csv --submission runs/20260216_full_local_run/submission_short10.csv --usalign-bin src/rna3d_local/evaluation/USalign --timeout-seconds 120 --ground-truth-mode best_of_gt_copies --score-json runs/20260217_score_short10_bestof_gt/score.json --report runs/20260217_score_short10_bestof_gt/report.json`
  - Full (28 targets, `ground_truth_mode=single` para custo controlado em ultra-longos):
    - `python -m rna3d_local score-local-bestof5 --ground-truth input/stanford-rna-3d-folding-2/validation_labels.csv --submission runs/20260216_full_local_run/submission.csv --usalign-bin src/rna3d_local/evaluation/USalign --timeout-seconds 900 --ground-truth-mode single --score-json runs/20260217_score_full28_single/score.json --report runs/20260217_score_full28_single/report.json`
- Versao de codigo/dados:
  - `git commit`: `b78b4f2`
  - Ground truth: `input/stanford-rna-3d-folding-2/validation_labels.csv`
- Artefatos:
  - `runs/20260217_score_short10_bestof_gt/score.json`
  - `runs/20260217_score_short10_bestof_gt/report.json`
  - `runs/20260217_score_full28_single/score.json`
  - `runs/20260217_score_full28_single/report.json`
- Metricas/score/custo:
  - Short10 (`best_of_gt_copies`): `score=0.04443` (n_targets=10)
  - Full28 (`single`): `score=0.042739285714285716` (n_targets=28)
- Conclusao:
  - O scorer USalign executou com sucesso e gerou relatórios por alvo; o modo `best_of_gt_copies` é muito mais caro em alvos ultra-longos (ex.: `9MME`), então o full foi medido em `single` nesta rodada.

## 2026-02-17 - marcusvinicius/Codex - ADHOC (Submit Kaggle via notebook + re-score do candidato)

- Data UTC:
  - Submit: `2026-02-17T15:29:55Z`
  - Re-score: `2026-02-17T15:34:44Z`
- Objetivo:
  - Submeter via notebook (code competition) o artefato `submission.csv` do kernel `marcux777/stanford-rna3d-submit-prod-v2` (versao `84`) e revalidar o score local “de verdade” no conjunto full (28 targets).
- Comandos executados:
  - `python -m compileall -q src/rna3d_local`
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan077_kernel_output_v84/submission.csv`
  - `python -m rna3d_local score-local-bestof5 --ground-truth input/stanford-rna-3d-folding-2/validation_labels.csv --submission runs/20260216_plan077_kernel_output_v84/submission.csv --usalign-bin src/rna3d_local/evaluation/USalign --timeout-seconds 900 --ground-truth-mode single --score-json runs/20260217_score_plan077_v84_recheck/score.json --report runs/20260217_score_plan077_v84_recheck/report.json`
  - `python -m rna3d_local submit-kaggle-notebook --competition stanford-rna-3d-folding-2 --notebook-ref marcux777/stanford-rna3d-submit-prod-v2 --notebook-version 84 --notebook-file submission.csv --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan077_kernel_output_v84/submission.csv --notebook-output-path runs/20260216_plan077_kernel_output_v84/submission.csv --score-json runs/20260216_plan077_kernel_output_v84/score_full.json --baseline-score 0.04443 --message "ADHOC: submit v84 (local USalign=0.18033 > baseline=0.04443)" --execute-submit`
- Versao de codigo/dados:
  - `git commit` (submit + re-score): `96d6074`
  - Ground truth: `input/stanford-rna-3d-folding-2/validation_labels.csv`
- Artefatos:
  - Kernel output: `runs/20260216_plan077_kernel_output_v84/submission.csv` (sha256=`657454b44169922f7c7cf3bee1bf38043e2085653b19a5873e8ab283617aa822`)
  - Submit report: `runs/20260216_plan077_kernel_output_v84/submit_notebook_report.json`
  - Re-score: `runs/20260217_score_plan077_v84_recheck/score.json`
  - Re-score report: `runs/20260217_score_plan077_v84_recheck/report.json`
- Metricas/score/custo:
  - `check-submission`: `ok=true`
  - Re-score Full28 (`single`): `score=0.17425` (n_targets=28)
  - Score preexistente do candidato (usado no gate do submit): `runs/20260216_plan077_kernel_output_v84/score_full.json` -> `score=0.18032857142857142`
- Observacoes:
  - O re-score (código atual) divergiu do score preexistente em 5 targets (`9EBP`, `9JFO`, `9LJN`, `9OD4`, `9G4J`); usar `runs/20260217_score_plan077_v84_recheck/*` como referencia atual para comparacoes futuras.

## 2026-02-17 - marcusvinicius/Codex - PLAN-123 (Sweep `diversity_lambda` no Top-5 híbrido)

- Data UTC: `2026-02-17T15:47:53Z`
- Plano: `PLAN-123`
- Objetivo:
  - Ajustar apenas `diversity_lambda` no `select-top5-hybrid` (sem mudar pipeline/modelos) e medir ganho no score local “de verdade” (USalign Best-of-5) no full28.
- Baseline:
  - Baseline full28 do mesmo pool híbrido (lambda do notebook): `score=0.17425` (exportado de `hybrid_top5.parquet` do kernel).
- Setup (inputs fixos):
  - Candidates: `runs/20260216_plan077_kernel_output_v84/run_phase1_phase2_full_v2/hybrid_candidates.parquet`
  - Sample: `input/stanford-rna-3d-folding-2/sample_submission.csv`
  - Ground truth: `input/stanford-rna-3d-folding-2/validation_labels.csv`
  - USalign: `src/rna3d_local/evaluation/USalign`
- Comandos executados (por variante):
  - `python -m rna3d_local select-top5-hybrid --candidates <hybrid_candidates.parquet> --out <hybrid_top5_lambda_X.parquet> --n-models 5 --diversity-lambda X`
  - `python -m rna3d_local export-submission --sample <sample_submission.csv> --predictions <hybrid_top5_lambda_X.parquet> --out <submission_lambda_X.csv>`
  - `python -m rna3d_local score-local-bestof5 --ground-truth <validation_labels.csv> --submission <submission_lambda_X.csv> --usalign-bin <USalign> --timeout-seconds 900 --ground-truth-mode single --score-json <score_lambda_X.json> --report <report_lambda_X.json>`
- Versao de codigo/dados:
  - `git commit`: `21ea3a1`
- Artefatos:
  - `runs/20260217_plan123_hybrid_lambda_sweep_20260217T154753Z/scores.csv`
  - `runs/20260217_plan123_hybrid_lambda_sweep_20260217T154753Z/sweep.log`
  - `runs/20260217_plan123_hybrid_lambda_sweep_20260217T154753Z/code_audit.md`
  - `runs/20260217_plan123_hybrid_lambda_sweep_20260217T154753Z/submission_baseline_lambda_0.35.csv`
  - `runs/20260217_plan123_hybrid_lambda_sweep_20260217T154753Z/score_baseline_lambda_0.35.json`
  - `runs/20260217_plan123_hybrid_lambda_sweep_20260217T154753Z/report_baseline_lambda_0.35.json`
  - Variantes (Top-5 + submission + score/report):
    - `lambda=0.0`, `0.15`, `0.55`, `0.75`
- Metricas/score:
  - Baseline `lambda=0.35`: `score=0.17425`
  - `lambda=0.0`: `score=0.1753785714285714`
  - `lambda=0.15`: `score=0.1753785714285714`
  - `lambda=0.55`: `score=0.1753785714285714`
  - `lambda=0.75`: `score=0.1753785714285714`
- Conclusao:
  - O sweep melhorou o score full28 em `+0.0011285714` vs baseline; qualquer um dos lambdas testados empatou no melhor score desta rodada.

## 2026-02-17 - marcusvinicius/Codex - PLAN-124 (Ablacao de `confidence` no pool híbrido)

- Data UTC: `2026-02-17T16:12:42Z`
- Plano: `PLAN-124`
- Objetivo:
  - Quantificar o impacto do campo `confidence` na selecao Top-5 híbrida, mantendo coordenadas fixas e medindo score USalign full28.
- Setup (inputs fixos):
  - Candidates base: `runs/20260216_plan077_kernel_output_v84/run_phase1_phase2_full_v2/hybrid_candidates.parquet`
  - Sample: `input/stanford-rna-3d-folding-2/sample_submission.csv`
  - Ground truth: `input/stanford-rna-3d-folding-2/validation_labels.csv`
  - USalign: `src/rna3d_local/evaluation/USalign`
  - Seletor: `select-top5-hybrid` com `n_models=5`, `diversity_lambda=0.0`
- Variantes:
  - `variant_conf_zero`: `confidence=0.0` em todas as linhas.
  - `variant_conf_center`: `confidence` centralizado por `target_id+source` (de-bias por fonte).
- Comandos executados (por variante):
  - Gerar candidates variante:
    - (polars) escrever `candidates.parquet` em `runs/<run_dir>/<variant>/candidates.parquet`
  - `python -m rna3d_local select-top5-hybrid --candidates <candidates.parquet> --out <hybrid_top5.parquet> --n-models 5 --diversity-lambda 0.0`
  - `python -m rna3d_local export-submission --sample <sample_submission.csv> --predictions <hybrid_top5.parquet> --out <submission.csv>`
  - `python -m rna3d_local score-local-bestof5 --ground-truth <validation_labels.csv> --submission <submission.csv> --usalign-bin <USalign> --timeout-seconds 900 --ground-truth-mode single --score-json <score.json> --report <report.json>`
- Versao de codigo/dados:
  - `git commit`: `f67dc3e`
- Artefatos:
  - `runs/20260217_plan124_conf_ablation_20260217T161242Z/scores.csv`
  - `runs/20260217_plan124_conf_ablation_20260217T161242Z/ablation.log`
  - `runs/20260217_plan124_conf_ablation_20260217T161242Z/variant_conf_zero/*`
  - `runs/20260217_plan124_conf_ablation_20260217T161242Z/variant_conf_center/*`
- Metricas/score:
  - `variant_conf_zero`: `score=0.1699`
  - `variant_conf_center`: `score=0.1699`
- Conclusao:
  - Remover/centralizar `confidence` degradou o score vs o melhor atual desta rodada (`0.1753785714`), sugerindo que `confidence` carrega sinal util neste pool e nao deve ser descartado sem substituir por um QA melhor.

## 2026-02-17 - marcusvinicius/Codex - PLAN-125 (Sweep de escala de `confidence`)

- Data UTC: `2026-02-17T16:23:31Z`
- Plano: `PLAN-125`
- Objetivo:
  - Calibrar o trade-off entre `confidence` e a penalidade de clash no seletor híbrido, sem alterar código (apenas escalando `confidence` no candidates).
- Setup (inputs fixos):
  - Candidates base: `runs/20260216_plan077_kernel_output_v84/run_phase1_phase2_full_v2/hybrid_candidates.parquet`
  - Sample: `input/stanford-rna-3d-folding-2/sample_submission.csv`
  - Ground truth: `input/stanford-rna-3d-folding-2/validation_labels.csv`
  - USalign: `src/rna3d_local/evaluation/USalign`
  - Seletor: `select-top5-hybrid` com `n_models=5`, `diversity_lambda=0.0`
- Variantes:
  - `confidence_scale in {0.5, 1.0, 2.0}` com `confidence := confidence_scale * confidence`.
- Artefatos:
  - `runs/20260217_plan125_conf_scale_20260217T162331Z/scores.csv`
  - `runs/20260217_plan125_conf_scale_20260217T162331Z/sweep.log`
  - `runs/20260217_plan125_conf_scale_20260217T162331Z/scale_*/{candidates.parquet,hybrid_top5.parquet,submission.csv,score.json,report.json}`
- Metricas/score (full28, `single`):
  - `confidence_scale=0.5`: `score=0.17423928571428574`
  - `confidence_scale=1.0`: `score=0.1753785714285714` (melhor desta rodada; coincide com PLAN-123)
  - `confidence_scale=2.0`: `score=0.17425`
- Conclusao:
  - O melhor ponto neste sweep foi manter a escala original (`confidence_scale=1.0`); reduzir ou aumentar piorou o score.

## 2026-02-17 - marcusvinicius/Codex - PLAN-126 (Kaggle TBM-first kernel v89 + score local)

- Data UTC: `2026-02-17T18:20:56Z`
- Plano: `PLAN-126`
- Objetivo:
  - Gerar `submission.csv` via TBM-first (sem hybrid/foundation) que passe contrato estrito e melhore o proxy local USalign vs o candidato `0.132`.
- Setup:
  - Kernel: `marcux777/stanford-rna3d-submit-prod-v2` (version `89`, `enable_internet=false`)
  - Output: `runs/20260217_plan126_kernel_output_v89/`
  - Sample: `input/stanford-rna-3d-folding-2/sample_submission.csv`
  - Ground truth (proxy): `input/stanford-rna-3d-folding-2/validation_labels.csv`
  - USalign: `src/rna3d_local/evaluation/USalign`
- Observacao (bug real encontrado):
  - `predict-tbm` pode retornar menos de `n_models` para alguns alvos (ex.: `9MME` com 4 templates validos), o que fazia `export-submission` falhar no modelo 5; a execucao do kernel v89 incluiu padding de TBM para cumprir contrato.
- Validacao local executada:
  - Contrato estrito:
    - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260217_plan126_kernel_output_v89/submission.csv`
  - Score “de verdade” (proxy full28, `single`):
    - `python -m rna3d_local score-local-bestof5 --ground-truth input/stanford-rna-3d-folding-2/validation_labels.csv --submission runs/20260217_plan126_kernel_output_v89/submission.csv --usalign-bin src/rna3d_local/evaluation/USalign --timeout-seconds 900 --ground-truth-mode single --score-json runs/20260217_plan126_score_v89/score.json --report runs/20260217_plan126_score_v89/report.json`
- Metricas/score:
  - `score=0.262925` (`runs/20260217_plan126_score_v89/score.json`)
- Submissao Kaggle:
  - `kaggle competitions submit -c stanford-rna-3d-folding-2 -k marcux777/stanford-rna3d-submit-prod-v2 -f submission.csv -v 89 -m "PLAN-126: TBM-first + pad missing model ids (local USalign 0.2629)"`
  - Status no momento do registro: `PENDING` (submetido em `2026-02-17 18:13:11` no CLI).

### 2026-02-17 - Update: kernel v90 (coverage split + DRfold2 fallback) + submit

- Data UTC: `2026-02-17T18:46:22Z`
- Kernel:
  - `marcux777/stanford-rna3d-submit-prod-v2` version `90`
- Mudanca:
  - Split de targets por cobertura real de templates (TBM vs fallback) para evitar crash no rerun hidden.
  - Fallback usa `DRfold_infer.py` do dataset `marcux777/stanford-rna3d-drfold2-official-v1` e exporta C1' replicado em `model_id=1..5`.
- Artefatos:
  - `runs/20260217_plan126_kernel_output_v90/submission.csv`
  - `runs/20260217_plan126_score_v90/score.json` (proxy local)
- Metricas/score:
  - `score=0.262925` (igual ao v89 no full28 public; fallback_targets=0 no dataset local)
- Submissao Kaggle:
  - `kaggle competitions submit -c stanford-rna-3d-folding-2 -k marcux777/stanford-rna3d-submit-prod-v2 -f submission.csv -v 90 -m "PLAN-126: TBM-first with coverage split + DRfold2 fallback for no-template targets"`
  - Status no momento do registro: `PENDING` (submetido em `2026-02-17 18:46:10` no CLI).
