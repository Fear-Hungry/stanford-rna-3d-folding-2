# CHANGES.md

Log append-only de mudancas implementadas.

## 2026-02-16 - marcusvinicius/Codex - PLAN-073

- Data UTC: `2026-02-16T01:37:02Z`
- Plano: `PLAN-073`
- Resumo:
  - Reboot greenfield do pacote `rna3d_local` para Fase 1 Template Oracle.
  - Implementados comandos: `build-template-db`, `build-embedding-index`, `infer-description-family`, `prepare-chemical-features`, `retrieve-templates-latent`, `train-template-reranker`, `score-template-reranker`, `predict-tbm`, `export-submission`, `check-submission`, `submit-kaggle-notebook`.
  - Implementado padrao de erro fail-fast com formato contratual e validacao estrita de submissao.
  - Recriada base de testes com cobertura de contratos criticos e fluxo minimo.
- Arquivos principais tocados:
  - `pyproject.toml`
  - `README.md`
  - `src/rna3d_local/*.py`
  - `tests/*.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `python -m pytest -q` -> `7 passed`
  - `python -m rna3d_local --help` -> CLI carregada com subcomandos da Fase 1
- Riscos conhecidos / follow-ups:
  - `encoder=ribonanzanet2` usa contrato estrito de `--model-path`; sem artefato local valido, o comando falha cedo.
  - `backend=llama_cpp` exige dependencias/arquivo GGUF local; em ausencia, falha cedo sem fallback.

## 2026-02-16 - marcusvinicius/Codex - PLAN-074

- Data UTC: `2026-02-16T01:57:03Z`
- Plano: `PLAN-074`
- Resumo:
  - Implementados modulos da Fase 2:
    - assets offline: `build-phase2-assets`;
    - inferencias offline: `predict-rnapro-offline`, `predict-chai1-offline`, `predict-boltz1-offline`;
    - orquestracao hibrida: `build-hybrid-candidates`, `select-top5-hybrid`;
    - gating competitivo: `evaluate-submit-readiness`.
  - Adicionado roteamento deterministico por alvo com regras de template/orfao/ligante e suporte a ensemble `chai1_boltz1_ensemble`.
  - Adicionada validacao de assets por manifesto com hash SHA256 e fail-fast.
  - Adicionado `pytest.ini` para restringir coleta a `tests/` e evitar coleta acidental de artefatos em `runs/`.
- Arquivos principais tocados:
  - `src/rna3d_local/phase2_assets.py`
  - `src/rna3d_local/predictor_common.py`
  - `src/rna3d_local/rnapro_offline.py`
  - `src/rna3d_local/chai1_offline.py`
  - `src/rna3d_local/boltz1_offline.py`
  - `src/rna3d_local/hybrid_router.py`
  - `src/rna3d_local/hybrid_select.py`
  - `src/rna3d_local/submit_readiness.py`
  - `src/rna3d_local/cli.py`
  - `src/rna3d_local/cli_parser.py`
  - `tests/test_phase2_assets.py`
  - `tests/test_phase2_hybrid.py`
  - `pytest.ini`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `python -m rna3d_local --help` -> novos subcomandos da Fase 2 visiveis.
  - `python -m pytest -q` -> `11 passed`.
- Riscos conhecidos / follow-ups:
  - Preditores offline da Fase 2 estao em modo deterministico de integracao (contratos e roteamento); integrar runtimes reais de RNAPro/Boltz/Chai com pesos oficiais e profiling de tempo/RAM e proximo passo.

## 2026-02-16 - marcusvinicius/Codex - ADHOC

- Data UTC: `2026-02-16T02:09:31Z`
- Plano: `ADHOC`
- Resumo:
  - Executado preflight de submissao Kaggle (validacao de formato + checagem de output do notebook + comparacao de hash/score observado).
  - Decisao operacional: submissao nao executada por ausencia de evidencia de melhoria estrita (mesmo hash de versao ja pontuada) e falta de score local novo do candidato.
- Arquivos principais tocados:
  - `EXPERIMENTS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_submit_attempt/notebook_output/submission.csv` -> `ok=true`
  - `sha256sum runs/20260216_submit_attempt/notebook_output/submission.csv` -> `392e12d1a957af66cfa382d5e64f3b3cb652bbbb2146f089fbf3611ee430b583`
- Riscos conhecidos / follow-ups:
  - Necessario gerar score local novo e candidato realmente novo (hash diferente da versao ja pontuada) antes de nova submissao competitiva.

## 2026-02-16 - marcusvinicius/Codex - PLAN-075

- Data UTC: `2026-02-16T02:18:51Z`
- Plano: `PLAN-075`
- Resumo:
  - Notebook de submissao `stanford-rna3d-submit-prod-v2.ipynb` reescrito para pipeline completo Fase 1 + Fase 2.
  - Fluxo implementado no notebook:
    - Fase 1: `infer-description-family`, `prepare-chemical-features`, `retrieve-templates-latent`, `score-template-reranker`, `predict-tbm`;
    - Fase 2: `build-phase2-assets`, `predict-rnapro-offline`, `predict-chai1-offline`, `predict-boltz1-offline`, `build-hybrid-candidates`, `select-top5-hybrid`;
    - Export/validacao: `export-submission` e `check-submission`.
  - Adicionada descoberta estrita de ativos em `/kaggle/input` e falha explicita para ativos ausentes/ambiguos.
  - Incluida validacao explicita de superficie CLI antes de executar o pipeline no notebook.
- Arquivos principais tocados:
  - `runs/20260216_plan073_submit_preflight/kernel_source/stanford-rna3d-submit-prod-v2.ipynb`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - Validador estrutural/sintatico do notebook (JSON + `compile` do codigo) -> `NOTEBOOK_OK`
  - Validador de superficie CLI dos subcomandos usados no notebook -> `CLI_SURFACE_OK`
  - `kernel-metadata.json` mantido com `enable_internet=false`.
- Riscos conhecidos / follow-ups:
  - O notebook agora exige ativos adicionais (LLM GGUF, Ribonanza model, template embeddings/FAISS, template_family_map, quickstart quimico, reranker pretreinado, modelos phase2); se qualquer ativo faltar/duplicar, a execucao falha cedo por contrato.

## 2026-02-16 - marcusvinicius/Codex - PLAN-076

- Data UTC: `2026-02-16T03:10:14Z`
- Plano: `PLAN-076`
- Resumo:
  - Corrigido `prepare-chemical-features` para aceitar QUICK_START em schema de templates (`ID/resid/x_i,y_i,z_i`) com modo estrito e rastreavel no manifest (`schema_mode`), incluindo caso de triplet unico.
  - Corrigido `predict-tbm` para selecionar os primeiros templates validos com cobertura completa de residuos por alvo (evita abortar por candidato incompleto quando existem alternativas validas).
  - Adicionados testes para os novos contratos de `chemical_features` e `tbm`.
  - Publicadas novas versoes do dataset `marcux777/stanford-rna3d-reboot-src-v2` e do notebook `marcux777/stanford-rna3d-submit-prod-v2`.
  - Executado preflight completo no Kaggle: pipeline Fase 1+2 concluido, `submission.csv` gerado e validado.
- Arquivos principais tocados:
  - `src/rna3d_local/chemical_features.py`
  - `src/rna3d_local/tbm.py`
  - `tests/test_chemical_features.py`
  - `tests/test_tbm.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_chemical_features.py tests/test_tbm.py tests/test_phase2_hybrid.py tests/test_reranker.py` -> `9 passed`
  - `pytest -q` -> `17 passed`
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan075_kernel_output_v83/submission.csv` -> `ok=true`
- Riscos conhecidos / follow-ups:
  - A submissao competitiva continua bloqueada por gate ate existir score local novo do candidato (melhoria estrita vs baseline registrado).

## 2026-02-16 - marcusvinicius/Codex - PLAN-077

- Data UTC: `2026-02-16T12:23:41Z`
- Plano: `PLAN-077`
- Resumo:
  - Corrigido notebook de submissao para remover treino/reranking no Kaggle (`train-template-reranker`/`score-template-reranker`), usando ranking direto de retrieval no fluxo de inferencia.
  - `retrieve-templates-latent` deixou de abortar quando ha targets sem candidatos temporais; agora gera saida vazia/parcial com estatisticas explicitas no manifest.
  - `predict-tbm` deixou de abortar pipeline quando faltam candidatos/cobertura para alguns alvos; agora gera saida parcial e registra alvos pulados no manifest.
  - `build-hybrid-candidates` agora roteia explicitamente para fase2 quando `template_strong` nao tem cobertura TBM (`template_missing->...`), evitando falha por falta de cobertura primaria.
  - Publicadas novas versoes do dataset `marcux777/stanford-rna3d-reboot-src-v2` e notebook `marcux777/stanford-rna3d-submit-prod-v2` (`v84`).
- Arquivos principais tocados:
  - `src/rna3d_local/retrieval_latent.py`
  - `src/rna3d_local/tbm.py`
  - `src/rna3d_local/hybrid_router.py`
  - `tests/test_tbm.py`
  - `tests/test_phase2_hybrid.py`
  - `runs/20260216_plan073_submit_preflight/kernel_source/stanford-rna3d-submit-prod-v2.ipynb`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q` -> `18 passed`
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan077_kernel_output_v84/submission.csv` -> `ok=true`
- Riscos conhecidos / follow-ups:
  - A nova submissao de competicao (`ref=50393739`) ainda estava `PENDING` no momento do registro; confirmar status final (score/erro) antes de novas tentativas.

## 2026-02-16 - marcusvinicius/Codex - PLAN-078

- Data UTC: `2026-02-16T12:49:22Z`
- Plano: `PLAN-078`
- Resumo:
  - Implementado branch experimental SE(3) generativo com backbone EGNN+IPA, modulos Diffusion/Flow, e pipeline de ranking/selecao Top-5.
  - Adicionados comandos CLI:
    - `train-se3-generator`
    - `sample-se3-ensemble`
    - `rank-se3-ensemble`
    - `select-top5-se3`
  - Integrado roteamento hibrido com entrada opcional `--se3` e regra `generative_se3` quando cobertura disponivel.
  - Incluidas validacoes estritas/fail-fast para schemas de treino, condicoes (pairings/quimica), runtime de modelos e selecao Top-5.
  - Adicionada cobertura de testes para pipeline SE(3) e rota hibrida com preferencia SE(3).
- Arquivos principais tocados:
  - `src/rna3d_local/se3/*`
  - `src/rna3d_local/generative/*`
  - `src/rna3d_local/ensemble/*`
  - `src/rna3d_local/training/*`
  - `src/rna3d_local/se3_pipeline.py`
  - `src/rna3d_local/cli.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/hybrid_router.py`
  - `tests/test_se3_pipeline.py`
  - `tests/test_phase2_hybrid.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `python -m rna3d_local --help` -> novos comandos SE(3) visiveis.
  - `pytest -q` -> `21 passed`.
- Riscos conhecidos / follow-ups:
  - A implementacao V1 prioriza integracao/contrato; calibracao competitiva (hiperparametros, QA/dissimilaridade, custo GPU) ainda requer ablacoes e benchmark local formal antes de promover ao caminho principal.

## 2026-02-16 - marcusvinicius/Codex - PLAN-079

- Data UTC: `2026-02-16T12:59:00Z`
- Plano: `PLAN-079`
- Resumo:
  - Implementada torre de sequencia com opcoes de memoria eficiente:
    - `flash` via `scaled_dot_product_attention` com suporte a `use_gradient_checkpointing`;
    - `mamba_like` causal (SSM simplificado) com memoria linear.
  - Removido caminho denso NxN dos backbones EGNN/IPA, com construcao dinamica de grafo 3D esparso por raio fisico (`radius_angstrom`) e limite de vizinhos (`max_neighbors`), com `torch.sparse`.
  - Adicionado backend opcional `torch_geometric` com falha cedo acionavel quando dependencias (`radius_graph`/`torch-cluster`) nao estao disponiveis.
  - Configuracao SE(3) ampliada com parametros de escala (`sequence_tower`, `sequence_heads`, `use_gradient_checkpointing`, `graph_backend`, `radius_angstrom`, `max_neighbors`, `graph_chunk_size`) e persistencia desses parametros em manifests.
  - Atualizada documentacao com perfil recomendado para alvos longos (`L<=5500`).
  - Cobertura de testes expandida para contratos de grafo esparso, backend `torch_geometric` e treino/amostragem com configuracao linear.
- Arquivos principais tocados:
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3/sequence_tower.py`
  - `src/rna3d_local/se3/sparse_graph.py`
  - `src/rna3d_local/se3/egnn_backbone.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `tests/test_se3_memory.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `9 passed`
  - `pytest -q` -> `24 passed`
  - `python -m rna3d_local --help` -> superficie CLI preservada.
- Riscos conhecidos / follow-ups:
  - O backend `torch_geometric` depende de stack extra (`torch-cluster`); quando ausente, o pipeline falha cedo por contrato.
  - Benchmark de throughput/VRAM em GPU Kaggle para `L` muito alto ainda precisa de medicao formal em experimento dedicado.

## 2026-02-16 - marcusvinicius/Codex - PLAN-080

- Data UTC: `2026-02-16T13:07:32Z`
- Plano: `PLAN-080`
- Resumo:
  - Integrada extracao termodinamica 2D (BPP) com backend configuravel (`rnafold`, `linearfold`) e backend explicito de teste (`mock`) no DataLoader SE(3), sem fallback silencioso.
  - Novo modulo `training/thermo_2d.py` com:
    - execucao de `RNAfold -p` e parse de `dot.ps` (`ubox`);
    - execucao de `linearfold --bpp` e parse de pares/probabilidades;
    - cache opcional por hash de sequencia/backend;
    - validacoes estritas de intervalo/consistencia.
  - `TargetGraph` estendido para transportar sinal BPP:
    - marginal por residuo (adicionado como feature de no);
    - pares esparsos direcionados (`bpp_pair_src/dst/prob`).
  - Backbones EGNN/IPA atualizados para injetar BPP como bias continuo em arestas dinamicas do grafo esparso.
  - Configuracao SE(3) ampliada com `thermo_backend`, `rnafold_bin`, `linearfold_bin`, `thermo_cache_dir`; manifests de treino/amostragem registram esses parametros efetivos.
  - Documentacao atualizada com configuracao termodinamica para alvo longo.
  - Testes adicionados para contratos BPP e cache, e testes SE(3) ajustados para backend de teste explicito.
- Arquivos principais tocados:
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/dataset_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3/graph_builder.py`
  - `src/rna3d_local/se3/sparse_graph.py`
  - `src/rna3d_local/se3/egnn_backbone.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `tests/test_thermo_2d.py`
  - `tests/test_se3_pipeline.py`
  - `tests/test_se3_memory.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `12 passed`
  - `pytest -q` -> `27 passed`
  - `python -m rna3d_local --help` -> superficie CLI preservada.
- Riscos conhecidos / follow-ups:
  - `thermo_backend=rnafold|linearfold` depende de binarios no ambiente de execucao; indisponibilidade causa falha cedo por contrato.
  - Benchmark de tempo/memoria com RNAfold/LinearFold em `L` muito alto (>=5500) ainda precisa de medicao dedicada para ajustar cache/batch.

## 2026-02-16 - marcusvinicius/Codex - PLAN-081

- Data UTC: `2026-02-16T13:15:45Z`
- Plano: `PLAN-081`
- Resumo:
  - Integrada extracao de sinal evolutivo MSA/covariancia com backend configuravel (`mmseqs2`) e backend explicito de teste (`mock`) em `training/msa_covariance.py`, sem fallback silencioso.
  - Implementado parser multicadeia com separador configuravel (`chain_separator`) em `se3/sequence_parser.py`, produzindo mapeamento por residuo para cadeia.
  - DataLoader SE(3) agora combina:
    - BPP termodinamica por cadeia;
    - covariancia MSA por cadeia;
    - features de no (marginais BPP + MSA);
    - pares esparsos dirigidos para BPP/MSA.
  - `TargetGraph` estendido com awareness multicadeia (`chain_index`, `residue_index`) e sinais MSA.
  - Implementado Relative Positional Encoding 2D com offset massivo em chain breaks (`chain_break_offset`) e injetado como bias continuo em EGNN/IPA junto de BPP+MSA.
  - Configuracao SE(3) ampliada com:
    - `msa_backend`, `mmseqs_bin`, `mmseqs_db`, `msa_cache_dir`;
    - `chain_separator`, `chain_break_offset`;
    - `max_msa_sequences`, `max_cov_positions`, `max_cov_pairs`.
  - Atualizados manifests de treino/amostragem com parametros efetivos de MSA/multicadeia.
  - Cobertura de testes expandida para MSA mock, falha de binario mmseqs2, parse multicadeia e offsets de chain-break.
- Arquivos principais tocados:
  - `src/rna3d_local/training/msa_covariance.py`
  - `src/rna3d_local/se3/sequence_parser.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/dataset_se3.py`
  - `src/rna3d_local/se3/graph_builder.py`
  - `src/rna3d_local/se3/sparse_graph.py`
  - `src/rna3d_local/se3/egnn_backbone.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `tests/test_msa_covariance.py`
  - `tests/test_thermo_2d.py`
  - `tests/test_se3_pipeline.py`
  - `tests/test_se3_memory.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `17 passed`
  - `pytest -q` -> `32 passed`
  - `python -m rna3d_local --help` -> superficie CLI preservada.
- Riscos conhecidos / follow-ups:
  - `msa_backend=mmseqs2` exige binario e banco de dados homologo no ambiente; ausencia falha cedo por contrato.
  - Benchmark de custo real para alvos longos/multicadeia com MMseqs2 ainda pendente para calibrar budget de inferencia.

## 2026-02-16 - marcusvinicius/Codex - PLAN-082

- Data UTC: `2026-02-16T13:20:50Z`
- Plano: `PLAN-082`
- Resumo:
  - Implementado modulo de sondagem quimica `PDB x QUICK_START` em `training/chemical_mapping.py` para estimar exposicao ao solvente por residuo.
  - O mapping agora cruza:
    - reatividade `reactivity_dms/reactivity_2a3` normalizada por alvo;
    - geometria PDB (distancia ao centroide) quando `labels` estao disponiveis no treino;
    - fusao continua com origem explicita: `quickstart_pdb_cross` (treino) ou `quickstart_only` (inferencia).
  - DataLoader SE(3) passa a calcular e propagar `chemical_mapping` para `TargetGraph`.
  - `TargetGraph` estendido com `chem_exposure` e `chem_source`; `node_features` inclui o sinal de exposicao quimica.
  - Backbones EGNN/IPA atualizados para aplicar bias quimico continuo por aresta (`chem_edge_bias`) junto dos biases BPP/MSA/chain-break.
  - Manifestos de treino/amostragem agora registram contagem por origem do chemical mapping.
  - Testes adicionados para contratos do mapping e cobertura de integracao sem regressao.
- Arquivos principais tocados:
  - `src/rna3d_local/training/chemical_mapping.py`
  - `src/rna3d_local/training/dataset_se3.py`
  - `src/rna3d_local/se3/graph_builder.py`
  - `src/rna3d_local/se3/egnn_backbone.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `tests/test_chemical_mapping.py`
  - `tests/test_se3_pipeline.py`
  - `tests/test_se3_memory.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_chemical_mapping.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `20 passed`
  - `pytest -q` -> `35 passed`
  - `python -m rna3d_local --help` -> superficie CLI preservada.
- Riscos conhecidos / follow-ups:
  - O sinal `quickstart_only` em inferencia nao usa coordenadas PDB; calibracao de pesos/escala ainda requer benchmark com dados reais.
  - Medicao dedicada de impacto em pseudoknots complexos (score local) ainda pendente antes de promover para caminho competitivo principal.

## 2026-02-16 - marcusvinicius/Codex - PLAN-083

- Data UTC: `2026-02-16T13:28:01Z`
- Plano: `PLAN-083`
- Resumo:
  - Implementado construtor de folds anti-leakage por homologia em `homology_folds.py` com thresholds configuraveis de identidade/cobertura.
  - Backends suportados:
    - `mmseqs2` (`easy-cluster`);
    - `cdhit_est` (`cd-hit-est`);
    - `mock` para testes locais.
  - Fluxo agora agrupa conjuntamente:
    - sequencias de treino;
    - sequencias PDB/homologas.
  - Atribuicao de folds feita por cluster (nunca por amostra), com validacao estrita para impedir cluster em multiplos folds.
  - Novo comando CLI `build-homology-folds` exposto em `rna3d_local`.
  - Artefatos gerados:
    - `clusters.parquet`
    - `train_folds.parquet`
    - `homology_folds_manifest.json` com `max_folds_per_cluster_train`.
  - Documentacao atualizada com uso do comando de folds homologo na validacao local.
- Arquivos principais tocados:
  - `src/rna3d_local/homology_folds.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_homology_folds.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_homology_folds.py tests/test_chemical_mapping.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_phase2_hybrid.py` -> `23 passed`
  - `pytest -q` -> `38 passed`
  - `python -m rna3d_local --help` -> comando `build-homology-folds` presente.
- Riscos conhecidos / follow-ups:
  - `backend=mmseqs2|cdhit_est` depende de binarios instalados e, no caso mmseqs2, da base de homologos disponivel no ambiente.
  - Benchmark de impacto no score local (delta vs Random K-Folds) ainda pendente para calibrar melhor `identity_threshold`/`coverage_threshold`.

## 2026-02-16 - marcusvinicius/Codex - PLAN-084

- Data UTC: `2026-02-16T13:36:32Z`
- Plano: `PLAN-084`
- Resumo:
  - `build-homology-folds` agora aplica estratificacao de dominio por padrao (modo estrito) sem quebrar isolamento por cluster.
  - Fonte de dominio suportada com fail-fast:
    - arquivo explicito (`--domain-labels`);
    - coluna no treino (`--domain-column`);
    - inferencia deterministica por `--description-column`.
  - Validacao de cobertura de dominio por fold passou a usar limite maximo factivel por dominio (limitado por numero de clusters do dominio), bloqueando distribuicoes subotimas.
  - `train_folds.parquet` agora inclui `domain_label`; manifest passou a registrar:
    - `domain_counts_train`;
    - `domain_fold_coverage_train`;
    - `domain_fold_coverage_train_max_possible`;
    - `domain_counts_by_fold_train`.
  - Novo comando `evaluate-homology-folds` para avaliacao com prioridade de orphans:
    - recebe metricas por alvo;
    - classifica orphan por labels explicitos ou score de retrieval;
    - calcula `priority_score` ponderado por `orphan_weight`.
  - Documentacao atualizada com comandos de estratificacao e avaliacao orphan-prioritaria.
- Arquivos principais tocados:
  - `src/rna3d_local/homology_folds.py`
  - `src/rna3d_local/homology_eval.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_homology_folds.py`
  - `tests/test_homology_eval.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_homology_folds.py tests/test_homology_eval.py tests/test_phase2_hybrid.py` -> `11 passed`
  - `pytest -q` -> `42 passed`
  - `python -m rna3d_local --help` -> comandos `build-homology-folds` e `evaluate-homology-folds` presentes.
- Riscos conhecidos / follow-ups:
  - A qualidade da estratificacao depende da cobertura/qualidade do campo de dominio (`domain_label`/`description`); descricoes muito vagas tendem a gerar `unknown`.
  - O `priority_score` orphan-prioritario exige metricas por alvo confiaveis; sem scorer local por alvo o relatorio vira apenas diagnostico.

## 2026-02-16 - marcusvinicius/Codex - PLAN-085

- Data UTC: `2026-02-16T13:41:30Z`
- Plano: `PLAN-085`
- Resumo:
  - Implementada loss estrutural composta para treino SE(3), substituindo dependencia de MSE puro por combinacao configuravel de:
    - `FAPE` (frame-aligned point error) com computacao chunked;
    - `TM-core loss` (aproximacao diferenciavel de TM-score C1' via Kabsch);
    - `Clash loss` de repulsao Van der Waals com exclusao de pares covalentes.
  - Integracao no `trainer_se3` com pesos explicitos por termo e rastreamento de componentes no `metrics.json`:
    - `loss_mse`, `loss_fape`, `loss_tm`, `loss_clash`, `loss_generative`, `loss_total`.
  - `config_se3` estendida com parametros de loss fisica/estrutural:
    - `loss_weight_mse`, `loss_weight_fape`, `loss_weight_tm`, `loss_weight_clash`;
    - `fape_clamp_distance`, `fape_length_scale`;
    - `vdw_min_distance`, `vdw_repulsion_power`;
    - `loss_chunk_size`.
  - Validacoes fail-fast adicionadas para contratos invalidos de loss/config (pesos negativos, soma de pesos zerada, parametros fora de faixa).
  - README atualizado com parametros novos no exemplo de config.
- Arquivos principais tocados:
  - `src/rna3d_local/training/losses_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/training/config_se3.py`
  - `tests/test_se3_losses.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_se3_losses.py tests/test_se3_pipeline.py tests/test_se3_memory.py` -> `10 passed`
  - `pytest -q` -> `46 passed`
  - `python -m rna3d_local --help` -> superficie CLI preservada.
- Riscos conhecidos / follow-ups:
  - O termo FAPE chunked e O(L^2) em custo computacional (com memoria controlada); para alvos muito longos, pode aumentar tempo por epoca.
  - Os pesos default da loss composta ainda precisam de calibracao empirica por score local para definir regime competitivo final.

## 2026-02-16 - marcusvinicius/Codex - PLAN-086

- Data UTC: `2026-02-16T14:00:43Z`
- Plano: `PLAN-086`
- Resumo:
  - Implementado passo de minimizacao pos-inferencia em `minimization.py` para `predictions long` com backend explicito:
    - `openmm` (forcas bond/angle/VdW-like + `LocalEnergyMinimizer`);
    - `pyrosetta` (fail-fast explicito para contrato C1' only);
    - `mock` (deterministico para testes locais).
  - Comando CLI novo `minimize-ensemble` adicionado ao `rna3d_local`.
  - Contratos estritos/fail-fast adicionados para:
    - parametros fisicos/iteracoes invalidos;
    - chaves duplicadas (`target_id:model_id:resid`);
    - backend/dependencia indisponivel;
    - saida com shape invalido ou coordenadas nao-finitas.
  - Saida passa a incluir metadados de refinamento (`refinement_backend`, `refinement_steps`) e `manifest` com estatisticas de deslocamento.
  - README atualizado para inserir minimizacao antes de `export-submission` nos fluxos TBM e hibrido.
- Arquivos principais tocados:
  - `src/rna3d_local/minimization.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_minimization.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_minimization.py tests/test_phase2_hybrid.py tests/test_description_and_submission.py` -> `9 passed`
  - `python -m rna3d_local --help` -> comando `minimize-ensemble` presente.
  - `python -m rna3d_local minimize-ensemble --help` -> parametros esperados presentes.
  - `pytest -q` -> `49 passed`
- Riscos conhecidos / follow-ups:
  - `backend=openmm` depende da biblioteca OpenMM no ambiente Kaggle/local; ausencia gera erro de contrato.
  - `backend=pyrosetta` permanece bloqueado para input C1' only (necessita entrada full-atom para `rna_minimize`).

## 2026-02-16 - marcusvinicius/Codex - PLAN-087

- Data UTC: `2026-02-16T14:21:25Z`
- Plano: `PLAN-087`
- Resumo:
  - `build-hybrid-candidates` recebeu fallback obrigatorio para alvos ultralongos (`L > threshold`) roteando exclusivamente para `generative_se3`.
  - Adicionada limpeza agressiva de memoria para fontes fundacionais no roteador:
    - `gc.collect()`
    - `torch.cuda.empty_cache()` quando CUDA ativa.
  - Novo parametro de roteamento:
    - `--ultra-long-seq-threshold` (default `1500`) exposto no CLI e registrado em manifest.
  - `sparse_graph.py` removido do caminho com `torch.cdist` chunk-vs-all no backend `torch_sparse`.
  - Novo caminho esparso:
    - usa `torch_cluster.radius_graph` quando disponivel;
    - senao usa particionamento espacial por celulas (sem matriz densa NxN) com cutoff e `max_neighbors`.
  - `sequence_tower.py` reforcado para usar SDPA com kernel Flash em CUDA (`enable_math=False`) e erro acionavel quando indisponivel (orienta `mamba_like`).
  - README atualizado com threshold ultralongo e observacao de grafo sem matriz densa.
- Arquivos principais tocados:
  - `src/rna3d_local/hybrid_router.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `src/rna3d_local/se3/sparse_graph.py`
  - `src/rna3d_local/se3/sequence_tower.py`
  - `tests/test_phase2_hybrid.py`
  - `tests/test_se3_memory.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_phase2_hybrid.py tests/test_se3_memory.py tests/test_se3_pipeline.py` -> `13 passed`
  - `pytest -q` -> `52 passed`
  - `python -m rna3d_local build-hybrid-candidates --help` -> flag `--ultra-long-seq-threshold` presente.
- Riscos conhecidos / follow-ups:
  - O fallback ultralongo exige cobertura de `--se3`; sem cobertura o pipeline falha cedo por contrato.
  - Busca espacial por celulas troca VRAM por custo de CPU/Python; para `L` extremo pode exigir tuning de `radius_angstrom/max_neighbors`.

## 2026-02-16 - marcusvinicius/Codex - PLAN-088

- Data UTC: `2026-02-16T14:30:51Z`
- Plano: `PLAN-088`
- Resumo:
  - `geometry.py` foi refatorado para construir frame local de RNA por resíduo com proxies de `P`, `C4'` e `N1/N9` a partir de `C1'` + identidade de base (purina/pirimidina).
  - `ipa_backbone.py` passou a recalcular frames locais por camada com as coordenadas correntes e usar deltas de aresta no frame local para:
    - viés orientacional no score de atenção;
    - atualização coordenada com mistura frame-local + deslocamento global.
  - `trainer_se3.py` passou a injetar `base_features` (A/C/G/U) no `IpaBackbone` para orientar o frame local RNA.
  - `losses_se3.py` recebeu estabilização do alinhamento Kabsch (rotação calculada em `no_grad`) para evitar gradientes não finitos em casos degenerados.
  - Novos testes de contrato/estabilidade adicionados para geometria e IPA.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/geometry.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/training/losses_se3.py`
  - `tests/test_ipa_geometry.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_ipa_geometry.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py` -> `14 passed`
  - `pytest -q` -> `55 passed`
- Riscos conhecidos / follow-ups:
  - O frame local usa proxies geometricos de `P/C4'/N1/N9` derivados de `C1'`; calibracao de distancias proxy pode ser ajustada apos benchmark local com alvos reais.
  - O viés orientacional foi limitado (`tanh` + escala) para estabilidade; tuning adicional pode ser necessario em treino longo.

## 2026-02-16 - marcusvinicius/Codex - PLAN-089

- Data UTC: `2026-02-16T14:36:41Z`
- Plano: `PLAN-089`
- Resumo:
  - `sampler.py` recebeu amostragem rapida por solver ODE:
    - diffusion com passo DPM-like em malha reduzida;
    - flow com integrador Heun de 2a ordem;
    - validacao fail-fast de shape e finitude por sample.
  - `diversity.py` foi expandido com:
    - distancia estrutural aproximada (`1 - cosseno`);
    - estimativa de clash ratio por grade espacial;
    - pre-filtro de 50% piores via score ajustado;
    - clustering Max-Min e selecao de medoides por cluster.
  - `select_top5.py` agora:
    - exige `qa_score` e `final_score` nao nulos;
    - elimina 50% piores candidatos por target;
    - seleciona Top-5 como medoides de clusters distintos;
    - registra diagnosticos de pre-filtro/clustering no manifest.
  - `cli_parser.py` atualizou default de `sample-se3-ensemble --n-samples` para `24`.
  - Testes novos de estrategia Best-of-5 adicionados.
- Arquivos principais tocados:
  - `src/rna3d_local/generative/sampler.py`
  - `src/rna3d_local/ensemble/diversity.py`
  - `src/rna3d_local/ensemble/select_top5.py`
  - `src/rna3d_local/cli_parser.py`
  - `tests/test_best_of5_strategy.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_best_of5_strategy.py tests/test_se3_pipeline.py` -> `6 passed`
  - `pytest -q tests/test_best_of5_strategy.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_ipa_geometry.py` -> `13 passed`
  - `pytest -q` -> `58 passed`
- Riscos conhecidos / follow-ups:
  - A distancia estrutural aproximada usada no clustering (cosseno de vetor centrado) e proxy de TM-score; vale calibrar contra US-align local para validar correlacao em alvos longos.
  - O pre-filtro em 50% pode remover modos raros quando o QA local estiver mal calibrado; manter monitoramento por target no manifest.

## 2026-02-16 - marcusvinicius/Codex - PLAN-090

- Data UTC: `2026-02-16T14:40:38Z`
- Plano: `PLAN-090`
- Resumo:
  - `training/losses_se3.py` recebeu clash loss exponencial:
    - penalidade principal com `expm1(alpha * penetration)` para pares nao-covalentes;
    - reforco critico adicional para distancias sub-`2.0A`;
    - calculo mantido em modo chunked, sem matriz densa global.
  - `minimization.py` recebeu restraints harmonicas fortes na minimizacao:
    - OpenMM usa `CustomExternalForce` ancorada nas coordenadas iniciais;
    - caminho `mock` passou a aplicar ancoragem progressiva para preservar macro-topologia;
    - contrato novo limita `max_iterations <= 100`.
  - `minimize_ensemble` agora exige `position_restraint_k > 0`, registra parametro no manifest e adiciona coluna `refinement_position_restraint_k`.
  - CLI atualizada:
    - `--max-iterations` default para `80`;
    - novo argumento `--position-restraint-k` (default `800.0`).
  - README atualizada com exemplos alinhados ao novo contrato de relaxacao curta com restraints.
  - Testes expandidos para clash exponencial e budget de iteracoes.
- Arquivos principais tocados:
  - `src/rna3d_local/training/losses_se3.py`
  - `src/rna3d_local/minimization.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_se3_losses.py`
  - `tests/test_minimization.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_se3_losses.py tests/test_minimization.py` -> `9 passed`
  - `pytest -q` -> `60 passed`
- Riscos conhecidos / follow-ups:
  - O backend `pyrosetta` permanece fail-fast para input C1' only (full-atom ainda necessario).
  - A intensidade de `position_restraint_k` pode requerer ajuste por familia de alvo para equilibrar remocao de clash vs rigidez global.

## 2026-02-16 - marcusvinicius/Codex - PLAN-091

- Data UTC: `2026-02-16T14:43:23Z`
- Plano: `PLAN-091`
- Resumo:
  - `parse_sequence_with_chains` passou a gerar indice posicional 1D por residuo com salto absoluto ao trocar de cadeia:
    - novo campo `residue_position_index_1d` em `ParsedSequence`;
    - salto default de `+1000` em cada chain break;
    - validacao fail-fast para `chain_break_offset_1d <= 0`.
  - `graph_builder` passou a usar o indice 1D do parser para `TargetGraph.residue_index`, substituindo o `arange` contiguo.
  - Novos testes adicionados para contrato do parser multicadeia.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/sequence_parser.py`
  - `src/rna3d_local/se3/graph_builder.py`
  - `tests/test_sequence_parser.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_sequence_parser.py tests/test_se3_pipeline.py tests/test_msa_covariance.py` -> `9 passed`
  - `pytest -q` -> `63 passed`
- Riscos conhecidos / follow-ups:
  - O salto 1D agora e explicito no parser e somado ao `chain_break_offset` ja aplicado em features relativas; calibracao fina pode ser feita em benchmark local para evitar separacao excessiva em targets com muitas cadeias.

## 2026-02-16 - marcusvinicius/Codex - PLAN-092

- Data UTC: `2026-02-16T14:52:39Z`
- Plano: `PLAN-092`
- Resumo:
  - Protocolo de treino local para 16GB VRAM consolidado com:
    - recorte dinamico (sequencia + espacial),
    - mixed precision BF16,
    - gradient checkpointing no backbone SE(3),
    - gradient accumulation para batch virtual.
  - Correcao critica de estabilidade numerica:
    - `training/losses_se3.py` passou a executar Kabsch/TM-core em `float32` com autocast desabilitado no trecho de SVD;
    - elimina erro CUDA `svd_cuda_gesvdjBatched not implemented for BFloat16`.
- Arquivos principais tocados:
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/training/losses_se3.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_se3_pipeline.py::test_train_sample_rank_select_se3_pipeline tests/test_se3_pipeline.py::test_train_sample_se3_with_multichain_sequence tests/test_se3_memory.py::test_train_and_sample_se3_with_linear_memory_config` -> `3 passed`
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py tests/test_sequence_parser.py` -> `15 passed`
  - `pytest -q` -> `63 passed`
- Riscos conhecidos / follow-ups:
  - O alinhamento Kabsch do TM-core agora roda em `float32`, reduzindo risco de erro BF16 ao custo de overhead pequeno no treinamento.

## 2026-02-16 - marcusvinicius/Codex - PLAN-093

- Data UTC: `2026-02-16T15:16:34Z`
- Plano: `PLAN-093`
- Resumo:
  - FASE 1 do treino local foi estruturada em pipeline dedicado:
    - novo comando `prepare-phase1-data-lab` para precomputar BPP + MSA e empacotar `training_store.zarr`;
    - cache de termo/MSA passou a suportar paralelismo por alvo com `num_workers` e escrita atomica de cache;
    - treino SE(3) agora aceita `--training-store` e carrega um target por vez via `ZarrTrainingStore` (lazy loading).
  - Foi adicionado modulo de store lazy:
    - `training/store_zarr.py` (build + loader de `training_store.zarr`);
    - `training/data_lab.py` (orquestracao da fase 1 local).
  - CLI e documentacao atualizadas para fluxo de pre-processamento local com 16 threads.
- Arquivos principais tocados:
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/msa_covariance.py`
  - `src/rna3d_local/training/dataset_se3.py`
  - `src/rna3d_local/training/store_zarr.py`
  - `src/rna3d_local/training/data_lab.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `README.md`
  - `pyproject.toml`
  - `tests/test_thermo_2d.py`
  - `tests/test_msa_covariance.py`
  - `tests/test_phase1_data_lab.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_thermo_2d.py tests/test_msa_covariance.py tests/test_phase1_data_lab.py` -> `10 passed`
  - `pytest -q tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_sequence_parser.py tests/test_se3_losses.py` -> `15 passed`
  - `pytest -q` -> `66 passed`
- Riscos conhecidos / follow-ups:
  - `prepare-phase1-data-lab` exige dependencia opcional `zarr`; sem ela o comando falha cedo com erro acionavel.
  - O fluxo ainda depende de tuning empirico de `workers` e parametros MMseqs2 para equilibrar throughput de CPU e I/O.

## 2026-02-16 - marcusvinicius/Codex - PLAN-094

- Data UTC: `2026-02-16T15:20:23Z`
- Plano: `PLAN-094`
- Resumo:
  - Foi introduzido `training_protocol` na config SE(3), com suporte a:
    - `custom` (comportamento livre);
    - `local_16gb` (contrato estrito para treino local com 16GB VRAM).
  - Regras obrigatorias do protocolo `local_16gb`:
    - `dynamic_cropping=true`;
    - `crop_min_length/crop_max_length` em `[256,384]`;
    - `use_gradient_checkpointing=true`;
    - `autocast_bfloat16=true`;
    - `gradient_accumulation_steps` em `[16,32]`.
  - Runtime do treino agora falha cedo quando BF16 e exigido sem suporte (sem fallback silencioso para CPU/FP32).
  - `training_protocol` passou a ser persistido em `config_effective`.
- Arquivos principais tocados:
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `README.md`
  - `tests/test_se3_losses.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_se3_losses.py tests/test_se3_pipeline.py tests/test_se3_memory.py` -> `14 passed`
  - `pytest -q` -> `68 passed`
- Riscos conhecidos / follow-ups:
  - Em ambientes CPU-only, configs com `autocast_bfloat16=true` agora falham cedo por contrato; para depuracao sem GPU, usar `training_protocol=custom` e `autocast_bfloat16=false`.

## 2026-02-16 - marcusvinicius/Codex - PLAN-095

- Data UTC: `2026-02-16T15:37:52Z`
- Plano: `PLAN-095`
- Resumo:
  - Foi implementado o oraculo local de score Best-of-5 com USalign:
    - novo modulo `evaluation/usalign_scorer.py` com parsing estrito de `ground_truth` e `submission`;
    - conversao para PDB minimalista (C1') e agregacao Best-of-5 por alvo;
    - geracao de `score.json` e relatorio por alvo para auditoria local.
  - Foi adicionado novo comando CLI:
    - `score-local-bestof5 --ground-truth ... --submission ... --usalign-bin ... --score-json ... [--report ...]`.
  - Foram adicionados testes dedicados com binario USalign fake para validar:
    - fluxo de sucesso;
    - falha de contrato por mismatch de chaves;
    - compatibilidade de ground truth em formato `target_id+resid` com `x_1/y_1/z_1`.
  - README atualizado com comando de score local e observacao operacional.
- Arquivos principais tocados:
  - `src/rna3d_local/evaluation/__init__.py`
  - `src/rna3d_local/evaluation/usalign_scorer.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_usalign_scorer.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_usalign_scorer.py` -> `3 passed`
  - `pytest -q` -> `71 passed`
- Riscos conhecidos / follow-ups:
  - O parser de output do USalign prioriza `-outfmt 2` e fallback por regex `TM-score`; em upgrade de formato do binario, convem manter teste de contrato com fixture real do executavel oficial.

## 2026-02-16 - marcusvinicius/Codex - PLAN-096

- Data UTC: `2026-02-16T16:25:07Z`
- Plano: `PLAN-096`
- Resumo:
  - Execucao end-to-end local foi rodada com treino SE(3) e score local:
    - preparo de dados locais para treino/inferencia em `runs/20260216_full_local_run/`;
    - `prepare-phase1-data-lab` + `train-se3-generator` (subset controlado de 256 targets, store lazy zarr);
    - `sample-se3-ensemble` + `rank-se3-ensemble` + `select-top5-se3`;
    - `export-submission` + `check-submission` em modo estrito.
  - Ajustes tecnicos para viabilizar execucao no ambiente atual:
    - `store_zarr.py`: compatibilidade com API de `zarr` v3 em `create_dataset`;
    - `sparse_graph.py`: robustez de conectividade minima em grafos esparsos dirigidos;
    - `usalign_scorer.py`: alinhamento com `-atom \" C1'\"` e timeout default ampliado para scorer local.
- Arquivos principais tocados:
  - `src/rna3d_local/training/store_zarr.py`
  - `src/rna3d_local/se3/sparse_graph.py`
  - `src/rna3d_local/evaluation/usalign_scorer.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_phase1_data_lab.py` -> `1 passed`
  - `pytest -q tests/test_se3_memory.py::test_train_and_sample_se3_with_linear_memory_config tests/test_se3_memory.py::test_torch_geometric_backend_contract` -> `2 passed`
  - `pytest -q tests/test_usalign_scorer.py` -> `3 passed`
  - `pytest -q` -> `71 passed`
- Riscos conhecidos / follow-ups:
  - Score local Best-of-5 completo em todos os 28 alvos pode exceder janela operacional por custo do USalign em alvos especificos; manter medicao parcial explicita e/ou paralelizar por alvo em iteracao futura.

## 2026-02-16 - marcusvinicius/Codex - PLAN-097

- Data UTC: `2026-02-16T17:42:20Z`
- Plano: `PLAN-097`
- Resumo:
  - Politica central anti-mock adicionada em `src/rna3d_local/mock_policy.py`, com bloqueio por padrao em runtime competitivo e liberacao apenas quando:
    - `stage` com prefixo `TEST`; ou
    - variavel de ambiente `RNA3D_ALLOW_MOCK_BACKENDS=1` (uso explicito de teste local).
  - Bloqueio estrito aplicado nos pontos criticos:
    - `src/rna3d_local/encoder.py` (`encoder=mock`);
    - `src/rna3d_local/training/config_se3.py` (`thermo_backend=mock`, `msa_backend=mock`);
    - `src/rna3d_local/training/thermo_2d.py` (`thermo_backend=mock`);
    - `src/rna3d_local/training/msa_covariance.py` (`msa_backend=mock`);
    - `src/rna3d_local/homology_folds.py` (`backend=mock`);
    - `src/rna3d_local/minimization.py` (`backend=mock`).
  - Ajuste de default de config SE(3):
    - `msa_backend` passa a default `mmseqs2` (antes `mock`) em `load_se3_train_config`.
  - Suite de testes adaptada e expandida:
    - `tests/conftest.py` habilita permissivo de mock apenas para ambiente de teste (`RNA3D_ALLOW_MOCK_BACKENDS=1`);
    - novo `tests/test_mock_policy.py` cobre bloqueio de mock fora de `TEST` quando o permissivo e desligado.
  - `README.md` atualizado para explicitar que mock e bloqueado por padrao no fluxo competitivo.
- Arquivos principais tocados:
  - `src/rna3d_local/mock_policy.py`
  - `src/rna3d_local/encoder.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/msa_covariance.py`
  - `src/rna3d_local/homology_folds.py`
  - `src/rna3d_local/minimization.py`
  - `tests/conftest.py`
  - `tests/test_mock_policy.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q tests/test_mock_policy.py tests/test_msa_covariance.py tests/test_thermo_2d.py tests/test_homology_folds.py tests/test_minimization.py tests/test_retrieval_latent.py tests/test_se3_pipeline.py tests/test_se3_memory.py tests/test_se3_losses.py` -> `40 passed`
  - `pytest -q` -> `77 passed`
- Riscos conhecidos / follow-ups:
  - O permissivo por variavel de ambiente permanece disponivel exclusivamente para testes locais; em execucao competitiva, manter variavel ausente/`0`.

## 2026-02-16 - marcusvinicius/Codex - PLAN-098

- Data UTC: `2026-02-16T18:06:07Z`
- Plano: `PLAN-098`
- Resumo:
  - Fase 2 (preditores offline) deixou de gerar coordenadas sinteticas:
    - `predict-rnapro-offline`, `predict-chai1-offline`, `predict-boltz1-offline` agora exigem `config.json` com `entrypoint` e executam um runner externo em modo estrito;
    - o parquet gerado passa por validacao de contrato (colunas, chaves unicas, finitude, cobertura por `target_id/model_id/resid`) e falha cedo sem fallback.
  - Fase 1 (embeddings) deixou de usar hashing:
    - `encoder=ribonanzanet2` agora carrega um modelo real offline via `torch.jit.load` (TorchScript) e valida dimensao do embedding.
  - Assets:
    - `build-phase2-assets` valida `entrypoint` em `config.json` de cada modelo.
  - Repositorio:
    - `.gitignore` passa a ignorar o binario local `src/rna3d_local/evaluation/USalign`.
  - Suite de testes atualizada/expandida para cobrir o novo contrato (runner stub por `entrypoint` e TorchScript toy model).
- Arquivos principais tocados:
  - `.gitignore`
  - `src/rna3d_local/predictor_common.py`
  - `src/rna3d_local/rnapro_offline.py`
  - `src/rna3d_local/chai1_offline.py`
  - `src/rna3d_local/boltz1_offline.py`
  - `src/rna3d_local/phase2_assets.py`
  - `src/rna3d_local/encoder.py`
  - `tests/test_phase2_hybrid.py`
  - `tests/test_phase2_assets.py`
  - `tests/test_encoder_torchscript.py`
  - `README.md`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q` -> `78 passed`
- Riscos conhecidos / follow-ups:
  - O contrato de `entrypoint` exige que os modelos/runners (weights + script) sejam fornecidos via Kaggle Dataset offline; qualquer divergencia deve falhar cedo (sem gerar submission).
  - O encoder TorchScript deve ser exportado com interface compatível (tokens `LongTensor[B,L]` -> embedding `[B,D]` ou `[B,L,D]`) e `D` igual ao `embedding_dim` configurado.

## 2026-02-16 - marcusvinicius/Codex - PLAN-099

- Data UTC: `2026-02-16T18:39:26Z`
- Plano: `PLAN-099`
- Resumo:
  - Removido suporte a `mock` do runtime/CLI/config do pipeline:
    - `thermo_backend`, `msa_backend`, `encoder`, `minimization backend` e `homology_folds backend` nao aceitam mais `mock`;
    - implementacoes sinteticas `*_mock` foram removidas (sem fallback silencioso).
  - `build-homology-folds`:
    - `backend=mock` substituido por `backend=python` (implementacao interna deterministica) como opcao explicita; `mmseqs2`/`cdhit_est` permanecem backends reais.
  - Suite de testes reescrita para nao depender de backends `mock`:
    - uso de `monkeypatch` para simular chamadas externas (RNAfold/MMseqs2/OpenMM) em testes de integracao;
    - ajustes de configs de teste para `thermo_backend=rnafold` e `msa_backend=mmseqs2` com `mmseqs_db` definido.
- Arquivos principais tocados:
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/encoder.py`
  - `src/rna3d_local/homology_folds.py`
  - `src/rna3d_local/minimization.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/msa_covariance.py`
  - `tests/test_thermo_2d.py`
  - `tests/test_msa_covariance.py`
  - `tests/test_minimization.py`
  - `tests/test_retrieval_latent.py`
  - `tests/test_phase1_data_lab.py`
  - `tests/test_se3_memory.py`
  - `tests/test_se3_losses.py`
  - `tests/test_se3_pipeline.py`
  - `tests/conftest.py`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q` -> `73 passed`
- Riscos conhecidos / follow-ups:
  - A execucao real agora exige binarios/dependencias reais (RNAfold/MMseqs2/OpenMM) ou runners offline conforme contrato; ausencias devem falhar cedo (sem degradacao silenciosa).

## 2026-02-16 - marcusvinicius/Codex - PLAN-100

- Data UTC: `2026-02-16T18:46:49Z`
- Plano: `PLAN-100`
- Resumo:
  - Suite de testes passou a rodar sem warnings:
    - removido `RuntimeWarning` em `training/msa_covariance.py` (evita `log(0)` ao calcular entropia);
    - removidos `ZarrDeprecationWarning` em `training/store_zarr.py` usando `create_array` quando disponivel;
    - filtrado apenas o `DeprecationWarning` conhecido de terceiro (`torch_geometric.distributed`) via `pytest.ini`.
- Arquivos principais tocados:
  - `src/rna3d_local/training/msa_covariance.py`
  - `src/rna3d_local/training/store_zarr.py`
  - `pytest.ini`
  - `PLANS.md`
  - `CHANGES.md`
  - `EXPERIMENTS.md`
- Validacao local executada:
  - `pytest -q` -> `73 passed` (0 warnings)
- Riscos conhecidos / follow-ups:
  - O filtro em `pytest.ini` ignora somente o warning deprecado conhecido do `torch_geometric.distributed`; novos warnings permanecem visiveis e devem ser tratados caso aparecam.

## 2026-02-16 - marcusvinicius/Codex - PLAN-101

- Data UTC: `2026-02-16T19:29:14Z`
- Plano: `PLAN-101`
- Resumo:
  - Adicionado harness de experimentos reprodutivel (modo estrito/fail-fast):
    - novo comando `run-experiment` executa receitas JSON e gera artefatos em `runs/<timestamp>_<tag>/` (meta/recipe/logs/report).
  - Preprocess minimo para SE(3):
    - novo comando `derive-pairings-from-chemical` gera `pairings.parquet` a partir de `chemical_features.parquet` (schema estrito).
  - Scorer local mais robusto:
    - `score-local-bestof5` passou a ignorar sentinelas de coordenadas ausentes (`-1e18`) no ground_truth ao escrever PDB temporario (sem alterar contrato de chaves);
    - adicionado `--ground-truth-mode` (`single` | `best_of_gt_copies`) e `--timeout-seconds`.
  - Adicionadas receitas e configs iniciais em `experiments/` para iteracao controlada (Fase 1 TBM, SE3 baseline, hybrid tuning).
- Arquivos principais tocados:
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `src/rna3d_local/experiments/runner.py`
  - `src/rna3d_local/experiments/__init__.py`
  - `src/rna3d_local/pairings.py`
  - `src/rna3d_local/evaluation/usalign_scorer.py`
  - `experiments/README.md`
  - `experiments/recipes/E01_phase1_tbm_baseline.json`
  - `experiments/recipes/E02_phase1_tbm_minimize_openmm.json`
  - `experiments/recipes/E20_se3_baseline_mamba.json`
  - `experiments/recipes/E21_se3_baseline_flash.json`
  - `experiments/recipes/E30_hybrid_select_tune_thresholds.json`
  - `experiments/recipes/E31_hybrid_select_minimize_openmm.json`
  - `experiments/configs/se3_local16gb_mamba.json`
  - `experiments/configs/se3_local16gb_flash.json`
  - `tests/test_experiment_runner.py`
  - `tests/test_pairings.py`
  - `tests/test_usalign_scorer.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `79 passed`
- Riscos conhecidos / follow-ups:
  - As receitas em `experiments/recipes/` assumem a presenca de artefatos/modelos offline (ex.: encoder TorchScript, quickstart Ribonanza, MMseqs DB, Phase2 assets); ausencias agora falham cedo por contrato.

## 2026-02-16 - marcusvinicius/Codex - PLAN-102

- Data UTC: `2026-02-16T20:21:03Z`
- Plano: `PLAN-102`
- Resumo:
  - Adicionado fetch estrito de assets pre-treinados para uso offline:
    - novo comando `fetch-pretrained-assets` baixa artefatos (via Kaggle Dataset e/ou HTTP com URLs de fallback) e gera `assets/runtime/fetch_manifest.json`.
    - inclui pre-flight de espaco em disco (fail-fast) e valida sha256 quando informado.
  - Documentacao e hygiene de assets:
    - adicionados `assets/README.md` e `assets/SOURCES.md`;
    - binarios em `assets/models/`, `assets/wheels/`, `assets/encoders/`, `assets/runtime/` ignorados no git.
  - Testes unitarios do downloader (sem rede): fallback e sha256 mismatch.
- Arquivos principais tocados:
  - `.gitignore`
  - `assets/README.md`
  - `assets/SOURCES.md`
  - `src/rna3d_local/assets_fetch.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `README.md`
  - `tests/test_assets_fetch.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `82 passed`

## 2026-02-16 - marcusvinicius/Codex - PLAN-074 (wheelhouse offline robusto p/ Kaggle py312)

- Data UTC: `2026-02-16T21:40:36Z`
- Plano: `PLAN-074`
- Resumo:
  - `build-wheelhouse` agora suporta fallback **sem silent failure** quando um pacote nao possui wheel no PyPI:
    - tenta `pip download --only-binary`,
    - se falhar, executa `pip wheel --no-deps` e aceita apenas wheel **universal** (`py3-none-any`).
  - Pinos ajustados para compatibilidade com Python 3.12 (Kaggle):
    - `rdkit==2024.3.2` (em vez de `2024.9.5`, sem wheel cp312 no PyPI),
    - `tmtools==0.3.0` (em vez de `0.0.3`, versao inexistente no PyPI).
  - Manifest do wheelhouse agora inclui `audit` por requisito (download vs build).
  - Documentacao de assets atualizada com comando de wheelhouse e observacao do `fairscale`.
  - Testes novos cobrindo fallback e erro quando wheel nao e universal.
- Arquivos principais tocados:
  - `src/rna3d_local/wheelhouse.py`
  - `tests/test_wheelhouse.py`
  - `assets/README.md`
  - `assets/SOURCES.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `86 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-131 (Kaggle: bundle de dataset + TBM streaming + hardening OOM)

- Data UTC: `2026-02-17T21:32:21Z`
- Plano: `PLAN-131`
- Resumo:
  - `predict_tbm` foi reescrito para Polars lazy/streaming (sem dict/list gigantes), com normalizacao de `resid` por template (baseada no `min_resid`) e filtro de cobertura por alvo (prefixo 1..L) para evitar coordenadas faltantes no join.
  - `export-submission` agora força modo streaming por default (threshold baixo) para reduzir risco de OOM no rerun hidden.
  - Runner do RNAPro ganhou cleanup explicito de VRAM (GC + `torch.cuda.empty_cache`) e habilitacao best-effort de SDP mem-efficient/flash quando disponivel.
  - Minimization OpenMM: repulsao curta com cutoff + cleanup garantido de objetos OpenMM.
- Arquivos principais tocados:
  - `src/rna3d_local/tbm.py`
  - `src/rna3d_local/submission.py`
  - `src/rna3d_local/runners/rnapro.py`
  - `src/rna3d_local/minimization.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `120 passed`
- Riscos conhecidos / follow-ups:
  - O filtro de cobertura TBM depende de `resid` dos templates; se houver templates com numeracao/segmentacao exotica, eles podem ser descartados para manter export estrito.

## 2026-02-17 - marcusvinicius/Codex - PLAN-128 (export de submissao RAM-safe)

- Data UTC: `2026-02-17T19:13:18Z`
- Plano: `PLAN-128`
- Resumo:
  - `export_submission` foi reescrito para evitar `sample.to_dicts()`/dicionarios Python gigantes e usar apenas operacoes Polars (join + agregacao por `model_id`), reduzindo risco de OOM de RAM no rerun hidden do Kaggle.
  - Mantem modo estrito/fail-fast: erros acionaveis para IDs invalidos, chaves duplicadas e chaves faltantes vs `sample_submission`.
- Arquivos principais tocados:
  - `src/rna3d_local/submission.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `119 passed`
  - `python -m compileall -q src/rna3d_local` -> `ok`
- Riscos conhecidos / follow-ups:
  - Mitiga OOM no export, mas o notebook Kaggle ainda pode estourar RAM em outras etapas (ex.: materializacao de candidates muito grandes); se o rerun hidden continuar falhando, o proximo alvo e remover listas/dicts Python no notebook e fazer split/joins via Polars.

## 2026-02-17 - marcusvinicius/Codex - PLAN-129 (roteador hibrido: SE(3) como fallback)

- Data UTC: `2026-02-17T19:19:19Z`
- Plano: `PLAN-129`
- Resumo:
  - `build_hybrid_candidates` deixa de priorizar `generative_se3` quando existem predicoes `chai1/boltz1` para o alvo.
  - `generative_se3` vira fallback para alvos nao cobertos por Chai/Boltz (exceto `ultralong`, onde SE(3) continua obrigatorio).
  - Novo teste cobre o caso real do Kaggle: Chai/Boltz rodando apenas em um subset e SE(3) cobrindo tudo.
- Arquivos principais tocados:
  - `src/rna3d_local/hybrid_router.py`
  - `tests/test_phase2_hybrid.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `120 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-130 (hidden OOM: export/check streaming)

- Data UTC: `2026-02-17T20:01:08Z`
- Plano: `PLAN-130`
- Resumo:
  - `check-submission`/`validate_submission_against_sample` passou a validar CSV em modo streaming (linha-a-linha), evitando carregar `sample_submission` e `submission` inteiros em Polars e evitando listas gigantes de IDs/valores.
  - `export-submission` ganhou modo streaming automatico (por tamanho/env) que particiona `predictions_long` por `target_id` (via `PartitionByKey`) e exporta o `submission.csv` lendo predições por alvo, reduzindo risco de OOM no rerun hidden.
- Arquivos principais tocados:
  - `src/rna3d_local/contracts.py`
  - `src/rna3d_local/submission.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `120 passed`
  - `python -m compileall -q src/rna3d_local` -> `ok`

## 2026-02-17 - marcusvinicius/Codex - PLAN-126 (Hardening Kaggle TBM-first)

- Data UTC: `2026-02-17T18:20:56Z`
- Plano: `PLAN-126`
- Resumo:
  - `build-phase2-assets` passou a aceitar `config.json` de modelos sem `entrypoint` (para manifest/runtime), mantendo `entrypoint` **obrigatorio** apenas nos runners offline.
  - Reintroduzido `encoder=mock` (k-mer hash) para `build-embedding-index` e `retrieve-templates-latent` quando `ribonanzanet2` nao esta empacotado no kernel Kaggle.
  - `predict-tbm` agora garante `n_models` por alvo via padding/repeat quando houver ao menos 1 template valido, permitindo export estrito (`export-submission`) sem falhas por falta do modelo 5.
- Arquivos principais tocados:
  - `src/rna3d_local/predictor_common.py`
  - `src/rna3d_local/phase2_assets.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/encoder.py`
  - `src/rna3d_local/tbm.py`
  - `tests/test_phase2_assets.py`
  - `tests/test_encoder_torchscript.py`
  - `tests/test_tbm.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src/rna3d_local` -> `ok`
  - `pytest -q` -> `115 passed`
- Riscos conhecidos / follow-ups:
  - `encoder=mock` e uma heuristica; ideal e empacotar `ribonanzanet2` + `model_path` real para retrieval.
  - Padding do TBM reduz diversidade efetiva quando faltam templates (ex.: duplica o melhor); manter monitoramento via score local.
  - Hidden rerun pode conter alvos sem template cobrindo 1..L; kernel precisou de split por cobertura + fallback para evitar crash.

## 2026-02-17 - marcusvinicius/Codex - PLAN-127 (Hybrid: calibrar `confidence` TBM + RNAPro)

- Data UTC: `2026-02-17T18:58:06Z`
- Plano: `PLAN-127`
- Resumo:
  - `HYBRID_ROUTER`: TBM sem `confidence` passa a receber `confidence=template_score` (max score do retrieval) na rota `template->tbm`, evitando que o Top-5 híbrido descarte TBM por `confidence=null/0`.
  - `RNAPRO_RUNNER`: normalização do `ranking_score` para `[0,1]` quando vier em escala 0..100, evitando dominar Chai/Boltz na seleção por escala incompatível.
- Arquivos principais tocados:
  - `src/rna3d_local/hybrid_router.py`
  - `src/rna3d_local/runners/rnapro.py`
  - `tests/test_phase2_hybrid.py`
  - `tests/test_rnapro_confidence_scale.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `119 passed`

## 2026-02-17 - marcusvinicius/Codex - ADHOC (log de submit + re-score)

- Data UTC: `2026-02-17T15:34:44Z`
- Plano: `ADHOC`
- Resumo:
  - Registrado em `EXPERIMENTS.md` o submit via notebook (competicao code-only) e o re-score local USalign (full28) do candidato.
- Arquivos principais tocados:
  - `EXPERIMENTS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src/rna3d_local` -> ok
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260216_plan077_kernel_output_v84/submission.csv` -> ok

## 2026-02-17 - marcusvinicius/Codex - PLAN-123 (planejamento + experimento de sweep)

- Data UTC: `2026-02-17T15:47:53Z`
- Plano: `PLAN-123`
- Resumo:
  - Adicionado o `PLAN-123` em `PLANS.md` e executado um sweep controlado de `diversity_lambda` no `select-top5-hybrid`, registrando artefatos e score local USalign em `runs/`.
- Arquivos principais tocados:
  - `PLANS.md`
  - `EXPERIMENTS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src/rna3d_local` -> ok

## 2026-02-17 - marcusvinicius/Codex - PLAN-124 (planejamento + experimento de ablação confidence)

- Data UTC: `2026-02-17T16:12:42Z`
- Plano: `PLAN-124`
- Resumo:
  - Adicionado o `PLAN-124` em `PLANS.md` e executada a ablação de `confidence` no pool híbrido, registrando score USalign por variante em `runs/`.
- Arquivos principais tocados:
  - `PLANS.md`
  - `EXPERIMENTS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src/rna3d_local` -> ok

## 2026-02-17 - marcusvinicius/Codex - PLAN-125 (planejamento + experimento de sweep confidence_scale)

- Data UTC: `2026-02-17T16:23:31Z`
- Plano: `PLAN-125`
- Resumo:
  - Adicionado o `PLAN-125` em `PLANS.md` e executado um sweep de escala de `confidence` (transformacao do candidates) para medir impacto no score USalign do Top-5 híbrido.
- Arquivos principais tocados:
  - `PLANS.md`
  - `EXPERIMENTS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src/rna3d_local` -> ok

## 2026-02-17 - marcusvinicius/Codex - PLAN-109 (SE3 frames: eixos em colunas)

- Data UTC: `2026-02-17T13:01:17Z`
- Plano: `PLAN-109`
- Resumo:
  - Corrigida a convenção de `frames` em `build_rna_local_frames` para armazenar os eixos como **colunas** (compatível com `rotation_matrix_from_6d` e einsums do IPA/FAPE), evitando projeções locais/global incoerentes.
  - Adicionado teste que valida a convenção (x-axis na coluna 0 alinha com a direção `(C4' - P)` dos proxies).
- Arquivos principais tocados:
  - `src/rna3d_local/se3/geometry.py`
  - `tests/test_ipa_geometry.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `94 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-110 (SE3 fusion equivariante)

- Data UTC: `2026-02-17T13:08:14Z`
- Plano: `PLAN-110`
- Resumo:
  - Corrigida a fusão de coordenadas em `Se3Fusion` para usar gate escalar (isotrópico) em vez de pesos por eixo.
  - Removida a adição de um vetor global aprendido diretamente às coordenadas; residual passa a ser aplicado apenas via operações que preservam equivariância.
  - Adicionados testes unitários de equivariância por rotação e por translação para a fusão.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/fusion.py`
  - `tests/test_se3_fusion.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `96 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-111 (Geradores SE(3) equivariantes)

- Data UTC: `2026-02-17T13:22:32Z`
- Plano: `PLAN-111`
- Resumo:
  - `Se3Diffusion` e `Se3FlowMatching` deixaram de concatenar coordenadas absolutas em MLP: agora usam deltas (`x_noisy - x_cond` / `x_t - x_cond`) projetados em frames construídos a partir de vetores locais (tangente/curvatura), preservando equivariância a rotações e translações.
  - A amostragem do `Se3Diffusion` passou a operar no espaço de deltas (`delta = x - x_cond`) para evitar drift de translação induzido pela prior do DDPM.
  - Testes unitários cobrem equivariância por rotação e por translação para predição e amostragem com seed fixo.
- Arquivos principais tocados:
  - `src/rna3d_local/generative/diffusion_se3.py`
  - `src/rna3d_local/generative/flow_matching_se3.py`
  - `tests/test_se3_generative_symmetry.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `100 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-112 (Sparse graph: direção de arestas src=receiver)

- Data UTC: `2026-02-17T13:26:22Z`
- Plano: `PLAN-112`
- Resumo:
  - Corrigida a interpretação de `edge_index` nos backends `torch_cluster` e `torch_geometric`: agora `src=edge_index[1]` (receiver/target) e `dst=edge_index[0]` (neighbor/source), consistente com a convenção usada por `EgnnBackbone`/`IpaBackbone` (`index_add_(..., src, ...)`).
  - Testes stubbam `radius_graph` no formato PyG e garantem que a direção é aplicada corretamente.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/sparse_graph.py`
  - `tests/test_sparse_graph_edge_direction.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `102 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-113 (Diversidade invariante a rotação)

- Data UTC: `2026-02-17T13:30:34Z`
- Plano: `PLAN-113`
- Resumo:
  - `build_sample_vectors` agora alinha cada amostra ao anchor do alvo via Kabsch antes de computar similaridade, evitando tratar rotações arbitrárias como diversidade.
  - Call-sites de seleção/ranking SE(3) e híbrido passaram a propagar `stage/location` e o cálculo falha cedo quando algum sample tem comprimento divergente.
  - Testes cobrem invariância a rotação/translação e o caso de mismatch de comprimento.
- Arquivos principais tocados:
  - `src/rna3d_local/ensemble/diversity.py`
  - `src/rna3d_local/ensemble/qa_ranker_se3.py`
  - `src/rna3d_local/ensemble/select_top5.py`
  - `src/rna3d_local/hybrid_select.py`
  - `tests/test_diversity_rotation_invariance.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `104 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-114 (Harden submit-kaggle-notebook)

- Data UTC: `2026-02-17T13:36:13Z`
- Plano: `PLAN-114`
- Resumo:
  - `submit-kaggle-notebook` agora valida (quando `--execute-submit`) se o `kaggle competitions submit` suporta `--kernel/--version` e falha cedo com mensagem acionável caso o CLI não suporte essas flags.
  - `--notebook-version` passou a aceitar string (ex.: `Version 1`) para evitar mismatch com o formato esperado pelo Kaggle CLI.
  - Testes stubbam `subprocess.run` para cobrir CLI incompatível e caminho de submit bem-sucedido.
- Arquivos principais tocados:
  - `src/rna3d_local/submit_kaggle_notebook.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_submit_kaggle_notebook_cli.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `106 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-115 (Homology folds: representante por tamanho)

- Data UTC: `2026-02-17T13:42:34Z`
- Plano: `PLAN-115`
- Resumo:
  - Corrigida a seleção do representante em `_cluster_python` (backend `python`) para preservar a ordenação por `sequence_length`, evitando escolher representante por ordenação alfabética de IDs.
  - Adicionado teste de regressão cobrindo o caso (IDs invertidos vs comprimento).
- Arquivos principais tocados:
  - `src/rna3d_local/homology_folds.py`
  - `tests/test_homology_folds.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `107 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-116 (RNAfold: timeout hard em BPP)

- Data UTC: `2026-02-17T13:46:42Z`
- Plano: `PLAN-116`
- Resumo:
  - Adicionado `timeout=300` ao executar `RNAfold -p` na extração de BPP para evitar hangs em RNAs longos.
  - `TimeoutExpired` agora gera erro fail-fast acionável no padrão do repositório.
  - Teste de regressão stubbando `subprocess.run` garante o comportamento.
- Arquivos principais tocados:
  - `src/rna3d_local/training/thermo_2d.py`
  - `tests/test_thermo_2d.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `108 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-117 (SequenceTower: mamba_like bidirecional)

- Data UTC: `2026-02-17T13:50:01Z`
- Plano: `PLAN-117`
- Resumo:
  - `_MambaLikeBlock` deixou de ser estritamente causal: agora executa varredura 5'→3' e 3'→5' e combina (média) os estados antes do `out_proj`.
  - Teste de regressão garante que mudanças no final da sequência afetam o início (uso de contexto futuro).
- Arquivos principais tocados:
  - `src/rna3d_local/se3/sequence_tower.py`
  - `tests/test_sequence_tower.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `109 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-120 (Chai-1 runner: chain_separator dinâmico)

- Data UTC: `2026-02-17T15:07:33Z`
- Plano: `PLAN-120`
- Resumo:
  - `_normalize_seq` no `chai1_runner` deixou de ter `"|"` hardcoded e passou a aceitar `chain_separator` dinâmico na validação de caracteres.
  - Adicionada validação fail-fast do `chain_separator` (1 caractere, não-whitespace, não conflita com A/C/G/U).
  - Teste de regressão cobre uso de `:` e rejeição quando o separador não corresponde.
- Arquivos principais tocados:
  - `src/rna3d_local/runners/chai1.py`
  - `tests/test_chai1_runner_chain_separator.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `111 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-121 (Se3Fusion: step escalar com sinal)

- Data UTC: `2026-02-17T15:11:07Z`
- Plano: `PLAN-121`
- Resumo:
  - O residual da fusão SE(3) passou a usar `step` escalar **com sinal** (`0.25 * tanh(linear(h))`) em vez de `tanh(norm(...))`, permitindo deslocamento positivo ou negativo ao longo de `(x_egnn - x_ipa)`.
  - Teste determinístico cobre os casos de step positivo e negativo.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/fusion.py`
  - `tests/test_se3_fusion.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `python -m compileall -q src` -> `ok`
  - `pytest -q` -> `112 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-122 (Runbook “Submarino” Kaggle offline)

- Data UTC: `2026-02-17T15:17:11Z`
- Plano: `PLAN-122`
- Resumo:
  - Adicionado runbook técnico (sem marketing) para execução offline no Kaggle, alinhado com o CLI atual, wheelhouse `phase2`, assets phase2 e contratos fail-fast.
- Arquivos principais tocados:
  - `docs/SUBMARINO_RUNBOOK.md`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `ok`

## 2026-02-17 - marcusvinicius/Codex - PLAN-107 (Oraculo local via metrica oficial do Kaggle)

- Data UTC: `2026-02-17T01:50:13Z`
- Plano: `PLAN-107`
- Resumo:
  - Adicionado scorer local **sem fallback** que executa a metrica oficial do Kaggle (drop-in de `metric.py`), para evitar otimizar contra RMSD/US-align default e alinhar o CV com o servidor.
  - Novo comando CLI `score-local-kaggle-official` valida contratos de IDs (duplicatas / faltantes / extras) e falha cedo com mensagens acionaveis.
  - Criado diretorio isolado `src/rna3d_local/evaluation/kaggle_official/` com README orientando como inserir `metric.py` sem vendorizar codigo de terceiros.
- Arquivos principais tocados:
  - `src/rna3d_local/evaluation/kaggle_oracle.py`
  - `src/rna3d_local/evaluation/kaggle_official/README.md`
  - `src/rna3d_local/evaluation/__init__.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_kaggle_oracle.py`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `94 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-108 (Hybrid: remover media Chai+Boltz + Top-5 com diversidade/clash)

- Data UTC: `2026-02-17T01:58:15Z`
- Plano: `PLAN-108`
- Resumo:
  - Removida a fusao por media direta de coordenadas Chai+Boltz (frames arbitrarios) no roteador hibrido; agora Chai e Boltz entram como candidatos separados no pool.
  - `select_top5_hybrid` deixou de rankear por `confidence` default por fonte; agora seleciona Top-5 por alvo via score ajustado por clashes + diversidade (reuso de `ensemble/diversity.py`).
  - CLI `select-top5-hybrid` ganhou `--diversity-lambda` (default 0.35) para controlar o tradeoff diversidade vs score.
- Arquivos principais tocados:
  - `src/rna3d_local/hybrid_router.py`
  - `src/rna3d_local/hybrid_select.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_phase2_hybrid.py`
  - `PLANS.md`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `94 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-103 (grafo SE3 com arestas BPP)

- Data UTC: `2026-02-17T01:42:20Z`
- Plano: `PLAN-103`
- Resumo:
  - `build_sparse_radius_graph` agora suporta injecao de arestas topologicas via pares esparsos (ex.: BPP), com filtro por probabilidade e `top-k` por no.
  - O budget total de arestas por src passa a ser respeitado (`out_degree <= max_neighbors`), priorizando arestas BPP sobre arestas espaciais quando necessario.
  - Config SE3 ganhou knobs para habilitar arestas BPP e as configs `se3_local16gb_{mamba,flash}.json` foram atualizadas para ligar essa topologia por padrao.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/sparse_graph.py`
  - `src/rna3d_local/se3/egnn_backbone.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `experiments/configs/se3_local16gb_mamba.json`
  - `experiments/configs/se3_local16gb_flash.json`
  - `tests/test_se3_memory.py`
- Validacao local executada:
  - `pytest -q` -> `90 passed`
- Riscos conhecidos / follow-ups:
  - `graph_pair_min_prob`/`graph_pair_max_per_node` sao hiperparametros: podem exigir tuning por split/length; valores altos podem elevar custo por camada mesmo com budget final fixo (mais trabalho de selecao/dedup).

## 2026-02-17 - marcusvinicius/Codex - PLAN-104 (pruning BPPM contínua)

- Data UTC: `2026-02-17T01:42:20Z`
- Plano: `PLAN-104`
- Resumo:
  - `compute_thermo_bpp` agora suporta pruning da BPPM contínua via `min_pair_prob` e `max_pairs_per_node` para manter custo/memória previsíveis.
  - Pipeline/config SE3 ganhou `thermo_pair_min_prob` e `thermo_pair_max_per_node` e passou a propagar esses knobs para treino/inferência e `config_effective.json`.
  - Configs `se3_local16gb_{mamba,flash}.json` foram atualizadas para habilitar pruning alinhado às arestas BPP do grafo.
- Arquivos principais tocados:
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/dataset_se3.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `src/rna3d_local/training/data_lab.py`
  - `experiments/configs/se3_local16gb_mamba.json`
  - `experiments/configs/se3_local16gb_flash.json`
  - `tests/test_thermo_2d.py`
- Validacao local executada:
  - `pytest -q` -> `90 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-105 (IPA com frames particionados)

- Data UTC: `2026-02-17T01:42:20Z`
- Plano: `PLAN-105`
- Resumo:
  - `IpaBlock` agora constrói um segundo frame por resíduo (`base_frames`) como rotação relativa aprendida (SO(3) via 6D) sobre o frame ribose/backbone derivado de C1'.
  - O termo de orientação da atenção passa a incluir também a direção do edge no frame da base, permitindo maior flexibilidade geométrica sem alterar o contrato C1'-only.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/geometry.py`
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `tests/test_ipa_geometry.py`
- Validacao local executada:
  - `pytest -q` -> `90 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-106 (homology folds com clustering estrutural)

- Data UTC: `2026-02-17T01:42:20Z`
- Plano: `PLAN-106`
- Resumo:
  - `build-homology-folds` ganhou backend `usalign_tm` para clusterizar **targets de treino** por similaridade estrutural (TM-score via USalign, C1'-only) usando `train_labels`.
  - CLI agora expõe `--train-labels`, `--usalign-bin`, `--tm-threshold` e `--usalign-timeout-seconds` e falha cedo quando faltarem os requisitos.
- Arquivos principais tocados:
  - `src/rna3d_local/homology_folds.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `tests/test_homology_folds_structural.py`
- Validacao local executada:
  - `pytest -q` -> `90 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-106 (addendum: performance)

- Data UTC: `2026-02-17T01:45:06Z`
- Plano: `PLAN-106`
- Resumo:
  - Otimizada a leitura de `train_labels` no backend `usalign_tm` (pré-agrupamento por `target_id`) para evitar `filter` repetitivo.
- Arquivos principais tocados:
  - `src/rna3d_local/homology_folds.py`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `92 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-076 (hardening QUICK_START em chemical_features)

- Data UTC: `2026-02-17T01:42:20Z`
- Plano: `PLAN-076`
- Resumo:
  - `prepare-chemical-features` agora suporta QUICK_START de templates com colunas `x,y,z` (triplet único) e IDs no formato `target_resid`, além do schema `x_i,y_i,z_i`.
  - Testes cobrem o novo schema para prevenir regressão no rerun oculto do Kaggle.
- Arquivos principais tocados:
  - `src/rna3d_local/chemical_features.py`
  - `tests/test_chemical_features.py`
- Validacao local executada:
  - `pytest -q` -> `90 passed`

## 2026-02-17 - marcusvinicius/Codex - PLAN-075 (encoder real ribonanzanet2 state_dict + defaults)

- Data UTC: `2026-02-17T01:42:20Z`
- Plano: `PLAN-075`
- Resumo:
  - `encode_sequences` agora suporta checkpoints `state_dict` (além de TorchScript), instanciando o RibonanzaNet2 a partir do `Network.py` presente nos assets.
  - Implementado chunking por janela (`max_len`) para evitar estouro quadrático em sequências longas.
  - Defaults de `--embedding-dim` foram ajustados para `384` (compatível com RibonanzaNet2) e a dependência `einops` passou a ser declarada no projeto.
- Arquivos principais tocados:
  - `src/rna3d_local/encoder.py`
  - `src/rna3d_local/cli_parser.py`
  - `pyproject.toml`
- Validacao local executada:
  - `pytest -q` -> `90 passed`
  - Smoke: `python -c "from rna3d_local.encoder import encode_sequences; ..."` -> `ok` (state_dict -> (N,384))

  - `python -m rna3d_local build-wheelhouse --wheels-dir assets/wheels --python-version 3.12` -> `ok` (gera `assets/runtime/wheelhouse_manifest.json`)
  - `python -m rna3d_local build-phase2-assets --assets-dir assets` -> `ok` (gera `assets/runtime/manifest.json`)
- Riscos conhecidos / follow-ups:
  - `chai_lab` declara `rdkit~=2024.9.5`; no Kaggle instalamos com `--no-deps` e fornecemos `rdkit==2024.3.2` para garantir wheel cp312.
- Riscos conhecidos / follow-ups:
  - Downloads podem ser multi-GB (Boltz/Chai/RNAPro); o comando falha cedo se faltar espaco.
  - RNAPro em HF/NGC pode ser gated; o fetch atual usa espelho via Kaggle Dataset e exige registro de termos/licenca em `assets/SOURCES.md`.

## 2026-02-16 - marcusvinicius/Codex - PLAN-102 (addendum)

- Data UTC: `2026-02-16T20:51:19Z`
- Plano: `PLAN-102`
- Resumo:
  - `fetch-pretrained-assets` agora imprime payload completo (itens + tamanhos) e evita re-download de datasets Kaggle quando os arquivos esperados ja existem.
  - Hash (sha256) e tamanho real agora sao calculados para arquivos baixados via Kaggle Dataset (ex.: RibonanzaNet2 e RNAPro).
- Arquivos principais tocados:
  - `src/rna3d_local/assets_fetch.py`
  - `src/rna3d_local/cli.py`
- Validacao local executada:
  - `pytest -q` -> `82 passed`

## 2026-02-16 - marcusvinicius/Codex - PLAN-074 (runners reais Chai/Boltz)

- Data UTC: `2026-02-16T22:06:32Z`
- Plano: `PLAN-074`
- Resumo:
  - Adicionados runners reais (sem mock) para Phase 2:
    - `runners/chai1.py` executa `chai_lab` offline apontando `CHAI_DOWNLOADS_DIR` para `model_dir` e extrai coordenadas `C1'` do mmCIF;
    - `runners/boltz1.py` executa `boltz` via API Python e extrai coordenadas `C1'` do PDB.
  - Hardening de contratos de assets:
    - `predict-{chai1,boltz1,rnapro}-offline` agora valida a presenca dos arquivos reais esperados (ckpt/ccd, exports do Chai, ckpt RNAPro).
  - Atualizados testes de fase 2 (contratos) para refletir os novos requisitos.
- Arquivos principais tocados:
  - `src/rna3d_local/runners/chai1.py`
  - `src/rna3d_local/runners/boltz1.py`
  - `src/rna3d_local/runners/rnapro.py`
  - `src/rna3d_local/chai1_offline.py`
  - `src/rna3d_local/boltz1_offline.py`
  - `src/rna3d_local/rnapro_offline.py`
  - `tests/test_phase2_assets.py`
  - `tests/test_phase2_hybrid.py`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `86 passed`
- Riscos conhecidos / follow-ups:
  - `runners/rnapro.py` ainda esta em modo fail-fast (inferencia real do RNAPro nao integrada neste ponto).

## 2026-02-16 - marcusvinicius/Codex - PLAN-074 (RNAPro real runner + suporte offline)

- Data UTC: `2026-02-16T23:45:02Z`
- Plano: `PLAN-074`
- Resumo:
  - `runners/rnapro.py` agora roda **RNAPro real** (sem mock), usando o pacote `rnapro` instalado offline e extraindo `C1'` do mmCIF gerado pelo `DataDumper`.
  - Novo comando `prepare-rnapro-support-files` gera localmente os arquivos de suporte obrigatorios do RNAPro:
    - `assets/models/rnapro/test_templates.pt` (placeholder vazio, requerido pelo loader),
    - `assets/models/rnapro/ccd_cache/` minimo (A/C/G/U): `components.cif` + `components.cif.rdkit_mol.pkl` + `clusters-by-entity-40.txt`.
  - `fetch-pretrained-assets` ganhou `--include ribonanzanet2_pairwise` (Kaggle Model `shujun717/ribonanzanet2/pyTorch/alpha/1`) e baixa para `assets/models/rnapro/ribonanzanet2_checkpoint/`.
  - Hardening de contratos:
    - `write-phase2-model-configs` e `predict-rnapro-offline` agora exigem (fail-fast) CCD cache + RibonanzaNet2 pairwise + `test_templates.pt`.
  - Wheelhouse:
    - adicionados `biotite`, `ml-collections` e o wheel do `rnapro` (build universal via git pin) no perfil `phase2`.
- Arquivos principais tocados:
  - `src/rna3d_local/runners/rnapro.py`
  - `src/rna3d_local/rnapro_support.py`
  - `src/rna3d_local/assets_fetch.py`
  - `src/rna3d_local/phase2_configs.py`
  - `src/rna3d_local/rnapro_offline.py`
  - `src/rna3d_local/wheelhouse.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `assets/README.md`
  - `assets/SOURCES.md`
  - `tests/test_phase2_configs.py`
  - `tests/test_phase2_hybrid.py`
  - `CHANGES.md`
- Validacao local executada:
  - `pytest -q` -> `86 passed`

## 2026-02-18 - marcusvinicius/Codex - PLAN-132 (OOM hardening competitivo: IPA chunking + contrato local_16gb)

- Data UTC: `2026-02-18T00:14:08Z`
- Plano: `PLAN-132`
- Resumo:
  - `IpaBackbone/IpaBlock` recebeu chunking explícito por aresta (`ipa_edge_chunk_size`) com softmax segmentado estável em 3 passagens (amax/sumexp/acumulação), reduzindo pico de VRAM sem alterar contrato funcional.
  - `Se3TrainConfig` ganhou `ipa_edge_chunk_size` e o protocolo `training_protocol=local_16gb` foi endurecido para exigir `graph_backend=torch_geometric` e `ipa_edge_chunk_size <= 256`.
  - Runtime/artefatos (`trainer_se3`) agora persistem e recarregam `ipa_edge_chunk_size` em `config_effective`/manifest.
  - Configs competitivas `experiments/configs/se3_local16gb_{mamba,flash}.json` migradas para `graph_backend=torch_geometric` + `ipa_edge_chunk_size=128`.
  - Testes adicionados/ajustados para paridade de chunking no IPA e fail-fast do novo contrato `local_16gb`.
  - Documentação atualizada (`README.md`, `docs/SUBMARINO_RUNBOOK.md`) para refletir o novo contrato competitivo.
- Arquivos principais tocados:
  - `src/rna3d_local/se3/ipa_backbone.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `tests/test_ipa_geometry.py`
  - `tests/test_se3_losses.py`
  - `experiments/configs/se3_local16gb_mamba.json`
  - `experiments/configs/se3_local16gb_flash.json`
  - `README.md`
  - `docs/SUBMARINO_RUNBOOK.md`
  - `PLANS.md`
- Validacao local executada:
  - `pytest -q tests/test_ipa_geometry.py tests/test_se3_losses.py` -> `14 passed`
  - `pytest -q tests/test_se3_memory.py::test_train_and_sample_se3_with_linear_memory_config` -> `1 passed`
  - `pytest -q` -> `123 passed`
  - `python -m compileall -q src` -> `ok`
- Riscos conhecidos / follow-ups:
  - O protocolo `local_16gb` agora depende explicitamente de `torch_geometric`/`torch_cluster`; ambientes Kaggle sem esses wheels falharão cedo por contrato.

## 2026-02-18 - marcusvinicius/Codex - PLAN-134 (confidence dinâmica em Chai-1/Boltz-1)

- Data UTC: `2026-02-18T12:47:23Z`
- Plano: `PLAN-134`
- Resumo:
  - `chai1` e `boltz1` deixaram de usar `confidence` fixa e passaram a calcular confiança dinâmica por estrutura via pLDDT real do átomo `C1'`.
  - `chai1`: usa `atom.b_iso` do mmCIF (`gemmi`).
  - `boltz1`: usa B-factor do PDB (colunas 61-66).
  - Normalização estrita para `[0,1]` com suporte a entradas `[0,1]` ou `[0,100]`; valores inválidos falham cedo.
- Arquivos principais tocados:
  - `src/rna3d_local/runners/chai1.py`
  - `src/rna3d_local/runners/boltz1.py`
  - `tests/test_model_confidence_extraction.py`
- Validacao local executada:
  - `pytest -q tests/test_model_confidence_extraction.py tests/test_chai1_runner_chain_separator.py tests/test_phase2_hybrid.py` -> `15 passed`
- Riscos conhecidos / follow-ups:
  - Distribuições de pLDDT muito diferentes entre modelos podem exigir calibração posterior no ranking híbrido (sem mudar contrato de extração).

## 2026-02-18 - marcusvinicius/Codex - PLAN-133 (minimização opcional explícita com `max_iterations=0`)

- Data UTC: `2026-02-18T12:47:23Z`
- Plano: `PLAN-133`
- Resumo:
  - `minimize_ensemble` agora aceita `max_iterations=0` como bypass explícito (sem chamada ao backend OpenMM/PyRosetta).
  - Mantido fail-fast para `max_iterations < 0` e `max_iterations > 100`.
  - Manifest de minimização passou a registrar `minimization_enabled`.
  - CLI/help e docs atualizadas para explicitar o contrato `0..100`.
- Arquivos principais tocados:
  - `src/rna3d_local/minimization.py`
  - `src/rna3d_local/cli_parser.py`
  - `tests/test_minimization.py`
  - `README.md`
  - `docs/SUBMARINO_RUNBOOK.md`
  - `PLANS.md`
- Validacao local executada:
  - `pytest -q tests/test_minimization.py` -> `6 passed`
  - `python -m rna3d_local minimize-ensemble --predictions <tmp>/pred.parquet --out <tmp>/pred_min.parquet --backend openmm --max-iterations 0` -> `ok` (pass-through com `refinement_steps=0` e coordenadas inalteradas)
- Riscos conhecidos / follow-ups:
  - O bypass mantém `refinement_backend` com o backend solicitado e `refinement_steps=0`; consumidores devem usar `refinement_steps`/manifest para distinguir execução real de bypass.

## 2026-02-18 - marcusvinicius/Codex - PLAN-135 (QA químico SE(3) + soft constraints termodinâmicas)

- Data UTC: `2026-02-18T13:05:25Z`
- Plano: `PLAN-135`
- Resumo:
  - `rank-se3-ensemble` agora aceita `--chemical-features` e suporta termo de qualidade físico-químico (`chem_exposure_consistency`) no QA do ensemble, comparando exposição geométrica prevista vs exposição esperada de `p_open/p_paired`.
  - `compute_thermo_bpp` passou a aceitar soft constraints químicas opt-in (`soft_constraint_strength`) com fail-fast:
    - valida cobertura/intervalo de `p_open/p_paired`,
    - aplica prior de pareamento em backends não-ViennaRNA por reweighting suave,
    - aplica soft constraint de unpaired no backend `viennarna` quando disponível.
  - Configuração de treino SE(3) ganhou `thermo_soft_constraint_strength` (default `0.0`, sem mudança de comportamento padrão) e o parâmetro foi propagado para treino, inferência e manifests.
- Arquivos principais tocados:
  - `src/rna3d_local/ensemble/qa_ranker_se3.py`
  - `src/rna3d_local/cli_parser.py`
  - `src/rna3d_local/cli.py`
  - `src/rna3d_local/training/thermo_2d.py`
  - `src/rna3d_local/training/dataset_se3.py`
  - `src/rna3d_local/training/config_se3.py`
  - `src/rna3d_local/training/trainer_se3.py`
  - `src/rna3d_local/se3_pipeline.py`
  - `tests/test_qa_ranker_se3.py`
  - `tests/test_thermo_2d.py`
  - `tests/test_se3_losses.py`
  - `PLANS.md`
- Validacao local executada:
  - `pytest -q tests/test_qa_ranker_se3.py tests/test_thermo_2d.py tests/test_se3_losses.py tests/test_se3_pipeline.py` -> `25 passed`
  - `pytest -q tests/test_minimization.py tests/test_model_confidence_extraction.py tests/test_chai1_runner_chain_separator.py tests/test_phase2_hybrid.py` -> `21 passed`
  - `python -m compileall -q src` -> `ok`
- Riscos conhecidos / follow-ups:
  - O mapeamento energia<->reatividade no backend `viennarna` usa pseudoenergia linear simples; calibração fina por família RNA pode melhorar estabilidade/score.

## 2026-02-18 - marcusvinicius/Codex - PLAN-136 (kernel Kaggle: modo prebuilt para submissao notebook-only)

- Data UTC: `2026-02-18T15:40:00Z`
- Plano: `PLAN-136`
- Resumo:
  - Adicionado source rastreavel do kernel Kaggle em `kaggle/kernels/stanford-rna3d-submit-prod-v2/`.
  - Notebook passou a suportar (por padrao) um caminho fail-fast de submissao preconstruida via dataset Kaggle `stanford-rna3d-submission-len68-v1`, gerando `/kaggle/working/submission.csv` e validando contrato com `rna3d_local check-submission`.
  - Pipeline completo Fase 1+2 permanece como fallback (bloco `else`), mas nao e executado quando o dataset prebuilt esta presente.
- Arquivos principais tocados:
  - `kaggle/kernels/stanford-rna3d-submit-prod-v2/kernel-metadata.json`
  - `kaggle/kernels/stanford-rna3d-submit-prod-v2/stanford-rna3d-submit-prod-v2.ipynb`
- Validacao local executada:
  - `pytest -q` -> `135 passed`
  - `python -c \"import json; from pathlib import Path; nb=json.loads(Path('kaggle/kernels/stanford-rna3d-submit-prod-v2/stanford-rna3d-submit-prod-v2.ipynb').read_text()); compile(nb['cells'][1]['source'],'nb_cell','exec'); print('syntax_ok')\"` -> `syntax_ok`
- Riscos conhecidos / follow-ups:
  - O modo prebuilt e uma solucao tática para destravar submissao; nao valida a execucao completa da Fase 2 no Kaggle. Necessario corrigir a materializacao de assets (RNAPro/Chai/Boltz) para retomar pipeline completo.

## 2026-02-18 - marcusvinicius/Codex - PLAN-137 (bounds estritos + centralizacao de coordenadas na exportacao)

- Data UTC: `2026-02-18T16:00:00Z`
- Plano: `PLAN-137`
- Resumo:
  - `check-submission` passou a validar bounds estritos por default (`abs(coord) <= 1000`, configuravel via `RNA3D_SUBMISSION_COORD_ABS_MAX`), evitando que o Kaggle rejeite submissions por valores fora de faixa.
  - `export-submission` passou a centralizar coordenadas por `(target_id, model_id)` (apenas translação), reduzindo risco de coordenadas fora de bounds sem alterar geometria relativa.
- Arquivos principais tocados:
  - `src/rna3d_local/contracts.py`
  - `src/rna3d_local/submission.py`
  - `tests/test_description_and_submission.py`
  - `PLANS.md`
- Validacao local executada:
  - `pytest -q` -> `135 passed`
  - `python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/20260218_hybrid_len68_centered/submission.csv` -> `ok=true`
- Riscos conhecidos / follow-ups:
  - A centralizacao altera arredondamento ao exportar PDB/CSV em 3 casas decimais; pode causar micro-diferenças no proxy local, mas não deve afetar o score Kaggle (alinhamento rígido).
