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
