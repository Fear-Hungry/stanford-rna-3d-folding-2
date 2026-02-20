# stanford-rna-3d-folding-2 (phase 1+2 reboot)

Repositorio reiniciado para implementar FASE 1 (Template Oracle) e FASE 2 (Arsenal Hibrido 3D) com modo estrito e fail-fast.

## Comandos disponiveis

```bash
python -m rna3d_local build-template-db --external-templates external_templates.csv --out-dir data/derived/template_db
python -m rna3d_local build-embedding-index --template-index data/derived/template_db/template_index.parquet --out-dir data/derived/template_db/emb --encoder ribonanzanet2 --model-path <local_model_file>
python -m rna3d_local infer-description-family --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out-dir runs/desc --backend llama_cpp --llm-model-path <model.gguf>
python -m rna3d_local prepare-chemical-features --quickstart <ribonanza_quickstart.csv> --out runs/chem/chemical_features.parquet
python -m rna3d_local build-homology-folds --train-targets input/stanford-rna-3d-folding-2/train_sequences.csv --pdb-sequences data/derived/template_db/template_index.parquet --out-dir runs/folds/homology --backend mmseqs2 --identity-threshold 0.40 --coverage-threshold 0.80 --n-folds 5 --description-column description
python -m rna3d_local evaluate-homology-folds --train-folds runs/folds/homology/train_folds.parquet --target-metrics runs/cv/per_target_scores.parquet --retrieval runs/retrieval/retrieval_candidates.parquet --report runs/folds/homology/eval_orphan_priority.json --orphan-score-threshold 0.65 --orphan-weight 0.70
python -m rna3d_local retrieve-templates-latent --template-index data/derived/template_db/template_index.parquet --template-embeddings data/derived/template_db/emb/template_embeddings.parquet --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/retrieval/retrieval_candidates.parquet --faiss-index data/derived/template_db/emb/faiss_ivfpq.index
python -m rna3d_local train-template-reranker --candidates runs/retrieval/retrieval_candidates.parquet --chemical-features runs/chem/chemical_features.parquet --templates data/derived/template_db/templates.parquet --out-dir runs/reranker
python -m rna3d_local score-template-reranker --candidates runs/retrieval/retrieval_candidates.parquet --chemical-features runs/chem/chemical_features.parquet --templates data/derived/template_db/templates.parquet --model runs/reranker/model.pt --config runs/reranker/config.json --out runs/reranker/reranked.parquet
python -m rna3d_local predict-tbm --retrieval runs/reranker/reranked.parquet --templates data/derived/template_db/templates.parquet --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/tbm/predictions.parquet --n-models 5 --allow-missing-targets --min-template-coverage 0.60
python -m rna3d_local minimize-ensemble --predictions runs/tbm/predictions.parquet --out runs/tbm/predictions_minimized.parquet --backend openmm --bond-length-angstrom 5.9 --vdw-min-distance-angstrom 2.1 --position-restraint-k 800.0
python -m rna3d_local export-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --predictions runs/tbm/predictions_minimized.parquet --out runs/tbm/submission.csv
python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/tbm/submission.csv
python -m rna3d_local score-local-bestof5 --ground-truth input/stanford-rna-3d-folding-2/validation_labels.csv --submission runs/tbm/submission.csv --usalign-bin src/rna3d_local/evaluation/USalign --score-json runs/tbm/score.json --report runs/tbm/score_report.json

# Fase 2 (assets + modelos offline + roteamento hibrido)
python -m rna3d_local fetch-pretrained-assets --assets-dir assets --include ribonanzanet2 --dry-run
python -m rna3d_local fetch-pretrained-assets --assets-dir assets --include ribonanzanet2 --include boltz1
python -m rna3d_local build-phase2-assets --assets-dir assets
python -m rna3d_local predict-rnapro-offline --model-dir assets/models/rnapro --targets runs/phase2/targets_short.csv --out runs/phase2/rnapro.parquet --n-models 5
python -m rna3d_local predict-chai1-offline --model-dir assets/models/chai1 --targets runs/phase2/targets_short.csv --out runs/phase2/chai1.parquet --n-models 5
python -m rna3d_local predict-boltz1-offline --model-dir assets/models/boltz1 --targets runs/phase2/targets_short.csv --out runs/phase2/boltz1.parquet --n-models 5
python -m rna3d_local build-hybrid-candidates --targets input/stanford-rna-3d-folding-2/test_sequences.csv --retrieval runs/retrieval/retrieval_candidates.parquet --tbm runs/tbm/predictions.parquet --rnapro runs/phase2/rnapro.parquet --chai1 runs/phase2/chai1.parquet --boltz1 runs/phase2/boltz1.parquet --se3-flash runs/se3/top5_flash.parquet --se3-mamba runs/se3/top5_mamba.parquet --out runs/phase2/hybrid_candidates.parquet --routing-out runs/phase2/routing.parquet --template-score-threshold 0.65 --short-max-len 350 --medium-max-len 600
python -m rna3d_local select-top5-hybrid --candidates runs/phase2/hybrid_candidates.parquet --out runs/phase2/hybrid_top5.parquet --n-models 5
python -m rna3d_local minimize-ensemble --predictions runs/phase2/hybrid_top5.parquet --out runs/phase2/hybrid_top5_minimized.parquet --backend openmm --bond-length-angstrom 5.9 --vdw-min-distance-angstrom 2.1 --position-restraint-k 800.0
python -m rna3d_local export-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --predictions runs/phase2/hybrid_top5_minimized.parquet --out runs/phase2/submission.csv
python -m rna3d_local evaluate-submit-readiness --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/phase2/submission.csv --score-json runs/phase2/score.json --baseline-score 0.0 --report runs/phase2/readiness.json

# Branch experimental SE(3) + gerativo (Best-of-5)
python -m rna3d_local prepare-phase1-data-lab --targets input/stanford-rna-3d-folding-2/train_sequences.csv --pairings runs/se3/pairings.parquet --chemical-features runs/chem/chemical_features.parquet --labels input/stanford-rna-3d-folding-2/train_labels.csv --out-dir runs/se3_phase1 --thermo-backend rnafold --msa-backend mmseqs2 --mmseqs-db /data/mmseqs/rna_db --workers 16
python -m rna3d_local train-se3-generator --targets input/stanford-rna-3d-folding-2/train_sequences.csv --pairings runs/se3/pairings.parquet --chemical-features runs/chem/chemical_features.parquet --labels input/stanford-rna-3d-folding-2/train_labels.csv --config runs/se3/config_train.json --out-dir runs/se3/model --seed 123 --training-store runs/se3_phase1/training_store.zarr
python -m rna3d_local sample-se3-ensemble --model-dir runs/se3/model --targets input/stanford-rna-3d-folding-2/test_sequences.csv --pairings runs/se3/pairings_test.parquet --chemical-features runs/chem/chemical_features.parquet --out runs/se3/candidates.parquet --method both --n-samples 24 --seed 123
python -m rna3d_local rank-se3-ensemble --candidates runs/se3/candidates.parquet --out runs/se3/ranked.parquet --diversity-lambda 0.35
python -m rna3d_local select-top5-se3 --ranked runs/se3/ranked.parquet --out runs/se3/top5.parquet --n-models 5 --diversity-lambda 0.35
```

## Configuracao de escala SE(3) para L longo

- `sequence_tower`:
  - `flash`: usa `scaled_dot_product_attention` (FlashAttention-2 quando backend CUDA suportar) e aceita `use_gradient_checkpointing=true`;
  - `mamba_like`: bloco SSM causal simplificado com memoria linear.
- `graph_backend`:
  - `torch_geometric`: usa `torch_geometric.nn.radius_graph` (falha cedo se dependencias extras nao estiverem instaladas);
  - `torch_sparse`: backend alternativo para experimentos nao-competitivos.
  - no protocolo `local_16gb`, `torch_geometric` e obrigatorio por contrato.
- `ipa_edge_chunk_size`:
  - controla chunking de arestas no IPA para reduzir pico de VRAM;
  - no protocolo `local_16gb`, deve ser `<= 256`.
- `thermo_backend`:
  - `rnafold`: usa ViennaRNA (`RNAfold -p`) para extrair BPP do ensemble de Boltzmann;
  - `linearfold`: executa `linearfold --bpp` para obter pares/probabilidades;
- `chemical mapping` (automatico no DataLoader SE(3)):
  - cruza `reactivity_dms/reactivity_2a3` com coordenadas PDB de treino;
  - gera exposicao ao solvente por residuo (`quickstart_pdb_cross`);
  - em inferencia sem PDB usa modo explicito `quickstart_only` (sem fallback silencioso).
- `msa_backend`:
  - `mmseqs2`: busca homologos e extrai covariancia de mutacoes compensatorias (WC/Hoogsteen proxy);
- `prepare-phase1-data-lab`:
  - precomputa BPP (RNAfold/LinearFold) e covariancia MSA (MMseqs2) em paralelo (`--workers`);
  - empacota dataset de treino em `training_store.zarr` para leitura lazy por target direto do SSD/NVMe;
  - requer dependencia opcional `zarr` (`pip install .[data_lab]`).
- Parametros fisicos: `radius_angstrom` (recomendado 12-15) e `max_neighbors`.
- Awareness multicadeia:
  - `chain_separator` define quebra de cadeia na sequencia (ex: `|`);
  - `chain_break_offset` aplica deslocamento massivo no RPE 2D para bloquear conectividade covalente entre cadeias.
- Exemplo de `config_train.json` para alvo longo:

```json
{
  "hidden_dim": 256,
  "num_layers": 4,
  "ipa_heads": 8,
  "diffusion_steps": 24,
  "flow_steps": 24,
  "epochs": 20,
  "learning_rate": 0.0002,
  "method": "both",
  "sequence_tower": "mamba_like",
  "sequence_heads": 8,
  "use_gradient_checkpointing": true,
  "graph_backend": "torch_geometric",
  "radius_angstrom": 14.0,
  "max_neighbors": 64,
  "graph_chunk_size": 512,
  "ipa_edge_chunk_size": 128,
  "thermo_backend": "rnafold",
  "rnafold_bin": "RNAfold",
  "linearfold_bin": "linearfold",
  "thermo_cache_dir": "runs/se3/thermo_cache",
  "msa_backend": "mmseqs2",
  "mmseqs_bin": "mmseqs",
  "mmseqs_db": "/data/mmseqs/rna_db",
  "msa_cache_dir": "runs/se3/msa_cache",
  "chain_separator": "|",
  "chain_break_offset": 1000,
  "max_msa_sequences": 96,
  "max_cov_positions": 256,
  "max_cov_pairs": 8192,
  "loss_weight_mse": 0.0,
  "loss_weight_fape": 1.0,
  "loss_weight_tm": 1.0,
  "loss_weight_clash": 5.0,
  "fape_clamp_distance": 10.0,
  "fape_length_scale": 10.0,
  "vdw_min_distance": 2.1,
  "vdw_repulsion_power": 4,
  "loss_chunk_size": 256,
  "dynamic_cropping": true,
  "crop_min_length": 256,
  "crop_max_length": 384,
  "crop_sequence_fraction": 0.60,
  "gradient_accumulation_steps": 16,
  "autocast_bfloat16": true,
  "training_protocol": "local_16gb"
}
```

## Observacoes operacionais

- Sem fallback silencioso: falhas de contrato interrompem execucao.
- `training_protocol=local_16gb` aplica contrato estrito para treino local:
  - `dynamic_cropping=true`;
  - `graph_backend=torch_geometric`;
  - `crop_min_length/crop_max_length` em `[256,384]`;
  - `ipa_edge_chunk_size <= 256`;
  - `use_gradient_checkpointing=true`;
  - `autocast_bfloat16=true`;
  - `gradient_accumulation_steps` em `[16,32]`.
- `autocast_bfloat16=true` falha cedo se CUDA/BF16 nao estiver disponivel (nao existe fallback silencioso para FP32/CPU).
- Backends `mock` foram removidos do runtime/CLI/config; configs com `mock` falham cedo (backend invalido).
- `backend=rules` e `ann_engine=numpy_bruteforce` continuam opcoes de teste local explicito.
- `build-homology-folds` aplica estratificacao de dominio por padrao (falha cedo sem fonte de dominio valida).
- `evaluate-homology-folds` prioriza score de orphans com ponderacao explicita (`orphan_weight`) e exige classificacao orphan valida.
- `minimize-ensemble` e opcional antes de `export-submission`; por padrão ele já roda em bypass (`max_iterations=0`). Evite passar `--max-iterations` no notebook de submissão, a menos que queira habilitar relaxação explicitamente.
- Quando habilitada, a minimizacao roda em regime curto (`0 <= max_iterations <= 100`) e `backend=openmm` falha cedo se dependencia/plataforma estiver indisponivel.
- `score-local-bestof5` usa o binario C++ do USalign para reproduzir Best-of-5 localmente e gerar `score.json` estrito para gating de submissao.
- O caminho padrao competitivo usa `encoder=ribonanzanet2`, `backend=llama_cpp` e `ann_engine=faiss_ivfpq`.
- FASE 2 usa roteamento deterministico por comprimento (prioridade maxima):
  - `L <= --short-max-len` -> foundation trio obrigatorio (`chai1 + boltz1 + rnapro`);
  - `--short-max-len < L <= --medium-max-len` -> `--se3-flash`;
  - `L > --medium-max-len` -> `--se3-mamba`, com fallback explicito para `tbm`;
  - `template_score` e `ligand_SMILES` permanecem apenas como metadados de diagnostico no `routing.parquet` (nao decidem rota primaria).
  - `--se3` legado permanece como alias temporario para `--se3-flash` e `--se3-mamba`.

## Contratos de runners offline (Fase 2)

- `predict-rnapro-offline`, `predict-chai1-offline`, `predict-boltz1-offline` nao geram saida sintetica.
- Cada `assets/models/<modelo>/config.json` deve conter:
  - `entrypoint`: lista de argumentos do comando a executar (runner offline), com placeholders:
    - `{model_dir}` `{targets}` `{out}` `{n_models}`.
- O comando e executado com `cwd=<model_dir>` e deve escrever um parquet no caminho `{out}` no schema long (`target_id,model_id,resid,resname,x,y,z[,confidence,source]`).
