# stanford-rna-3d-folding-2 (phase 1+2 reboot)

Repositorio reiniciado para implementar FASE 1 (Template Oracle) e FASE 2 (Arsenal Hibrido 3D) com modo estrito e fail-fast.

## Comandos disponiveis

```bash
python -m rna3d_local build-template-db --external-templates external_templates.csv --out-dir data/derived/template_db
python -m rna3d_local build-embedding-index --template-index data/derived/template_db/template_index.parquet --out-dir data/derived/template_db/emb --encoder ribonanzanet2 --model-path <local_model_file>
python -m rna3d_local infer-description-family --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out-dir runs/desc --backend llama_cpp --llm-model-path <model.gguf>
python -m rna3d_local prepare-chemical-features --quickstart <ribonanza_quickstart.csv> --out runs/chem/chemical_features.parquet
python -m rna3d_local retrieve-templates-latent --template-index data/derived/template_db/template_index.parquet --template-embeddings data/derived/template_db/emb/template_embeddings.parquet --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/retrieval/retrieval_candidates.parquet --faiss-index data/derived/template_db/emb/faiss_ivfpq.index
python -m rna3d_local train-template-reranker --candidates runs/retrieval/retrieval_candidates.parquet --chemical-features runs/chem/chemical_features.parquet --out-dir runs/reranker
python -m rna3d_local score-template-reranker --candidates runs/retrieval/retrieval_candidates.parquet --chemical-features runs/chem/chemical_features.parquet --model runs/reranker/model.pt --config runs/reranker/config.json --out runs/reranker/reranked.parquet
python -m rna3d_local predict-tbm --retrieval runs/reranker/reranked.parquet --templates data/derived/template_db/templates.parquet --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/tbm/predictions.parquet --n-models 5
python -m rna3d_local export-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --predictions runs/tbm/predictions.parquet --out runs/tbm/submission.csv
python -m rna3d_local check-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/tbm/submission.csv

# Fase 2 (assets + modelos offline + roteamento hibrido)
python -m rna3d_local build-phase2-assets --assets-dir assets
python -m rna3d_local predict-rnapro-offline --model-dir assets/models/rnapro --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/phase2/rnapro.parquet --n-models 5
python -m rna3d_local predict-chai1-offline --model-dir assets/models/chai1 --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/phase2/chai1.parquet --n-models 5
python -m rna3d_local predict-boltz1-offline --model-dir assets/models/boltz1 --targets input/stanford-rna-3d-folding-2/test_sequences.csv --out runs/phase2/boltz1.parquet --n-models 5
python -m rna3d_local build-hybrid-candidates --targets input/stanford-rna-3d-folding-2/test_sequences.csv --retrieval runs/retrieval/retrieval_candidates.parquet --tbm runs/tbm/predictions.parquet --rnapro runs/phase2/rnapro.parquet --chai1 runs/phase2/chai1.parquet --boltz1 runs/phase2/boltz1.parquet --se3 runs/se3/top5.parquet --out runs/phase2/hybrid_candidates.parquet --routing-out runs/phase2/routing.parquet --template-score-threshold 0.65
python -m rna3d_local select-top5-hybrid --candidates runs/phase2/hybrid_candidates.parquet --out runs/phase2/hybrid_top5.parquet --n-models 5
python -m rna3d_local export-submission --sample input/stanford-rna-3d-folding-2/sample_submission.csv --predictions runs/phase2/hybrid_top5.parquet --out runs/phase2/submission.csv
python -m rna3d_local evaluate-submit-readiness --sample input/stanford-rna-3d-folding-2/sample_submission.csv --submission runs/phase2/submission.csv --score-json runs/phase2/score.json --baseline-score 0.0 --report runs/phase2/readiness.json

# Branch experimental SE(3) + gerativo (Best-of-5)
python -m rna3d_local train-se3-generator --targets input/stanford-rna-3d-folding-2/train_sequences.csv --pairings runs/se3/pairings.parquet --chemical-features runs/chem/chemical_features.parquet --labels input/stanford-rna-3d-folding-2/train_labels.csv --config runs/se3/config_train.json --out-dir runs/se3/model --seed 123
python -m rna3d_local sample-se3-ensemble --model-dir runs/se3/model --targets input/stanford-rna-3d-folding-2/test_sequences.csv --pairings runs/se3/pairings_test.parquet --chemical-features runs/chem/chemical_features.parquet --out runs/se3/candidates.parquet --method both --n-samples 24 --seed 123
python -m rna3d_local rank-se3-ensemble --candidates runs/se3/candidates.parquet --out runs/se3/ranked.parquet --diversity-lambda 0.35
python -m rna3d_local select-top5-se3 --ranked runs/se3/ranked.parquet --out runs/se3/top5.parquet --n-models 5 --diversity-lambda 0.35
```

## Configuracao de escala SE(3) para L longo

- `sequence_tower`:
  - `flash`: usa `scaled_dot_product_attention` (FlashAttention-2 quando backend CUDA suportar) e aceita `use_gradient_checkpointing=true`;
  - `mamba_like`: bloco SSM causal simplificado com memoria linear.
- `graph_backend`:
  - `torch_sparse`: radius graph com `torch.sparse` e processamento em chunks (`graph_chunk_size`);
  - `torch_geometric`: usa `torch_geometric.nn.radius_graph` (falha cedo se dependencias extras nao estiverem instaladas).
- Parametros fisicos: `radius_angstrom` (recomendado 12-15) e `max_neighbors`.
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
  "graph_backend": "torch_sparse",
  "radius_angstrom": 14.0,
  "max_neighbors": 64,
  "graph_chunk_size": 512
}
```

## Observacoes operacionais

- Sem fallback silencioso: falhas de contrato interrompem execucao.
- `encoder=mock`, `backend=rules` e `ann_engine=numpy_bruteforce` existem apenas para teste local explicito.
- O caminho padrao competitivo usa `encoder=ribonanzanet2`, `backend=llama_cpp` e `ann_engine=faiss_ivfpq`.
- FASE 2 usa roteamento deterministico:
  - template forte -> TBM;
  - template/órfão fraco com `--se3` -> `generative_se3`;
  - orfao sem ligante -> ensemble `chai1_boltz1_ensemble`;
  - alvo com `ligand_SMILES` -> Boltz-1 primario;
  - RNAPro entra como candidato suplementar quando fornecido.
