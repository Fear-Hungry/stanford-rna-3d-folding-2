# Research Harness

Estrutura leve para automacao de pesquisa orientada a verificacao.

Diretorios versionados:
- `configs/research/`: configuracoes base dos experimentos.
- `scripts/`: wrappers de execucao para os comandos `research-*`.
- `research/`: documentacao e manifests estaticos.

Artefatos de execucao (append-only, fora do git):
- `runs/research/literature/<run_id>/...`
- `runs/research/experiments/<run_id>/...`
- `runs/research/reports/<run_id>.md`

Gate obrigatorio:
1. solver executa e retorna status aceito;
2. checks estruturais passam;
3. reproducao por comando unico passa.

Comandos:
- `python -m rna3d_local research-sync-literature ...`
- `python -m rna3d_local research-run ...`
- `python -m rna3d_local research-verify ...`
- `python -m rna3d_local research-report ...`
