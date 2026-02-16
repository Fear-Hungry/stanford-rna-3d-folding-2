# Experimentos (iteracao controlada, sem promessas de ranking)

Este diretorio padroniza **experimentos reprodutiveis** para iterar no score local (USalign Best-of-5) com o pipeline do repositorio.

Objetivo pratico: reduzir risco de *overfitting* no Public LB e acelerar ablaçoes com artefatos em `runs/`.

## Como rodar

1) Validar/inspecionar uma receita (sem executar):

```bash
python -m rna3d_local run-experiment --recipe experiments/recipes/E01_phase1_tbm_baseline.json --dry-run
```

2) Executar, sobrescrevendo variaveis de caminho quando necessario:

```bash
python -m rna3d_local run-experiment \
  --recipe experiments/recipes/E01_phase1_tbm_baseline.json \
  --var ribonanzanet2_model=assets/models/ribonanzanet2_encoder.ts
```

- Cada execucao cria `runs/<timestamp>_<tag>/` com:
  - `recipe.json` (receita resolvida),
  - `meta.json` (git, python, plataforma),
  - `step_XX_<name>.log` (log por etapa),
  - `run_report.json` (status + tempos + artefatos).

## Regras do repositorio (reforco)

- **Fail-fast**: qualquer falha de contrato/artefato bloqueia o experimento; nao existe fallback silencioso.
- **Antes de submeter Kaggle**: sempre passar por `check-submission` + `score-local-bestof5` + gate de melhoria estrita (ver `AGENTS.md`).
- **Treino no Kaggle e proibido**: use Kaggle apenas para notebook de submissao (inferencia/export).

## Receitas (ordem recomendada)

1) **Fase 1 (TBM)**: `E01_*` e `E02_*`
   - valida Template DB + embeddings + retrieval + TBM + score local.
2) **SE(3) gerativo (orfans/ultra-long)**: `E20_*`
   - valida data lab (zarr), treino, amostragem, diversidade Top-5 e score local.
3) **Hybrid router (Fase 1 + Fase 2 + SE3)**: `E30_*`
   - tuning de thresholds e composicao do Top-5 usando artefatos precomputados.

## Checklist antes de registrar no `EXPERIMENTS.md`

- anotar o commit (`git rev-parse HEAD`) e a receita usada (`runs/.../recipe.json`);
- anexar paths de artefatos em `runs/...`;
- registrar score local (`runs/.../score.json`) e custo/tempo.

