# Assets offline (Kaggle sem internet)

Este diretorio armazena **apenas artefatos locais** (pesos, wheels e manifests) para rodar inferencia no Kaggle com `enable_internet=false`.

Regras:

- **Nao commitar** binarios: `assets/models/`, `assets/wheels/`, `assets/encoders/`, `assets/runtime/` sao ignorados via `.gitignore`.
- Registre a origem/licenca em `assets/SOURCES.md` antes de usar em competicao.
- Gere um manifest verificavel com:
  - `python -m rna3d_local build-phase2-assets --assets-dir assets`

Fetch automatizado:

- `python -m rna3d_local fetch-pretrained-assets --assets-dir assets --include ribonanzanet2 --dry-run`
- `python -m rna3d_local fetch-pretrained-assets --assets-dir assets --include ribonanzanet2`

RNAPro (extras obrigatorios p/ inferencia real):

- RibonanzaNet2 (pairwise) para RNAPro:
  - `python -m rna3d_local fetch-pretrained-assets --assets-dir assets --include ribonanzanet2_pairwise`
- CCD cache minimo + `test_templates.pt` (gerado localmente; nao roda no Kaggle):
  - `python -m rna3d_local prepare-rnapro-support-files --model-dir assets/models/rnapro`

Wheelhouse (pip offline no Kaggle):

- `python -m rna3d_local build-wheelhouse --wheels-dir assets/wheels --python-version 3.12`
- Observacao: alguns pacotes (ex.: `fairscale==0.4.13`) nao possuem wheel no PyPI para cp312; o builder faz fallback e **builda um wheel universal** via `pip wheel` (registrado no `wheelhouse_manifest.json`).
