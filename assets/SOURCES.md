# Sources & licenses (obrigatorio)

Este arquivo registra **fontes e licencas** dos pesos/artefatos baixados para rodar offline no Kaggle.

Preencha/atualize antes de anexar `assets/` a um Kaggle Dataset.

## Boltz

- Codigo: MIT (repo oficial do Boltz).
- Pesos: fornecidos publicamente pelos autores (ex.: `boltz-community/boltz-1` no Hugging Face; fallback `model-gateway.boltz.bio`).

## Chai-1 (chai-lab)

- Codigo + pesos: Apache-2.0 (repo oficial do `chai-lab`).
- Artefatos baixados via `chaiassets.com` (modelo exportado + dependencias de inferencia).

## RNAPro (NVIDIA)

- Codigo: Apache-2.0 (repo oficial `NVIDIA-Digital-Bio/RNAPro`).
- Pesos: `nvidia/RNAPro` (Hugging Face) e NGC; pode ser **gated** (exige login/aceite de termos).
- Se usar espelho via Kaggle Dataset, registrar **tambem** a fonte original (Hugging Face/NGC) e os termos aplicaveis.
- Dependencias de runtime para inferencia offline (incluidas no wheelhouse):
  - `rnapro` wheel construido via git (universal `py3-none-any`),
  - `ml-collections`, `biotite`, `rdkit`, `scipy`, etc (sem `deepspeed/cuequivariance/triton`).
- Assets adicionais obrigatorios para o runner `runners/rnapro.py`:
  - RibonanzaNet2 checkpoint via Kaggle Model `shujun717/ribonanzanet2/pyTorch/alpha/1` (arquivos `pairwise.yaml` + `pytorch_model_fsdp.bin`),
  - CCD cache minimo (`components.cif` + `components.cif.rdkit_mol.pkl`) gerado localmente a partir do CCD oficial do PDB.

## RibonanzaNet2

- Pesos/artefatos encontrados tipicamente via Kaggle Dataset (ex.: `shujun717/ribonanzanet2-ddpm-v2`).
- Registrar referencia do dataset e, quando possivel, o paper/origem do modelo.
  - Para RNAPro: Kaggle Model `shujun717/ribonanzanet2` (pairwise encoder).

## Wheels (dependencias Python)

- Wheels em `assets/wheels/` sao baixados do PyPI para instalacao offline no Kaggle (`enable_internet=false`).
- Excecao: quando um pacote nao fornece wheel para o alvo (ex.: `fairscale==0.4.13`), o `build-wheelhouse` faz build local de um **wheel universal** a partir do sdist.

## CCD (Chemical Component Dictionary)

- Fonte: `components.cif.gz` do wwPDB (`files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz`).
- Neste repo usamos um **cache minimo** (apenas bases RNA A/C/G/U) para reduzir tamanho e acelerar inferencia offline do RNAPro.
