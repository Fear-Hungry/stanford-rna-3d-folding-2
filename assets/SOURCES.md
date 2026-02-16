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

- Pesos: `nvidia/RNAPro` (Hugging Face) e NGC; pode ser **gated** (exige login/aceite de termos).
- Se usar espelho via Kaggle Dataset, registrar **tambem** a fonte original (Hugging Face/NGC) e os termos aplicaveis.

## RibonanzaNet2

- Pesos/artefatos encontrados tipicamente via Kaggle Dataset (ex.: `shujun717/ribonanzanet2-ddpm-v2`).
- Registrar referencia do dataset e, quando possivel, o paper/origem do modelo.

## Wheels (dependencias Python)

- Wheels em `assets/wheels/` sao baixados do PyPI para instalacao offline no Kaggle (`enable_internet=false`).
- Excecao: quando um pacote nao fornece wheel para o alvo (ex.: `fairscale==0.4.13`), o `build-wheelhouse` faz build local de um **wheel universal** a partir do sdist.
