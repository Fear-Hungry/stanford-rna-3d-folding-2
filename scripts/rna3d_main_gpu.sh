#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Uso:
  scripts/rna3d_main_gpu.sh <comando-cli> [args...]

Descricao:
  Wrapper GPU-first para `python -m rna3d_local`.
  - Injeta flags CUDA em comandos GPU-capable.
  - Falha cedo se CUDA nao estiver disponivel.
  - Nao altera comandos CPU-only.

Exemplos:
  scripts/rna3d_main_gpu.sh retrieve-templates --help
  scripts/rna3d_main_gpu.sh predict-tbm --templates t.parquet --targets x.csv --out tbm.parquet
  scripts/rna3d_main_gpu.sh train-qa-rnrank --help
EOF
}

contains_help_flag() {
  local token
  for token in "$@"; do
    if [[ "${token}" == "--help" || "${token}" == "-h" ]]; then
      return 0
    fi
  done
  return 1
}

is_gpu_capable_command() {
  local cmd="$1"
  case "${cmd}" in
    retrieve-templates|predict-tbm|train-rnapro|predict-rnapro|build-candidate-pool|train-qa-rnrank|score-qa-rnrank|select-top5-global|train-qa-gnn-ranker|score-qa-gnn-ranker)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

check_cuda_ready() {
  python - <<'PY'
import sys

stage = "[MAIN_GPU] [scripts/rna3d_main_gpu.sh:check_cuda_ready]"
try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(
        f"{stage} torch indisponivel para modo GPU | impacto=1 | exemplos={type(exc).__name__}:{exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)

if not torch.cuda.is_available():
    print(
        f"{stage} CUDA indisponivel (torch.cuda.is_available=False) | impacto=1 | exemplos=valide driver/runtime NVIDIA",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

COMMAND="$1"
shift

USER_ARGS=("$@")
FORCED_ARGS=()

case "${COMMAND}" in
  retrieve-templates|train-rnapro|build-candidate-pool)
    FORCED_ARGS+=(--compute-backend cuda)
    ;;
  predict-tbm|predict-rnapro)
    FORCED_ARGS+=(--qa-device cuda --compute-backend cuda)
    ;;
  train-qa-rnrank|score-qa-rnrank|select-top5-global|train-qa-gnn-ranker|score-qa-gnn-ranker)
    FORCED_ARGS+=(--device cuda)
    ;;
esac

if is_gpu_capable_command "${COMMAND}" && ! contains_help_flag "${USER_ARGS[@]}"; then
  check_cuda_ready
fi

cd "${REPO_ROOT}"
exec python -m rna3d_local "${COMMAND}" "${USER_ARGS[@]}" "${FORCED_ARGS[@]}"
