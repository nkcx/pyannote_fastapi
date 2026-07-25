#!/usr/bin/env bash
set -euo pipefail

# Usage (Hub download at runtime):
#   export IMAGE="${IMAGE:-ghcr.io/nkcx/pyannote_fastapi:latest}"
#   export API_KEYS="replace-me"
#   export HF_TOKEN="replace-me"   # after accepting HF model access terms
#   ./examples/docker-run.sh
#
# Offline alternative: clone per the model card, then mount and set MODEL_PATH:
#   export MODEL_PATH="/models/pipeline"
#   docker run ... -e MODEL_PATH=/models/pipeline -v "$PWD/pyannote-speaker-diarization-community-1:/models/pipeline:ro" ...

IMAGE="${IMAGE:?Set IMAGE to your CUDA image reference, e.g. ghcr.io/nkcx/pyannote_fastapi:latest}"
API_KEYS="${API_KEYS:?Set API_KEYS to a comma-separated list}"

RUN_ARGS=(
  --rm -it
  --gpus all
  -e "API_KEYS=${API_KEYS}"
  -e "LOG_LEVEL=${LOG_LEVEL:-INFO}"
  -p "8000:8000"
  -v "pyannote_hf_cache:/opt/huggingface"
)

if [[ -n "${MODEL_PATH:-}" ]]; then
  RUN_ARGS+=(-e "MODEL_PATH=${MODEL_PATH}")
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  RUN_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi

if [[ -z "${MODEL_PATH:-}" && -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN (after accepting HF terms) or MODEL_PATH + a volume mount for offline checkout." >&2
  exit 1
fi

docker run "${RUN_ARGS[@]}" "${IMAGE}"
