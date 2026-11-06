#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# Activate virtualenv if present
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  source "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/Repo2RLEnv/.venv/bin/activate" ]; then
  source "$REPO_ROOT/Repo2RLEnv/.venv/bin/activate"
fi

# Load .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

BASE_IMAGE="${BASE_IMAGE:-carlesoctav/swegemma}"
LLM="${LLM:-openai/thinkingmachines/Inkling-Small}"
LLM_ENDPOINT="${LLM_ENDPOINT:-https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1}"
LLM_KEY_ENV="${LLM_KEY_ENV:-TINKER_API_KEY}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-docker.io/carlesoctav}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/workspace/bootstrap}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
MAX_SECONDS="${MAX_SECONDS:-86400}"
MAX_SPEND_USD="${MAX_SPEND_USD:-0}"
SUCCESS_LIMIT="${SUCCESS_LIMIT:-50}"
LIMIT="${LIMIT:-500}"
MAX_FILES="${MAX_FILES:-30}"
OUT_BASE="${OUT_BASE:-train_new}"

# Target repositories
REPOS=(
  "encode/httpx"
  "psf/requests"
  "tiangolo/fastapi"
  "Textualize/rich"
  "encode/starlette"
  "tiangolo/sqlmodel"
  "tiangolo/typer"
  "pallets/flask"
  "pallets/click"
)

# Allow passing a single repo as an argument: ./mining.sh [repo_name]
if [ $# -ge 1 ]; then
  TARGET="$1"
  MATCHED=()
  for r in "${REPOS[@]}"; do
    repo_name="${r##*/}"
    if [ "$TARGET" = "$r" ] || [ "$TARGET" = "$repo_name" ]; then
      MATCHED=("$r")
      break
    fi
  done
  if [ ${#MATCHED[@]} -eq 0 ]; then
    MATCHED=("$TARGET")
  fi
  RUN_REPOS=("${MATCHED[@]}")
else
  RUN_REPOS=("${REPOS[@]}")
fi

for repo in "${RUN_REPOS[@]}"; do
  repo_name="${repo##*/}"
  out_dir="$REPO_ROOT/$OUT_BASE/$repo_name"
  mkdir -p "$out_dir"

  echo "=========================================================="
  echo "Incremental Mining $repo -> $out_dir (limit=$LIMIT)"
  echo "Outputs: $out_dir/hard, $out_dir/easy, and $out_dir/fix-desc"
  echo "=========================================================="

  EXTRA_ARGS=()
  if [ -n "$BASE_IMAGE" ]; then
    EXTRA_ARGS+=(--base-image "$BASE_IMAGE")
  fi

  uv run python "$REPO_ROOT/scripts/incremental_miner.py" \
    --repo "$repo" \
    --out "$out_dir" \
    --limit "$LIMIT" \
    --max-success "$SUCCESS_LIMIT" \
    --max-source-files "$MAX_FILES" \
    --llm "$LLM" \
    --llm-endpoint "$LLM_ENDPOINT" \
    --llm-key-env "$LLM_KEY_ENV" \
    --reasoning-effort "$REASONING_EFFORT" \
    --image-registry "$IMAGE_REGISTRY" \
    --cache-dir "$CACHE_DIR" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-seconds "$MAX_SECONDS" \
    --max-spend-usd "$MAX_SPEND_USD" \
    "${EXTRA_ARGS[@]}"
done
