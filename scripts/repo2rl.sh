#!/usr/bin/env bash
# Run Repo2RLEnv Incremental Mining across predefined competition repositories
set -euo pipefail

REAL_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
REAL_DIR="$(cd "$(dirname "$REAL_SCRIPT")" && pwd)"
if [ -f "$REAL_DIR/scripts/incremental_miner.py" ]; then
  REPO_ROOT="$REAL_DIR"
elif [ -f "$REAL_DIR/../scripts/incremental_miner.py" ]; then
  REPO_ROOT="$(cd "$REAL_DIR/.." && pwd)"
else
  REPO_ROOT="$(pwd)"
fi

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

REPO_KEY="${1:-}"

if [ -z "$REPO_KEY" ]; then
  echo "Usage: $0 <repo_name> [prs_or_out_dir] [out_dir]"
  echo ""
  echo "Predefined repo_name:"
  echo "  - fastapi   (tiangolo/fastapi, 67 PRs)"
  echo "  - httpx     (encode/httpx, 1 PR)"
  echo "  - rich      (Textualize/rich, 48 PRs)"
  echo "  - requests  (psf/requests, 13 PRs)"
  echo ""
  echo "Examples:"
  echo "  $0 fastapi                       # All PRs -> test/hard, test/easy, and test/fix"
  echo "  $0 fastapi my_test_dir           # All PRs -> my_test_dir/hard, my_test_dir/easy, and my_test_dir/fix"
  echo "  $0 fastapi 11194                 # PR 11194 -> test/hard, test/easy, and test/fix"
  echo "  $0 fastapi 5077,5624 my_test_dir # PRs -> my_test_dir/hard, my_test_dir/easy, and my_test_dir/fix"
  exit 1
fi

case "${REPO_KEY,,}" in
  fastapi|fastapi/fastapi|tiangolo/fastapi)
    REPO="tiangolo/fastapi"
    DEFAULT_PRS="5077,5624,9425,9555,9753,11194,11355,12942,13207,13537,13713,13786,13920,14077,14099,14186,14246,14258,14262,14266,14297,14301,14303,14306,14349,14356,14360,14361,14371,14372,14419,14430,14448,14455,14458,14459,14463,14479,14482,14485,14487,14492,14512,14583,14605,14609,14616,14786,14791,14794,14851,14873,14953,14962,14964,14978,14986,15023,15030,15280,15588,15589,15661,15745,15763,15785,15800"
    ;;
  httpx|encode/httpx)
    REPO="encode/httpx"
    DEFAULT_PRS="3672"
    ;;
  rich|textualize/rich)
    REPO="Textualize/rich"
    DEFAULT_PRS="2725,2943,3006,3043,3052,3061,3063,3064,3067,3105,3130,3180,3278,3296,3454,3468,3469,3470,3471,3472,3480,3486,3506,3518,3521,3535,3675,3676,3718,3772,3777,3782,3882,3894,3905,3930,3934,3935,3938,3942,3944,3953,4006,4070,4075,4076,4077,4079"
    ;;
  requests|psf/requests)
    REPO="psf/requests"
    DEFAULT_PRS="6589,6592,6629,6644,6757,7205,7309,7315,7328,7427,7433,7502,7505"
    ;;
  *)
    echo "Error: Unknown repository '$REPO_KEY'"
    echo "Supported: fastapi, httpx, rich, requests"
    exit 1
    ;;
esac

ARG2="${2:-}"
ARG3="${3:-}"

if [ -n "$ARG2" ]; then
  if [ "$ARG2" = "-o" ] || [ "$ARG2" = "--out" ]; then
    PRS="$DEFAULT_PRS"
    OUT_DIR="${ARG3:-$REPO_ROOT/test}"
  elif [[ "$ARG2" =~ ^[0-9]+([,0-9[:space:]]*)$ ]]; then
    PRS="$ARG2"
    OUT_DIR="${ARG3:-$REPO_ROOT/test}"
  else
    # ARG2 is not digits, so treat it directly as the custom output directory
    PRS="$DEFAULT_PRS"
    OUT_DIR="$ARG2"
  fi
else
  PRS="$DEFAULT_PRS"
  OUT_DIR="${ARG3:-$REPO_ROOT/test}"
fi

BASE_IMAGE="${BASE_IMAGE:-carlesoctav/swegemm-base}"
LLM="${LLM:-openai/thinkingmachines/Inkling-Small}"
LLM_ENDPOINT="${LLM_ENDPOINT:-https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1}"
LLM_KEY_ENV="${LLM_KEY_ENV:-TINKER_API_KEY}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-docker.io/carlesoctav}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/workspace/bootstrap}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
MAX_SECONDS="${MAX_SECONDS:-86400}"
MAX_SPEND_USD="${MAX_SPEND_USD:-0}"
MAX_FILES="${MAX_FILES:-30}"

echo "=========================================================="
echo "Incremental Mining $REPO"
echo "PRs count: $(echo "$PRS" | tr ',' '\n' | wc -l) PRs"
echo "Output base: $OUT_DIR"
echo "Outputs: $OUT_DIR/hard, $OUT_DIR/easy, and $OUT_DIR/fix-desc"
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$BASE_IMAGE" ]; then
  EXTRA_ARGS+=(--base-image "$BASE_IMAGE")
fi

exec uv run python "$REPO_ROOT/scripts/incremental_miner.py" \
  --repo "$REPO" \
  --prs "$PRS" \
  --out "$OUT_DIR" \
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
