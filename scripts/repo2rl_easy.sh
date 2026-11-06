#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Note: repo2rl.sh now generates BOTH easy and hard modes into \$OUT_DIR/easy and \$OUT_DIR/hard in a single pass."
exec "$SCRIPT_DIR/repo2rl.sh" "$@"
