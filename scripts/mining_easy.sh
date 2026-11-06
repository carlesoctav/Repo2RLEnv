#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

echo "Note: mining.sh now generates BOTH easy and hard modes into \$out_dir/easy and \$out_dir/hard in a single pass."
exec "$REPO_ROOT/mining.sh" "$@"
