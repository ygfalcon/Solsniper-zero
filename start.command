#!/usr/bin/env bash
set -euo pipefail
# Launcher wrapper.

cd "$(dirname "$0")" || exit 1

exec python3 -m solhunter_zero.launcher "$@" || {
  echo "Error: failed to start the SolHunter Zero launcher." >&2
  exit 1
}

