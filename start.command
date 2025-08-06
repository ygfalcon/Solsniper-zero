#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Use an existing virtual environment if available
if [[ -x ".venv/bin/python3" ]]; then
  PYTHON=".venv/bin/python3"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

if [[ "$(uname)" == "Darwin" ]]; then
  THREADS="$($PYTHON -m solhunter_zero.system cpu-count 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
  export RAYON_NUM_THREADS="$THREADS"
  exec arch -arm64 "$PYTHON" scripts/startup.py --one-click --full-deps "$@"
else
  exec "$PYTHON" scripts/startup.py --one-click --full-deps "$@"
fi

