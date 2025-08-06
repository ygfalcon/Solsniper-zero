#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Use an existing virtual environment if available
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

if [[ "$(uname)" == "Darwin" ]]; then
  exec arch -arm64 "$PYTHON" scripts/startup.py --one-click "$@"
else
  exec "$PYTHON" scripts/startup.py --one-click "$@"
fi

