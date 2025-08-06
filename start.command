#!/usr/bin/env bash

# Start Solhunter Zero via the bootstrap module in one-click mode.
set -euo pipefail
cd "$(dirname "$0")"

# Use an existing virtual environment if available
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

if [[ "$(uname)" == "Darwin" ]]; then
  # Ensure macOS uses the ARM64 architecture
  exec arch -arm64 "$PYTHON" -m solhunter_zero.bootstrap --one-click "$@"
else
  exec "$PYTHON" -m solhunter_zero.bootstrap --one-click "$@"
fi

