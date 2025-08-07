#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python" ]; then
  PYTHON=".venv/Scripts/python"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
    USE_ARCH=1
  else
    echo "Error: No suitable Python interpreter found. Install Python 3.11+ or create a virtual environment in .venv." >&2
    exit 1
  fi
fi

if [ -n "$USE_ARCH" ]; then
  exec arch -arm64 "$PYTHON" -m solhunter_zero.launcher "$@"
else
  exec "$PYTHON" -m solhunter_zero.launcher "$@"
fi
