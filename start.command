#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: Neither python3.11 nor python is available in PATH." >&2
  exit 1
fi

exec "$PYTHON_BIN" scripts/startup.py --auto "$@"

