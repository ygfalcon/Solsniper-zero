#!/usr/bin/env bash
set -euo pipefail
# Launcher wrapper.

cd "$(dirname "$0")" || exit 1

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Error: Python 3.11 is required. Please install Python 3.11 and try again." >&2
  exit 1
fi

exec "$PYTHON" -m solhunter_zero.launcher "$@" || {
  echo "Error: failed to start the SolHunter Zero launcher." >&2
  exit 1
}
