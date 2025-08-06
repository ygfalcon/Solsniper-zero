#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Find a suitable Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: could not find python interpreter (python3, python3.11, or python)." >&2
    exit 1
fi

exec "$PYTHON" start.py --one-click --full-deps "$@"

