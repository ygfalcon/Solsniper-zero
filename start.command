#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Ensure script is running on arm64
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "Error: unsupported architecture '$ARCH'; this script requires arm64." >&2
    exit 1
fi

# Find a suitable Python interpreter
if command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
else
    echo "Error: could not find python interpreter (python, python3, or python3.11)." >&2
    exit 1
fi

exec arch -arm64 "$PYTHON" start.py --one-click --full-deps "$@"

