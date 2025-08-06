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
find_python() {
    if command -v python >/dev/null 2>&1; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python3.11 >/dev/null 2>&1; then
        PYTHON=python3.11
    else
        PYTHON=""
    fi
}

find_python
if [[ -z "$PYTHON" ]]; then
    echo "Could not find python interpreter (python, python3, or python3.11). Attempting installation..." >&2
    python scripts/mac_setup.py --non-interactive || true
    find_python
    if [[ -z "$PYTHON" ]]; then
        echo "Error: could not find python interpreter (python, python3, or python3.11) even after installation attempt." >&2
        exit 1
    fi
fi

exec arch -arm64 "$PYTHON" start.py --one-click --full-deps "$@"

