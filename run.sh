#!/usr/bin/env bash
# Deprecated wrapper that forwards to the Python launcher.
# The previous complex logic now lives in solhunter_zero/launcher.py.
# This script will be removed in a future release.
set -euo pipefail
exec > >(tee -a startup.log) 2>&1
PY=$(command -v python3 || command -v python || true)
if [ -z "${PY}" ]; then
    echo "Python interpreter not found. Please install Python 3.11 or newer." >&2
    exit 1
fi
exec "$PY" -m solhunter_zero.launcher "$@"
