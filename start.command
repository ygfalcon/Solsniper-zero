#!/usr/bin/env bash
# This launcher finds a Python interpreter and ensures it is at least version 3.11.
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null; then
  PY=$(command -v python3)
elif command -v python >/dev/null; then
  PY=$(command -v python)
else
  echo "Error: Python interpreter not found. Install Python 3.11 or higher." >&2
  exit 1
fi

PY_VERSION=$("$PY" -V 2>&1 | awk '{print $2}')
IFS='.' read -r PY_MAJOR PY_MINOR _ <<< "$PY_VERSION"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 11) )); then
  echo "Error: Python 3.11 or higher is required (found $PY_VERSION)." >&2
  exit 1
fi

"$PY" scripts/startup.py --one-click
