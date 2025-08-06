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

if [ "$(uname -m)" != "arm64" ]; then
  if command -v arch >/dev/null 2>&1; then
    arch -arm64 "$PY" scripts/startup.py --one-click && exit 0
    echo "Error: arm64 Python is required. Install the arm64 version of Python." >&2
    exit 1
  else
    echo "Error: arm64 Python is required. Install the arm64 version of Python." >&2
    exit 1
  fi
fi

"$PY" scripts/startup.py --one-click
