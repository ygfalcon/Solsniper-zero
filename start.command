#!/usr/bin/env bash
# This launcher expects Python 3.11 to be available as `python3.11` or fall back to
# the default `python3` interpreter.
cd "$(dirname "$0")"

command -v python3.11 >/dev/null && PY=python3.11 || PY=python3
if ! command -v "$PY" >/dev/null; then
  echo "Error: Python interpreter not found. Install Python 3.11 or python3." >&2
  exit 1
fi

"$PY" scripts/startup.py --one-click
