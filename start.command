#!/bin/bash
cd "$(dirname "$0")"
PY=python
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
fi
"$PY" scripts/startup.py --auto
