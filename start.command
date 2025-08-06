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

# Configure Rayon thread pool if not already set
if [ -z "$RAYON_NUM_THREADS" ]; then
  if command -v nproc >/dev/null 2>&1; then
    export RAYON_NUM_THREADS="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    export RAYON_NUM_THREADS="$(getconf _NPROCESSORS_ONLN)"
  else
    export RAYON_NUM_THREADS="$("$PY" - <<'EOF'
import os
print(os.cpu_count() or 1)
EOF
)"
  fi
fi

# Enable MPS fallback when running on macOS
if [ "$(uname -s)" = "Darwin" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

if [ ! -f "config.toml" ]; then
  cp config.example.toml config.toml
  echo "Created default config.toml from config.example.toml"
fi

# On macOS ensure Homebrew, rustup and Xcode command-line tools are available
if [ "$(uname -s)" = "Darwin" ]; then
  missing=0
  command -v brew >/dev/null 2>&1 || missing=1
  command -v rustup >/dev/null 2>&1 || missing=1
  command -v xcode-select >/dev/null 2>&1 || missing=1
  xcode-select -p >/dev/null 2>&1 || missing=1

  if [ $missing -ne 0 ]; then
    echo "Missing required macOS tools. Running scripts/mac_setup.sh ..."
    scripts/mac_setup.sh
    exec "$0" "$@"
  fi
fi

"$PY" scripts/startup.py --one-click
