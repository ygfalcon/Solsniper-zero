#!/usr/bin/env bash
# This launcher finds a Python interpreter and ensures it is at least version 3.11.
set -euo pipefail
cd "$(dirname "$0")"

# Rotate logs before redirecting output
rotate_logs() {
  local logfile="startup.log"
  local timestamp="$(date +'%Y%m%d-%H%M%S')"
  if [ -f "$logfile" ]; then
    mv "$logfile" "${logfile%.log}-$timestamp.log"
  fi
  local max_logs=5
  ls -1t ${logfile%.log}-*.log 2>/dev/null | tail -n +$((max_logs+1)) | xargs -r rm -- || true
}

rotate_logs
exec > >(tee -a startup.log) 2>&1

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
if [ -z "${RAYON_NUM_THREADS:-}" ]; then
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

if [ "$(uname -s)" = "Darwin" ]; then
  if ! command -v brew >/dev/null 2>&1 || ! command -v rustup >/dev/null 2>&1; then
    echo "Missing Homebrew or rustup. Running mac setup..."
    scripts/mac_setup.sh
  fi
fi

"$PY" scripts/startup.py --one-click
