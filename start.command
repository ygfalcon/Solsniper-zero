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

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    PY=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PY=$(command -v python)
  else
    return 1
  fi

  PY_VERSION=$("$PY" -V 2>&1 | awk '{print $2}')
  IFS='.' read -r PY_MAJOR PY_MINOR _ <<< "$PY_VERSION"
  if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 11) )); then
    return 2
  fi
  return 0
}

if ! find_python; then
  STATUS=$?
  if [[ $STATUS -eq 1 ]]; then
    echo "Python interpreter not found. Attempting to install Python 3.11..."
  else
    echo "Python 3.11 or higher is required (found $PY_VERSION). Attempting installation..."
  fi
  scripts/mac_setup.sh --non-interactive || true
  echo "Retrying Python interpreter discovery..."
  if ! find_python; then
    STATUS=$?
    if [[ $STATUS -eq 1 ]]; then
      echo "Error: Python interpreter not found. Install Python 3.11 or higher." >&2
    else
      echo "Error: Python 3.11 or higher is required (found $PY_VERSION)." >&2
    fi
    exit 1
  fi
fi

# Configure GPU-related environment variables
eval "$("$PY" -m solhunter_zero.device --setup-env)"

# Configure Rayon thread pool if not already set
if [ -z "${RAYON_NUM_THREADS:-}" ]; then
  export RAYON_NUM_THREADS="$("$PY" -m scripts.threading)"
fi

"$PY" - <<'PY'
from scripts.startup import ensure_config
ensure_config()
PY

if [ "$(uname -s)" = "Darwin" ]; then
  if ! command -v brew >/dev/null 2>&1 || ! command -v rustup >/dev/null 2>&1; then
    echo "Missing Homebrew or rustup. Running mac setup..."
    scripts/mac_setup.sh --non-interactive
    if command -v brew >/dev/null 2>&1; then
      eval "$(brew shellenv)"
    elif [ -x /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
      eval "$(/usr/local/bin/brew shellenv)"
    else
      echo "Homebrew not found after mac setup. Re-running mac setup..."
      exec scripts/mac_setup.sh --non-interactive
    fi
  fi

  for cmd in brew python3.11 rustup; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Error: $cmd not found in PATH after mac setup." >&2
      exit 1
    fi
  done
fi

"$PY" scripts/startup.py --one-click
