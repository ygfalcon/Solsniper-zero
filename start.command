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

# If SOLANA_RPC_URL is set, ensure it's reachable within timeout
if [ -n "${SOLANA_RPC_URL:-}" ]; then
  "$PY" - <<'PY'
import os, sys, urllib.request
url = os.environ["SOLANA_RPC_URL"]
if url.startswith("ws://"):
    url = "http://" + url[5:]
elif url.startswith("wss://"):
    url = "https://" + url[6:]
try:
    req = urllib.request.Request(url, method="HEAD")
    urllib.request.urlopen(req, timeout=5)
except Exception as e:
    print(f"Error: SOLANA_RPC_URL '{os.environ['SOLANA_RPC_URL']}' is unreachable: {e}", file=sys.stderr)
    sys.exit(1)
PY
fi

# Enable MPS fallback when running on macOS
if [ "$(uname -s)" = "Darwin" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

"$PY" scripts/startup.py --one-click
