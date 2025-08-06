#!/usr/bin/env bash
# Requires Python 3.11 or higher.

set -euo pipefail

PY=$(command -v python3 || command -v python)

if [ ! -f "config.toml" ]; then
    cp config.example.toml config.toml
    echo "Created default config.toml from config.example.toml"
fi

"$PY" - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("Error: Python 3.11 or higher is required.", file=sys.stderr)
    sys.exit(1)
PY

export DEPTH_SERVICE=${DEPTH_SERVICE:-true}

# On macOS, enable PyTorch's MPS fallback so unsupported ops run on the CPU
if [ "$(uname -s)" = "Darwin" ]; then
    # PYTORCH_ENABLE_MPS_FALLBACK=1 allows PyTorch to fall back to CPU for
    # operations that are not implemented for Apple's MPS backend
    export PYTORCH_ENABLE_MPS_FALLBACK=1
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

# Ensure Python dependencies are installed
"$PY" scripts/deps.py --install

# Detect if a GPU is present before moving FAISS indexes to GPU memory
if "$PY" -m solhunter_zero.device --check-gpu >/dev/null 2>&1; then
    export GPU_MEMORY_INDEX="${GPU_MEMORY_INDEX:-1}"
else
    echo "No GPU detected; using CPU mode"
fi

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

if ! command -v cargo >/dev/null 2>&1; then
    echo "Installing Rust toolchain via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh ||
        { echo "rustup download failed" >&2; exit 1; }
    sh /tmp/rustup.sh -y
    rm /tmp/rustup.sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: 'cargo' is not installed." >&2
    exit 1
fi

# Determine expected native library name based on platform
uname_s=$(uname -s)
uname_m=$(uname -m)
case "$uname_s" in
    Darwin*) libfile="libroute_ffi.dylib" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT*) libfile="route_ffi.dll" ;;
    *) libfile="libroute_ffi.so" ;;
esac

# Build and copy the library if it is not already present
if [ ! -f "solhunter_zero/$libfile" ]; then
    cargo build --manifest-path route_ffi/Cargo.toml --release --features=parallel
    cp "route_ffi/target/release/$libfile" solhunter_zero/ 2>/dev/null
    if [ ! -f "solhunter_zero/$libfile" ]; then
        echo "Error: $libfile was not copied to solhunter_zero." >&2
        if [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
            echo "Try building for macOS arm64 with:" >&2
            echo "  cargo build --manifest-path route_ffi/Cargo.toml --release --target aarch64-apple-darwin" >&2
        fi
        exit 1
    fi
fi

cargo build --manifest-path depth_service/Cargo.toml --release

# Allow skipping the metrics aggregator for debugging
NO_METRICS=0
args=()
for arg in "$@"; do
    if [ "$arg" = "--no-metrics" ]; then
        NO_METRICS=1
    else
        args+=("$arg")
    fi
done
set -- "${args[@]}"

if [ $NO_METRICS -eq 0 ]; then
    METRICS_LOG=$(mktemp)
    "$PY" -m solhunter_zero.metrics_aggregator >"$METRICS_LOG" 2>&1 &
    AGG_PID=$!
    sleep 1
    if ! kill -0 "$AGG_PID" 2>/dev/null; then
        echo "metrics_aggregator failed to start" >&2
        cat "$METRICS_LOG"
        rm -f "$METRICS_LOG"
        exit 1
    fi
    trap 'kill $AGG_PID 2>/dev/null; rm -f "$METRICS_LOG"' EXIT
fi

first_arg="${1-}"

if [ "$first_arg" = "--daemon" ]; then
    shift
    "$PY" -m solhunter_zero.train_cli --daemon "$@"
elif [ "$first_arg" = "--start-all" ] || { [ "$#" -eq 0 ] && [ "$uname_s" = "Darwin" ]; }; then
    if [ "$first_arg" = "--start-all" ]; then
        shift
    fi
    "$PY" scripts/start_all.py autopilot
elif [ "$#" -eq 0 ] || [ "$first_arg" = "--auto" ]; then
    if [ "$first_arg" = "--auto" ]; then
        shift
    fi
    "$PY" -m solhunter_zero.main --auto "$@"
else
    "$PY" -m solhunter_zero.main "$@"
fi
