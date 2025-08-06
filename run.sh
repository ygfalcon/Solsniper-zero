#!/usr/bin/env bash
# Requires Python 3.11 or higher.

set -euo pipefail

# Rotate logs before redirecting output
source scripts/rotate_logs.sh
rotate_logs
exec > >(tee -a startup.log) 2>&1

PY=$(command -v python3 || command -v python)

"$PY" - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("Error: Python 3.11 or higher is required.", file=sys.stderr)
    sys.exit(1)
PY

"$PY" - <<'PY'
from scripts.startup import ensure_config
ensure_config()
PY

export DEPTH_SERVICE=${DEPTH_SERVICE:-true}

# Configure GPU-related environment variables
eval "$("$PY" -m solhunter_zero.device --setup-env)"

# Configure Rayon thread pool if not already set
if [ -z "${RAYON_NUM_THREADS:-}" ]; then
    export RAYON_NUM_THREADS="$("$PY" -m scripts.threading)"
fi

check_deps() {
    "$PY" scripts/deps.py "$@"
}

run_cargo_build() {
    local output status
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        if command -v rustup >/dev/null 2>&1; then
            if ! rustup target list --installed 2>/dev/null | grep -q '^aarch64-apple-darwin$'; then
                rustup target add aarch64-apple-darwin
            fi
        fi
    fi
    set +e
    output=$(cargo build "$@" 2>&1)
    status=$?
    set -e
    if [ $status -ne 0 ]; then
        echo "$output" >&2
        if echo "$output" | grep -qi 'aarch64-apple-darwin'; then
            echo "Hint: run 'rustup target add aarch64-apple-darwin'." >&2
        fi
        if echo "$output" | grep -qi 'linker.*not found'; then
            echo "Hint: ensure the Xcode command line tools are installed (xcode-select --install)." >&2
        fi
    fi
    return $status
}

set +e
missing_opt_json=$(check_deps)
deps_status=$?
set -e
if [ $deps_status -ne 0 ]; then
    missing_opt=$("$PY" - "$missing_opt_json" <<'PY'
import json,sys
print(' '.join(json.loads(sys.argv[1]).get("optional", [])))
PY
)
    missing_extras=$("$PY" - "$missing_opt_json" <<'PY'
import json,sys
mods=set(json.loads(sys.argv[1]).get("optional", []))
extras=[]
if 'orjson' in mods:
    extras.append('fastjson')
if {'lz4','zstandard'} & mods:
    extras.append('fastcompress')
if 'msgpack' in mods:
    extras.append('msgpack')
print(','.join(extras))
PY
)
    echo "Installing dependencies..."
    command -v pip >/dev/null 2>&1 || { echo "pip not found" >&2; exit 1; }
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        "$PY" - <<'EOF'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('torch') else 1)
EOF
        if [ $? -ne 0 ]; then
            "$PY" -m pip install torch==2.1.0 torchvision==0.16.0 \
              --extra-index-url https://download.pytorch.org/whl/metal
        fi
        "$PY" - <<'EOF'
import torch, sys
sys.exit(0 if torch.backends.mps.is_available() else 1)
EOF
        if [ $? -ne 0 ]; then
            echo "MPS unavailable. Reinstalling Metal wheel..."
            "$PY" -m pip install --force-reinstall torch==2.1.0 torchvision==0.16.0 \
              --extra-index-url https://download.pytorch.org/whl/metal
            "$PY" - <<'EOF'
import torch, sys
sys.exit(0 if torch.backends.mps.is_available() else 1)
EOF
            if [ $? -ne 0 ]; then
                echo "Error: MPS backend remains unavailable" >&2
                exit 1
            fi
        fi
    fi
    if [ -n "$missing_extras" ]; then
        "$PY" -m pip install ".[${missing_extras}]"
    else
        "$PY" -m pip install .
    fi
    if [ -n "$missing_opt" ]; then
        echo "Installed optional modules: $missing_opt"
    fi
fi

set +e
post_check_json=$(check_deps)
post_check_status=$?
set -e
if [ $post_check_status -ne 0 ]; then
    "$PY" - "$post_check_json" <<'PY'
import json,sys
data=json.loads(sys.argv[1])
req=data.get("required", [])
opt=data.get("optional", [])
if req:
    print("Missing required modules: " + ' '.join(req))
if opt:
    print("Missing optional modules: " + ' '.join(opt))
PY
    exit 1
fi

if [ "$(uname -s)" = "Darwin" ]; then
    if "$PY" - <<'EOF'
import torch, sys
sys.exit(0 if torch.backends.mps.is_available() else 1)
EOF
    then
        export SOLHUNTER_MPS=1
    else
        export SOLHUNTER_MPS=0
    fi
else
    export SOLHUNTER_MPS=0
fi

if "$PY" -m solhunter_zero.device --check-gpu >/dev/null 2>&1; then
    [ "$(uname -s)" = "Darwin" ] && export TORCH_DEVICE="mps"
    export GPU_MEMORY_INDEX="${GPU_MEMORY_INDEX:-1}"
fi

if [ -n "${SOLANA_RPC_URL:-}" ]; then
    "$PY" - <<'PY'
import os, sys, urllib.request, time
url = os.environ["SOLANA_RPC_URL"]
if url.startswith("ws://"):
    url = "http://" + url[5:]
elif url.startswith("wss://"):
    url = "https://" + url[6:]
req = urllib.request.Request(url, method="HEAD")
for attempt in range(3):
    try:
        urllib.request.urlopen(req, timeout=5)
        break
    except Exception as e:
        if attempt == 2:
            print(
                f"Error: SOLANA_RPC_URL '{os.environ['SOLANA_RPC_URL']}' is unreachable: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        wait = 2**attempt
        print(
            f"Attempt {attempt + 1} failed to reach SOLANA_RPC_URL at {url}: {e}.",
            f" Retrying in {wait} seconds...",
            file=sys.stderr,
        )
        time.sleep(wait)
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
    run_cargo_build --manifest-path route_ffi/Cargo.toml --release --features=parallel
    cp "route_ffi/target/release/$libfile" solhunter_zero/ 2>/dev/null
    if [ ! -f "solhunter_zero/$libfile" ] && [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
        # The initial build might emit the library under the target triple directory
        cp "route_ffi/target/aarch64-apple-darwin/release/$libfile" solhunter_zero/ 2>/dev/null
        if [ ! -f "solhunter_zero/$libfile" ]; then
            echo "Rebuilding for aarch64-apple-darwin..."
            run_cargo_build --manifest-path route_ffi/Cargo.toml --release --features=parallel --target aarch64-apple-darwin
            cp "route_ffi/target/aarch64-apple-darwin/release/$libfile" solhunter_zero/ 2>/dev/null
        fi
    fi
    if [ ! -f "solhunter_zero/$libfile" ]; then
        echo "Error: $libfile was not copied to solhunter_zero." >&2
        if [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
            echo "Try building for macOS arm64 with:" >&2
            echo "  cargo build --manifest-path route_ffi/Cargo.toml --release --target aarch64-apple-darwin" >&2
        fi
        exit 1
    fi
    if [ "$uname_s" = "Darwin" ]; then
        set +e
        codesign --force --sign - solhunter_zero/libroute_ffi.dylib
        codesign_status=$?
        set -e
        if [ $codesign_status -ne 0 ]; then
            echo "Error: failed to codesign libroute_ffi.dylib. Ensure the Xcode command line tools are installed and run 'codesign --force --sign - solhunter_zero/libroute_ffi.dylib'." >&2
            exit 1
        fi
    fi
fi

run_cargo_build --manifest-path depth_service/Cargo.toml --release

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
