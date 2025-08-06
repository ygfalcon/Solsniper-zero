#!/usr/bin/env bash
# Requires Python 3.11 or higher.

set -euo pipefail

if [ ! -f "config.toml" ]; then
    cp config.example.toml config.toml
    echo "Created default config.toml from config.example.toml"
fi

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("Error: Python 3.11 or higher is required.", file=sys.stderr)
    sys.exit(1)
PY

export DEPTH_SERVICE=${DEPTH_SERVICE:-true}

# Detect if a GPU is present before moving FAISS indexes to GPU memory
if python -m solhunter_zero.device --check-gpu >/dev/null 2>&1; then
    export GPU_MEMORY_INDEX="${GPU_MEMORY_INDEX:-1}"
else
    echo "No GPU detected; using CPU mode"
fi

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
        export RAYON_NUM_THREADS="$(python - <<'EOF'
import os
print(os.cpu_count() or 1)
EOF
)"
    fi
fi

check_deps() {
  python - <<'PY'
import pkgutil, re, sys, tomllib, json
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
missing_required = []
for dep in deps:
    mod = re.split('[<=>]', dep)[0].replace('-', '_')
    if pkgutil.find_loader(mod) is None:
        missing_required.append(mod)
optional = ['faiss', 'sentence_transformers', 'torch', 'orjson', 'lz4', 'zstandard', 'msgpack']
missing_optional = [m for m in optional if pkgutil.find_loader(m) is None]
print(json.dumps({"required": missing_required, "optional": missing_optional}))
sys.exit(1 if missing_required or missing_optional else 0)
PY
}

set +e
missing_opt_json=$(check_deps)
deps_status=$?
set -e
if [ $deps_status -ne 0 ]; then
    missing_opt=$(python - "$missing_opt_json" <<'PY'
import json,sys
print(' '.join(json.loads(sys.argv[1]).get("optional", [])))
PY
)
    missing_extras=$(python - "$missing_opt_json" <<'PY'
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
        python - <<'EOF'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('torch') else 1)
EOF
        if [ $? -ne 0 ]; then
            pip install torch==2.1.0 torchvision==0.16.0 \
              --extra-index-url https://download.pytorch.org/whl/metal
        fi
    fi
    if [ -n "$missing_extras" ]; then
        pip install ".[${missing_extras}]"
    else
        pip install .
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
    python - "$post_check_json" <<'PY'
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

if [ -n "${SOLANA_RPC_URL:-}" ]; then
    python - <<'PY'
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
    python -m solhunter_zero.metrics_aggregator >"$METRICS_LOG" 2>&1 &
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
    python -m solhunter_zero.train_cli --daemon "$@"
elif [ "$first_arg" = "--start-all" ] || { [ "$#" -eq 0 ] && [ "$uname_s" = "Darwin" ]; }; then
    if [ "$first_arg" = "--start-all" ]; then
        shift
    fi
    python scripts/start_all.py autopilot
elif [ "$#" -eq 0 ] || [ "$first_arg" = "--auto" ]; then
    if [ "$first_arg" = "--auto" ]; then
        shift
    fi
    python -m solhunter_zero.main --auto "$@"
else
    python -m solhunter_zero.main "$@"
fi
