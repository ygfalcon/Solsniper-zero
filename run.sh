#!/usr/bin/env bash
# Requires Python 3.11 or higher.

set -e

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("Error: Python 3.11 or higher is required.", file=sys.stderr)
    sys.exit(1)
PY

export DEPTH_SERVICE=${DEPTH_SERVICE:-true}

# Detect if a GPU is present before moving FAISS indexes to GPU memory
has_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        return 0
    elif [ -d /proc/driver/nvidia ]; then
        return 0
    else
        return 1
    fi
}

if has_gpu; then
    export GPU_MEMORY_INDEX="1"
else
    echo "No GPU detected; using CPU mode"
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
print(json.dumps(missing_optional))
sys.exit(1 if missing_required or missing_optional else 0)
PY
}

missing_opt_json=$(check_deps)
if [ $? -ne 0 ]; then
    missing_opt=$(python - "$missing_opt_json" <<'PY'
import json,sys
print(' '.join(json.loads(sys.argv[1])))
PY
)
    missing_extras=$(python - "$missing_opt_json" <<'PY'
import json,sys
mods=set(json.loads(sys.argv[1]))
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

if ! command -v cargo >/dev/null 2>&1; then
    echo "Installing Rust toolchain via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
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

if [ "$1" = "--daemon" ]; then
    shift
    python -m solhunter_zero.train_cli --daemon "$@"
elif [ "$#" -eq 0 ] || [ "$1" = "--auto" ]; then
    if [ "$1" = "--auto" ]; then
        shift
    fi
    python -m solhunter_zero.main --auto "$@"
else
    python -m solhunter_zero.main "$@"
fi
