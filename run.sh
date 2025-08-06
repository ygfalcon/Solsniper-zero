#!/usr/bin/env bash
set -e

# Parse CLI flags
SKIP_BUILD=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--skip-build" ]; then
        SKIP_BUILD=true
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

export DEPTH_SERVICE=${DEPTH_SERVICE:-true}

# Ensure FAISS indexes are moved to GPU memory when available
export GPU_MEMORY_INDEX="1"

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

if [ "$SKIP_BUILD" != true ]; then
    if [ ! -f solhunter_zero/libroute_ffi.so ]; then
        if ! cargo build --manifest-path route_ffi/Cargo.toml --release --features=parallel; then
            echo "Error: Failed to build libroute_ffi.so. Ensure the Rust toolchain is installed or run with --skip-build if you have a precompiled binary." >&2
            exit 1
        fi
        if ! cp route_ffi/target/release/libroute_ffi.so solhunter_zero/ 2>/dev/null; then
            echo "Error: libroute_ffi.so was not copied to solhunter_zero." >&2
            exit 1
        fi
    fi

    if ! cargo build --manifest-path depth_service/Cargo.toml --release; then
        echo "Error: Failed to build depth_service. Use --skip-build to bypass compilation if binaries are available." >&2
        exit 1
    fi
else
    if [ ! -f solhunter_zero/libroute_ffi.so ]; then
        echo "Error: --skip-build specified but solhunter_zero/libroute_ffi.so not found." >&2
        exit 1
    fi
fi

python -m solhunter_zero.metrics_aggregator &
AGG_PID=$!
trap 'kill $AGG_PID 2>/dev/null' EXIT

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
