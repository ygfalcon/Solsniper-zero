#!/usr/bin/env bash
set -e

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
optional = ['faiss', 'sentence_transformers', 'torch']
missing_optional = [m for m in optional if pkgutil.find_loader(m) is None]
print(json.dumps({'required': missing_required, 'optional': missing_optional}))
sys.exit(0 if not missing_required else 1)
PY
}

set +e
deps_json=$(check_deps)
status=$?
set -e
if [ $status -ne 0 ]; then
    missing_req=$(python - "$deps_json" <<'PY'
import json,sys
print(' '.join(json.loads(sys.argv[1])['required']))
PY
)
    echo "Error: missing required Python modules: $missing_req" >&2
    echo "Install them manually, e.g. 'pip install .', then re-run." >&2
    exit 1
else
    missing_opt=$(python - "$deps_json" <<'PY'
import json,sys
print(' '.join(json.loads(sys.argv[1])['optional']))
PY
)
    if [ -n "$missing_opt" ]; then
        echo "Warning: optional Python modules not found: $missing_opt" >&2
        echo "Install them manually for additional features." >&2
    fi
fi

if [ ! -f solhunter_zero/libroute_ffi.so ]; then
    echo "Error: missing required Rust library 'solhunter_zero/libroute_ffi.so'." >&2
    echo "Install Rust from https://www.rust-lang.org/tools/install and build manually:" >&2
    echo "  cargo build --manifest-path route_ffi/Cargo.toml --release --features=parallel" >&2
    echo "  cp route_ffi/target/release/libroute_ffi.so solhunter_zero/" >&2
    exit 1
fi

if [ "${DEPTH_SERVICE,,}" = "true" ]; then
    if [ ! -x depth_service/target/release/depth_service ]; then
        echo "Error: DEPTH_SERVICE requested but depth service binary is missing." >&2
        echo "Install Rust from https://www.rust-lang.org/tools/install and build manually:" >&2
        echo "  cargo build --manifest-path depth_service/Cargo.toml --release" >&2
        exit 1
    fi
fi

RUN_AGG=false
args=()
for arg in "$@"; do
    if [ "$arg" = "--with-aggregator" ]; then
        RUN_AGG=true
    else
        args+=("$arg")
    fi
done
if $RUN_AGG; then
    python -m solhunter_zero.metrics_aggregator &
    AGG_PID=$!
    trap 'kill $AGG_PID 2>/dev/null' EXIT
fi
set -- "${args[@]}"

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
