#!/usr/bin/env bash
set -e

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
    pip install .
    if [ -n "$missing_opt" ]; then
        echo "Installed optional modules: $missing_opt"
    fi
fi

if [ "${DEPTH_SERVICE,,}" = "true" ]; then
    if ! command -v cargo >/dev/null 2>&1; then
        # DEPTH_SERVICE requested
        echo "Warning: DEPTH_SERVICE requested but 'cargo' is not installed." >&2
        echo "Disabling DEPTH_SERVICE" >&2
        DEPTH_SERVICE=false
    else
        cargo build --manifest-path depth_service/Cargo.toml --release
    fi
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
