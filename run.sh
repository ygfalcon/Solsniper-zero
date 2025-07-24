#!/usr/bin/env bash
set -e

check_deps() {
python - <<'PY'
import pkgutil, re, sys, tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
missing = False
for dep in deps:
    mod = re.split('[<=>]', dep)[0].replace('-', '_')
    if pkgutil.find_loader(mod) is None:
        missing = True
        break
sys.exit(1 if missing else 0)
PY
}

if ! check_deps; then
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
fi

if [ "${DEPTH_SERVICE,,}" = "true" ]; then
    if ! command -v cargo >/dev/null 2>&1; then
        echo "Error: DEPTH_SERVICE requested but 'cargo' is not installed." >&2
        echo "Install Rust from https://www.rust-lang.org/tools/install" >&2
        exit 1
    fi
    cargo build --manifest-path depth_service/Cargo.toml --release
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
