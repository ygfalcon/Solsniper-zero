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
    pip install .
fi

if [ "$#" -eq 0 ] || [ "$1" = "--auto" ]; then
    if [ "$1" = "--auto" ]; then
        shift
    fi
    python -m solhunter_zero.main --auto "$@"
else
    python -m solhunter_zero.main "$@"
fi
