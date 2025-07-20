#!/usr/bin/env bash
set -e

check_deps() {
python - <<'PY'
import pkgutil, re, sys
missing=False
with open('requirements.txt') as f:
    for line in f:
        pkg=line.strip()
        if not pkg or pkg.startswith('#'):
            continue
        mod=re.split('[<=>]', pkg)[0].replace('-', '_')
        if pkgutil.find_loader(mod) is None:
            missing=True
            break
sys.exit(1 if missing else 0)
PY
}

if ! check_deps; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

if [ "$1" = "--auto" ]; then
    shift
    python -m solhunter_zero.main --auto "$@"
else
    python -m solhunter_zero.main "$@"
fi
