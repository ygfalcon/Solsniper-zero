#!/usr/bin/env bash
# macOS helper that delegates to ``scripts/launcher.py``.  All environment
# preparation is performed by ``solhunter_zero.startup.prepare_environment``.
set -euo pipefail
cd "$(dirname "$0")"
exec python scripts/launcher.py --one-click --full-deps "$@"

