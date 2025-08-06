#!/usr/bin/env bash
# Thin wrapper around ``scripts/launcher.py`` which now handles all environment
# preparation via ``solhunter_zero.startup.prepare_environment``.
set -euo pipefail
exec python scripts/launcher.py --auto "$@"

