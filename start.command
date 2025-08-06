#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python scripts/launcher.py --one-click --full-deps "$@"

