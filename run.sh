#!/usr/bin/env bash
set -euo pipefail
exec python scripts/launcher.py --one-click --full-deps "$@"

