#!/usr/bin/env bash
set -euo pipefail
exec python scripts/startup.py --one-click --full-deps "$@"

