#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python scripts/startup.py --auto "$@"

