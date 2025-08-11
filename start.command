#!/bin/sh
set -euo pipefail
cd "$(dirname "$0")" || exit 1
exec /usr/bin/env python3 scripts/setup_one_click.py "$@" || exit 1
