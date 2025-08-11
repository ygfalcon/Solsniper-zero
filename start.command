#!/bin/sh
set -euo pipefail
cd "$(dirname "$0")" || exit 1
exec /usr/bin/env python3 -m solhunter_zero.launcher "$@" || exit 1
