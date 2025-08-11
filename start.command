#!/usr/bin/env bash
set -euo pipefail
# macOS launcher wrapper.

cd "$(dirname "$0")" || exit 1
exec python3 -m solhunter_zero.launcher "$@"

