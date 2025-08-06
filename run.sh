#!/usr/bin/env bash

# Launch Solhunter Zero via the bootstrap module in one-click mode
set -euo pipefail
exec python -m solhunter_zero.bootstrap --one-click "$@"

