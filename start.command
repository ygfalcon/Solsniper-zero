#!/usr/bin/env bash
set -euo pipefail
# macOS launcher wrapper.

cd "$(dirname "$0")" || exit 1

exec arch -arm64 python3 -m solhunter_zero.launcher "$@" || {
  echo "Error: failed to start using 'arch -arm64'. Are you running this script in the correct CPU mode?" >&2
  exit 1
}

