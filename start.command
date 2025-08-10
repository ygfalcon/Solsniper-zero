#!/usr/bin/env bash
set -euo pipefail
# macOS launcher wrapper.

cd "$(dirname "$0")" || exit 1
exec ./start.py "$@" || exit 1

