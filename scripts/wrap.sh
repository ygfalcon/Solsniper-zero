#!/usr/bin/env bash
# Generic macOS launcher wrapper.
# Usage: source scripts/wrap.sh <target> [args...]

set -e

TARGET="$1"
shift

CALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
cd "$CALLER_DIR"

exec "./$TARGET" "$@"
