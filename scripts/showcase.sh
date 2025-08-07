#!/usr/bin/env bash
# Run the investor showcase module.

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
exec python -m solhunter_zero.investor_showcase "$@"
