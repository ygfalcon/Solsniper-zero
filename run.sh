#!/usr/bin/env bash
# Shell entry point for SolHunter Zero.
#
# This script simply delegates to ``start.py`` so that all platforms
# share the same Python-based launcher.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/start.py" "$@"

