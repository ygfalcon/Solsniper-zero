#!/usr/bin/env bash
cd "$(dirname "$0")"
exec arch -arm64 python3 start.py "$@"
