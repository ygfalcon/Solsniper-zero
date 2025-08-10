#!/usr/bin/env bash
# macOS launcher wrapper.

cd "$(dirname "$0")"
exec ./start.py "$@"

