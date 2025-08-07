#!/usr/bin/env bash
# macOS launcher wrapper for the one-click trading demo

cd "$(dirname "$0")"
exec solhunter-one-click "$@"
