#!/usr/bin/env bash
# Cross-platform launcher wrapper for the one-click trading demo

cd "$(dirname "$0")"
exec solhunter-one-click "$@"
