#!/usr/bin/env bash
# Deprecated macOS launcher.
# Use "python start.py" or the "solhunter-start" console script instead.

echo "start.command is deprecated; invoking Python launcher..." >&2
exec python start.py "$@"

