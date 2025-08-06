#!/usr/bin/env bash
# Deprecated launcher retained for backwards compatibility.
# Use "python start.py" or the "solhunter-start" console script instead.

echo "run.sh is deprecated; invoking Python launcher..." >&2
exec python start.py "$@"

