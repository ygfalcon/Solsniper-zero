#!/bin/sh
set -eu
# Launcher wrapper. Interpreter validation is delegated to
# solhunter_zero.python_env.

cd "$(dirname "$0")" || exit 1

exec /usr/bin/env python3.11 -m solhunter_zero.launcher "$@" \
  || exec /usr/bin/env python3 -m solhunter_zero.launcher "$@" \
  || exit 1
