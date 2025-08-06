#!/usr/bin/env bash
set -euo pipefail

source scripts/rotate_logs.sh
rotate_logs
exec > >(tee -a startup.log) 2>&1
exec python start.py --auto "$@"

