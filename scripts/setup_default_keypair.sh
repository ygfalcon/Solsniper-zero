#!/usr/bin/env bash
set -e

MNEMONIC="${MNEMONIC:-}"
PASSPHRASE="${PASSPHRASE:-}"

if [ -z "$MNEMONIC" ]; then
  echo "MNEMONIC environment variable is required" >&2
  exit 1
fi

solhunter-wallet derive default "$MNEMONIC" --passphrase "$PASSPHRASE"
solhunter-wallet select default
