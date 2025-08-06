#!/usr/bin/env bash
set -euo pipefail

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode command line tools..."
    xcode-select --install
    exit 1  # prompt user to re-run after installation
fi

# Change to repository root
cd "$(dirname "$0")/.."

# Install Homebrew if not installed
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ -d /opt/homebrew/bin ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -d /usr/local/bin ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi

# Update Homebrew and install packages
brew update
brew install python@3.11 rustup-init pkg-config cmake protobuf

# Install Rust toolchain if missing
if ! command -v rustup >/dev/null 2>&1; then
  rustup-init -y
  source "$HOME/.cargo/env"
fi

# Optionally upgrade pip for Python 3.11
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m pip install --upgrade pip || true
  python3.11 -m pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/metal
fi

