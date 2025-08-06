#!/usr/bin/env bash
set -euo pipefail

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode command line tools..."
    xcode-select --install
    until xcode-select -p >/dev/null 2>&1; do
        read -p "Command line tools not yet installed. Press Enter to re-check or type 'c' to cancel: " ans
        if [[ "${ans}" == "c" || "${ans}" == "C" ]]; then
            echo "Please re-run this script after the tools are installed."
            exit 1
        fi
    done
fi

# Change to repository root
cd "$(dirname "$0")/.."

# Determine shell profile file
if [[ "$SHELL" == */zsh ]]; then
  PROFILE_FILE="$HOME/.zprofile"
else
  PROFILE_FILE="$HOME/.bash_profile"
fi

# Install Homebrew if not installed
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ -d /opt/homebrew/bin ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -d /usr/local/bin ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
  if [[ $SHELL == *bash* ]]; then
    profile_file="$HOME/.bash_profile"
  else
    profile_file="$HOME/.zprofile"
  fi
  brew shellenv >> "$profile_file"
  echo "Updated $profile_file with Homebrew environment."
fi

# Persist Homebrew environment for future shells
if ! grep -Fq 'HOMEBREW_PREFIX' "$PROFILE_FILE" 2>/dev/null; then
  brew shellenv >> "$PROFILE_FILE"
fi

# Update Homebrew and install packages
brew update
brew install python@3.11 rustup-init pkg-config cmake protobuf

# Install Rust toolchain if missing
if ! command -v rustup >/dev/null 2>&1; then
  rustup-init -y
  source "$HOME/.cargo/env"
fi

# Make Rust tools available in future shells
if ! grep -Fq 'source "$HOME/.cargo/env"' "$PROFILE_FILE" 2>/dev/null; then
  echo 'source "$HOME/.cargo/env"' >> "$PROFILE_FILE"
fi

# Optionally upgrade pip for Python 3.11
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m pip install --upgrade pip || true
  python3.11 -m pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/metal
fi

