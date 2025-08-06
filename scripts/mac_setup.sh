#!/usr/bin/env bash
set -euo pipefail

NON_INTERACTIVE=0
for arg in "$@"; do
    if [[ "$arg" == "--non-interactive" ]]; then
        NON_INTERACTIVE=1
        break
    fi
done

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode command line tools..."
    xcode-select --install
    elapsed=0
    timeout=300
    interval=10
    until xcode-select -p >/dev/null 2>&1; do
        if (( NON_INTERACTIVE )); then
            if (( elapsed >= timeout )); then
                echo "Command line tools installation timed out after ${timeout}s." >&2
                exit 1
            fi
            sleep "$interval"
            elapsed=$((elapsed + interval))
        else
            read -p "Command line tools not yet installed. Press Enter to re-check or type 'c' to cancel: " ans
            if [[ "${ans}" == "c" || "${ans}" == "C" ]]; then
                echo "Please re-run this script after the tools are installed."
                exit 1
            fi
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
fi

# Refresh Homebrew environment for the current shell.
# This is idempotent and safe even if Homebrew was already installed.
eval "$(brew shellenv)"

# Update Homebrew and install packages
brew update
brew install python@3.11 rustup-init pkg-config cmake protobuf

# Install Rust toolchain if missing
if ! command -v rustup >/dev/null 2>&1; then
  rustup-init -y
  source "$HOME/.cargo/env"
fi

# Verify required tools are on PATH
missing=()
for cmd in python3.11 pip3.11 rustup; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing+=("$cmd")
  fi
done
if ((${#missing[@]})); then
  brew_prefix="$(brew --prefix)"
  echo "Missing ${missing[*]} on PATH. Ensure ${brew_prefix}/bin is in your PATH and re-run this script."
  exit 1
fi

# Persist Homebrew environment for future shells
if ! grep -Fq 'HOMEBREW_PREFIX' "$PROFILE_FILE" 2>/dev/null; then
  brew shellenv >> "$PROFILE_FILE"
  echo "Updated $PROFILE_FILE with Homebrew environment."
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

# Verify required tools are installed
missing=()
for tool in python3.11 brew rustup; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing+=("$tool")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Error: Missing required tool(s): ${missing[*]}" >&2
  exit 1
fi

