#!/bin/sh
# Install missing OS prerequisites, then prepare the project environment.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
admin() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}
python_ok() {
    "$1" -c 'import sys, venv, ensurepip; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 15)))' >/dev/null 2>&1
}
case "$(uname -s)" in
    Darwin)
        mac_major=$(sw_vers -productVersion | cut -d . -f 1)
        if [ "$mac_major" -lt 14 ]; then
            echo "Ollama requires macOS 14 or newer." >&2
            exit 1
        fi
        if [ "$(id -u)" -eq 0 ]; then
            echo "Run this installer as your normal user, without sudo." >&2
            exit 1
        fi
        python_executable=${LOCAL_HELPER_PYTHON:-python3}
        if [ -n "${LOCAL_HELPER_PYTHON:-}" ] && ! python_ok "$python_executable"; then
            echo "LOCAL_HELPER_PYTHON must point to Python 3.10–3.14 with venv and ensurepip." >&2
            exit 1
        fi
        if ! command -v brew >/dev/null 2>&1; then
            echo "Installing Homebrew; its installer may ask for your password."
            installer=$(mktemp)
            trap 'rm -f "$installer"' EXIT HUP INT TERM
            curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh -o "$installer"
            /bin/bash "$installer"
            rm -f "$installer"
            trap - EXIT HUP INT TERM
        fi
        # Make a new Homebrew installation available in this same session.
        eval "$(brew shellenv)"
        if ! python_ok "$python_executable"; then
            if ! brew list python@3.12 >/dev/null 2>&1; then brew install python@3.12; fi
            python_executable="$(brew --prefix python@3.12)/bin/python3.12"
        fi
        # The Ollama server is a global Homebrew executable, never a pip package.
        # Also recognize the official application before installing another copy.
        if ! command -v ollama >/dev/null 2>&1 && [ -x /Applications/Ollama.app/Contents/Resources/ollama ]; then
            export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"
        fi
        if ! command -v ollama >/dev/null 2>&1; then brew install ollama; fi
        ;;
    Linux)
        python_executable=${LOCAL_HELPER_PYTHON:-python3}
        if ! python_ok "$python_executable" || ! command -v curl >/dev/null 2>&1 ||
            ! "$python_executable" -c 'import ensurepip' >/dev/null 2>&1; then
            if command -v apt-get >/dev/null 2>&1; then
                admin apt-get update
                admin apt-get install -y python3 python3-venv python3-pip curl ca-certificates
            elif command -v dnf >/dev/null 2>&1; then
                admin dnf install -y python3 python3-pip curl ca-certificates
            elif command -v pacman >/dev/null 2>&1; then
                admin pacman -S --needed --noconfirm python python-pip curl ca-certificates
            else
                echo "Install Python 3.10–3.14 with venv/pip, curl and CA certificates, then rerun." >&2
                exit 1
            fi
        fi
        if ! command -v ollama >/dev/null 2>&1; then
            installer=$(mktemp)
            curl -fsSL https://ollama.com/install.sh -o "$installer"
            sh "$installer"
            rm -f "$installer"
        fi
        ;;
    *) echo "Use install.bat on Windows. This script supports macOS and Linux." >&2; exit 1 ;;
esac
if ! python_ok "$python_executable"; then
    echo "Set LOCAL_HELPER_PYTHON to a Python 3.10–3.14 executable and rerun." >&2
    exit 1
fi
exec "$python_executable" "$PROJECT_DIR/scripts/install_python.py" "$@"
