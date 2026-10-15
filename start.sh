#!/bin/sh
# macOS / Linux: prefer the project venv, then the active/system Python.
set -eu
export PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/Ollama.app/Contents/Resources:$PATH"
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${LOCAL_HELPER_PYTHON:-}" ]; then
    exec "$LOCAL_HELPER_PYTHON" "$PROJECT_DIR/scripts/start_local.py"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/start_local.py"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$PROJECT_DIR/scripts/start_local.py"
else
    echo "Python 3 is missing. Install Python or set LOCAL_HELPER_PYTHON to its executable." >&2
    exit 1
fi
