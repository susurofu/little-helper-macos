#!/bin/sh
# Finder-friendly macOS entry point.
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$PROJECT_DIR/start.sh"
launcher_exit=$?
if [ "$launcher_exit" -ne 0 ]; then
    printf '\nPress Enter to close…'
    read -r ignored
fi
exit "$launcher_exit"
