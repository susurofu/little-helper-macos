#!/bin/sh
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$PROJECT_DIR/install.sh" "$@"
installer_exit=$?
printf '\nPress Enter to close…'
read -r ignored
exit "$installer_exit"
