#!/usr/bin/env bash
# Build Jamie's EventKit Reminders helper and install it where Jamie looks for it.
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
  echo "The Reminders helper builds only on macOS." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
destination="${1:-$HOME/Library/Application Support/Jamie/bin}"

cd "$root/native/reminders-helper"
swift build -c release
mkdir -p "$destination"
install -m 0755 ".build/release/reminders-helper" "$destination/reminders-helper"
echo "Installed $destination/reminders-helper"
echo "The first reminders request will ask for Reminders access."
