#!/bin/bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: macos/build-app.sh [DESTINATION]' \
    '' \
    'Build Jamie.app around this checkout and its .venv, by default in ~/Applications.' \
    'The app runs the code in place: rebuild only after moving the checkout or .venv.'
}

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
python="$root/.venv/bin/python"
destination=${1:-$HOME/Applications/Jamie.app}
bundle_id=me.jameshliu.jamie

if [[ ! -x $python ]]; then
  printf 'Error: %s is missing. Create the virtual environment first (see README.md).\n' \
    "$python" >&2
  exit 1
fi
if ! "$python" -c 'import assistant, webview' 2>/dev/null; then
  printf '%s\n' "Error: install Jamie into .venv first: $python -m pip install -e ." >&2
  exit 1
fi
if [[ -e $destination ]]; then
  existing=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' \
    "$destination/Contents/Info.plist" 2>/dev/null || true)
  if [[ $existing != "$bundle_id" ]]; then
    printf 'Error: %s exists and is not a Jamie build; not replacing it.\n' "$destination" >&2
    exit 1
  fi
fi

version=$("$python" -c 'import assistant; print(assistant.__version__)')
base=$("$python" -c 'import sys; print(sys.base_prefix)')
config="$base/bin/python3-config"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
app="$work/Jamie.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$work/Jamie.iconset"

# The executable embeds this Python and starts it with the checkout's .venv.
# shellcheck disable=SC2046 # python3-config prints several flags
clang -O2 -Wall -o "$app/Contents/MacOS/Jamie" "$root/macos/launcher.c" \
  -DJAMIE_PYTHON="\"$python\"" \
  $("$config" --cflags) $("$config" --embed --ldflags)

sed "s/@VERSION@/$version/g" "$root/macos/Info.plist" > "$app/Contents/Info.plist"

icons=()
for size in 16 32 128 256 512; do
  icons+=("$work/Jamie.iconset/icon_${size}x${size}.png" "$size")
  icons+=("$work/Jamie.iconset/icon_${size}x${size}@2x.png" $((size * 2)))
done
"$python" "$root/macos/render_icon.py" "${icons[@]}"
iconutil -c icns "$work/Jamie.iconset" -o "$app/Contents/Resources/Jamie.icns"

codesign --force --sign - "$app" >/dev/null
mkdir -p "$(dirname "$destination")"
rm -rf "$destination"
mv "$app" "$destination"
# Ask Launch Services to pick up the new icon and name.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
  -f "$destination" >/dev/null 2>&1 || true
printf 'Built %s\nOpen it from Spotlight or Launchpad, or run: open "%s"\n' \
  "$destination" "$destination"
