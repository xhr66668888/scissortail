#!/usr/bin/env bash
# Makes the app icons in app/src-tauri/icons from the logo:
#
#   scripts/icons.sh                 from app/src-tauri/icons/logo.png
#   scripts/icons.sh new-logo.png    copies it to logo.png first
#
# The logo is a square PNG with a transparent background. macOS gets it on a
# white rounded tile, Windows the logo alone (scripts/icons.swift draws both);
# `cargo tauri icon` then makes every size.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ICONS="$ROOT/app/src-tauri/icons"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ $# -gt 0 && ! $1 -ef $ICONS/logo.png ]]; then
  cp "$1" "$ICONS/logo.png"
fi
swift "$ROOT/scripts/icons.swift" "$ICONS/logo.png" "$TMP"
cd "$ROOT/app/src-tauri"
cargo tauri icon "$TMP/macos.png" -o "$TMP/macos" >/dev/null
cargo tauri icon "$TMP/windows.png" -o "$TMP/windows" >/dev/null
cp "$TMP"/macos/{32x32,64x64,128x128,128x128@2x,icon}.png "$TMP/macos/icon.icns" "$ICONS/"
cp "$TMP"/windows/Square*Logo.png "$TMP/windows/StoreLogo.png" "$TMP/windows/icon.ico" "$ICONS/"
echo "icons updated in app/src-tauri/icons"
