#!/bin/bash
# Builds the README cover: procedural artwork -> cover.html -> assets/cover.png (2560 x 1280).
# Needs Python 3 with numpy and Pillow, and Google Chrome or Chromium (headless). Usage: bash assets/src/build_cover.sh
set -euo pipefail
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
CHROME=${CHROME:-}
for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" google-chrome chromium chromium-browser; do
  [ -n "$CHROME" ] && break
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CHROME=$c; fi
done
[ -n "$CHROME" ] || { echo "Chrome/Chromium not found (set CHROME=/path/to/chrome)"; exit 1; }
"$PY" "$SRC/make_cover_art.py"
"$PY" "$SRC/make_cover.py"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=1280,640 \
  --virtual-time-budget=8000 --screenshot="$SRC/../cover.png" "file://$SRC/cover.html" 2>/dev/null
echo "wrote assets/cover.png"
