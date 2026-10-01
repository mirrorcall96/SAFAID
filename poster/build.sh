#!/bin/bash
# Builds the poster: poster/poster.html -> assets/poster.pdf (A3, vector) and assets/poster.png (preview).
# Needs Python 3, Google Chrome or Chromium (headless) and pdftoppm (poppler). Usage: bash poster/build.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$HERE/..
CHROME=${CHROME:-}
for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" google-chrome chromium chromium-browser; do
  [ -n "$CHROME" ] && break
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CHROME=$c; fi
done
[ -n "$CHROME" ] || { echo "Chrome/Chromium not found (set CHROME=/path/to/chrome)"; exit 1; }
python3 "$HERE/make_poster.py"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=8000 \
  --print-to-pdf="$REPO/assets/poster.pdf" "file://$HERE/poster.html" 2>/dev/null
pdftoppm -png -r 120 -singlefile "$REPO/assets/poster.pdf" "$REPO/assets/poster"
echo "wrote assets/poster.pdf and assets/poster.png"
