#!/bin/bash
# Renders assets/safaid_pipeline.gif from assets/src/pipeline_anim.tex.
# Needs pdflatex (TikZ), pdftoppm (poppler) and ImageMagick (magick). Usage: bash assets/src/make_animation.sh
set -euo pipefail
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$SRC/../safaid_pipeline.gif
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
frames=()
for variant in 0 1; do
  for stage in 1 2 3 4 5 6; do
    job=f_${variant}_${stage}
    pdflatex -interaction=batchmode -halt-on-error -output-directory "$TMP" -jobname "$job" \
      "\\def\\stage{$stage}\\def\\variant{$variant}\\input{$SRC/pipeline_anim.tex}" > /dev/null
    pdftoppm -png -r 170 -singlefile "$TMP/$job.pdf" "$TMP/$job"
    # delay in 1/100 s: the three model stages are short, the answer is held
    case $stage in 1|2) d=110 ;; 3|4|5) d=45 ;; 6) d=230 ;; esac
    frames+=(-delay "$d" "$TMP/$job.png")
  done
done
# one palette for all frames, no dithering (flat colours stay flat)
magick "${frames[@]}" -loop 0 +dither +remap -layers Optimize "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
