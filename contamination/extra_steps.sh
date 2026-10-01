#!/bin/bash
# Post-cutoff contamination test on the GPU machine (see contamination/README.md). STATUS: pending, no results yet.
#
# Run it on its own, after the models it evaluates exist (r11, r3, r4 from scripts/queue.sh; r5, r6 from queue2.sh):
#   WIKIMEDIA_CONTACT="<your email or URL>" bash contamination/extra_steps.sh
# or at the end of scripts/queue2.sh with SAFAID_POSTCUTOFF=1.
#
# Code: every script of this folder (PC_CODE=<dir> overrides). Evaluation code checked against safaid/eval.py.
# Stages (each logs and continues on failure; re-running resumes every stage):
#   0 preflight: file check, main-venv pip freeze snapshot, eval_post.py vs safaid/eval.py inference-code check
#   1 generator venv $PC_GENV (diffusers 0.38.0, transformers 5.7.0), never touching the training venv
#   2 reals: Wikimedia Commons fetch (CPU/network) in the background
#   3 fakes: one generator at a time (FLUX.2-klein-4B, ERNIE-Image-Turbo, Z-Image-Turbo), CALIBRATION line each,
#     that generator's HF cache deleted afterwards
#   4 manifest (benchmarks PostCutoff / PostCutoffModelPreCutoffVAE / PostCutoffReal) + generation_meta.json
#   5 eval_post.py for the 5 models with the training venv's python (per-model failures are logged and skipped)
#   6 score_postcutoff.py -> RESULTS_postcutoff.md
#   7 outputs in $SAFAID_WORKDIR/runs/postcutoff/ and the DONE marker
# Env:  WIKIMEDIA_CONTACT  email or URL for the Wikimedia User-Agent (their policy asks for one; never hard-code it)
#       PC_N (1000) PC_GENS PC_BATCH (4) PC_SKIP_REALS=1 PC_SKIP_GEN=1 PC_FORCE=1
#       PC_GEN_TIMEOUT (4h) PC_REALS_TIMEOUT (6h) PC_EVAL_TIMEOUT (3h)
#       PC_WORKSPACE (default $SAFAID_WORKDIR or /workspace/safaid)  PC_GENV (default /workspace/genv)
#       PC_MPY (training-venv python, default /venv/main/bin/python)  PC_EVAL_PY (default <repo>/safaid/eval.py)
set -u
export PATH=/venv/main/bin:/opt/instance-tools/bin:$PATH
W=${PC_WORKSPACE:-${SAFAID_WORKDIR:-/workspace/safaid}}
mkdir -p "$W/logs"
LOG=$W/logs/extra_steps.log
exec > >(tee -a "$LOG") 2>&1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PC=${PC_CODE:-$HERE/postcutoff}
[ -f "$PC/gen_fakes.py" ] || PC=$HERE
EVAL_PY=${PC_EVAL_PY:-$(cd "$HERE/.." && pwd)/safaid/eval.py}
DATA_ROOT=$W/data
DATA=$DATA_ROOT/postcutoff                  # eval_post.py maps zip "postcutoff" to <data_root>/postcutoff
OUT=$W/runs/postcutoff                      # synced by scripts/watch_vast.sh (runs/ minus model/)
GENV=${PC_GENV:-/workspace/genv}
GPY=$GENV/bin/python
MPY=${PC_MPY:-/venv/main/bin/python}
GEN_CACHE=$W/hf_gen_cache                   # generator weights only; the default HF cache (Qwen3-VL) is not touched
N=${PC_N:-1000}
GENS=${PC_GENS:-"flux2_klein_4b ernie_image_turbo z_image_turbo"}
BATCH=${PC_BATCH:-4}
CONTACT=${WIKIMEDIA_CONTACT:-}
MAN=$DATA/postcutoff_manifest.csv
mkdir -p "$DATA" "$OUT"

STATUS=()
ERRORS=0
log() { echo "[extra_steps $(date -u +%FT%TZ)] $*"; }
note() { STATUS+=("$1"); log "$1"; }
err() { STATUS+=("ERROR: $1"); ERRORS=$((ERRORS + 1)); log "ERROR: $1"; }
count_png() { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l | tr -d ' '; }
# every genv command: no PYTHONPATH leakage, its own HF cache, xet high-performance downloads
GENV_ENV=(env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 HF_HUB_CACHE="$GEN_CACHE" HF_XET_CACHE="$GEN_CACHE/xet"
          HF_HUB_DISABLE_TELEMETRY=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1)
PIP=(env -u PYTHONPATH -u PYTHONHOME -u PIP_TARGET -u PIP_PREFIX -u PIP_USER PYTHONNOUSERSITE=1
     PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 "$GPY" -m pip)

finish() {
  "$MPY" -m pip freeze 2>/dev/null | sort > "$OUT/venv_main_freeze_after.txt"
  if cmp -s "$OUT/venv_main_freeze_before.txt" "$OUT/venv_main_freeze_after.txt"; then
    note "/venv/main packages unchanged (pip freeze identical before/after)"
  else
    err "/venv/main pip freeze changed: $(diff "$OUT/venv_main_freeze_before.txt" "$OUT/venv_main_freeze_after.txt" | head -5 | tr '\n' ' ')"
  fi
  local st=ok
  [ "$ERRORS" -gt 0 ] && st=errors
  { echo "status: $st"; echo "finished_utc: $(date -u +%FT%TZ)"; printf '%s\n' "${STATUS[@]}"; } > "$OUT/DONE"
  log "=== finished (status: $st); marker $OUT/DONE"
  cat "$OUT/DONE"
}

log "=== post-cutoff contamination test: code $PC, N=$N, generators: $GENS"
if [ -f "$OUT/DONE" ] && grep -q '^status: ok' "$OUT/DONE" && [ "${PC_FORCE:-0}" != 1 ]; then
  log "DONE (status ok) already present -> nothing to do (PC_FORCE=1 to rerun)"; exit 0
fi

# ---------------- 0. preflight ----------------
for f in gen_fakes.py fetch_recent_reals.py eval_manifest_builder.py eval_post.py score_postcutoff.py \
         postcutoff_common.py prompts_postcutoff.py; do
  if [ ! -f "$PC/$f" ]; then err "missing $PC/$f"; finish; exit 0; fi
done
"$MPY" -m pip freeze 2>/dev/null | sort > "$OUT/venv_main_freeze_before.txt"
log "disk: $(df -h "$W" | awk 'NR==2{print $4" free of "$2}')"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
"$MPY" "$PC/eval_post.py" --manifest - --model_path - --run_dir - --test_prompt investigative \
  --check_against "$EVAL_PY" --check_only && note "eval_post.py inference code identical to safaid/eval.py" \
  || note "WARNING: eval_post.py inference code differs from safaid/eval.py, or eval.py is missing (see log)"

# ---------------- 1. generator venv ----------------
genv_ok() {
  [ -x "$GPY" ] || return 1
  "${GENV_ENV[@]}" "$GPY" - <<'EOF'
import sys
import diffusers, transformers, torch, PIL, requests
ok = diffusers.__version__ == "0.38.0" and transformers.__version__ == "5.7.0"
ok = ok and transformers.__file__.startswith(sys.prefix) and diffusers.__file__.startswith(sys.prefix)
ok = ok and all(hasattr(diffusers, c) for c in ("Flux2KleinPipeline", "ErnieImagePipeline", "ZImagePipeline"))
print("genv:", sys.prefix, "python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      torch.cuda.is_available(), "| diffusers", diffusers.__version__, "| transformers", transformers.__version__,
      transformers.__file__, "| PIL", PIL.__version__)
sys.exit(0 if ok and torch.cuda.is_available() else 1)
EOF
}

build_genv() {
  local base mode=isolated
  for base in "$MPY" /usr/bin/python3; do "$base" -c 'import venv, ensurepip' 2>/dev/null && break; done
  # Reuse /venv/main's torch via --system-site-packages ONLY if that exposes torch but NOT transformers (4.57.3).
  local probe=$GENV.probe
  rm -rf "$probe"
  if "$base" -m venv --system-site-packages "$probe" >/dev/null 2>&1 && \
     env -u PYTHONPATH PYTHONNOUSERSITE=1 "$probe/bin/python" -c '
import importlib.util as u, sys
t, tr = u.find_spec("torch"), u.find_spec("transformers")
print("probe --system-site-packages: torch", t and t.origin, "| transformers", tr and tr.origin)
sys.exit(0 if (t is not None and tr is None) else 1)'; then
    mode=system_site
  fi
  rm -rf "$probe"
  log "genv: base $base, mode $mode ($( [ $mode = isolated ] && echo 'torch==2.9.1 cu128 installed into genv' || echo 'torch reused, no transformers visible'))"
  rm -rf "$GENV"
  if [ $mode = system_site ]; then "$base" -m venv --system-site-packages "$GENV" || return 1
  else "$base" -m venv "$GENV" || return 1; fi
  timeout 10m "${PIP[@]}" install -q --upgrade pip || return 1
  if [ $mode = isolated ]; then
    timeout 45m "${PIP[@]}" install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 || return 1
  fi
  timeout 30m "${PIP[@]}" install -q diffusers==0.38.0 transformers==5.7.0 accelerate==1.15.0 safetensors==0.8.0 \
    huggingface_hub==1.33.0 tokenizers==0.22.2 hf_transfer pillow==12.1.0 requests "numpy<3" sentencepiece protobuf \
    || return 1
  echo "$mode" > "$GENV/.safaid_mode"
}

if [ "${PC_SKIP_GEN:-0}" = 1 ] && [ "${PC_SKIP_REALS:-0}" = 1 ]; then
  note "genv: not needed (PC_SKIP_GEN=1 and PC_SKIP_REALS=1)"
elif genv_ok; then
  note "genv: reused existing $GENV ($(cat "$GENV/.safaid_mode" 2>/dev/null))"
  "${PIP[@]}" freeze > "$OUT/genv_freeze.txt" 2>/dev/null
else
  log "building $GENV"
  if build_genv && genv_ok; then note "genv: built ($(cat "$GENV/.safaid_mode"))"
  else err "genv build failed; no fakes or reals can be made"; finish; exit 0; fi
  "${PIP[@]}" freeze > "$OUT/genv_freeze.txt" 2>/dev/null
fi

# ---------------- 2. reals in the background ----------------
REALS_PID=""
if [ "${PC_SKIP_REALS:-0}" != 1 ]; then
  [ -n "$CONTACT" ] || log "WARNING: WIKIMEDIA_CONTACT not set; the Wikimedia User-Agent will carry no contact"
  rm -f "$DATA/.reals_rc"
  ( "${GENV_ENV[@]}" timeout "${PC_REALS_TIMEOUT:-6h}" "$GPY" "$PC/fetch_recent_reals.py" --out "$DATA" --n "$N" \
      ${CONTACT:+--contact "$CONTACT"} > "$W/logs/postcutoff_reals.log" 2>&1; echo $? > "$DATA/.reals_rc" ) &
  REALS_PID=$!
  log "reals fetch started (pid $REALS_PID, log logs/postcutoff_reals.log)"
fi

# ---------------- 3. fakes, one generator at a time ----------------
if [ "${PC_SKIP_GEN:-0}" != 1 ]; then
  "${GENV_ENV[@]}" timeout 20m "$GPY" "$PC/gen_fakes.py" --check_env --generators "$(echo $GENS | tr ' ' ,)" \
    && note "gen_fakes --check_env OK" || err "gen_fakes --check_env reported a problem (continuing)"
  for g in $GENS; do
    log "--- generator $g; disk free $(df -h "$W" | awk 'NR==2{print $4}')"
    "${GENV_ENV[@]}" timeout "${PC_GEN_TIMEOUT:-4h}" "$GPY" "$PC/gen_fakes.py" --out "$DATA" --generators "$g" --n "$N" \
      --batch_size "$BATCH" --purge_cache 2>&1 | tee -a "$W/logs/postcutoff_fakes.log"
    rc=${PIPESTATUS[0]}
    rm -rf "$GEN_CACHE"/models--* "$GEN_CACHE"/xet 2>/dev/null   # whatever is left of this generator's weights
    n=$(count_png "$DATA/$g/1_fake")
    cal=$(grep -h "CALIBRATION.*\[$g\]" "$W/logs/postcutoff_fakes.log" 2>/dev/null | tail -1)
    if [ "$rc" -eq 0 ] && [ "$n" -ge "$N" ]; then note "fakes $g: $n/$N ($cal)"
    else err "fakes $g: rc=$rc, $n/$N images ($cal)"; fi
  done
fi
NFAKE=0
for g in $GENS; do NFAKE=$((NFAKE + $(count_png "$DATA/$g/1_fake"))); done
if [ -n "$REALS_PID" ]; then
  if [ "$NFAKE" -eq 0 ]; then
    log "no fakes at all -> stopping the reals fetch"; pkill -f fetch_recent_reals.py 2>/dev/null
  fi
  log "waiting for the reals fetch (pid $REALS_PID)"
  wait "$REALS_PID"
fi
NREAL=$(count_png "$DATA/wikimedia_2026/0_real")
RRC=$(cat "$DATA/.reals_rc" 2>/dev/null || echo "n/a")
if [ "$NREAL" -ge "$N" ]; then note "reals: $NREAL/$N (fetch rc $RRC)"
elif [ "$NREAL" -ge $((N / 2)) ]; then note "WARNING: reals: only $NREAL/$N (fetch rc $RRC; see logs/postcutoff_reals.log)"
else err "reals: $NREAL/$N (fetch rc $RRC; see logs/postcutoff_reals.log)"; fi
tail -3 "$W/logs/postcutoff_reals.log" 2>/dev/null

# ---------------- 4. manifest + generation_meta.json ----------------
[ -f "$DATA/wikimedia_2026/attribution.csv" ] && cp "$DATA/wikimedia_2026/attribution.csv" "$OUT/attribution.csv"
if "$MPY" "$PC/eval_manifest_builder.py" --root "$DATA" --out "$MAN" --verify --expect_n "$N" \
     --meta_out "$OUT/generation_meta.json"; then
  cp "$MAN" "$OUT/postcutoff_manifest.csv"
  note "manifest: $(($(wc -l < "$MAN") - 1)) rows"
else
  err "manifest build failed; no evaluation"; finish; exit 0
fi

# ---------------- 5. evaluate the 5 models (sequentially on the GPU) ----------------
MODELS=(
  "r11_safaid_seed4_|investigative"
  "r3_safaid_seed2_|investigative"
  "r4_safaid_seed3_|investigative"
  "r5_catAIDprompt_seed2_|vanilla"
  "r6_catAIDprompt_seed3_|vanilla"
)
find_run() {  # name prefix -> run dir holding model/checkpoint-1600 (runs_prev first, then runs)
  local d
  for d in "$W"/runs_prev/"$1"* "$W"/runs/"$1"*; do
    [ -d "$d/model/checkpoint-1600" ] && { echo "$d"; return 0; }
  done
  return 1
}
for spec in "${MODELS[@]}"; do
  IFS='|' read -r PFX TP <<< "$spec"
  if ! RUN=$(find_run "$PFX"); then err "eval ${PFX}*: no model/checkpoint-1600 in runs_prev/ or runs/"; continue; fi
  NAME=$(basename "$RUN"); RD=$OUT/$NAME
  mkdir -p "$RD"
  if [ -f "$RD/EVAL_DONE_$TP" ]; then note "eval $NAME: already done"; continue; fi
  log "eval $NAME (prompt $TP) -> $RD"
  ( cd "$W" && timeout "${PC_EVAL_TIMEOUT:-3h}" "$MPY" "$PC/eval_post.py" --manifest "$MAN" \
      --model_path "$RUN/model/checkpoint-1600" --run_dir "$RD" --test_prompt "$TP" --tag "$TP" \
      --data_root "$DATA_ROOT" --check_against "$EVAL_PY" ) > "$RD/eval_log_$TP.txt" 2>&1
  rc=$?
  tail -25 "$RD/eval_log_$TP.txt"
  if [ "$rc" -eq 0 ] && [ -f "$RD/EVAL_DONE_$TP" ]; then note "eval $NAME: ok"
  else err "eval $NAME: rc=$rc (see $RD/eval_log_$TP.txt)"; fi
done

# ---------------- 6. score ----------------
if "$MPY" "$PC/score_postcutoff.py" --runs_root "$OUT" --context_root "$W/runs_prev" --context_root "$W/runs" \
     --attribution "$OUT/attribution.csv" --generation_meta "$OUT/generation_meta.json" \
     --out_md "$OUT/RESULTS_postcutoff.md" --out_json "$OUT/results_postcutoff.json" > "$OUT/score_log.txt" 2>&1; then
  note "results: $OUT/RESULTS_postcutoff.md"
  cat "$OUT/RESULTS_postcutoff.md"
else
  err "scoring failed (see $OUT/score_log.txt)"; tail -20 "$OUT/score_log.txt"
fi

# ---------------- 7. outputs + DONE ----------------
for f in RESULTS_postcutoff.md attribution.csv generation_meta.json; do
  [ -s "$OUT/$f" ] || err "missing output $OUT/$f"
done
finish
exit 0
