#!/bin/bash
# Batch 1 of the paper experiments, one after another (train -> eval -> next run). Resumable: finished steps
# are skipped. Batch 2 (control seeds 2-3, QLoRA, ablations) is scripts/queue2.sh.
#
#   r11, r3, r4  SAFAID (investigative prompt, fake-only augmentation), runs 1, 2, 3 (seeds 4, 2, 3)
#   r2           CatAID-prompt control on the same backbone, seed 1 (trained with the category prompt, tested without)
#
# Start on the GPU machine (survives a closed SSH session; the tmux session name "queue" is what
# scripts/watch_vast.sh ends once everything is done):
#   tmux new -d -s queue "bash scripts/queue.sh > /workspace/safaid/logs/queue.log 2>&1"
#   (or: setsid nohup bash scripts/queue.sh > /workspace/safaid/logs/queue.log 2>&1 &)
# Env:
#   SAFAID_WORKDIR       working directory with data/, runs/, logs/                        (default /workspace/safaid)
#   SAFAID_RUNS          run folders                                                         (default $SAFAID_WORKDIR/runs)
#   SAFAID_AUTOSTOP_MIN  vast.ai only: stop this instance (GPU billing ends, disk is kept) this many minutes after the
#                        queue finishes, unless scripts/watch_vast.sh is still alive; 0 disables  (default 45)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SAFAID_WORKDIR=${SAFAID_WORKDIR:-/workspace/safaid}
export SAFAID_RUNS=${SAFAID_RUNS:-$SAFAID_WORKDIR/runs}
AUTOSTOP_MIN=${SAFAID_AUTOSTOP_MIN:-45}
export PATH=/venv/main/bin:/opt/instance-tools/bin:$PATH   # vast.ai PyTorch image; harmless elsewhere
cd "$SAFAID_WORKDIR" || exit 1
mkdir -p logs
rm -f ALL_DONE
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
SPEC="qwen3vl4b_fullFT_lr5e-6_ebs8_step1600of3750_notebookAug_rtxpro6000s"
# name | seed | train prompt | test prompt
RUNS=(
  "r11_safaid_seed4_${SPEC}|4|investigative|investigative"
  "r2_catAIDprompt_seed1_${SPEC}|1|category|vanilla"
  "r3_safaid_seed2_${SPEC}|2|investigative|investigative"
  "r4_safaid_seed3_${SPEC}|3|investigative|investigative"
)
for spec in "${RUNS[@]}"; do
  IFS='|' read -r NAME SEED TRP TEP <<< "$spec"
  RD=$SAFAID_RUNS/$NAME; mkdir -p "$RD"
  echo "=== $(date -u +%FT%T) $NAME"
  if [ ! -f "$RD/TRAIN_DONE" ]; then
    python "$REPO/safaid/train.py" --run_name "$NAME" --seed "$SEED" --train_prompt "$TRP" --stop_step 1600 \
      --out_root "$SAFAID_RUNS" \
      > "$RD/train_log.txt" 2>&1 || { echo "TRAIN FAILED $NAME"; echo failed > "$RD/FAILED"; continue; }
  fi
  # effective TrainingArguments after Unsloth's patches, as JSON (results/<run>/training_args.json)
  [ -f "$RD/training_args.json" ] || python -c "import unsloth, trl, torch, json; a=torch.load('$RD/model/checkpoint-1600/training_args.bin', weights_only=False); open('$RD/training_args.json','w').write(json.dumps(json.loads(a.to_json_string()), indent=1))" > /dev/null 2>&1
  if [ ! -f "$RD/EVAL_DONE_$TEP" ]; then
    python "$REPO/safaid/eval.py" --model_path "$RD/model/checkpoint-1600" --run_dir "$RD" --test_prompt "$TEP" \
      > "$RD/eval_log_$TEP.txt" 2>&1 || { echo "EVAL FAILED $NAME"; echo failed > "$RD/FAILED"; continue; }
  fi
  echo "=== $(date -u +%FT%T) $NAME done: $(grep AVERAGE_19 "$RD/eval_log_$TEP.txt")"
done
date -u +%FT%T > ALL_DONE
echo "ALL_DONE"

# vast.ai self-stop fallback, in case the laptop-side watcher is not running. The watcher touches WATCHER_HEARTBEAT
# every minute and destroys the instance itself once all models are downloaded, so we only stop after it has been
# silent for 5 minutes. CONTAINER_ID / CONTAINER_API_KEY are the instance-scoped values vast.ai injects at boot.
CID=$(tr '\0' '\n' 2>/dev/null < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
KEY=$(tr '\0' '\n' 2>/dev/null < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
if [ "$AUTOSTOP_MIN" -gt 0 ] && [ -n "$CID" ] && [ -n "$KEY" ] && command -v vastai >/dev/null 2>&1; then
  sleep $((AUTOSTOP_MIN * 60))
  while [ -n "$(find WATCHER_HEARTBEAT -mmin -5 2>/dev/null)" ]; do sleep 60; done
  vastai stop instance "$CID" --api-key "$KEY"
fi
