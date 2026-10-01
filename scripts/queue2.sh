#!/bin/bash
# Second batch of experiments, one after another (train -> eval -> next run). Resumable: finished steps are
# skipped. Batch 1 (the three SAFAID runs + CatAID-prompt control seed 1) is scripts/queue.sh.
#
#   r5, r6   CatAID-prompt control, seeds 2 and 3            (with r2: the 3-run same-backbone control)
#   r16      controlled QLoRA (4-bit, r=16, alpha=16, LR 2e-4), seed of SAFAID run 1
#   r12      ablation: no investigative prompt (plain question for training and test), seed of SAFAID run 1
#   r13      ablation: uniform augmentation (real AND fake images get 3 copies; 16,000 samples), same seed
#   r14      ablation: no augmentation (4,000 samples), same seed
# r13 and r14 fix the cosine schedule at 3,750 optimizer steps (--schedule_steps 3750), the length of SAFAID's 3-epoch
# schedule, so every run follows the same learning-rate trajectory up to the evaluated step 1600.
#
# Start on the GPU machine (survives a closed SSH session; the tmux session name "queue" is what
# scripts/watch_vast.sh ends once everything is done):
#   tmux new -d -s queue "bash scripts/queue2.sh > /workspace/safaid/logs/queue2.log 2>&1"
#   (or: setsid nohup bash scripts/queue2.sh > /workspace/safaid/logs/queue2.log 2>&1 &)
# Env:
#   SAFAID_WORKDIR          working directory with data/, runs/, logs/                     (default /workspace/safaid)
#   SAFAID_RUNS             run folders                                                      (default $SAFAID_WORKDIR/runs)
#   SAFAID_KEEP_ALL_MODELS  1 = keep the checkpoints of the ablations r12-r14 (default: deleted after their
#                           evaluation, and marked NO_DOWNLOAD so the watcher does not fetch them)
#   SAFAID_POSTCUTOFF       1 = run the post-cutoff contamination test (contamination/extra_steps.sh) at the end;
#                           it needs WIKIMEDIA_CONTACT and the r11/r3/r4/r5/r6 checkpoints (default 0)
#   SAFAID_AUTOSTOP_MIN     vast.ai only: stop this instance this many minutes after the queue finishes, unless
#                           scripts/watch_vast.sh is still alive; 0 disables                 (default 45)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SAFAID_WORKDIR=${SAFAID_WORKDIR:-/workspace/safaid}
export SAFAID_RUNS=${SAFAID_RUNS:-$SAFAID_WORKDIR/runs}
AUTOSTOP_MIN=${SAFAID_AUTOSTOP_MIN:-45}
KEEP_ALL=${SAFAID_KEEP_ALL_MODELS:-0}
export PATH=/venv/main/bin:/opt/instance-tools/bin:$PATH   # vast.ai PyTorch image; harmless elsewhere
cd "$SAFAID_WORKDIR" || exit 1
mkdir -p logs
rm -f ALL_DONE
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
SPEC="qwen3vl4b_fullFT_lr5e-6_ebs8_step1600of3750_notebookAug_rtxpro6000s"
# name | seed | train prompt | test prompt | method | aug | schedule_steps | keep model
RUNS=(
  "r5_catAIDprompt_seed2_${SPEC}|2|category|vanilla|full|fake_only|0|keep"
  "r6_catAIDprompt_seed3_${SPEC}|3|category|vanilla|full|fake_only|0|keep"
  "r16_qlora_seed4_qwen3vl4b_qlora4bit_r16a16_lr2e-4_ebs8_step1600of3750_notebookAug_rtxpro6000s|4|investigative|investigative|qlora|fake_only|0|keep"
  "r12_ablNoInvestigativePrompt_seed4_${SPEC}|4|vanilla|vanilla|full|fake_only|0|drop"
  "r13_ablUniformAug_seed4_qwen3vl4b_fullFT_lr5e-6_ebs8_step1600of3750steps_uniformAug_rtxpro6000s|4|investigative|investigative|full|uniform|3750|drop"
  "r14_ablNoAug_seed4_qwen3vl4b_fullFT_lr5e-6_ebs8_step1600of3750steps_noAug_rtxpro6000s|4|investigative|investigative|full|none|3750|drop"
)
for spec in "${RUNS[@]}"; do
  IFS='|' read -r NAME SEED TRP TEP METHOD AUG SCHED KEEP <<< "$spec"
  [ "$KEEP_ALL" = 1 ] && KEEP=keep
  RD=$SAFAID_RUNS/$NAME; mkdir -p "$RD"; rm -f "$RD/FAILED"
  [ "$KEEP" = drop ] && touch "$RD/NO_DOWNLOAD"
  echo "=== $(date -u +%FT%T) $NAME"
  if [ ! -f "$RD/TRAIN_DONE" ]; then
    python "$REPO/safaid/train.py" --run_name "$NAME" --seed "$SEED" --train_prompt "$TRP" --stop_step 1600 \
      --method "$METHOD" --aug_mode "$AUG" --schedule_steps "$SCHED" --out_root "$SAFAID_RUNS" \
      > "$RD/train_log.txt" 2>&1 || { echo "TRAIN FAILED $NAME"; echo failed > "$RD/FAILED"; continue; }
  fi
  # the model to evaluate: checkpoint-1600, or checkpoint-1600-merged16 for QLoRA (train.py writes it to TRAIN_DONE)
  MODEL=$(head -1 "$RD/TRAIN_DONE")
  # effective TrainingArguments after Unsloth's patches, as JSON (results/<run>/training_args.json)
  python -c "import unsloth, trl, torch, json; a=torch.load('$RD/model/checkpoint-1600/training_args.bin', weights_only=False); open('$RD/training_args.json','w').write(json.dumps(json.loads(a.to_json_string()), indent=1))" > /dev/null 2>&1
  if [ ! -f "$RD/EVAL_DONE_$TEP" ]; then
    python "$REPO/safaid/eval.py" --model_path "$MODEL" --run_dir "$RD" --test_prompt "$TEP" \
      > "$RD/eval_log_$TEP.txt" 2>&1 || { echo "EVAL FAILED $NAME"; echo failed > "$RD/FAILED"; continue; }
  fi
  [ "$KEEP" = drop ] && rm -rf "$RD/model"
  echo "=== $(date -u +%FT%T) $NAME done: $(grep AVERAGE_19 "$RD/eval_log_$TEP.txt")"
done

# optional: post-cutoff contamination test (pending; see contamination/README.md)
[ "${SAFAID_POSTCUTOFF:-0}" = 1 ] && bash "$REPO/contamination/extra_steps.sh"

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
