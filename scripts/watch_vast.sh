#!/bin/bash
# Laptop-side watcher for a vast.ai instance running scripts/queue.sh or scripts/queue2.sh. Every 60 s it
#   - rsyncs every run folder (config, logs, loss.csv, progress.json, predictions, metrics; not the model) to $LOCAL
#   - records GPU / disk / queue status and the account credit in $LOCAL/_instance/
#   - downloads each finished checkpoint (~8.9 GB; the LoRA adapter for QLoRA) to $MODELS, one at a time, resumable;
#     runs marked NO_DOWNLOAD by the queue (ablations whose checkpoint is deleted after evaluation) are skipped
#   - stops the instance if the credit drops below $SAFAID_MIN_CREDIT
#   - when all runs are done and downloaded: ends the queue's tmux session (so its self-stop fallback cannot fire while
#     this watcher owns the shutdown), final sync, then destroys the instance (or only stops it, keeping the disk, if a
#     run FAILED or the final sync failed). Every stop/destroy is verified against the API, because the vastai CLI
#     can exit 0 on failure.
#
# Usage:   bash scripts/watch_vast.sh <instance_id>
#          keep it alive with:  nohup bash scripts/watch_vast.sh <id> >> watcher.log 2>&1 &   (macOS: caffeinate -i bash ...)
# Env:     SAFAID_SSH_HOST        ssh destination of the instance: root@<ip> or a Host alias in ~/.ssh/config  (required)
#          SAFAID_SSH_OPTS        extra ssh options, e.g. "-p <port> -i $HOME/.ssh/<key>"; `vastai ssh-url <id>` prints
#                                 the host and port
#          SAFAID_REMOTE_WORKDIR  SAFAID_WORKDIR on the instance        (default /workspace/safaid)
#          SAFAID_LOCAL_RUNS      local copy of the run folders          (default ./runs_synced)
#          SAFAID_LOCAL_MODELS    local folder for the checkpoints       (default ./models)
#          SAFAID_MIN_CREDIT      stop the instance below this credit    (default 1.5, USD)
#          SAFAID_QUEUE_TMUX      tmux session of the queue on the box   (default queue)
#          VASTAI                 vastai CLI                             (default: vastai on PATH)
INST=${1:?usage: watch_vast.sh <instance_id>}
R=${SAFAID_SSH_HOST:?set SAFAID_SSH_HOST, e.g. root@<ip> or a Host alias from ~/.ssh/config}
SSH="ssh ${SAFAID_SSH_OPTS:-} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
RW=${SAFAID_REMOTE_WORKDIR:-/workspace/safaid}; RW=${RW%/}
LOCAL=${SAFAID_LOCAL_RUNS:-./runs_synced}
MODELS=${SAFAID_LOCAL_MODELS:-./models}
MIN_CREDIT=${SAFAID_MIN_CREDIT:-1.5}
VAST=${VASTAI:-vastai}
QTMUX=${SAFAID_QUEUE_TMUX:-queue}
ST=$LOCAL/_instance/status.txt
ALERT=$LOCAL/_instance/ALERT.txt
DLPID=""
mkdir -p "$LOCAL/_instance" "$MODELS"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }

inst_status() {  # prints actual_status, or GONE if the instance no longer exists
  $VAST show instance "$INST" --raw 2>/dev/null | python3 -c '
import json,sys
try: o=json.load(sys.stdin)
except Exception: print("UNKNOWN"); sys.exit()
print("GONE" if (not o or o.get("instances",1) is None or "id" not in o) else o.get("actual_status"))' 2>/dev/null
}

verified() {  # verified stop|destroy  -> retries until the API confirms
  local action=$1 i s
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if [ "$action" = destroy ]; then $VAST destroy instance "$INST" -y >/dev/null 2>&1; else $VAST stop instance "$INST" >/dev/null 2>&1; fi
    sleep 30
    s=$(inst_status)
    if [ "$action" = destroy ] && [ "$s" = GONE ]; then log "instance $INST destroyed (verified)"; return 0; fi
    if [ "$action" = stop ] && { [ "$s" = exited ] || [ "$s" = stopped ] || [ "$s" = GONE ]; }; then log "instance $INST stopped (verified: $s)"; return 0; fi
    log "$action not confirmed yet (status=$s), retry $i"
  done
  echo "$(date '+%F %T') $action of $INST NOT CONFIRMED - check the vast.ai console" | tee -a "$ALERT"
  return 1
}

dl_running() { [ -n "$DLPID" ] && kill -0 "$DLPID" 2>/dev/null; }

log "watcher start: instance $INST"
while true; do
  rsync -az --exclude 'model/' -e "$SSH" "$R:$RW/runs/" "$LOCAL/" 2>/dev/null
  rsync -az -e "$SSH" "$R:$RW/logs/" "$LOCAL/_instance/logs/" 2>/dev/null
  # the heartbeat tells queue.sh that this watcher is alive and will stop/destroy the instance itself
  if $SSH "$R" "touch $RW/WATCHER_HEARTBEAT;
             nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader;
             df -h / | tail -1; for f in $RW/runs/*/progress.json; do [ -f \"\$f\" ] && echo \"\$f\" && cat \"\$f\"; done;
             tail -4 $RW/logs/queue*.log 2>/dev/null; ls $RW/ALL_DONE 2>/dev/null; true" > "$ST.tmp" 2>/dev/null; then
    mv "$ST.tmp" "$ST"
    head -1 "$ST" >> "$LOCAL/_instance/gpu_log.csv"
  else
    s=$(inst_status)
    echo "$(date '+%F %T') instance unreachable (status=$s)" >> "$ST"
    if [ "$s" = GONE ]; then log "instance is gone - watcher exiting"; exit 0; fi
  fi
  CREDIT=$($VAST show user --raw 2>/dev/null | python3 -c "import json,sys;print(round(json.load(sys.stdin)['credit'],2))" 2>/dev/null)
  echo "$(date '+%F %T') credit=\$$CREDIT" >> "$ST"
  if [ -n "$CREDIT" ] && python3 -c "import sys; sys.exit(0 if float('$CREDIT') < float('$MIN_CREDIT') else 1)"; then
    echo "$(date '+%F %T') LOW CREDIT \$$CREDIT -> stopping instance (data kept, top up and restart it)" | tee -a "$ALERT"
    verified stop
    exit 1
  fi

  # download trained models in the background, one at a time (resumable with --partial)
  if ! dl_running; then
    for d in "$LOCAL"/r*_*/; do
      [ -d "$d" ] || continue
      n=$(basename "$d")
      if [ -f "$d/NO_DOWNLOAD" ] && [ ! -f "$MODELS/$n/DOWNLOADED" ]; then
        mkdir -p "$MODELS/$n" && echo "not kept (NO_DOWNLOAD: checkpoint deleted on the box after evaluation)" > "$MODELS/$n/DOWNLOADED"
        continue
      fi
      if [ -f "$d/TRAIN_DONE" ] && [ ! -f "$MODELS/$n/DOWNLOADED" ]; then
        ( mkdir -p "$MODELS/$n" && rsync -a --partial -e "$SSH" "$R:$RW/runs/$n/model/checkpoint-1600" "$MODELS/$n/" \
            && date > "$MODELS/$n/DOWNLOADED" ) &
        DLPID=$!
        log "downloading model $n"
        break
      fi
    done
  fi

  # the queue deletes ALL_DONE when it starts and writes it when it ends; only the file counts (not the log tail)
  if grep -qx "$RW/ALL_DONE" "$ST"; then
    # While this watcher is alive it owns the shutdown: end the queue's tmux session, which is only waiting for its
    # self-stop fallback now (ending the session, not just its sleep, so the fallback's stop command never runs).
    $SSH "$R" "tmux kill-session -t $QTMUX 2>/dev/null; true" 2>/dev/null
    PENDING=0
    for d in "$LOCAL"/r*_*/; do
      [ -d "$d" ] || continue
      n=$(basename "$d")
      [ -f "$d/FAILED" ] && [ ! -f "$d/TRAIN_DONE" ] && continue
      [ -f "$MODELS/$n/DOWNLOADED" ] || PENDING=1
    done
    if [ $PENDING -eq 0 ] && ! dl_running; then
      ok=0
      for i in 1 2 3 4 5; do
        rsync -az --exclude 'model/' -e "$SSH" "$R:$RW/runs/" "$LOCAL/" \
          && rsync -az -e "$SSH" "$R:$RW/logs/" "$LOCAL/_instance/logs/" && { ok=1; break; }
        sleep 30
      done
      if [ $ok -eq 1 ] && ! ls "$LOCAL"/r*_*/FAILED >/dev/null 2>&1; then
        log "ALL DONE, final sync ok -> destroying instance"
        verified destroy && exit 0
      else
        echo "$(date '+%F %T') a run FAILED or final sync failed -> stopping only (disk kept for a retry)" | tee -a "$ALERT"
        verified stop
      fi
      exit 1
    fi
  fi
  sleep 60
done
