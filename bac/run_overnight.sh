#!/bin/bash
# Overnight queue: the remaining runs, then refresh the summary tables and figures.
# Waits for the TTA chain to finish first so only one GPU job ever runs.
cd "$(dirname "$0")"
PY=../.venv/bin/python
FILTER='warn|pytree|Redirects|Special tokens|Found '
TTA_LOG=$1
log() { echo "[$(date +%H:%M)] $*"; }

if [ -n "$TTA_LOG" ]; then
  until grep -q TTA_DONE "$TTA_LOG" 2>/dev/null; do sleep 30; done
fi
log "overnight queue starting"

run() {  # name, train args...
  local name=$1; shift
  if [ -f ../results/${name}_val.json ]; then log "skip $name (done)"; return; fi
  log "train $name"
  [ -f ../runs/$name/connector.pt ] || $PY train.py --name $name --steps 200 --amp "$@" 2>&1 | grep -Evi "$FILTER" | grep -E "^step=(1|100|200) |saved|Error|Traceback"
  log "eval $name"
  $PY evaluate.py --name $name --weights ../runs/$name/connector.pt 2>&1 | grep -E '"gIoU"|"cIoU"|Error|Traceback'
}

run uniform16     --kind uniform --tokens 16
run tome64        --kind tome    --tokens 64
run random64      --kind random  --tokens 64
run bac64_sobel   --kind bac     --tokens 64 --alpha 0
run bac64_learned --kind bac     --tokens 64 --alpha 1
run bac64_nonms   --kind bac     --tokens 64 --no-nms
run bac64_nope    --kind bac     --tokens 64 --no-pe
run bac64_rho25   --kind bac     --tokens 64 --rho 0.25
run bac32         --kind bac     --tokens 32
run pool32        --kind pool    --tokens 32
run bac128        --kind bac     --tokens 128
run pool128       --kind pool    --tokens 128

log "refreshing summary"
$PY analyze.py > /dev/null 2>&1
log "OVERNIGHT_DONE"
