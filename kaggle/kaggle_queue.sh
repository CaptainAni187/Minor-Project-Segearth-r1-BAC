#!/bin/bash
# Reproduce the project on Kaggle (GPU T4 x2): two lanes, one per GPU.
#   bash kaggle/kaggle_queue.sh quick   # core claims only, ~1 h
#   bash kaggle/kaggle_queue.sh full    # all 29 configurations, ~4-6 h
# Results land in /kaggle/working/Minor/results; runs whose results already exist are skipped (safe to re-run).
# fp32 throughout: T4/P100 have no native bf16, so --amp (bf16) is not used here.
MODE=${1:-quick}
export USE_FLAX=0 USE_TF=0
cd "$(dirname "$0")/../bac"
mkdir -p ../results ../runs
F='warn|pytree|Redirects|Special tokens|Found '
log() { echo "[$(date +%H:%M)] $*"; }

ev() {  # gpu name evaluate-args...
  local gpu=$1 name=$2; shift 2
  [ -f ../results/${name}_val.json ] && { log "skip $name"; return; }
  log "eval $name (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --name $name "$@" 2>&1 | grep -E '"gIoU"|"cIoU"|Error|Traceback'
}
tr() {  # gpu name train-args...
  local gpu=$1 name=$2; shift 2
  [ -f ../results/${name}_val.json ] && { log "skip $name"; return; }
  log "train $name (GPU $gpu)"
  [ -f ../runs/$name/connector.pt ] || CUDA_VISIBLE_DEVICES=$gpu python train.py --name $name --steps 200 "$@" 2>&1 \
      | grep -Evi "$F" | grep -E "^step=(1|100|200) |saved|Error|Traceback"
  ev $gpu $name --weights ../runs/$name/connector.pt
}

lane0() {
  # inference-only controls: released weights, no training
  ev 0 released
  ev 0 zero  --kind zero
  ev 0 wrong --kind wrong
  ev 0 wrong_far --kind wrong_far
  ev 0 none  --kind none
  ev 0 dproj_zero --kind dproj_zero
  ev 0 wrongq --shuffle-questions
  [ "$MODE" = quick ] && return
  ev 0 dproj --kind dproj
  ev 0 dproj_zero_wrongq --kind dproj_zero --shuffle-questions
  ev 0 released_tta2 --tta 2
  ev 0 none_tta2 --kind none --tta 2
  tr 0 bac16     --kind bac --tokens 16
  tr 0 pool16    --kind pool --tokens 16
  tr 0 uniform16 --kind uniform --tokens 16
  tr 0 bac32     --kind bac --tokens 32
  tr 0 pool32    --kind pool --tokens 32
  tr 0 bac128    --kind bac --tokens 128
  tr 0 pool128   --kind pool --tokens 128
}
lane1() {
  tr $G1 bac64  --kind bac  --tokens 64
  tr $G1 pool64 --kind pool --tokens 64
  [ "$MODE" = quick ] && return
  tr $G1 uniform64     --kind uniform --tokens 64
  tr $G1 tome64        --kind tome --tokens 64
  tr $G1 random64      --kind random --tokens 64
  tr $G1 bac64_lam0    --kind bac --tokens 64 --sal-weight 0
  tr $G1 bac64_sobel   --kind bac --tokens 64 --alpha 0
  tr $G1 bac64_learned --kind bac --tokens 64 --alpha 1
  tr $G1 bac64_nonms   --kind bac --tokens 64 --no-nms
  tr $G1 bac64_nope    --kind bac --tokens 64 --no-pe
  tr $G1 bac64_rho25   --kind bac --tokens 64 --rho 0.25
  tr $G1 bac64_rho75   --kind bac --tokens 64 --rho 0.75
}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
log "mode=$MODE, GPUs=$NGPU"
if [ "$NGPU" -ge 2 ]; then          # T4 x2: both lanes in parallel
  G1=1
  lane0 > ../results/lane0.log 2>&1 &
  lane1 > ../results/lane1.log 2>&1 &
  wait
else                                # single GPU (P100 / one T4): the same work, one lane after the other
  G1=0
  lane0 > ../results/lane0.log 2>&1
  lane1 > ../results/lane1.log 2>&1
fi
cat ../results/lane0.log ../results/lane1.log
log "QUEUE_DONE"
