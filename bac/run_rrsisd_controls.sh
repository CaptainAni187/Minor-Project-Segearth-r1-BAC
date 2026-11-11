#!/bin/bash
# RRSIS-D token controls: does the referring model use its visual tokens?
cd "$(dirname "$0")/.."
export SEGEARTH_CKPT=$PWD/checkpoints/SegEarth-R1-RRSIS-D
run() { local n=$1; shift
  [ -f results/rrsisd/${n}_val.json ] && { echo "skip $n"; return; }
  .venv/bin/python bac/evaluate.py --dataset rrsisd --name $n --out results/rrsisd "$@" 2>&1 | grep -E '"gIoU"|"cIoU"|Error|Traceback' | sed "s/^/$n /"; }
run rrsisd_released
run rrsisd_zero      --kind zero
run rrsisd_wrong_far --kind wrong_far
run rrsisd_none      --kind none
run rrsisd_wrongq    --shuffle-questions
echo RRSISD_DONE
