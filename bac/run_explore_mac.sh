#!/bin/bash
# Mac side of the exploration: (1) does the LLM's *text answer* use the visual tokens, and does the connector
# matter there? (2) mask-path improvements without retraining: held-out threshold calibration, 4-way flip TTA.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
O=results/answer_nll
nll() { local n=$1; shift
  [ -f $O/${n}_val.json ] && { echo "skip $n"; return; }
  $PY bac/evaluate.py --name $n --answer-nll --out $O "$@" 2>&1 | grep -E '"answer_nll"|"answer_top1"|Error|Traceback' | sed "s/^/$n /"; }
nll released
nll zero      --kind zero
nll wrong_far --kind wrong_far
nll none      --kind none
nll wrongq    --shuffle-questions
for r in uniform64 bac64 pool64 bac64_learned uniform16 bac16 pool16 tome64 random64; do nll $r --weights runs/$r/connector.pt; done
echo NLL_DONE
[ -f results/calibration/summary.json ] || $PY bac/calibrate_threshold.py 2>&1 | grep -v -i "warn\|redirect\|special tokens\|Found " | tail -30
echo CALIB_DONE
[ -f results/released_tta4_val.json ] || $PY bac/evaluate.py --name released_tta4 --tta 4 2>&1 | grep -E '"gIoU"|"cIoU"|Error|Traceback'
echo EXPLORE_MAC_DONE
