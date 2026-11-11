#!/bin/bash
# Train + evaluate every configuration sequentially (one GPU job at a time).
# Usage: bash bac/run_queue.sh [steps]      -- skips runs whose results already exist
cd "$(dirname "$0")"
PY=../.venv/bin/python
STEPS=${1:-200}
FILTER='warn|pytree|Redirects|Special tokens|Found '

run() {  # name, train args...
  local name=$1; shift
  if [ ! -f ../results/${name}_val.json ]; then
    [ -f ../runs/$name/connector.pt ] || $PY train.py --name $name --steps $STEPS --amp "$@" 2>&1 | grep -Evi "$FILTER"
    $PY evaluate.py --name $name --weights ../runs/$name/connector.pt 2>&1 | grep -Evi "$FILTER" | tail -20
  fi
}

run bac64      --kind bac     --tokens 64
run pool64     --kind pool    --tokens 64
refresh() { $PY analyze.py > /dev/null 2>&1; }
refresh
run bac16      --kind bac     --tokens 16
run pool16     --kind pool    --tokens 16
refresh
run bac64_lam0 --kind bac     --tokens 64 --sal-weight 0
run uniform64  --kind uniform --tokens 64
$PY analyze.py
echo QUEUE_DONE
