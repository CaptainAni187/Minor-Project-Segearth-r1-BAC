#!/bin/bash
# Token-bottleneck testbed grid (seed given as $1, default 0). Skips runs whose results exist.
cd "$(dirname "$0")/../.."
SEED=${1:-0}
PY=${PY:-.venv/bin/python}
run() { local name=$1; shift
  [ -f results/bottleneck/${name}_s${SEED}.json ] && { echo "skip $name"; return; }
  $PY bac/token_bottleneck/testbed.py --name ${name}_s${SEED} --seed $SEED "$@" 2>&1 | grep -E "gIoU|Error|Traceback"; }
run tb_pool1024        --kind pool    --tokens 1024
run tb_pool16          --kind pool    --tokens 16
run tb_bac16           --kind bac     --tokens 16
run tb_bac16_learned   --kind bac     --tokens 16 --alpha 1
run tb_uniform16       --kind uniform --tokens 16
run tb_pool64          --kind pool    --tokens 64
run tb_bac64           --kind bac     --tokens 64
run tb_bac64_learned   --kind bac     --tokens 64 --alpha 1
run tb_uniform64       --kind uniform --tokens 64
run tb_tome64          --kind tome    --tokens 64
run tb_random64        --kind random  --tokens 64
run tb_tome16          --kind tome    --tokens 16
run tb_random16        --kind random  --tokens 16
echo GRID_DONE
