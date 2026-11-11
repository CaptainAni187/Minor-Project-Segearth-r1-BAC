#!/bin/bash
# run.sh NAME EPOCHS SEED testbed-args...   (bash, so argument handling is predictable)
# Full output goes to results/logs/runs/NAME_sSEED.log; the console gets progress lines, and the log tail on failure.
cd "$(dirname "$0")/../.."
NAME=$1 EPOCHS=$2 SEED=$3; shift 3
OUT=results/bottleneck
for ((i = 1; i <= $#; i++)); do [ "${!i}" = "--out-dir" ] && { j=$((i + 1)); OUT=${!j}; }; done
[ -f $OUT/${NAME}_s${SEED}.json ] && { echo "skip ${NAME}_s${SEED}"; exit 0; }
mkdir -p results/logs/runs
L=results/logs/runs/${NAME}_s${SEED}.log
${PY:-.venv/bin/python} bac/token_bottleneck/testbed.py --name ${NAME}_s${SEED} --epochs $EPOCHS --seed $SEED "$@" > $L 2>&1
rc=$?
grep -E "steps on|step': (0|600|1200|1775)|gIoU" $L
if [ $rc -ne 0 ] || [ ! -f $OUT/${NAME}_s${SEED}.json ]; then
  echo "FAILED ${NAME}_s${SEED} (exit $rc) -- last lines of $L:"; tail -40 $L; exit 1
fi
