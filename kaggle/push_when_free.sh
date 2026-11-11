#!/bin/bash
# push_when_free.sh FOLDER: push a GPU kernel as soon as fewer than 2 of our GPU kernels are running.
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
while true; do
  n=0
  for k in bac-reproduction bac-followup bac-token-testbed; do
    $K kernels status animeshkumar187/$k 2>&1 | grep -qE "RUNNING|QUEUED" && n=$((n + 1))
  done
  echo "[$(date +%H:%M)] GPU kernels running: $n"
  [ $n -lt 2 ] && break
  sleep 120
done
$K kernels push -p "$1" 2>&1 | tail -1
