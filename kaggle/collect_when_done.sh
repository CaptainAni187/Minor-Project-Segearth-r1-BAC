#!/bin/bash
# collect_when_done.sh KERNEL OUTDIR: wait for a kernel to finish, then download its log and results zip.
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
while $K kernels status animeshkumar187/$1 2>&1 | grep -qE "RUNNING|QUEUED"; do sleep 300; done
$K kernels status animeshkumar187/$1
mkdir -p $2; $K kernels output animeshkumar187/$1 -p $2 --page-size 200 --file-pattern '.*\.(zip|txt)$' 2>&1 | tail -2
for z in $2/*.zip; do [ -f "$z" ] && unzip -oq "$z" -d $2/unz; done
python3 -c "
import json,sys; t=''.join(e.get('data','') for e in json.load(open('$2/$1.log'))); print(t[-4000:])"
