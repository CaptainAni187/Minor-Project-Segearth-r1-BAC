#!/bin/bash
# Larger fallback models on the refusals of all three validation sets (7B names): SAM-L, OWLv2-L, both.
cd "$(dirname "$0")/../.."
for cfg in "large base" "base large" "large large"; do
  set -- $cfg
  .venv/bin/python exploration/scripts/ov_branch.py calib300 test300 valrest --nouns Qwen2.5-7B-Instruct --mode vocab --refusals-only --sam $1 --owl $2 2>&1 | grep -E "done|Error|Traceback" | sed "s/^/sam=$1 owl=$2 /"
done
echo OV_LARGE_DONE
