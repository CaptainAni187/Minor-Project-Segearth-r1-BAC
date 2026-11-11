#!/bin/bash
# WiSE-FT screening on calib300 only (selection set). alpha 1.0 reproduces the Kaggle ft_d4 run (sanity check).
cd "$(dirname "$0")/../.."
for a in 1.0 0.5 0.25 0.75; do
  .venv/bin/python -u exploration/scripts/eval_interp.py calib300 --alpha $a --ft ft_d4 2>&1 | grep -E '^\{|interpolated|Error|Traceback'
done
echo INTERP_CALIB_DONE
