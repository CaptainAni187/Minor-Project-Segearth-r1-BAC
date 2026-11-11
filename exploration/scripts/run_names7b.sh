#!/bin/bash
# After the val-rest dump: 7B names on the development sets (refusals + held-out for accuracy), names for val-rest
# refusals with both models, and the OV branch on all refusal sets that need it.
cd "$(dirname "$0")/../.."
L=exploration/logs; mkdir -p $L; LLM=exploration/.venv-llm/bin/python
$LLM exploration/scripts/predict_noun.py calib300 test300 --model Qwen/Qwen2.5-7B-Instruct --refusals-or-heldout > $L/nouns7b_dev.log 2>&1
$LLM exploration/scripts/predict_noun.py valrest --refusals-only > $L/nouns3b_valrest.log 2>&1
$LLM exploration/scripts/predict_noun.py valrest --model Qwen/Qwen2.5-7B-Instruct --refusals-only > $L/nouns7b_valrest.log 2>&1
for nm in Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct; do
  .venv/bin/python exploration/scripts/ov_branch.py valrest --nouns $nm --mode vocab --refusals-only 2>&1 | grep -E "done|Error|Traceback"
done
.venv/bin/python exploration/scripts/ov_branch.py calib300 test300 --nouns Qwen2.5-7B-Instruct --mode vocab --refusals-only 2>&1 | grep -E "done|Error|Traceback"
echo NAMES7B_DONE
