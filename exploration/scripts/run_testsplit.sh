#!/bin/bash
# Pre-registered final check of policy P* on the official EarthReason test split (see NOTES.md).
cd "$(dirname "$0")/../.."
set -e
L=exploration/logs; mkdir -p $L
.venv/bin/python exploration/scripts/dump_logits.py testsplit --no-flip --meta-only 2>&1 | grep -v -i "warn\|redirect\|special tokens\|Found " > $L/dump_testsplit.log
tail -1 $L/dump_testsplit.log
exploration/.venv-llm/bin/python exploration/scripts/predict_noun.py testsplit --refusals-only > $L/nouns_testsplit.log 2>&1
tail -1 $L/nouns_testsplit.log
.venv/bin/python exploration/scripts/ov_branch.py testsplit --nouns Qwen2.5-3B-Instruct --mode vocab --refusals-only 2>&1 | grep -v -i "warn\|Found " > $L/ov_testsplit.log
tail -1 $L/ov_testsplit.log
echo TESTSPLIT_DONE
