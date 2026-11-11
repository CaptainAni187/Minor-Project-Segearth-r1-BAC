#!/bin/bash
# Phase 2 on Kaggle (GPU T4 x2): diagnostics + testbed follow-ups. Skips anything already done.
export USE_FLAX=0 USE_TF=0 PY=python
cd "$(dirname "$0")/.."
mkdir -p results/bottleneck results/logs
log() { echo "[$(date +%H:%M)] $*"; }
log "diagnostics"; CUDA_VISIBLE_DEVICES=0 python bac/diagnostics.py 24 2>&1 | tail -25
[ -f data/bottleneck/train_q.pt ] || { log "extracting features"; CUDA_VISIBLE_DEVICES=0 python bac/token_bottleneck/extract_features.py > results/logs/extract.log 2>&1 || { tail -40 results/logs/extract.log; exit 1; }; grep done results/logs/extract.log; }
R="bash bac/token_bottleneck/run.sh"; E=${EPOCHS:-6}
nm() { case $1 in *[0-9]) echo "tb_$1_$2";; *) echo "tb_$1$2";; esac; }   # tb_pool32, but tb_bac2_32 (not tb_bac232)
laneA() { g=$1
  for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$g $R tb_bacq16 $E $s --kind bacq --tokens 16; CUDA_VISIBLE_DEVICES=$g $R tb_bacq64 $E $s --kind bacq --tokens 64; done
  for k in uniform pool bac2 bacq; do for m in 32 128; do CUDA_VISIBLE_DEVICES=$g $R $(nm $k $m) $E 0 --kind $k --tokens $m; done; done
  CUDA_VISIBLE_DEVICES=$g $R tb_pool16 $E 0 --kind pool --tokens 16
  CUDA_VISIBLE_DEVICES=$g $R tb_uniform16 $E 0 --kind uniform --tokens 16; }
laneB() { g=$1
  for s in 0 1 2; do for k in tome random; do for m in 16 64; do CUDA_VISIBLE_DEVICES=$g $R tb_${k}${m} $E $s --kind $k --tokens $m; done; done; done
  CUDA_VISIBLE_DEVICES=$g $R tb_bac16 $E 0 --kind bac --tokens 16
  CUDA_VISIBLE_DEVICES=$g $R tb_bac2_16 $E 0 --kind bac2 --tokens 16
  CUDA_VISIBLE_DEVICES=$g $R tb_bac2_16_randsel $E 0 --kind bac2 --tokens 16 --select random; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
log "follow-up queue, GPUs=$NGPU, epochs=$E"
if [ "$NGPU" -ge 2 ]; then laneA 0 2>&1 | tee results/logs/fu_laneA.log & laneB 1 2>&1 | tee results/logs/fu_laneB.log & wait
else laneA 0 2>&1 | tee results/logs/fu_laneA.log; laneB 0 2>&1 | tee results/logs/fu_laneB.log; fi
log "FOLLOWUP_DONE"
