#!/bin/bash
# Exploration on Kaggle (GPU T4 x2): what would make boundary-aware allocation beat the learned strided conv?
#   orb / orr : oracle allocation (boundary tokens placed on the GT target boundary / region): the ceiling
#   supb/supr : question-aware BAC-q whose saliency is supervised with the GT boundary / region (training only)
#   r75       : BAC-v2 with 75% of the budget on boundary tokens
export USE_FLAX=0 USE_TF=0 PY=python
cd "$(dirname "$0")/.."
mkdir -p results/bottleneck results/logs
log() { echo "[$(date +%H:%M)] $*"; }
[ -f data/bottleneck/train_q.pt ] || { log "extracting features"; python bac/token_bottleneck/extract_features.py > results/logs/extract.log 2>&1 || { tail -40 results/logs/extract.log; exit 1; }; grep done results/logs/extract.log; }
R="bash bac/token_bottleneck/run.sh"; E=${EPOCHS:-6}
lane() { g=$1 t=$2
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g $R tb_bac2_${t}_orb $E $s --kind bac2 --tokens $t --select oracle_boundary
    CUDA_VISIBLE_DEVICES=$g $R tb_bac2_${t}_orr $E $s --kind bac2 --tokens $t --select oracle_region
    CUDA_VISIBLE_DEVICES=$g $R tb_bacq${t}_supb $E $s --kind bacq --tokens $t --sup boundary
    CUDA_VISIBLE_DEVICES=$g $R tb_bacq${t}_supr $E $s --kind bacq --tokens $t --sup region
    CUDA_VISIBLE_DEVICES=$g $R tb_bac2_${t}_r75 $E $s --kind bac2 --tokens $t --rho 0.75
  done; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
log "explore queue, GPUs=$NGPU, epochs=$E"
if [ "$NGPU" -ge 2 ]; then lane 0 16 2>&1 | tee results/logs/ex_lane16.log & lane 1 64 2>&1 | tee results/logs/ex_lane64.log & wait
else lane 0 16 2>&1 | tee results/logs/ex_lane16.log; lane 0 64 2>&1 | tee results/logs/ex_lane64.log; fi
log "EXPLORE_DONE"
