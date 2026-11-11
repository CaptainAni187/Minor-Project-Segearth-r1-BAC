#!/bin/bash
# Controls for the supervised-allocation result (boundary-supervised BAC-q beat the strided conv):
#   supbrand : identical model and supervision, but the boundary tokens are placed at random -> value of allocation
#   supbr0   : identical trunk + supervised question-aware saliency head, all tokens from the strided conv (rho=0)
#              -> value of the auxiliary supervision alone (budget prior off: its target would be 0)
export USE_FLAX=0 USE_TF=0 PY=python
cd "$(dirname "$0")/.."
mkdir -p results/bottleneck results/logs
log() { echo "[$(date +%H:%M)] $*"; }
[ -f data/bottleneck/train_q.pt ] || { log "extracting features"; python bac/token_bottleneck/extract_features.py > results/logs/extract.log 2>&1 || { tail -40 results/logs/extract.log; exit 1; }; grep done results/logs/extract.log; }
R="bash bac/token_bottleneck/run.sh"; E=${EPOCHS:-6}
lane() { g=$1 t=$2
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g $R tb_bacq${t}_supbrand $E $s --kind bacq --tokens $t --sup boundary --select random
    CUDA_VISIBLE_DEVICES=$g $R tb_bacq${t}_supbr0   $E $s --kind bacq --tokens $t --sup boundary --rho 0 --sal-weight 0
  done; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
log "explore2 queue, GPUs=$NGPU, epochs=$E"
if [ "$NGPU" -ge 2 ]; then lane 0 16 2>&1 | tee results/logs/ex2_lane16.log & lane 1 64 2>&1 | tee results/logs/ex2_lane64.log & wait
else lane 0 16 2>&1 | tee results/logs/ex2_lane16.log; lane 0 64 2>&1 | tee results/logs/ex2_lane64.log; fi
log "EXPLORE2_DONE"
