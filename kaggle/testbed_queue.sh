#!/bin/bash
# Token-bottleneck testbed on Kaggle (GPU T4 x2): extract features once, then two lanes x 3 seeds.
export USE_FLAX=0 USE_TF=0
cd "$(dirname "$0")/.."
export PY=python
mkdir -p results/bottleneck results/logs
log() { echo "[$(date +%H:%M)] $*"; }
[ -f data/bottleneck/train_q.pt ] || { log "extracting features"; python bac/token_bottleneck/extract_features.py > results/logs/extract.log 2>&1 || { tail -40 results/logs/extract.log; exit 1; }; grep done results/logs/extract.log; }
R="bash bac/token_bottleneck/run.sh"      # via bash: Kaggle drops the executable bit when it unpacks the dataset
E=${EPOCHS:-6}
core64() {  # gpu seed
  CUDA_VISIBLE_DEVICES=$1 $R tb_pool64           $E $2 --kind pool    --tokens 64
  CUDA_VISIBLE_DEVICES=$1 $R tb_uniform64        $E $2 --kind uniform --tokens 64
  CUDA_VISIBLE_DEVICES=$1 $R tb_bac64            $E $2 --kind bac     --tokens 64
  CUDA_VISIBLE_DEVICES=$1 $R tb_bac2_64          $E $2 --kind bac2    --tokens 64
  CUDA_VISIBLE_DEVICES=$1 $R tb_bac2_64_randsel  $E $2 --kind bac2    --tokens 64 --select random
}
laneA() { core64 $1 0; core64 $1 1; }
laneB() { core64 $1 2
  CUDA_VISIBLE_DEVICES=$1 $R tb_pool1024 $E 1 --kind pool --tokens 1024
  CUDA_VISIBLE_DEVICES=$1 $R tb_pool1024 $E 2 --kind pool --tokens 1024; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
log "testbed queue, GPUs=$NGPU, epochs=$E"
if [ "$NGPU" -ge 2 ]; then laneA 0 2>&1 | tee results/logs/tb_laneA.log & laneB 1 2>&1 | tee results/logs/tb_laneB.log & wait
else laneA 0 2>&1 | tee results/logs/tb_laneA.log; laneB 0 2>&1 | tee results/logs/tb_laneB.log; fi
log "TESTBED_DONE"
