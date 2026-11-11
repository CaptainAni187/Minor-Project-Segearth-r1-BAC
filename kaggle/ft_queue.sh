#!/bin/bash
# Mask-path fine-tuning from the released SegEarth-R1 weights (encoder, LLM and connector frozen).
#   ft_d4   : D4 augmentation (flips + 90-degree rotations); upstream trained with no geometric augmentation
#   ft_ctrl : identical but no augmentation (isolates augmentation from simply training longer)
# Evaluated on the full validation split (calib300 / test300 are sliced out offline), plus 8-way D4 TTA.
export USE_FLAX=0 USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
mkdir -p results/logs
log() { echo "[$(date +%H:%M)] $*"; }
S=${STEPS:-1500}; LR=${LR:-3e-5}
F='warn|pytree|Redirects|Special tokens|Found '
tr() { CUDA_VISIBLE_DEVICES=$1 python bac/train.py --name $2 --kind released --mask-path --steps $S --batch 1 --accum 4 --lr $LR "${@:3}" 2>&1 | grep -Evi "$F" | grep -E "mask path|step=(1|100|500|1000|1500) |saved|Error|Traceback"; }
ev() { CUDA_VISIBLE_DEVICES=$1 python bac/evaluate.py --name $2 "${@:3}" 2>&1 | grep -Evi "$F" | grep -E '"gIoU"|"cIoU"|Error|Traceback' | sed "s/^/$2 /"; }
laneA() { tr $1 ft_d4 --aug d4
  ev $1 ft_d4_full --weights runs/ft_d4/connector.pt --limit 0
  ev $1 ft_d4_tta8_test300 --weights runs/ft_d4/connector.pt --tta 8
  ev $1 ft_d4_tta8_calib300 --weights runs/ft_d4/connector.pt --tta 8 --subset calib300; }
laneB() { tr $1 ft_ctrl --aug none
  ev $1 ft_ctrl_full --weights runs/ft_ctrl/connector.pt --limit 0
  ev $1 released_tta8_test300 --tta 8
  ev $1 released_tta8_calib300 --tta 8 --subset calib300; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
log "ft queue, GPUs=$NGPU, steps=$S, lr=$LR"
if [ "$NGPU" -ge 2 ]; then laneA 0 2>&1 | tee results/logs/ft_laneA.log & laneB 1 2>&1 | tee results/logs/ft_laneB.log & wait
else laneA 0 2>&1 | tee results/logs/ft_laneA.log; laneB 0 2>&1 | tee results/logs/ft_laneB.log; fi
log "FT_DONE"
