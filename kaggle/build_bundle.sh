#!/bin/bash
# Rebuild kaggle/upload/dataset/bac_code.zip from the current code (+ Mac results as reference_results/).
set -e
cd "$(dirname "$0")/.."
U=kaggle/upload; S=$(mktemp -d)
mkdir -p $S/reference_results
cp results/*_val.csv results/*_val.json results/summary.csv results/stratified.csv results/val_subset_ids.txt $S/reference_results/ 2>/dev/null || true
rm -f $U/dataset/bac_code.zip
zip -qr $U/dataset/bac_code.zip bac/*.py bac/*.sh bac/token_bottleneck/*.py bac/token_bottleneck/*.sh kaggle/*.sh \
  requirements.txt README.md SegEarth-R1 -x 'SegEarth-R1/.git/*' '*/__pycache__/*'
(cd $S && zip -qr "$OLDPWD/$U/dataset/bac_code.zip" reference_results)
rm -rf $S
cp $U/dataset/bac_code.zip kaggle/bac_code.zip
unzip -l $U/dataset/bac_code.zip | tail -1
