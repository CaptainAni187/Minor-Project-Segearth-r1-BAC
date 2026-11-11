#!/bin/bash
# Core 16-token comparison, seeds given as arguments (default 1 2). Skips finished runs.
cd "$(dirname "$0")/../.."
R=bac/token_bottleneck/run.sh
for s in "${@:-1 2}"; do
  $R tb_pool16        6 $s --kind pool    --tokens 16
  $R tb_uniform16     6 $s --kind uniform --tokens 16
  $R tb_bac16         6 $s --kind bac     --tokens 16
  $R tb_bac2_16       6 $s --kind bac2    --tokens 16
  $R tb_bac2_16_randsel 6 $s --kind bac2  --tokens 16 --select random
done
echo CORE16_DONE
