#!/usr/bin/env bash
# Repair comparison with real clickbait vectors instead of prototypes drawn from warm items, on the saved clean-trained
# checkpoints of seeds 2025..2039 (no training). STRENGTH=1.0 replaces the item's text instead of mixing it (sensitivity check).
set -euo pipefail
export PYTHONPATH=src
DATA_PATH=${DATA_PATH:-processed_data_baby/cold}
PROTOS=${PROTOS:-data_maps/clickbait_prototypes.npy}
STRENGTH=${STRENGTH:-0.8}
OUT=${OUT:-results/rq2_real_clickbait}
PY=${PY:-python}
SEEDS=${SEEDS:-"2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039"}
n=0
for seed in $SEEDS; do
  $PY -m mamex_robustness.repair_cli --data_path "$DATA_PATH" --seed "$seed" --noise_type realtext --noise_rate 0.4 \
    --noise_strength "$STRENGTH" --real_proto_file "$PROTOS" --out_dir "$OUT" &
  n=$((n+1)); if [ $((n % 5)) -eq 0 ]; then wait; fi
done
wait
