#!/usr/bin/env bash
# Image-only baseline for the repairs: the text of suspect cold items is dropped (text weight 0) and the items are ranked
# by their image embedding alone. Runs on the saved clean-trained checkpoints (no training), hashtag noise at rate 0.4.
set -euo pipefail
export PYTHONPATH=src
DATA_PATH=${DATA_PATH:-processed_data_baby/cold}
OUT=${OUT:-results/rq2_image_only}
PY=${PY:-python}
SEEDS=${SEEDS:-"2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039"}
n=0
for seed in $SEEDS; do
  $PY -m mamex_robustness.repair_cli --data_path "$DATA_PATH" --seed "$seed" --noise_type hashtag --noise_rate 0.4 \
    --image_only 1 --out_dir "$OUT" &
  n=$((n+1)); if [ $((n % 5)) -eq 0 ]; then wait; fi
done
wait
