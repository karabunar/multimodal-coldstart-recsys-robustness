#!/usr/bin/env bash
# other designs of noise-augmented training (A shuffle, B fixed prototypes, C image side), seeds 2025 to 2027, nine jobs at once.
# Needs torch 2.9.1 (see docs/reproducibility.md). Afterwards: gate cause analysis per checkpoint and scripts/rq4_augmentation_designs_table.py.
set -euo pipefail
export PYTHONPATH=src
PY=${PY:-python}
D="--data_path ${DATA_PATH:-processed_data_baby/cold}"
COMMON="--mode stress_test --noise_rate 0.4 --save_final 1 --results_dir results/rq4_augmentation_designs"
if [ "${STAGE:-all}" != after ]; then
  for s in 2025 2026 2027; do
    $PY main.py $D $COMMON --seed $s --train_noise_rate 0.4 --train_noise_type shuffle --suffix design_A > /dev/null 2>&1 &
    $PY main.py $D $COMMON --seed $s --train_noise_rate 0.4 --train_fixed_prototypes 1 --suffix design_B > /dev/null 2>&1 &
    $PY main.py $D $COMMON --seed $s --train_image_noise_rate 0.4 --suffix design_C > /dev/null 2>&1 &
  done
  wait
fi
for d in A B C; do
  for s in 2025 2026 2027; do
    $PY -m mamex_robustness.gate_cause_cli $D --seed $s --out results/rq4_augmentation_designs/gate_cause/${d}_seed${s}.json \
      --ckpt results/rq4_augmentation_designs/checkpoints/stress_test_hashtag_r0.40_g8.0_seed${s}_design_${d}.pt
  done
done
$PY scripts/rq4_augmentation_designs_table.py
