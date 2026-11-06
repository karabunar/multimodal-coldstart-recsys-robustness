#!/usr/bin/env bash
# ten training jobs share one GPU, then the repair evaluations, the gate analyses and the tables.
# clean baselines, seeds 2035-2039 (with checkpoints). Seed 2027 for the augmented model (rate 0.4),
# the held-out-item probe and the lower augmentation rates. The gate-subspace analysis reads the checkpoints afterwards.
# Needs torch 2.9.1 (see docs/reproducibility.md). Training takes several hours; set STAGE=after to skip it.
set -euo pipefail
export PYTHONPATH=src
PY=${PY:-python}
D="--data_path ${DATA_PATH:-processed_data_baby/cold}"
if [ "${STAGE:-all}" != after ]; then
  for s in 2035 2036 2037 2038 2039; do
    $PY main.py $D --mode baseline --seed $s --save_final 1 --results_dir results/ckpt291 > /dev/null 2>&1 &
  done
  $PY main.py $D --mode stress_test --seed 2027 --noise_rate 0.4 --train_noise_rate 0.4 --save_final 1 --results_dir results/ckpt_aug > /dev/null 2>&1 &
  $PY main.py $D --mode stress_test --seed 2027 --noise_rate 0.4 --train_noise_rate 0.4 --holdout_warm 0.1 --holdout_probe 1 --suffix heldout_aug --results_dir results/rq4_held_out > /dev/null 2>&1 &
  $PY main.py $D --mode stress_test --seed 2027 --noise_rate 0.4 --train_noise_rate 0.0 --holdout_warm 0.1 --holdout_probe 1 --suffix heldout_clean --results_dir results/rq4_held_out > /dev/null 2>&1 &
  for r in 0.1 0.2; do
    $PY main.py $D --mode stress_test --seed 2027 --noise_rate 0.4 --train_noise_rate $r --save_final 1 --suffix rate$r --results_dir results/rq4_lower_rates > /dev/null 2>&1 &
  done
  wait
fi
for s in 2035 2036 2037 2038 2039; do
  $PY -m mamex_robustness.repair_cli $D --seed $s --noise_rate 0.4
done
for r in 0.1 0.2; do
  $PY -m mamex_robustness.gate_cause_cli $D --seed 2027 --out results/rq4_lower_rates/gate_cause/r${r}_seed2027.json \
    --ckpt results/rq4_lower_rates/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2027_rate${r}.pt
done
$PY scripts/rq2_repair_seeds.py 2039 rq2_repair_fifteen_seeds      # results/rq2_repair_fifteen_seeds.md (15 seeds)
$PY scripts/rq4_held_out_table.py
$PY scripts/rq4_lower_rates_table.py
for sc in rq4_dose_response rq4_trace rq4_user_direction rq4_user_subspace rq4_gate_subspace; do
  $PY scripts/$sc.py
done
