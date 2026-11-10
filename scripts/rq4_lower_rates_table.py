"""noise-augmented training at lower rates: noisy test Recall@20 and the gate's text weight on clean cold items.

usage: python scripts/rq4_lower_rates_table.py   (reads results/rq4_lower_rates/*.json and results/rq4_lower_rates/gate_cause/*.json, writes results/rq4_lower_rates.md)
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CLEAN_TRAINED = {2025: 0.0667, 2026: 0.0652, 2027: 0.0685}  # noisy Recall@20 of the clean-trained model of the same seed (results/rq2_repair_ten_seeds.md)


def main():
    out = ['## noise-augmented training at lower rates (hashtag test noise 0.4)', '',
           '| training rate | seed | best epoch | noisy Recall@20 | clean-trained model, same seed | clean Recall@20 | w_t on clean cold items | w_t on warm items |', '|---|---|---|---|---|---|---|---|']
    for rate in ('0.1', '0.2'):
        for seed in (2025, 2026, 2027):
            name = f'stress_test_hashtag_r0.40_g8.0_seed{seed}_rate{rate}'
            with open(os.path.join(ROOT, 'results', 'rq4_lower_rates', name + '.json')) as f:
                run = json.load(f)
            with open(os.path.join(ROOT, 'results', 'rq4_lower_rates', 'gate_cause', f'r{rate}_seed{seed}.json')) as f:
                gate = json.load(f)['report']
            out.append(f"| {rate} | {seed} | {run['best_epoch']} | {run['report']['noisy']['recall']['20']:.4f} | {CLEAN_TRAINED[seed]:.4f} | "
                       f"{run['report']['clean']['recall']['20']:.4f} | {gate['w_t_cold']:.3f} | {gate['w_t_warm']:.3f} |")
    open(os.path.join(ROOT, 'results', 'rq4_lower_rates.md'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
