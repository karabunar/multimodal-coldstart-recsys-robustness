"""other designs of noise-augmented training (A shuffle, B fixed prototypes, C image side) against the clean-trained model and its M3 repair.

usage: python scripts/rq4_augmentation_designs_table.py   (reads results/rq4_augmentation_designs/*.json, results/rq4_augmentation_designs/gate_cause/*.json, results/repair/*.json; writes results/rq4_augmentation_designs.md)
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SEEDS = (2025, 2026, 2027)
DESIGNS = {'A': 'A shuffle', 'B': 'B fixed prototypes', 'C': 'C image side'}


def load(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


def main():
    clean_trained, m3 = {}, {}
    for s in SEEDS:
        r = load('results', 'repair', f'repair_hashtag_r0.40_seed{s}.json')['report']['noisy']
        clean_trained[s], m3[s] = r['unrepaired']['recall'], r['m3']['recall']
    out = ['## other designs of noise-augmented training at rate 0.4 (hashtag test noise 0.4)', '',
           '| design | seed | best epoch | noisy Recall@20 | clean-trained, same seed | clean-trained + M3 | clean Recall@20 | w_t on clean cold items | w_t on warm items |',
           '|---|---|---|---|---|---|---|---|---|']
    for d, label in DESIGNS.items():
        for s in SEEDS:
            run = load('results', 'rq4_augmentation_designs', f'stress_test_hashtag_r0.40_g8.0_seed{s}_design_{d}.json')
            gate = load('results', 'rq4_augmentation_designs', 'gate_cause', f'{d}_seed{s}.json')['report']
            out.append(f"| {label} | {s} | {run['best_epoch']} | {run['report']['noisy']['recall']['20']:.4f} | {clean_trained[s]:.4f} | {m3[s]:.4f} | "
                       f"{run['report']['clean']['recall']['20']:.4f} | {gate['w_t_cold']:.3f} | {gate['w_t_warm']:.3f} |")
    open(os.path.join(ROOT, 'results', 'rq4_augmentation_designs.md'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
