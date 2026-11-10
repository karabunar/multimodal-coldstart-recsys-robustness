"""mean gate text weight on trained warm items, held-out warm items and cold items (clean text), from results/rq4_held_out/*.json.

usage: python scripts/rq4_held_out_table.py   (writes results/rq4_held_out.md)
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    rows = []
    for kind, label in (('heldout_aug', 'noise-augmented (rate 0.4)'), ('heldout_clean', 'clean-trained')):
        for path in sorted(glob.glob(os.path.join(ROOT, 'results', 'rq4_held_out', f'*{kind}*.json'))):
            with open(path) as f:
                data = json.load(f)
            probe = data['report']['holdout_probe']
            rows.append((label, data['args']['seed'], probe, data['best_epoch']))
    out = ['## gate text weight w_t by group when 10% of the warm items have no training interactions (clean text)', '',
           '| model | seed | best epoch | trained warm | held-out warm | cold | held-out items |', '|---|---|---|---|---|---|---|']
    for label, seed, p, epoch in rows:
        out.append(f"| {label} | {seed} | {epoch} | {p['trained_warm']['w_t']:.3f} | {p['heldout_warm']['w_t']:.3f} | {p['cold']['w_t']:.3f} | {p['n_heldout_warm']} |")
    text = '\n'.join(out) + '\n'
    with open(os.path.join(ROOT, 'results', 'rq4_held_out.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
