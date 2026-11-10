"""confirmatory evaluation on unused seeds. Usage: PYTHONPATH=src python scripts/rq3_confirmation_run.py 2040 ... 2044
Reads results/rq3_confirmation/checkpoints (mode baseline, --save_final 1); writes results/rq3_confirmation.json and results/rq3_confirmation.md."""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import evaluate_repairs

GRID = [1, 2, 4, 8, 16]


def run(seed):
    ctx = load_session(seed, extra=['--gamma_grid'] + [str(g) for g in GRID], ckpt='results/rq3_confirmation/checkpoints/baseline_hashtag_r0.00_g8.0_seed{seed}.pt')
    args, data, model, session = ctx.args, ctx.data, ctx.model, ctx.session
    assert model.control_scope is not None
    rep = session.final_report(0.4, 'hashtag', args.noise_seed, GRID)
    r = evaluate_repairs(session, 0.4, 'hashtag', args.noise_seed)
    g = lambda cond, m: r[cond][m]['recall']
    out = {k: rep[k]['recall']['20'] for k in ('clean', 'noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant')}
    out.update({'m1': g('noisy', 'm1'), 'm3': g('noisy', 'm3'), 'auroc_cosine': r['auroc']['vgr_cosine']})
    return out


if __name__ == '__main__':
    res = {s: run(int(s)) for s in sys.argv[1:]}
    json.dump(res, open('results/rq3_confirmation.json', 'w'), indent=1)
    n = len(res); m = lambda f: np.mean([res[s][f] for s in res]); w = lambda a, b: sum(res[s][a] > res[s][b] for s in res)
    lines = ['| quantity | value |', '|---|---|',
             f"| clean / noisy / VGR / M1 / M3 (mean Recall@20) | {m('clean'):.4f} / {m('noisy'):.4f} / {m('noisy_vgr'):.4f} / {m('m1'):.4f} / {m('m3'):.4f} |",
             f"| a: noisy < clean (seeds); mean loss | {w('clean', 'noisy')}/{n}; {m('clean') - m('noisy'):.4f} |",
             f"| b: VGR > noisy | {w('noisy_vgr', 'noisy')}/{n} |",
             f"| c: M3 > noisy | {w('m3', 'noisy')}/{n} |",
             f"| d: M3 >= M1 (seeds); mean M3-M1 | {sum(res[s]['m3'] >= res[s]['m1'] for s in res)}/{n}; {m('m3') - m('m1'):+.4f} |",
             f"| e: VGR > shuffled; mean VGR-shuffled | {w('noisy_vgr', 'noisy_vgr_shuffled')}/{n}; {m('noisy_vgr') - m('noisy_vgr_shuffled'):+.4f} |",
             f"| f: AUROC cosine >= 0.99 (seeds); mean | {sum(res[s]['auroc_cosine'] >= 0.99 for s in res)}/{n}; {m('auroc_cosine'):.4f} |"]
    open('results/rq3_confirmation.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
