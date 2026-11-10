"""do the shuffled and constant VGR controls change when they act among the cold items only?

Loads the clean-trained checkpoints (results/ckpt291/checkpoints), selects gamma on noisy validation data (as in the main grid),
and evaluates VGR and both controls on noisy test text with the control scope 'all' (earlier runs: whole catalogue) and 'cold'.
Usage: PYTHONPATH=src python scripts/rq2_vgr_control_scope.py 2025 2026 ...   (writes results/rq2_vgr_control_scope.json and results/rq2_vgr_control_scope.md)
"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session

GRID = [1, 2, 4, 8, 16]


def run(seed):
    ctx = load_session(seed, extra=['--gamma_grid'] + [str(g) for g in GRID])
    args, model, session = ctx.args, ctx.model, ctx.session
    out = {}
    all_scope = None
    cold_scope = model.control_scope
    assert cold_scope is not None
    for name, scope in (('all', all_scope), ('cold', cold_scope)):
        model.control_scope = scope
        rep = session.final_report(0.4, 'hashtag', args.noise_seed, GRID)
        out[name] = {k: rep[k]['recall']['20'] for k in
                     ('noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant')}
        out[name]['ndcg20'] = {k: rep[k]['ndcg']['20'] for k in ('noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant')}
        out[name]['gamma'] = rep['gamma_selection']['selected']
    return out


if __name__ == '__main__':
    res = {s: run(int(s)) for s in sys.argv[1:]}
    json.dump(res, open('results/rq2_vgr_control_scope.json', 'w'), indent=1)
    rows = ['| scope | noisy | VGR | shuffled | constant | VGR-shuffled | VGR-constant |', '|---|---|---|---|---|---|---|']
    for sc in ('all', 'cold'):
        m = {k: np.mean([res[s][sc][k] for s in res]) for k in ('noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant')}
        rows.append(f"| {sc} | {m['noisy']:.4f} | {m['noisy_vgr']:.4f} | {m['noisy_vgr_shuffled']:.4f} | {m['noisy_vgr_constant']:.4f} | "
                    f"{m['noisy_vgr']-m['noisy_vgr_shuffled']:+.4f} | {m['noisy_vgr']-m['noisy_vgr_constant']:+.4f} |")
    open('results/rq2_vgr_control_scope.md', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))
