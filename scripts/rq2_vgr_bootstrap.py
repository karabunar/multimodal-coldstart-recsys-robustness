"""Bootstrap over test users with the controls acting among the cold items (seeds 2025 to 2027, ckpt291 checkpoints).
Usage: PYTHONPATH=src python scripts/rq2_vgr_bootstrap.py   (writes results/rq2_vgr_bootstrap.md)"""
import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.reporting.ranking_metrics import bootstrap_means, user_metrics

GRID = [1, 2, 4, 8, 16]
CONDS = ['clean', 'noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant']
per_seed = []
for seed in (2025, 2026, 2027):
    ctx = load_session(seed, extra=[])
    args, data, session = ctx.args, ctx.data, ctx.session
    session.final_report(0.4, 'hashtag', args.noise_seed, GRID)  # selects gamma on noisy validation data and keeps it
    diag = session.diagnostics(0.4, 'hashtag', args.noise_seed)
    per_seed.append({c: user_metrics(diag['users'], diag['ranks'][c]) for c in CONDS})
values = {c: np.stack([s[c]['recall@20'] for s in per_seed]) for c in CONDS}
diffs = [('noisy_vgr', 'noisy'), ('noisy_vgr', 'noisy_vgr_shuffled'), ('noisy_vgr', 'noisy_vgr_constant'), ('noisy_vgr_shuffled', 'noisy'), ('noisy_vgr_constant', 'noisy')]
res = bootstrap_means(values, n_boot=1000, seed=0, diffs=diffs)
lines = ['| Recall@20 quantity (3 seeds, cold-scope controls) | mean | 95% interval over test users |', '|---|---|---|']
for k, v in res.items():
    if ' - ' in k:
        lines.append(f"| {k} | {v['mean']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] |")
open('results/rq2_vgr_bootstrap.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
