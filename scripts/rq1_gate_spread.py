"""spread of the gate's text weight w_t across cold items (clean-trained checkpoints)."""
import json
import sys

import numpy as np
import torch

from mamex_robustness.application.loading import load_session


@torch.no_grad()
def wt(session, text):
    m = session.model
    m.eval(); m.use_vgr = False
    m.set_eval_text(text)
    try:
        m()
    finally:
        m.set_eval_text(None)
    return m.last_gate_weights[:, 1].cpu().numpy()


def run(seed):
    ctx = load_session(seed, extra=[])
    args, data, s = ctx.args, ctx.data, ctx.session
    cold = np.asarray(data.cold_item_index); warm = np.asarray(data.warm_item_index)
    noisy = data.noisy_text_feat(0.4, 'hashtag', args.noise_seed)
    bad = cold[(noisy[cold] != data.text_feat[cold]).any(axis=1)]
    clean_w, noisy_w = wt(s, None), wt(s, noisy)
    return {'std_cold': float(clean_w[cold].std()), 'std_warm': float(clean_w[warm].std()), 'mean_cold': float(clean_w[cold].mean()),
            'corrupted_clean': float(clean_w[bad].mean()), 'corrupted_noisy': float(noisy_w[bad].mean()),
            'p5_p95_cold': [float(np.percentile(clean_w[cold], 5)), float(np.percentile(clean_w[cold], 95))]}


if __name__ == '__main__':
    res = {s: run(int(s)) for s in sys.argv[1:]}
    json.dump(res, open('results/rq1_gate_spread.json', 'w'), indent=1)
    a = lambda k: np.array([res[s][k] for s in res])
    lines = ['| quantity | mean over seeds | range |', '|---|---|---|']
    for k in ('std_cold', 'std_warm', 'mean_cold', 'corrupted_clean', 'corrupted_noisy'):
        lines.append(f'| {k} | {a(k).mean():.3f} | {a(k).min():.3f} to {a(k).max():.3f} |')
    d = a('corrupted_noisy') - a('corrupted_clean')
    lines.append(f'| shift of w_t on corrupted items | {d.mean():+.3f} | {d.min():+.3f} to {d.max():+.3f} |')
    lines.append(f"| seeds with std_cold < 0.10 | {int((a('std_cold') < 0.10).sum())}/{len(res)} | |")
    open('results/rq1_gate_spread.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
