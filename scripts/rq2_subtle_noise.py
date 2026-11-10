"""subtler corruptions (docs/hypotheses.md). Usage: PYTHONPATH=src python scripts/rq2_subtle_noise.py 2025 ... 2029"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import evaluate_repairs
from mamex_robustness.domain.noise import CorruptionSpec, TextNoiseSimulator

KINDS = ('hashtag_s0.4', 'nearshuffle')


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def make_noise(data, kind, rate, seed):
    cold, warm = np.asarray(data.cold_item_index), np.asarray(data.warm_item_index)
    if kind == 'hashtag_s0.4':
        return TextNoiseSimulator(CorruptionSpec(rate, 'hashtag', seed, strength=0.4))(data.text_feat, cold, warm)
    rng = np.random.default_rng(seed)
    bad = rng.choice(cold, size=int(round(rate * len(cold))), replace=False)  # same mask rule as the simulator
    sims = _unit(data.text_feat[bad].astype(np.float32)) @ _unit(data.text_feat[warm].astype(np.float32)).T
    near = warm[np.argpartition(-sims, 19, axis=1)[:, :20]]
    donors = near[np.arange(len(bad)), rng.integers(0, 20, size=len(bad))]
    out = data.text_feat.copy()
    out[bad] = data.text_feat[donors]
    return out


def run(seed, kind):
    ctx = load_session(seed, extra=[])
    args, data, session = ctx.args, ctx.data, ctx.session
    data.noisy_text_feat = lambda rate, noise_type='hashtag', s=2025: make_noise(data, kind, rate, s)
    r = evaluate_repairs(session, 0.4, 'hashtag', args.noise_seed)
    g = lambda cond, m: r[cond][m]['recall']
    return {'clean': g('clean', 'unrepaired'), 'noisy': g('noisy', 'unrepaired'), 'vgr': g('noisy', 'vgr'), 'm1': g('noisy', 'm1'),
            'm3': g('noisy', 'm3'), 'm3_random': g('noisy', 'm3_random'), 'm3_shuffled': g('noisy', 'm3_shuffled'),
            'auroc_cosine': r['auroc']['vgr_cosine'], 'n_corrupted': r['n_corrupted_cold']}


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1:]]
    res = {k: {s: run(s, k) for s in seeds} for k in KINDS}
    json.dump(res, open('results/rq2_subtle_noise.json', 'w'), indent=1)
    lines = ['| noise | clean | noisy | VGR | M1 | M3 | AUROC cosine | seeds: noisy<clean | M3>noisy | M3>=M1 | M3>both controls |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for k in KINDS:
        d = res[k]; n = len(d); m = lambda f: np.mean([d[s][f] for s in d]); w = lambda a, b: sum(d[s][a] > d[s][b] for s in d)
        lines.append(f"| {k} | {m('clean'):.4f} | {m('noisy'):.4f} | {m('vgr'):.4f} | {m('m1'):.4f} | {m('m3'):.4f} | {m('auroc_cosine'):.3f} | "
                     f"{w('clean', 'noisy')}/{n} | {w('m3', 'noisy')}/{n} | {sum(d[s]['m3'] >= d[s]['m1'] for s in d)}/{n} | "
                     f"{sum(d[s]['m3'] > max(d[s]['m3_random'], d[s]['m3_shuffled']) for s in d)}/{n} |")
    open('results/rq2_subtle_noise.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
