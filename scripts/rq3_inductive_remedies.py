"""an inductive repair for warm-only-trained models (docs/hypotheses.md).

Usage: PYTHONPATH=src python scripts/rq3_inductive_remedies.py 2025 2026 2027   (writes results/rq3_inductive_remedies.json and results/rq3_inductive_remedies.md)
Models: results/rq3_warm_only/checkpoints (cold items left out of the auxiliary losses). Detector: ridge from raw image to raw text
features fitted on warm items only. Repair: blend flagged text toward the unit mean warm text ("muting").
"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import COSINE_GRID, LAM_GRID, _scores, _select, _vgr_cosine
from mamex_robustness.domain import repair

RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def fit_ridge(image, text, warm, seed=0):
    """Dual ridge image -> text on warm items; ridge weight picked on 20% held-out warm items by mean cosine."""
    X, Y = _unit(image[warm].astype(np.float64)), _unit(text[warm].astype(np.float64))
    xm, ym = X.mean(0), Y.mean(0)
    X, Y = X - xm, Y - ym
    perm = np.random.default_rng(seed).permutation(len(warm))
    hold, fit = perm[:len(warm) // 5], perm[len(warm) // 5:]

    def solve(idx, lam, Xq):
        K = X[idx] @ X[idx].T
        alpha = np.linalg.solve(K + lam * np.trace(K) / len(idx) * np.eye(len(idx)), Y[idx])
        return Xq @ X[idx].T @ alpha

    best = max(RIDGE_GRID, key=lambda lam: (_unit(solve(fit, lam, X[hold]) + ym) * _unit(Y[hold] + ym)).sum(1).mean())
    alpha = np.linalg.solve(X @ X.T + best * np.trace(X @ X.T) / len(warm) * np.eye(len(warm)), Y)
    return (lambda img: (_unit(img.astype(np.float64)) - xm) @ X.T @ alpha + ym), best


def run(seed):
    ctx = load_session(seed, extra=['--warm_only_aux', '1'], ckpt='results/rq3_warm_only/checkpoints/stress_test_hashtag_r0.40_gsel_seed{seed}_warmonly.pt')
    args, data, session = ctx.args, ctx.data, ctx.session
    ks = (10, 20)
    cold, warm = np.asarray(data.cold_item_index), np.asarray(data.warm_item_index)
    noisy_test = data.noisy_text_feat(0.4, 'hashtag', args.noise_seed)
    noisy_val = data.noisy_text_feat(0.4, 'hashtag', session.VAL_NOISE_SEED_OFFSET + args.noise_seed)
    corrupted = (noisy_test[cold] != data.text_feat[cold]).any(axis=1)
    predict, ridge_lam = fit_ridge(data.image_feat, data.text_feat, warm)
    predicted = predict(data.image_feat[cold])
    neutral = _unit(_unit(data.text_feat[warm].astype(np.float64)).mean(0, keepdims=True))
    neutral = np.repeat(neutral, len(cold), axis=0)

    def ridge_score(text):
        return (_unit(text[cold].astype(np.float64)) * _unit(predicted)).sum(1)

    def build(method, text, param):
        rows = text[cold]
        if method == 'mute_ridge':
            lam = repair.quantile_lambda(ridge_score(text), param[0], param[2], param[1])
        elif method == 'mute_vgr':
            lam = repair.quantile_lambda(_vgr_cosine(session, text)[cold], param[0], param[2], param[1])
        else:  # mute_oracle
            lam = np.where((rows != data.text_feat[cold]).any(axis=1), param, 0.0)
        return repair.with_cold_rows(text, cold, repair.blend(rows, neutral, lam))

    grids = {'mute_ridge': COSINE_GRID, 'mute_vgr': COSINE_GRID, 'mute_oracle': LAM_GRID}
    out = {'ridge_lambda': ridge_lam, 'clean': _scores(session, None, 'test', ks=ks)['recall'],
           'noisy': _scores(session, noisy_test, 'test', ks=ks)['recall'],
           'auroc_ridge': repair.auroc(-ridge_score(noisy_test), corrupted),
           'auroc_vgr_cosine': repair.auroc(-_vgr_cosine(session, noisy_test)[cold], corrupted)}
    for method, grid in grids.items():
        param, _ = _select(session, {p: build(method, noisy_val, p) for p in grid}, ks)
        out[method] = _scores(session, build(method, noisy_test, param), 'test', ks=ks)['recall']
        out[method + '_clean'] = _scores(session, build(method, data.text_feat, param), 'test', ks=ks)['recall']
    return out


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1:]]
    res = {s: run(s) for s in seeds}
    json.dump(res, open('results/rq3_inductive_remedies.json', 'w'), indent=1)
    n = len(res); m = lambda f: np.mean([res[s][f] for s in res])
    lines = ['| quantity | mean |', '|---|---|']
    for f in ('clean', 'noisy', 'mute_ridge', 'mute_vgr', 'mute_oracle', 'mute_ridge_clean', 'auroc_ridge', 'auroc_vgr_cosine'):
        lines.append(f'| {f} | {m(f):.4f} |')
    for f in ('mute_ridge', 'mute_vgr', 'mute_oracle'):
        gains = [res[s][f] - res[s]['noisy'] for s in res]
        lines.append(f'| {f} gain over noisy (seeds positive) | {np.mean(gains):+.4f} ({sum(g > 0 for g in gains)}/{n}) |')
    open('results/rq3_inductive_remedies.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
