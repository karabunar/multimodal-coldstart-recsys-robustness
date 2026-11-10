"""H43: a contrastive image-text detector for the warm-only models (docs/hypotheses.md).

Usage: PYTHONPATH=src python scripts/rq3_contrastive_detector.py 2025 2026 ...   (writes results/rq3_contrastive_detector.json and .md)
Models: results/rq3_warm_only/checkpoints. Detector: two MLP towers trained with InfoNCE on warm items only (clean text).
Repair: M3, the blend toward the text of the ten image-nearest warm items, with the weight from the detector's score.
Controls: VGR's cosine, a random score, the true mask. Parameters are chosen on noisy validation data, never on the test set.
"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import COSINE_GRID, LAM_GRID, _scores, _select, _vgr_cosine
from mamex_robustness.domain import contrastive, repair


def run(seed):
    ctx = load_session(seed, extra=['--warm_only_aux', '1'], ckpt='results/rq3_warm_only/checkpoints/stress_test_hashtag_r0.40_gsel_seed{seed}_warmonly.pt')
    args, data, session = ctx.args, ctx.data, ctx.session
    ks = (10, 20)
    cold, warm = np.asarray(data.cold_item_index), np.asarray(data.warm_item_index)
    noisy_test = data.noisy_text_feat(0.4, 'hashtag', args.noise_seed)
    noisy_val = data.noisy_text_feat(0.4, 'hashtag', session.VAL_NOISE_SEED_OFFSET + args.noise_seed)
    corrupted = (noisy_test[cold] != data.text_feat[cold]).any(axis=1)
    score_fn = contrastive.fit(data.image_feat[warm], data.text_feat[warm], seed=seed)
    neighbour = repair.neighbor_text(data.image_feat, data.text_feat, warm, cold, k=10)
    random_score = np.random.default_rng(seed).normal(size=len(cold))  # one fixed random score per seed

    def score(method, text):
        if method == 'm3_contrastive':
            return score_fn(data.image_feat[cold], text[cold])
        if method == 'm3_vgr':
            return _vgr_cosine(session, text)[cold]
        return random_score  # m3_random

    def build(method, text, param):
        rows = text[cold]
        if method == 'm3_oracle':
            lam = np.where((rows != data.text_feat[cold]).any(axis=1), param, 0.0)
        else:
            lam = repair.quantile_lambda(score(method, text), param[0], param[2], param[1])
        return repair.with_cold_rows(text, cold, repair.blend(rows, neighbour, lam))

    out = {'clean': _scores(session, None, 'test', ks=ks)['recall'], 'noisy': _scores(session, noisy_test, 'test', ks=ks)['recall'],
           'auroc_contrastive': repair.auroc(-score('m3_contrastive', noisy_test), corrupted),
           'auroc_vgr_cosine': repair.auroc(-score('m3_vgr', noisy_test), corrupted)}
    for method, grid in (('m3_contrastive', COSINE_GRID), ('m3_vgr', COSINE_GRID), ('m3_random', COSINE_GRID), ('m3_oracle', LAM_GRID)):
        param, _ = _select(session, {p: build(method, noisy_val, p) for p in grid}, ks)
        out[method] = _scores(session, build(method, noisy_test, param), 'test', ks=ks)['recall']
        out[method + '_clean'] = _scores(session, build(method, data.text_feat, param), 'test', ks=ks)['recall']
    return out


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1:]]
    res = {}
    for s in seeds:
        res[s] = run(s)
        json.dump(res, open('results/rq3_contrastive_detector.json', 'w'), indent=1)
        print(s, {k: round(v, 4) for k, v in res[s].items() if k in ('auroc_contrastive', 'auroc_vgr_cosine')}, flush=True)
    n = len(res); m = lambda f: np.mean([res[s][f] for s in res])
    lines = ['| quantity | mean |', '|---|---|']
    for f in ('clean', 'noisy', 'm3_contrastive', 'm3_vgr', 'm3_random', 'm3_oracle', 'm3_contrastive_clean', 'auroc_contrastive', 'auroc_vgr_cosine'):
        lines.append(f'| {f} | {m(f):.4f} |')
    for f in ('m3_contrastive', 'm3_vgr', 'm3_random', 'm3_oracle'):
        gains = [res[s][f] - res[s]['noisy'] for s in res]
        lines.append(f'| {f} gain over noisy (seeds positive) | {np.mean(gains):+.4f} ({sum(g > 0 for g in gains)}/{n}) |')
    better = sum(res[s]['m3_contrastive'] > res[s]['m3_random'] for s in res)
    lines.append(f'| contrastive above random (seeds) | {better}/{n} |')
    open('results/rq3_contrastive_detector.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
