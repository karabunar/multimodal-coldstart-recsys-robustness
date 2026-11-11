"""Control for Hypothesis 5 (added after the results): does the contrastive detector read image-text agreement or the look of hashtag text?

Same detectors as rq5_contrastive_detector.py (same seeds). AUROC on the cold items for each noise type at rate 0.4:
shuffle puts another item's clean text in place (the text stays natural, only the match is wrong), gaussian adds noise, hashtag appends hashtags.
Usage: PYTHONPATH=src python scripts/rq5_contrastive_noise_types.py 2025 ...   (writes results/rq5_contrastive_noise_types.json and .md)
"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import _vgr_cosine
from mamex_robustness.domain import contrastive, repair

TYPES = ('hashtag', 'shuffle', 'gaussian')


def run(seed):
    ctx = load_session(seed, extra=['--warm_only_aux', '1'], ckpt='results/rq3_warm_only/checkpoints/stress_test_hashtag_r0.40_gsel_seed{seed}_warmonly.pt')
    data, session = ctx.data, ctx.session
    cold, warm = np.asarray(data.cold_item_index), np.asarray(data.warm_item_index)
    score_fn = contrastive.fit(data.image_feat[warm], data.text_feat[warm], seed=seed)
    out = {}
    for t in TYPES:
        noisy = data.noisy_text_feat(0.4, t, ctx.args.noise_seed)
        corrupted = (noisy[cold] != data.text_feat[cold]).any(axis=1)
        out[t] = {'share': float(corrupted.mean()),
                  'auroc_contrastive': repair.auroc(-score_fn(data.image_feat[cold], noisy[cold]), corrupted),
                  'auroc_vgr_cosine': repair.auroc(-_vgr_cosine(session, noisy)[cold], corrupted)}
    return out


if __name__ == '__main__':
    res = {}
    for s in map(int, sys.argv[1:]):
        res[s] = run(s)
        json.dump(res, open('results/rq5_contrastive_noise_types.json', 'w'), indent=1)
        print(s, {t: round(res[s][t]['auroc_contrastive'], 3) for t in TYPES}, flush=True)
    lines = ['| noise type (rate 0.4) | share corrupted | AUROC contrastive | AUROC VGR cosine |', '|---|---|---|---|']
    for t in TYPES:
        m = lambda k: np.mean([res[s][t][k] for s in res])
        lines.append(f"| {t} | {m('share'):.2f} | {m('auroc_contrastive'):.3f} | {m('auroc_vgr_cosine'):.3f} |")
    open('results/rq5_contrastive_noise_types.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
