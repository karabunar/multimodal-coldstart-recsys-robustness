"""Real item text check: corrupted product titles, embedded again, on the clean-trained checkpoints (docs/hypotheses.md).

Usage: PYTHONPATH=src python scripts/rq2_real_item_text.py 2025 2026 ...   (writes results/rq2_real_item_text.json and .md)
Needs data_maps/real_text_variants.npz from scripts/build_real_text_variants.py. S = the cold items with a 2023 catalogue entry.
Every item of S uses the embedding of its clean title (the reference); a share of S gets the wrapped or the mismatched title.
Parameters are chosen on the validation mask, never on the test set; no training.
"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import _scores, evaluate_image_only, evaluate_repairs

RATE = 0.8  # share of S that gets a corrupted title; about 40% of all cold items, as in the main grid
CONDITIONS = ('wrap', 'mismatch')


def run(seed, variants):
    ctx = load_session(seed)
    data, session = ctx.data, ctx.session
    items = variants['items']
    base = data.text_feat.copy()
    base[items] = variants['clean']
    data.text_feat = base  # the reference: every item of S comes from the same pipeline

    def corrupt(condition, mask_seed):
        rng = np.random.default_rng(mask_seed)
        chosen = rng.choice(len(items), size=int(round(RATE * len(items))), replace=False)
        out = base.copy()
        out[items[chosen]] = variants[condition][chosen]
        return out

    res = {'original_clean': _scores(session, None, 'test')['recall'], 'title_only_clean': _scores(session, base, 'test')['recall'],
           'n_titles': int(len(items))}
    for condition in CONDITIONS:
        data.noisy_text_feat = lambda rate, typ, s, c=condition: corrupt(c, s)
        rep = evaluate_repairs(session, RATE, 'hashtag', ctx.args.noise_seed)
        img = evaluate_image_only(session, RATE, 'hashtag', ctx.args.noise_seed)
        res[condition] = {'n_corrupted_cold': rep['n_corrupted_cold'], 'auroc_vgr': rep['auroc']['vgr_cosine'],
                          **{m: rep['noisy'][m]['recall'] for m in ('unrepaired', 'vgr', 'm1', 'm1_random', 'm3', 'm3_random', 'm3_shuffled', 'oracle')},
                          'image_only': img['noisy']['image_only_flagged']['recall']}
    return res


if __name__ == '__main__':
    variants = dict(np.load('data_maps/real_text_variants.npz'))
    res = {}
    for s in map(int, sys.argv[1:]):
        res[s] = run(s, variants)
        json.dump(res, open('results/rq2_real_item_text.json', 'w'), indent=1)
        print(s, {c: round(res[s][c]['unrepaired'], 4) for c in CONDITIONS}, flush=True)
    n = len(res)
    mean = lambda f: float(np.mean([f(r) for r in res.values()]))
    lines = [f'Seeds: {n}. Items with a title: {next(iter(res.values()))["n_titles"]}.',
             f'Recall@20, original features {mean(lambda r: r["original_clean"]):.4f}; title-only reference {mean(lambda r: r["title_only_clean"]):.4f}.', '',
             '| quantity | wrap | mismatch |', '|---|---|---|']
    row = lambda name, f: lines.append(f'| {name} | ' + ' | '.join(f'{mean(lambda r, c=c: f(r[c], r)):.4f}' for c in CONDITIONS) + ' |')
    row('corrupted cold items', lambda c, r: c['n_corrupted_cold'])
    row('AUROC of VGR cosine', lambda c, r: c['auroc_vgr'])
    for m in ('unrepaired', 'vgr', 'm1', 'm3', 'image_only', 'm3_random', 'm3_shuffled', 'oracle'):
        row(f'{m} Recall@20', lambda c, r, m=m: c[m])
    for m in ('vgr', 'm1', 'm3', 'image_only', 'oracle'):
        row(f'{m} gain over unrepaired', lambda c, r, m=m: c[m] - c['unrepaired'])
    row('loss from corruption', lambda c, r: r['title_only_clean'] - c['unrepaired'])
    wins = lambda c, a, b: sum(res[s][c][a] > res[s][c][b] for s in res)
    for c in CONDITIONS:
        lines.append(f'| {c}: seeds with M3 above unrepaired / random / permuted | {wins(c, "m3", "unrepaired")} / {wins(c, "m3", "m3_random")} / {wins(c, "m3", "m3_shuffled")} of {n} |  |')
    open('results/rq2_real_item_text.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
