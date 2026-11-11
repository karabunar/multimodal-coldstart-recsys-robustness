"""Why M3 trails M1 on real titles (docs/hypotheses.md, "Real item text check").

Usage: PYTHONPATH=src python scripts/rq2_real_item_text_decomposition.py 2025 2026 ...
Writes results/rq2_real_item_text_decomposition.json and .md. Same data, checkpoints and masks as rq2_real_item_text.py, no training.
M1's weight is chosen on the validation mask; it is then applied to one group of cold items at a time. For M3 the precision and recall
of its flagged set (weight above 0.5) against the true mask are reported. Parameters are never chosen on the test set.
"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.application.repair_eval import COSINE_GRID, LAM_GRID, _scores, _select, _vgr_cosine
from mamex_robustness.domain import repair

RATE = 0.8
CONDITIONS = ('wrap', 'mismatch')
GROUPS = ('all', 'corrupted', 's_clean', 'outside_s')


def run(seed, variants):
    ctx = load_session(seed)
    data, session = ctx.data, ctx.session
    items = variants['items']
    base = data.text_feat.copy()
    base[items] = variants['clean']
    data.text_feat = base
    cold, warm = np.asarray(data.cold_item_index), np.asarray(data.warm_item_index)
    in_s = np.isin(cold, items)
    knn = repair.neighbor_text(data.image_feat, base, warm, cold, 10)

    def corrupt(condition, mask_seed):
        rng = np.random.default_rng(mask_seed)
        chosen = rng.choice(len(items), size=int(round(RATE * len(items))), replace=False)
        out = base.copy()
        out[items[chosen]] = variants[condition][chosen]
        return out

    def blend_on(text, lam, subset):
        rows = text[cold]
        new = rows.copy()
        new[subset] = repair.blend(rows[subset], knn[subset], lam)
        return repair.with_cold_rows(text, cold, new)

    res = {}
    for condition in CONDITIONS:
        noisy_test = corrupt(condition, ctx.args.noise_seed)
        noisy_val = corrupt(condition, session.VAL_NOISE_SEED_OFFSET + ctx.args.noise_seed)
        corrupted = (noisy_test[cold] != base[cold]).any(axis=1)
        groups = {'all': np.ones(len(cold), bool), 'corrupted': corrupted, 's_clean': in_s & ~corrupted, 'outside_s': ~in_s}
        everyone = groups['all']
        lam, _ = _select(session, {l: blend_on(noisy_val, l, everyone) for l in LAM_GRID}, (10, 20))
        unrepaired = _scores(session, noisy_test, 'test')['recall']
        out = {'lambda': lam, 'unrepaired': unrepaired, 'n': {g: int(m.sum()) for g, m in groups.items()}}
        for g, mask in groups.items():
            out['gain_' + g] = _scores(session, blend_on(noisy_test, lam, mask), 'test')['recall'] - unrepaired

        def m3_lambda(text, param):
            return repair.quantile_lambda(_vgr_cosine(session, text)[cold], param[0], param[2], param[1])

        def m3(text, param):
            lam_vec = m3_lambda(text, param)
            return repair.with_cold_rows(text, cold, repair.blend(text[cold], knn, lam_vec))

        param, _ = _select(session, {p: m3(noisy_val, p) for p in COSINE_GRID}, (10, 20))
        flagged = m3_lambda(noisy_test, param) > 0.5
        out['m3_gain'] = _scores(session, m3(noisy_test, param), 'test')['recall'] - unrepaired
        out['m3_flagged_share'] = float(flagged.mean())
        out['m3_precision'] = float((flagged & corrupted).sum() / max(flagged.sum(), 1))
        out['m3_recall'] = float((flagged & corrupted).sum() / corrupted.sum())
        out['m3_flagged_outside_s'] = float(flagged[~in_s].mean())
        res[condition] = out
    return res


if __name__ == '__main__':
    variants = dict(np.load('data_maps/real_text_variants.npz'))
    res = {}
    for s in map(int, sys.argv[1:]):
        res[s] = run(s, variants)
        json.dump(res, open('results/rq2_real_item_text_decomposition.json', 'w'), indent=1)
        print(s, {c: round(res[s][c]['gain_all'], 4) for c in CONDITIONS}, flush=True)
    n = len(res)
    mean = lambda c, k: float(np.mean([r[c][k] for r in res.values()]))
    lines = [f'Seeds: {n}. Gains are Recall@20 over the unrepaired model; M1 weight chosen on validation data, applied to one group at a time.', '',
             '| quantity | wrap | mismatch |', '|---|---|---|']
    for g in GROUPS:
        lines.append(f'| items in group "{g}" | ' + ' | '.join(f'{np.mean([r[c]["n"][g] for r in res.values()]):.0f}' for c in CONDITIONS) + ' |')
    for g in GROUPS:
        lines.append(f'| M1 gain, blended into "{g}" | ' + ' | '.join(f'{mean(c, "gain_" + g):+.4f}' for c in CONDITIONS) + ' |')
    for k in ('m3_gain', 'm3_flagged_share', 'm3_precision', 'm3_recall', 'm3_flagged_outside_s'):
        lines.append(f'| {k} | ' + ' | '.join(f'{mean(c, k):.4f}' for c in CONDITIONS) + ' |')
    for c in CONDITIONS:
        a = sum(r[c]['gain_corrupted'] < r[c]['gain_all'] for r in res.values())
        b = sum(r[c]['gain_s_clean'] > r[c]['gain_outside_s'] for r in res.values())
        lines.append(f'| {c}: seeds with (a) corrupted-only below all, (b) S-clean above outside-S | {a} of {n}, {b} of {n} |  |')
    open('results/rq2_real_item_text_decomposition.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
