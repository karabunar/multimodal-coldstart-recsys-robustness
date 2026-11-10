"""is the neighbour-text repair a repair or a general regulariser? (docs/hypotheses.md)
Usage: PYTHONPATH=src python scripts/rq2_repair_regulariser.py 2025 ... 2039   (writes results/rq2_repair_regulariser.json and results/rq2_repair_regulariser.md)"""
import json
import sys

import numpy as np

from mamex_robustness.application.loading import load_session
from mamex_robustness.domain import repair

LAMS = (0.1, 0.25, 0.5, 1.0)


def recall20(session, text):
    _, recall, _, _ = session.test(mode='test', top_list=[10, 20], noisy_text=text, use_vgr=False)
    return float(recall[20])


def run(seed):
    ctx = load_session(seed, extra=[])
    args, data, session = ctx.args, ctx.data, ctx.session
    cold = np.asarray(data.cold_item_index)
    clean = data.text_feat
    noisy = data.noisy_text_feat(0.4, 'hashtag', args.noise_seed)
    corrupted = (noisy[cold] != clean[cold]).any(axis=1)
    knn = repair.neighbor_text(data.image_feat, clean, data.warm_item_index, cold, 10)
    out = {'clean': recall20(session, None), 'noisy': recall20(session, noisy)}
    for lam in LAMS:
        out[f'clean_blend_{lam}'] = recall20(session, repair.with_cold_rows(clean, cold, repair.blend(clean[cold], knn, lam)))
    out['oracle'] = recall20(session, repair.with_cold_rows(noisy, cold, repair.blend(noisy[cold], knn, np.where(corrupted, 1.0, 0.0))))
    out['oracle_plus_blend_0.25'] = recall20(session, repair.with_cold_rows(noisy, cold, repair.blend(noisy[cold], knn, np.where(corrupted, 1.0, 0.25))))
    return out


if __name__ == '__main__':
    res = {s: run(int(s)) for s in sys.argv[1:]}
    json.dump(res, open('results/rq2_repair_regulariser.json', 'w'), indent=1)
    n = len(res)
    mean = lambda k: np.mean([res[s][k] for s in res])
    wins = lambda a, b: sum(res[s][a] > res[s][b] for s in res)
    lines = ['| condition | mean Recall@20 | seeds above reference |', '|---|---|---|',
             f"| clean text (reference a) | {mean('clean'):.4f} | |"]
    for lam in LAMS:
        k = f'clean_blend_{lam}'
        lines.append(f"| clean text blended with t_knn, lambda {lam} | {mean(k):.4f} ({mean(k)-mean('clean'):+.4f}) | {wins(k, 'clean')}/{n} |")
    lines += [f"| noisy text, unrepaired | {mean('noisy'):.4f} | |",
              f"| oracle: corrupted items lambda 1 (reference b) | {mean('oracle'):.4f} | |",
              f"| oracle plus clean items lambda 0.25 | {mean('oracle_plus_blend_0.25'):.4f} ({mean('oracle_plus_blend_0.25')-mean('oracle'):+.4f}) | {wins('oracle_plus_blend_0.25', 'oracle')}/{n} |"]
    open('results/rq2_repair_regulariser.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
