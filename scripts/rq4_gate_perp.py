"""what the part of the gate-sensitive subspace outside the user directions encodes (augmented models, clean text).

G = top-20 right singular vectors of the gate's first-layer weights on the image output; the user subspace U = top-20 right singular vectors of
the user embedding; G-perp = the 10 directions of span(G) with the largest component outside U (left singular vectors of G - U U^T G).
(a) swap of the G-perp projection between cold and warm items: drop of the mean text weight on cold items;
(b) AUROC of each G-perp coordinate for cold against warm items;
(c) Spearman of the item-level change of w_t on warm items (G-perp projection replaced by the cold mean) with log n_i.

usage: PYTHONPATH=src python scripts/rq4_gate_perp.py   (writes results/rq4_gate_perp.md and results/rq4_gate_perp.json)
"""
import json
import os

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

K, K_PERP = 20, 10
SEEDS = (2025, 2026, 2027)


def swap_subspace(v, basis, target_mean):
    return v + (target_mean - v @ basis) @ basis.T


def auroc(cold_scores, warm_scores):
    ranks = rankdata(np.concatenate([cold_scores, warm_scores]))
    n_c, n_w = len(cold_scores), len(warm_scores)
    return float((ranks[:n_c].sum() - n_c * (n_c + 1) / 2) / (n_c * n_w))


def main():
    rows = []
    for seed in SEEDS:
        path = f'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed{seed}.pt'
        args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--use_gpu', '0', '--data_path', 'processed_data_baby/cold']))
        env = Env(args)
        data = Loader4MM(env)
        counts = np.zeros(data.m_item)
        for items in data.train_data.values():
            for i in items:
                counts[i] += 1
        model = MOME_model(env, data)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        warm_np, cold_np = np.asarray(data.warm_item_index), np.asarray(data.cold_item_index)
        warm, cold = torch.as_tensor(warm_np), torch.as_tensor(cold_np)
        with torch.no_grad():
            img, _ = model.image_linear(model.image_feat)
            txt, _ = model.text_linear(model.text_feat)
            vw, vc, tc, tw = img[warm], img[cold], txt[cold], txt[warm]
            wt = lambda v, t: torch.softmax(model.gate(torch.cat([v, t], dim=1)), dim=1)[:, 1]
            dim = vc.shape[1]
            user = torch.linalg.svd(model.user_emb.weight, full_matrices=False)[2][:K].T
            gate = torch.linalg.svd(model.gate[0].weight[:, :dim], full_matrices=False)[2][:K].T
            residual = gate - user @ (user.T @ gate)
            perp = torch.linalg.svd(residual, full_matrices=False)[0][:, :K_PERP]
            base = float(wt(vc, tc).mean())
            drop = base - float(wt(swap_subspace(vc, perp, (vw @ perp).mean(dim=0)), tc).mean())
            # (b) per-direction AUROC, cold against warm
            coords_c, coords_w = (vc @ perp).numpy(), (vw @ perp).numpy()
            aucs = [auroc(coords_c[:, j], coords_w[:, j]) for j in range(K_PERP)]
            separating = sum(1 for a in aucs if a >= 0.75 or a <= 0.25)
            # (c) item-level change on warm items
            change = (wt(swap_subspace(vw, perp, (vc @ perp).mean(dim=0)), tw) - wt(vw, tw)).numpy()
            rho = float(spearmanr(change, np.log(counts[warm_np])).correlation)
            perp_norm = float((residual.norm(dim=0)[:K_PERP]).mean())
            # exploratory, not part of the written hypotheses: the most separating direction against the interaction count and the output norm
            j = int(np.argmax([abs(x - 0.5) for x in aucs]))
            explore = {'direction': j, 'auroc': aucs[j], 'spearman_logn': float(spearmanr(coords_w[:, j], np.log(counts[warm_np])).correlation),
                       'spearman_norm': float(spearmanr(coords_w[:, j], vw.norm(dim=1).numpy()).correlation)}
        rows.append({'seed': seed, 'w_t_cold': base, 'drop_perp': drop, 'aucs': aucs, 'n_separating': separating,
                     'spearman_change_logn': rho, 'exploratory': explore, 'mean_change_warm': float(change.mean()), 'residual_norm_top10': perp_norm})
        print({k: v for k, v in rows[-1].items() if k != 'aucs'}, flush=True)
    with open('results/rq4_gate_perp.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## the part of the gate-sensitive subspace outside the user directions (augmented models, clean text)', '',
           '| seed | w_t cold | drop for G-perp (10 directions) | G-perp directions with AUROC >= 0.75 or <= 0.25 (of 10) | AUROC range | Spearman(change of w_t on warm items, log n) | mean change on warm items |',
           '|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['seed']} | {r['w_t_cold']:.3f} | {r['drop_perp']:+.3f} | {r['n_separating']} | {min(r['aucs']):.2f} to {max(r['aucs']):.2f} | "
                   f"{r['spearman_change_logn']:+.3f} | {r['mean_change_warm']:+.3f} |")
    out += ['', 'Exploratory (not a written hypothesis): the G-perp direction whose coordinate separates cold from warm items most, on warm items.', '',
            '| seed | direction | AUROC cold against warm | Spearman with log n | Spearman with output norm |', '|---|---|---|---|---|']
    for r in rows:
        e = r['exploratory']
        out.append(f"| {r['seed']} | {e['direction']} | {e['auroc']:.2f} | {e['spearman_logn']:+.3f} | {e['spearman_norm']:+.3f} |")
    open('results/rq4_gate_perp.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
