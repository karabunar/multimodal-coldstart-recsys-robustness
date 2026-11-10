"""swap the projection of the image expert output onto two other 20-dimensional subspaces between cold and warm items:
G = the top right singular vectors of the gate's first-layer weights that read the image output, P = the top principal components of the
warm items' image outputs; the user subspace of rq4_user_subspace.py is the reference. Reads the gate's text weight on cold items (clean text).

usage: PYTHONPATH=src python scripts/rq4_gate_subspace.py   (writes results/rq4_gate_subspace.md and results/rq4_gate_subspace.json)
"""
import json
import os

import numpy as np
import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

K = 20
SEEDS = (2025, 2026, 2027)


def swap_subspace(v, basis, target_mean):
    return v + (target_mean - v @ basis) @ basis.T


def main():
    rows = []
    for seed in SEEDS:
        path = f'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed{seed}.pt'
        if not os.path.exists(path):
            continue
        args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--use_gpu', '0', '--data_path', 'processed_data_baby/cold']))
        env = Env(args)
        data = Loader4MM(env)
        model = MOME_model(env, data)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        warm, cold = torch.as_tensor(np.asarray(data.warm_item_index)), torch.as_tensor(np.asarray(data.cold_item_index))
        with torch.no_grad():
            img, _ = model.image_linear(model.image_feat)
            txt, _ = model.text_linear(model.text_feat)
            vw, vc, tc = img[warm], img[cold], txt[cold]

            def w_t(v):
                return float(torch.softmax(model.gate(torch.cat([v, tc], dim=1)), dim=1)[:, 1].mean())
            base = w_t(vc)
            dim = vc.shape[1]
            user = torch.linalg.svd(model.user_emb.weight, full_matrices=False)[2][:K].T
            gate = torch.linalg.svd(model.gate[0].weight[:, :dim], full_matrices=False)[2][:K].T
            pca = torch.linalg.svd(vw - vw.mean(dim=0), full_matrices=False)[2][:K].T

            def drop(basis):
                return base - w_t(swap_subspace(vc, basis, (vw @ basis).mean(dim=0)))
            overlap = lambda a, b: float((a.T @ b).pow(2).sum() / K)
            rows.append({'seed': seed, 'w_t_cold': base, 'drop_user': drop(user), 'drop_gate': drop(gate), 'drop_pca': drop(pca),
                         'overlap_gate_user': overlap(gate, user), 'overlap_pca_user': overlap(pca, user), 'overlap_gate_pca': overlap(gate, pca),
                         'chance': K / dim})
        print(rows[-1], flush=True)
    with open('results/rq4_gate_subspace.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## drop of the mean text weight on cold items when the projection onto a 20-dimensional subspace is replaced by the warm mean (augmented models, clean text)', '',
           '| seed | w_t cold | user subspace | gate-input subspace G | PCA subspace P | overlap G with user | overlap P with user | overlap G with P | chance |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['seed']} | {r['w_t_cold']:.3f} | {r['drop_user']:+.3f} | {r['drop_gate']:+.3f} | {r['drop_pca']:+.3f} | {r['overlap_gate_user']:.2f} | "
                   f"{r['overlap_pca_user']:.2f} | {r['overlap_gate_pca']:.2f} | {r['chance']:.2f} |")
    open('results/rq4_gate_subspace.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
