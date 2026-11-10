"""Table and prediction check for the real clickbait run (Recall@20, noise rate 0.4, seeds 2025..2039).

usage: python scripts/rq2_real_clickbait_table.py [result dir [output name]]
       (defaults: results/rq2_real_clickbait, results/rq2_real_clickbait.md; STRENGTH=1.0 results go to another dir)
The thresholds are the ones written in docs/hypotheses.md before the first run.
"""
import json
import os
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SYNTHETIC_M3 = 0.0078      # mean M3 gain on the synthetic hashtag noise, 15 seeds (rq2_repair_fifteen_seeds.md)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else 'results/rq2_real_clickbait'
    name = sys.argv[2] if len(sys.argv) > 2 else 'rq2_real_clickbait'
    reports = {}
    for seed in range(2025, 2040):
        path = os.path.join(ROOT, folder, f'repair_realtext_r0.40_seed{seed}.json')
        if os.path.exists(path):
            reports[seed] = json.load(open(path))['report']
    seeds = sorted(reports)
    rec = lambda s, block, m: reports[s][block][m]['recall']
    gain = lambda s, m: rec(s, 'noisy', m) - rec(s, 'noisy', 'unrepaired')
    loss = lambda s: (rec(s, 'clean', 'unrepaired') - rec(s, 'noisy', 'unrepaired')) / rec(s, 'clean', 'unrepaired')
    out = [f'## Real clickbait vectors, Recall@20, noise rate 0.4, {len(seeds)} seeds', '',
           '| seed | clean | noisy | loss | VGR | M1 | M3 | M3 beats both controls | AUROC of the cosine |', '|---|---|---|---|---|---|---|---|---|']
    for s in seeds:
        both = gain(s, 'm3') > gain(s, 'm3_random') and gain(s, 'm3') > gain(s, 'm3_shuffled')
        out.append(f"| {s} | {rec(s, 'clean', 'unrepaired'):.4f} | {rec(s, 'noisy', 'unrepaired'):.4f} | {loss(s):.1%} | {gain(s, 'vgr'):+.4f} | "
                   f"{gain(s, 'm1'):+.4f} | {gain(s, 'm3'):+.4f} | {'yes' if both else 'no'} | {reports[s]['auroc']['vgr_cosine']:.4f} |")
    mean = lambda f: st.mean(f(s) for s in seeds)
    out.append(f"| mean | {mean(lambda s: rec(s, 'clean', 'unrepaired')):.4f} | {mean(lambda s: rec(s, 'noisy', 'unrepaired')):.4f} | {mean(loss):.1%} | "
               f"{mean(lambda s: gain(s, 'vgr')):+.4f} | {mean(lambda s: gain(s, 'm1')):+.4f} | {mean(lambda s: gain(s, 'm3')):+.4f} | | "
               f"{mean(lambda s: reports[s]['auroc']['vgr_cosine']):.4f} |")
    n = len(seeds)
    a_loss, b_auc = mean(loss), mean(lambda s: reports[s]['auroc']['vgr_cosine'])
    c_vgr = sum(gain(s, 'vgr') > 0 for s in seeds)
    d_m3, d_mean = sum(gain(s, 'm3') > 0 for s in seeds), mean(lambda s: gain(s, 'm3'))
    ok = lambda b: 'holds' if b else 'does not hold'
    out += ['', 'Predictions written before the first run (docs/hypotheses.md):',
            f"- a. loss between 5% and 25%: {a_loss:.1%}, {ok(0.05 <= a_loss <= 0.25)}.",
            f"- b. mean AUROC of the cosine at least 0.95: {b_auc:.4f}, {ok(b_auc >= 0.95)}.",
            f"- c. VGR beats the unrepaired model in at least 12 of 15 seeds: {c_vgr} of {n}, {ok(c_vgr >= 12)}.",
            f"- d. M3 beats the unrepaired model in at least 12 of 15 seeds with mean gain at least +0.003: {d_m3} of {n}, mean {d_mean:+.4f}, {ok(d_m3 >= 12 and d_mean >= 0.003)}.",
            f"- e. mean M3 gain within a factor of two of the synthetic {SYNTHETIC_M3:+.4f} (between {SYNTHETIC_M3 / 2:+.4f} and {SYNTHETIC_M3 * 2:+.4f}): {d_mean:+.4f}, {ok(SYNTHETIC_M3 / 2 <= d_mean <= SYNTHETIC_M3 * 2)}."]
    text = '\n'.join(out) + '\n'
    with open(os.path.join(ROOT, 'results', name + '.md'), 'w') as f:
        f.write(text)
    print(text)


main()
