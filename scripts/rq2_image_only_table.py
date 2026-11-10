"""Table of the image-only baseline against the unrepaired model, M1 and M3 (15 seeds, hashtag noise 0.4).

Reads results/rq2_image_only/imageonly_hashtag_r0.40_seed*.json and results/repair/repair_hashtag_r0.40_seed*.json."""
import json
import math
import statistics as st

SEEDS = range(2025, 2040)
rows = []
for seed in SEEDS:
    a = json.load(open(f'results/rq2_image_only/imageonly_hashtag_r0.40_seed{seed}.json'))['report']
    b = json.load(open(f'results/repair/repair_hashtag_r0.40_seed{seed}.json'))['report']
    rows.append((seed, a, b))


def mean(f):
    return st.mean(f(a, b) for _, a, b in rows)


noisy = lambda a, b: a['noisy']['unrepaired']['recall']
clean = lambda a, b: a['clean']['unrepaired']['recall']
loss = mean(clean) - mean(noisy)
cond = [
    ('Unrepaired', lambda a, b: a['noisy']['unrepaired'], lambda a, b: a['clean']['unrepaired']),
    ('Image only, flagged items', lambda a, b: a['noisy']['image_only_flagged'], lambda a, b: a['clean']['image_only_flagged']),
    ('Image only, all cold items', lambda a, b: a['noisy']['image_only_all'], None),
    ('Image only, true mask (oracle)', lambda a, b: a['noisy']['image_only_oracle'], None),
    ('M1 (blind blend)', lambda a, b: b['noisy']['m1'], lambda a, b: b['clean']['m1']),
    ('M3 (flagged blend)', lambda a, b: b['noisy']['m3'], lambda a, b: b['clean']['m3']),
]
lines = ['## Image-only baseline, Recall@20, hashtag noise 0.4, 15 seeds', '',
         '| condition | noisy Recall@20 | share of the loss recovered | noisy NDCG@20 | clean-input Recall@20 |', '|---|---|---|---|---|']
for name, nf, cf in cond:
    r = st.mean(nf(a, b)['recall'] for _, a, b in rows)
    n = st.mean(nf(a, b)['ndcg'] for _, a, b in rows)
    share = '' if name == 'Unrepaired' else f'{(r - mean(noisy)) / loss * 100:.0f}%'
    c = f'{st.mean(cf(a, b)["recall"] for _, a, b in rows):.4f}' if cf else ''
    lines.append(f'| {name} | {r:.4f} | {share} | {n:.4f} | {c} |')
diff = [b['noisy']['m3']['recall'] - a['noisy']['image_only_flagged']['recall'] for _, a, b in rows]
lines += ['', f'Clean (no corruption) Recall@20 {mean(clean):.4f}; unrepaired noisy {mean(noisy):.4f}.',
          f'M3 minus image-only on flagged items: {st.mean(diff):+.4f} (std over seeds {st.stdev(diff):.4f}, s.e. {st.stdev(diff) / math.sqrt(len(diff)):.4f}), '
          f'M3 higher in {sum(d > 0 for d in diff)} of {len(diff)} seeds.',
          'The flagged share q is chosen on noisy validation data (NDCG@10) from 0.3, 0.4, 0.5, as for M3. The baseline was added after the other results.']
open('results/rq2_image_only.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
