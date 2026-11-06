"""(docs/hypotheses.md): which text do the LATTICE text features resemble, and is the repair's premise true for real titles.

usage: python -I rq2_real_text_source.py <split dir> <item_asin.txt> <meta_Baby_Products.jsonl> <Baby_5.json> <out prefix>
Descriptive only: no model, no test metric, nothing tuned. The input files are read as data (.txt, .json, .jsonl, .npy).
"""
import collections
import json
import math
import re
import sys

import numpy as np
import scipy.sparse as sp

split_dir, asin_file, meta_file, reviews_file, out = sys.argv[1:6]
STOP = set('a an and are as at be by for from has have in is it its of on or that the this to was were will with you your i my we our they them he she his her their not but if so can just very all any more most than then there these those which who what when where how about up out into over also one two get got would could should do does did been being am no yes too'.split())


def tokens(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOP]


def tfidf(docs):
    """Sublinear tf, idf = log((1 + n) / (1 + df)) + 1, L2-normalised rows (scipy sparse)."""
    vocab, rows, cols, vals = {}, [], [], []
    for r, d in enumerate(docs):
        for w, c in collections.Counter(tokens(d)).items():
            rows.append(r), cols.append(vocab.setdefault(w, len(vocab))), vals.append(1 + math.log(c))
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)))
    df = np.asarray((m > 0).sum(axis=0)).ravel()
    m = m.multiply(np.log((1 + len(docs)) / (1 + df)) + 1).tocsr()
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()) + 1e-12
    return sp.diags(1 / norm) @ m


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


asin_of = {int(n): a for a, n in (l.split() for l in open(asin_file))}
item_of = {a: n for n, a in asin_of.items()}
text_feat = np.load(f'{split_dir}/text_feat.npy').astype(np.float32)
image_feat = np.load(f'{split_dir}/image_feat.npy').astype(np.float32)
cold = np.load(f'{split_dir}/cold_item_index.npy')
warm = np.setdiff1d(np.arange(len(text_feat)), cold)

meta = {}
for line in open(meta_file):
    d = json.loads(line)
    a = d.get('parent_asin')
    if a in item_of and a not in meta and d.get('title'):
        meta[a] = d
reviews = collections.defaultdict(list)
for line in open(reviews_file):
    d = json.loads(line)
    reviews[d['asin']].append((d.get('reviewText') or '') + ' ' + (d.get('summary') or ''))

S = np.array(sorted(item_of[a] for a in meta))          # items with a 2023 catalogue entry
in_S = np.zeros(len(text_feat), bool)
in_S[S] = True
title = [meta[asin_of[i]]['title'] for i in S]
product = [' '.join([meta[asin_of[i]]['title']] + list(meta[asin_of[i]].get('features') or []) + list(meta[asin_of[i]].get('description') or [])) for i in S]
review = [' '.join(reviews[asin_of[i]]) for i in S]
results = {'n_items': int(len(text_feat)), 'n_subset': int(len(S)), 'n_cold': int(len(cold)), 'n_cold_in_subset': int(in_S[cold].sum()),
           'n_warm_in_subset': int(in_S[warm].sum())}


def top_k(sim, k=10):
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / a.shape[1] for x, y in zip(a, b)]))


# ---------------- source of the text features
feat_sim = unit(text_feat[S]) @ unit(text_feat[S]).T
nn_feat = top_k(feat_sim)
source_overlap = {}
for name, docs in (('T title', title), ('P title+features+description', product), ('R reviews (2014)', review)):
    m = tfidf(docs)
    source_overlap[name] = overlap(nn_feat, top_k((m @ m.T).toarray()))
rng = np.random.default_rng(0)
mp = tfidf(product)[rng.permutation(len(S))]
source_overlap['permuted control (P)'] = overlap(nn_feat, top_k((mp @ mp.T).toarray()))
source_overlap['chance'] = 10 / (len(S) - 1)
results['overlap_at_10'] = source_overlap
k = 'P title+features+description'
results['source_overlap'] = {'a_P_over_R': source_overlap[k] / source_overlap['R reviews (2014)'], 'b_P_minus_T': source_overlap[k] - source_overlap['T title'],
                  'c_best': max(source_overlap['T title'], source_overlap[k], source_overlap['R reviews (2014)']), 'c_control': source_overlap['permuted control (P)']}

# ---------------- premise of the repair
mt = tfidf(title)                                           # rows follow S
row_of = {int(i): r for r, i in enumerate(S)}
cat2 = {int(i): (meta[asin_of[i]].get('categories') or [None, None])[1:2] for i in S}
cat2 = {i: c[0] for i, c in cat2.items() if c and c[0]}
cold_S = [int(i) for i in cold if in_S[i]]
warm_S = np.array([int(i) for i in warm if in_S[i]])


def title_sim(i, js):
    return float((mt[row_of[i]] @ mt[[row_of[j] for j in js]].T).toarray().mean())


def nearest(feat, queries, k=10):
    sims = unit(feat[queries]) @ unit(feat[warm]).T
    return warm[np.argpartition(-sims, k - 1, axis=1)[:, :k]]


img_nn = nearest(image_feat, np.array(cold_S))
txt_nn = nearest(text_feat, np.array(cold_S))
rng = np.random.default_rng(0)
stats = {'image': [[], [], 0], 'text': [[], [], 0], 'random': [[], [], 0]}
for r, i in enumerate(cold_S):
    sets = {'image': img_nn[r], 'text': txt_nn[r], 'random': rng.choice(warm_S, size=10, replace=False)}
    for name, js in sets.items():
        js = [int(j) for j in js if in_S[j]]
        if not js:
            continue
        stats[name][0].append(title_sim(i, js))
        stats[name][2] += len(js)
        if i in cat2:
            shared = [cat2[j] == cat2[i] for j in js if j in cat2]
            if shared:
                stats[name][1].append(np.mean(shared))
# the random reference is the mean over 20 further draws per item for stability
ref_title, ref_cat = [], []
for _ in range(20):
    t, c = [], []
    for i in cold_S:
        js = [int(j) for j in rng.choice(warm_S, size=10, replace=False)]
        t.append(title_sim(i, js))
        if i in cat2:
            shared = [cat2[j] == cat2[i] for j in js if j in cat2]
            if shared:
                c.append(np.mean(shared))
    ref_title.append(np.mean(t)), ref_cat.append(np.mean(c))
premise = {name: {'title_similarity': float(np.mean(v[0])), 'same_category_share': float(np.mean(v[1])), 'kept_per_item': v[2] / len(cold_S)}
       for name, v in stats.items() if name != 'random'}
premise['random reference (20 draws)'] = {'title_similarity': float(np.mean(ref_title)), 'same_category_share': float(np.mean(ref_cat))}
results['premise'] = premise
results['ratios'] = {'a_title_image_over_random': premise['image']['title_similarity'] / premise['random reference (20 draws)']['title_similarity'],
                         'b_category_image_over_random': premise['image']['same_category_share'] / premise['random reference (20 draws)']['same_category_share'],
                         'c_text_minus_image_title': premise['text']['title_similarity'] - premise['image']['title_similarity']}
results['n_cat2_items'] = len(cat2)
json.dump(results, open(out + '.json', 'w'), indent=2)

lines = ['## real text for the items of the Baby split', '',
         f"Items with a 2023 catalogue entry: {results['n_subset']} of {results['n_items']}; cold items among them {results['n_cold_in_subset']} of {results['n_cold']}, warm items {results['n_warm_in_subset']}. Items with a second-level category: {len(cat2)}.", '',
         '### overlap@10 of the 10 nearest items in `text_feat` with the 10 nearest in TF-IDF space', '', '| source | overlap@10 |', '|---|---|']
lines += [f'| {k} | {v:.4f} |' for k, v in source_overlap.items()]
r = results['source_overlap']
lines += ['', f"H32a: P/R = {r['a_P_over_R']:.2f} (predicted at least 1.25). H32b: P - T = {r['b_P_minus_T']:+.4f} (predicted at least 0). H32c: best source {r['c_best']:.3f} (predicted 0.05 to 0.30), control {r['c_control']:.4f} (predicted below 0.01).", '',
          '### neighbours of the cold items in the subset (10 nearest warm items by each space; those with a catalogue entry are kept)', '',
          '| neighbours | title similarity (TF-IDF cosine) | same second-level category | kept per item |', '|---|---|---|---|']
for name, v in premise.items():
    lines.append(f"| {name} | {v['title_similarity']:.4f} | {v['same_category_share']:.3f} | {v.get('kept_per_item', 10):.1f} |")
r = results['ratios']
lines += ['', f"H33a: title similarity image / random = {r['a_title_image_over_random']:.2f} (predicted at least 1.5). H33b: category share image / random = {r['b_category_image_over_random']:.2f} (predicted at least 2). H33c: text minus image title similarity = {r['c_text_minus_image_title']:+.4f} (predicted above 0)."]
open(out + '.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
