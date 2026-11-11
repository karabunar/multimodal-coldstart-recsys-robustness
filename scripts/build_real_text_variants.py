"""Embed real product titles, clean and corrupted at the text level, with the encoder identified for the distributed features.

usage: python -I build_real_text_variants.py <split dir> <item_asin.txt> <meta_Baby_Products.jsonl> <clickbait_strings.txt> <out.npz> [encoder]

Items: the cold items with a 2023 catalogue entry (the set S). For each: the clean title, the title with a clickbait string in
front (wrap) and the title of another item (mismatch), embedded with mean pooling at the raw scale of text_feat.
Needs transformers and torch (no sentence-transformers). The npz holds embeddings only, no titles; it is not committed
because the catalogue is not ours (see .gitignore).
"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from mamex_robustness.domain.real_text import corrupted_titles

split_dir, asin_file, meta_file, strings_file, out_file = sys.argv[1:6]
encoder = sys.argv[6] if len(sys.argv) > 6 else 'sentence-transformers/roberta-large-nli-stsb-mean-tokens'
item_of = {a: int(n) for a, n in (l.split() for l in open(asin_file))}
cold = np.load(f'{split_dir}/cold_item_index.npy')
is_cold = set(map(int, cold))
titles = {}
for line in open(meta_file):
    d = json.loads(line)
    a = d.get('parent_asin')
    if a in item_of and item_of[a] in is_cold and item_of[a] not in titles and d.get('title'):
        titles[item_of[a]] = d['title'].strip()
items = np.array(sorted(titles))
clean = [titles[i] for i in items]
strings = [l.strip() for l in open(strings_file) if l.strip()]
variants = {'clean': clean, **corrupted_titles(clean, strings, seed=0)}

tok, model = AutoTokenizer.from_pretrained(encoder), AutoModel.from_pretrained(encoder).eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


emb = {k: embed(v) for k, v in variants.items()}
text = np.load(f'{split_dir}/text_feat.npy').astype(np.float32)
unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
print(f'{len(items)} cold items with a title of {len(cold)}')
print('row norm: title embeddings', {k: round(float(np.linalg.norm(v, axis=1).mean()), 1) for k, v in emb.items()}, 'text_feat', round(float(np.linalg.norm(text, axis=1).mean()), 1))
print('mean cosine of the title embedding to the item\'s distributed text_feat', round(float((unit(emb['clean']) * unit(text[items])).sum(1).mean()), 3))
np.savez(out_file, items=items, **emb)
