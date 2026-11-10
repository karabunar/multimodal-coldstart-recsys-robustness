"""Embed the fixed clickbait strings with the encoder that reproduces the distributed text features.

usage: python -I build_clickbait_prototypes.py <strings.txt> <text_feat.npy> <out.npy> [encoder]

The encoder defaults to sentence-transformers/roberta-large-nli-stsb-mean-tokens, which gave the best match to the
distributed text_feat in rq2_text_encoder_probe.py. Needs sentence-transformers (not part of requirements.txt).
The output keeps the raw encoder scale, as text_feat does; the noise simulator normalises both before mixing.
"""
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

strings_file, feat_file, out_file = sys.argv[1:4]
encoder = sys.argv[4] if len(sys.argv) > 4 else 'sentence-transformers/roberta-large-nli-stsb-mean-tokens'
strings = [line.strip() for line in open(strings_file) if line.strip()]
model = SentenceTransformer(encoder)
protos = model.encode(strings, batch_size=16, normalize_embeddings=False, convert_to_numpy=True).astype(np.float32)
text = np.load(feat_file).astype(np.float32)
unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
rng = np.random.default_rng(0)
sample = text[rng.choice(len(text), 2000, replace=False)]
print(f'{len(strings)} strings, shape {protos.shape}')
print(f'row norm: prototypes {np.linalg.norm(protos, axis=1).mean():.1f}, text_feat {np.linalg.norm(text, axis=1).mean():.1f}')
print(f'mean cosine of a prototype to 2000 item texts {float((unit(protos) @ unit(sample).T).mean()):.3f}; '
      f'between item texts {float((unit(sample[:500]) @ unit(sample[500:1000]).T).mean()):.3f}')
np.save(out_file, protos)
print('written', out_file)
