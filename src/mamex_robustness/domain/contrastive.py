"""Image-text agreement score learned with an InfoNCE loss on warm items only (no knowledge of the corruption)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tower(d_in, hidden, out, dropout):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))


def _embed(tower, x):
    return F.normalize(tower(F.normalize(x, dim=-1)), dim=-1)


def _info_nce(a, b, temperature):
    logits = a @ b.T / temperature
    target = torch.arange(len(a))
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def fit(image, text, seed=0, hidden=512, out=128, dropout=0.3, lr=1e-3, weight_decay=1e-2, temperature=0.07,
        batch=512, max_epochs=100, patience=10, holdout=0.2):
    """Train two towers on matching (image, text) rows; 20% of the rows stop the training early by held-out loss.

    Returns score(image, text): the cosine of the two tower outputs for each row pair (a numpy array).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = torch.as_tensor(np.asarray(image), dtype=torch.float32), torch.as_tensor(np.asarray(text), dtype=torch.float32)
    perm = rng.permutation(len(x))
    n_hold = max(2, int(len(x) * holdout))
    hold, train = perm[:n_hold], perm[n_hold:]
    f, g = _tower(x.shape[1], hidden, out, dropout), _tower(y.shape[1], hidden, out, dropout)
    opt = torch.optim.AdamW([*f.parameters(), *g.parameters()], lr=lr, weight_decay=weight_decay)
    best, best_state, wait = float('inf'), None, 0
    for _ in range(max_epochs):
        f.train(); g.train()
        order = rng.permutation(train)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue
            loss = _info_nce(_embed(f, x[idx]), _embed(g, y[idx]), temperature)
            opt.zero_grad(); loss.backward(); opt.step()
        f.eval(); g.eval()
        with torch.no_grad():
            held = _info_nce(_embed(f, x[hold]), _embed(g, y[hold]), temperature).item()
        if held < best - 1e-4:
            best, wait = held, 0
            best_state = ({k: v.clone() for k, v in f.state_dict().items()}, {k: v.clone() for k, v in g.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
    f.load_state_dict(best_state[0]); g.load_state_dict(best_state[1]); f.eval(); g.eval()

    def score(image_rows, text_rows):
        with torch.no_grad():
            a = _embed(f, torch.as_tensor(np.asarray(image_rows), dtype=torch.float32))
            b = _embed(g, torch.as_tensor(np.asarray(text_rows), dtype=torch.float32))
            return (a * b).sum(-1).numpy()

    return score
