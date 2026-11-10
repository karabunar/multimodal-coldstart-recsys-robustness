"""Visual Grounding Regularization (VGR) for the two-way modality gate."""
import torch
import torch.nn.functional as F

_EPS = 1e-8


def cross_modal_cosine(image_emb, text_emb):
    """Per-item cosine similarity of the two modality embeddings; carries no gradient."""
    return F.cosine_similarity(image_emb.detach(), text_emb.detach(), dim=1, eps=_EPS)


VGR_CONTROLS = ('cosine', 'shuffled', 'constant')


def vgr_reweight(gate_weights, image_emb, text_emb, gamma, control='cosine', scope=None):
    """Down-weight the textual expert when text disagrees with the visual evidence.

    gate_weights: [N, 2] rows [w_v, w_t] summing to 1.
    w_t_reg = w_t * sigmoid(gamma * cos(v, t)), then both weights are renormalized to sum to 1.

    control isolates where the benefit comes from:
      'cosine'   - the method itself (per-item factor);
      'shuffled' - the same factors, but each item receives another item's factor (fixed permutation);
      'constant' - every item receives the mean factor.

    scope: optional index tensor. The controls then permute / average the factors only among these items
    (the cold items at evaluation), so each control's factors have the distribution of the evaluated items;
    items outside the scope keep their own factor. None (legacy) uses every row.
    """
    if control not in VGR_CONTROLS:
        raise ValueError(f'control must be one of {VGR_CONTROLS}, got {control!r}')
    factor = torch.sigmoid(gamma * cross_modal_cosine(image_emb, text_emb))
    if control != 'cosine':
        idx = torch.arange(len(factor), device=factor.device) if scope is None else torch.as_tensor(scope, device=factor.device).long()
        sub = factor[idx]
        if control == 'shuffled':
            generator = torch.Generator().manual_seed(0)
            sub = sub[torch.randperm(len(sub), generator=generator).to(factor.device)]
        else:
            sub = sub.mean().expand_as(sub)
        factor = factor.clone()
        factor[idx] = sub
    w_v, w_t = gate_weights[:, 0], gate_weights[:, 1]
    w_t_reg = w_t * factor
    total = w_v + w_t_reg
    return torch.stack([w_v / total, w_t_reg / total], dim=1)
