"""Diagnostic probes of the gate (where it puts its weight, where cold items sit, interventions); mixed into Session.

They read the model's best state and never train. Split out of session.py so that the training and evaluation code stays short.
"""
import numpy as np
import torch


class ProbesMixin:
    def gate_probe(self, noise_rate, noise_type, noise_seed):
        """Mean gate weights on warm items, split by a training-style corruption (model must hold the best state).

        The gate scores one item at a time, so corrupted and uncorrupted items are measured in the same pass.
        """
        noisy = self.dataset.train_noisy_text_feat(noise_rate, noise_type, self.PROBE_NOISE_SEED_OFFSET + noise_seed)
        warm = np.asarray(self.dataset.warm_item_index)
        changed = (noisy[warm] != self.dataset.text_feat[warm]).any(axis=1)
        self.model.eval()
        return {'n_warm_corrupted': int(changed.sum()), 'n_warm_uncorrupted': int((~changed).sum()),
                'warm_clean': self._gate_mean(None, False, warm),
                'warm_corrupted': self._gate_mean(noisy, False, warm[changed]),
                'warm_uncorrupted': self._gate_mean(noisy, False, warm[~changed]),
                'cold_clean': self._gate_mean(None, False, np.asarray(self.dataset.cold_item_index))}

    @torch.no_grad()
    def representation_probe(self):
        """Where the cold items sit relative to the warm items in the space the gate reads (model must hold the best state).

        Clean text, eval mode. The gate input is the concatenation of the image and text expert outputs.
        """
        from scipy.stats import spearmanr
        self.model.eval()
        image_emb, _ = self.model.image_linear(self.model.image_feat)
        text_emb, _ = self.model.text_linear(self.model.text_feat)
        combined = torch.cat([image_emb, text_emb], dim=1)
        logits = self.model.gate(combined)
        weights = torch.softmax(logits, dim=1)
        warm = torch.as_tensor(np.asarray(self.dataset.warm_item_index), device=self.env.device)
        cold = torch.as_tensor(np.asarray(self.dataset.cold_item_index), device=self.env.device)
        distance = (combined - combined[warm].mean(0)).norm(dim=1)

        def group(index):
            return {'norm_v': float(image_emb[index].norm(dim=1).mean()),
                    'norm_t': float(text_emb[index].norm(dim=1).mean()),
                    'distance': float(distance[index].mean()),
                    'logit_gap': float((logits[index, 1] - logits[index, 0]).mean()),
                    'w_t': float(weights[index, 1].mean())}

        warm_stats, cold_stats = group(warm), group(cold)
        rho = spearmanr(distance[cold].cpu().numpy(), weights[cold, 1].cpu().numpy())[0]
        return {'warm': warm_stats, 'cold': cold_stats,
                'distance_ratio': cold_stats['distance'] / warm_stats['distance'],
                'spearman_distance_wt_cold': None if np.isnan(rho) else float(rho)}

    def holdout_probe(self):
        """Mean gate weights on trained warm items, held-out warm items (no training interactions) and cold items, on clean text."""
        warm = np.asarray(self.dataset.warm_item_index)
        held = np.asarray(self.dataset.heldout_warm_index)
        trained = np.setdiff1d(warm, held)
        return {'n_trained_warm': int(len(trained)), 'n_heldout_warm': int(len(held)),
                'trained_warm': self._gate_mean(None, False, trained),
                'heldout_warm': self._gate_mean(None, False, held),
                'cold': self._gate_mean(None, False, np.asarray(self.dataset.cold_item_index))}

    @torch.no_grad()
    def gate_cause(self, seed=0):
        """Interventions on the gate input of the cold items (clean text, eval mode; model must hold the best state).

        The gate reads the concatenated image and text expert outputs. Each intervention changes only that input and
        reports the mean text weight w_t of the cold items: the unchanged value, each expert output rescaled per item to
        the mean warm norm (both, image only, text only), both outputs replaced by those of a random warm item, and only the image or only the text output replaced (the last two were added after the first results were read).
        zero_gap is the text-minus-image gate logit with both expert outputs set to zero (the gate's own bias).
        """
        self.model.eval()
        image_emb, _ = self.model.image_linear(self.model.image_feat)
        text_emb, _ = self.model.text_linear(self.model.text_feat)
        warm = torch.as_tensor(np.asarray(self.dataset.warm_item_index), device=self.env.device)
        cold = torch.as_tensor(np.asarray(self.dataset.cold_item_index), device=self.env.device)

        def w_t(img, txt):
            return float(torch.softmax(self.model.gate(torch.cat([img, txt], dim=1)), dim=1)[:, 1].mean())

        def to_norm(x, target):
            return x * (target / x.norm(dim=1, keepdim=True).clamp_min(1e-12))

        warm_v, warm_t = float(image_emb[warm].norm(dim=1).mean()), float(text_emb[warm].norm(dim=1).mean())
        v_c, t_c = image_emb[cold], text_emb[cold]
        generator = torch.Generator().manual_seed(seed)
        pick = warm[torch.randint(len(warm), (len(cold),), generator=generator).to(self.env.device)]
        zero = torch.zeros(1, image_emb.shape[1], device=self.env.device)
        logit = self.model.gate(torch.cat([zero, zero], dim=1))[0]
        return {'zero_gap': float(logit[1] - logit[0]),
                'warm_norm': {'v': warm_v, 't': warm_t},
                'cold_norm': {'v': float(v_c.norm(dim=1).mean()), 't': float(t_c.norm(dim=1).mean())},
                'w_t_warm': w_t(image_emb[warm], text_emb[warm]),
                'w_t_cold': w_t(v_c, t_c),
                'w_t_cold_scaled_both': w_t(to_norm(v_c, warm_v), to_norm(t_c, warm_t)),
                'w_t_cold_scaled_image': w_t(to_norm(v_c, warm_v), t_c),
                'w_t_cold_scaled_text': w_t(v_c, to_norm(t_c, warm_t)),
                'w_t_cold_replaced_by_warm': w_t(image_emb[pick], text_emb[pick]),
                'w_t_cold_image_replaced_by_warm': w_t(image_emb[pick], t_c),
                'w_t_cold_text_replaced_by_warm': w_t(v_c, text_emb[pick])}
