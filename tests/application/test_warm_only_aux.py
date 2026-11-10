"""--warm_only_aux: cold items must not influence training (alignment, expert balance, gate balance)."""
import copy

import numpy as np
import torch

from mamex_robustness.models.mamex import MOME_model


def one_step_state(session, cold_seed, warm_only):
    session.env.args.warm_only_aux = int(warm_only)
    data = session.dataset
    data.text_feat = data.text_feat.copy(); data.image_feat = data.image_feat.copy()
    cold = np.asarray(data.cold_item_index)
    rng = np.random.RandomState(cold_seed)  # the model L2-normalises features, so change the direction, not the scale
    data.text_feat[cold] = rng.randn(len(cold), data.text_feat.shape[1])
    data.image_feat[cold] = rng.randn(len(cold), data.image_feat.shape[1])
    torch.manual_seed(1); np.random.seed(1)
    session.model = MOME_model(session.env, data)
    session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.01)
    session.model.train()
    session.train_epoch()
    return copy.deepcopy(session.model.state_dict())


def test_cold_features_do_not_change_training_when_warm_only(session):
    a = one_step_state(session, 1, True)
    b = one_step_state(session, 2, True)
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_cold_features_change_training_by_default(session):
    a = one_step_state(session, 1, False)
    b = one_step_state(session, 2, False)
    assert any(not torch.allclose(a[k], b[k], atol=1e-6) for k in a)
