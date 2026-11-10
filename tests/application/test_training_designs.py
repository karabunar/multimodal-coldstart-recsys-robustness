"""Ablation designs of noise-augmented training: corruption type, fixed prototypes, image-side noise."""
import numpy as np
import pytest
import torch

from mamex_robustness.domain.noise import CorruptionSpec, TextNoiseSimulator, mask_prototype_items


def record(session, **args):
    """Run 3 epochs with these training-noise arguments; return what the model used in each train_epoch."""
    session.env.args.__dict__.update(noise_type='hashtag', noise_seed=7, noise_rate=0.4, **args)
    seen = []
    original = session.train_epoch

    def train_epoch():
        text, image = session.model._eval_text, session.model._eval_image
        seen.append((None if text is None else text.clone(), None if image is None else image.clone()))
        return original()

    session.train_epoch = train_epoch
    session.train(3)
    return seen


def test_prototype_helper_matches_the_prototypes_the_test_mask_uses(session):
    cold, warm, clean = (np.asarray(session.dataset.cold_item_index), np.asarray(session.dataset.warm_item_index), session.dataset.text_feat)
    noisy = session.dataset.noisy_text_feat(1.0, 'hashtag', 11)
    items = mask_prototype_items(1.0, 11, cold, warm)
    protos = clean[items] / np.linalg.norm(clean[items], axis=1, keepdims=True)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    for row in cold:
        mixes = [unit((0.2 * unit(clean[row:row + 1]) + 0.8 * p[None]))[0] for p in protos]
        assert any(np.allclose(noisy[row], m, atol=1e-5) for m in mixes)


def test_fixed_prototypes_repeat_across_epochs_and_the_default_does_not(session):
    session.env.args.train_noise_rate = 0.5
    fixed = [session.dataset.train_noisy_text_feat(0.5, 'hashtag', s, mask_prototype_items(0.4, 7, session.dataset.cold_item_index, session.dataset.warm_item_index))
             for s in (1, 2)]
    free = [session.dataset.train_noisy_text_feat(0.5, 'hashtag', s) for s in (1, 2)]
    clean = session.dataset.text_feat

    items = mask_prototype_items(0.4, 7, session.dataset.cold_item_index, session.dataset.warm_item_index)
    protos = clean[items] / np.linalg.norm(clean[items], axis=1, keepdims=True)
    for noisy in fixed:
        changed = np.where(~np.isclose(noisy, clean).all(axis=1))[0]
        assert len(changed) == round(0.5 * len(session.dataset.warm_item_index))
        cos = (noisy[changed] / np.linalg.norm(noisy[changed], axis=1, keepdims=True)) @ protos.T
        assert (cos.max(axis=1) > 0.7).all()  # each corrupted row points at one of the evaluation prototypes
    assert not np.allclose(fixed[0], free[0])


def test_training_noise_type_overrides_the_evaluation_type(session):
    seen = record(session, train_noise_rate=0.5, train_noise_type='shuffle')
    clean = session.model.text_feat
    warm_rows = {tuple(np.round(r, 5)) for r in clean.numpy()}
    for text, _ in seen:
        changed = ~torch.isclose(text, clean).all(dim=1)
        assert changed.sum() > 0
        assert all(tuple(np.round(r, 5)) in warm_rows for r in text[changed].numpy())  # shuffle copies the text of another item


def test_image_noise_touches_only_warm_image_rows_and_leaves_text_clean(session):
    seen = record(session, train_noise_rate=0.0, train_image_noise_rate=0.5)
    clean = session.model.image_feat
    cold = torch.as_tensor(session.dataset.cold_item_index)
    for text, image in seen:
        assert text is None
        changed = ~torch.isclose(image, clean).all(dim=1)
        assert int(changed.sum()) == round(0.5 * len(session.dataset.warm_item_index))
        assert not changed[cold].any()
    assert session.model._eval_image is None and session.model._eval_text is None


def test_validation_stays_clean_with_image_noise(session):
    session.env.args.__dict__.update(noise_type='hashtag', noise_seed=7, noise_rate=0.4, train_image_noise_rate=0.5, train_noise_rate=0.0)
    seen = []
    original = session.test
    session.test = lambda *a, **k: (seen.append(session.model._eval_image), original(*a, **k))[1]
    session.train(2)
    assert all(image is None for image in seen)
