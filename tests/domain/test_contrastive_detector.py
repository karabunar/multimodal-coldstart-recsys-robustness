import numpy as np

from mamex_robustness.domain import contrastive


def _paired(n=400, seed=0):
    """Text is a noisy linear function of the image, so matching pairs agree and shuffled ones do not."""
    rng = np.random.default_rng(seed)
    image = rng.normal(size=(n, 12))
    text = image @ rng.normal(size=(12, 8)) + 0.1 * rng.normal(size=(n, 8))
    return image.astype(np.float32), text.astype(np.float32)


def test_matching_pairs_score_above_shuffled_pairs():
    image, text = _paired()
    score = contrastive.fit(image[:300], text[:300], seed=0, hidden=32, out=8, dropout=0.0, lr=3e-3, max_epochs=300, patience=40)
    matched = score(image[300:], text[300:])
    shuffled = score(image[300:], text[300:][np.random.default_rng(1).permutation(100)])
    assert matched.mean() > shuffled.mean() + 0.2


def test_score_is_a_cosine():
    image, text = _paired(60)
    score = contrastive.fit(image, text, seed=0, hidden=16, out=4, max_epochs=3)
    s = score(image, text)
    assert s.shape == (60,) and np.all(np.abs(s) <= 1 + 1e-6)


def test_fit_is_deterministic_for_a_seed():
    image, text = _paired(80)
    a = contrastive.fit(image, text, seed=3, hidden=16, out=4, max_epochs=4)(image, text)
    b = contrastive.fit(image, text, seed=3, hidden=16, out=4, max_epochs=4)(image, text)
    assert np.allclose(a, b)
