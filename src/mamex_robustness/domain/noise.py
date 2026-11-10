"""Text corruption for stress-testing cold-start item embeddings.

Only pre-extracted text embeddings exist, so clickbait text ("#siyaset #viral")
is simulated at the embedding level (realtext: with embeddings of real clickbait strings). Pure numpy; no I/O.
"""
from dataclasses import dataclass

import numpy as np

NOISE_TYPES = ('hashtag', 'shuffle', 'gaussian', 'realtext')
_EPS = 1e-8


@dataclass(frozen=True)
class CorruptionSpec:
    """A StressScenario's corruption settings.

    noise_rate: fraction of cold items whose text is corrupted.
    strength: hashtag -> weight of the clickbait prototype in the mix;
        gaussian -> noise std as a multiple of the feature std.
    n_prototypes: number of shared clickbait vectors (hashtag only).
    """
    noise_rate: float
    noise_type: str = 'hashtag'
    seed: int = 2025
    strength: float = 0.8
    n_prototypes: int = 5

    def __post_init__(self):
        if not 0.0 <= self.noise_rate <= 1.0:
            raise ValueError(f'noise_rate must be in [0, 1], got {self.noise_rate}')
        if self.noise_type not in NOISE_TYPES:
            raise ValueError(f'noise_type must be one of {NOISE_TYPES}, got {self.noise_type!r}')


def mask_prototype_items(noise_rate, seed, cold_item_index, warm_item_index, n_prototypes=5):
    """The warm items whose text the hashtag corruption of the test (or validation) mask with this seed uses as prototypes."""
    rng = np.random.default_rng(seed)
    rng.choice(np.asarray(cold_item_index), size=int(round(noise_rate * len(cold_item_index))), replace=False)
    return rng.choice(warm_item_index, size=n_prototypes, replace=False)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + _EPS)


class TextNoiseSimulator:
    def __init__(self, spec, prototypes=None):
        """prototypes (realtext only): rows of embeddings of real clickbait strings, in the space of text_feat."""
        self.spec = spec
        self.prototypes = prototypes

    def __call__(self, text_feat, cold_item_index, warm_item_index, proto_items=None):
        """Return a corrupted copy of text_feat; only a noise_rate share of cold rows change.
        proto_items (hashtag only): rows of text_feat to use as the prototypes instead of drawing them."""
        spec = self.spec
        rng = np.random.default_rng(spec.seed)
        cold = np.asarray(cold_item_index)
        # The mask is drawn before any type-specific randomness so it is the same for every noise_type.
        n_corrupt = int(round(spec.noise_rate * len(cold)))
        corrupted = rng.choice(cold, size=n_corrupt, replace=False)

        out = text_feat.copy()
        if n_corrupt == 0:
            return out
        if spec.noise_type == 'hashtag':
            drawn = rng.choice(warm_item_index, size=spec.n_prototypes, replace=False)
            protos = _unit(text_feat[drawn if proto_items is None else np.asarray(proto_items)])
            chosen = protos[rng.integers(0, spec.n_prototypes, size=n_corrupt)]
            out[corrupted] = _unit((1 - spec.strength) * _unit(text_feat[corrupted]) + spec.strength * chosen)
        elif spec.noise_type == 'shuffle':
            donors = rng.choice(warm_item_index, size=n_corrupt, replace=True)
            out[corrupted] = text_feat[donors]
        elif spec.noise_type == 'realtext':
            if self.prototypes is None:
                raise ValueError("noise_type 'realtext' needs prototypes (--real_proto_file)")
            chosen = _unit(np.asarray(self.prototypes, dtype=text_feat.dtype))[rng.integers(0, len(self.prototypes), size=n_corrupt)]
            out[corrupted] = _unit((1 - spec.strength) * _unit(text_feat[corrupted]) + spec.strength * chosen)
        else:
            std = spec.strength * text_feat.std()
            out[corrupted] = text_feat[corrupted] + rng.normal(0.0, std, size=(n_corrupt, text_feat.shape[1]))
        return out
