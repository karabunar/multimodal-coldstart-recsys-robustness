"""Text-level corruption of real product titles (before they are embedded)."""
import numpy as np


def corrupted_titles(titles, strings, seed=0):
    """Two corrupted versions of every title.

    wrap: a clickbait string in front of the title (the product stays in the text).
    mismatch: the title of another item (natural text of the wrong product); never equal to the item's own title.
    """
    rng = np.random.default_rng(seed)
    titles = list(titles)
    wrap = [f'{strings[int(rng.integers(len(strings)))]} {t}' for t in titles]
    mismatch = []
    for i, t in enumerate(titles):
        j = int(rng.integers(len(titles)))
        for _ in range(100):
            if titles[j] != t:
                break
            j = int(rng.integers(len(titles)))
        else:
            raise ValueError('every title is equal; no mismatch exists')
        mismatch.append(titles[j])
    return {'wrap': wrap, 'mismatch': mismatch}
