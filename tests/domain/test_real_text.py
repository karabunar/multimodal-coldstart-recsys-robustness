from mamex_robustness.domain import real_text

TITLES = [f'product {i}' for i in range(20)]
STRINGS = ['WOW!!! order now', 'You will not believe this']


def test_wrap_puts_a_clickbait_string_in_front_of_the_title():
    wrapped = real_text.corrupted_titles(TITLES, STRINGS, seed=0)['wrap']
    assert all(w.endswith(t) and w[:-len(t)].strip() in STRINGS for w, t in zip(wrapped, TITLES))


def test_mismatch_is_the_title_of_another_item():
    mismatched = real_text.corrupted_titles(TITLES, STRINGS, seed=0)['mismatch']
    assert all(m != t and m in TITLES for m, t in zip(mismatched, TITLES))


def test_mismatch_never_returns_an_equal_title_even_with_duplicates():
    titles = ['same', 'same', 'other', 'same', 'other']
    mismatched = real_text.corrupted_titles(titles, STRINGS, seed=1)['mismatch']
    assert all(m != t for m, t in zip(mismatched, titles))


def test_is_deterministic_for_a_seed():
    assert real_text.corrupted_titles(TITLES, STRINGS, seed=3) == real_text.corrupted_titles(TITLES, STRINGS, seed=3)
    assert real_text.corrupted_titles(TITLES, STRINGS, seed=3) != real_text.corrupted_titles(TITLES, STRINGS, seed=4)
