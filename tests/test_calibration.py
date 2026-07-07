"""
Tier 1 — membrane cutoff calibration: leave-one-cell-out honesty
(webui.calibration). Synthetic ring measurements; no images needed.
"""
import numpy as np
from webui import calibration as C


def _synth_cells(seed=0, n_pos=30, n_neg=30):
    """Positive cells have a bright DAB arc on part of the ring; negatives are diffuse
    low. rv = ring DAB OD pixels, rh = hematoxylin OD pixels."""
    rng = np.random.default_rng(seed)
    def cell(label):
        n = int(rng.integers(120, 240))
        rh = rng.uniform(0.2, 0.5, n)
        if label == 1:
            k = int(n * 0.6)
            rv = np.r_[rng.uniform(0.02, 0.15, k), rng.uniform(0.6, 1.2, n - k)]
        else:
            rv = rng.uniform(0.02, 0.25, n)
        return (label, rv[:n], rh)
    return [cell(1) for _ in range(n_pos)] + [cell(0) for _ in range(n_neg)]


def test_loo_f1_is_not_more_optimistic_than_in_sample():
    cells = _synth_cells()
    t = C._neg_t_pix(cells, 99.0)
    frac = C._ring_frac(cells, t)
    y = np.array([c[0] for c in cells])
    _, f1_insample = C._best_f1_cut(frac, y)
    loo_f1, loo_auc, n = C._loo_f1_auc(cells, 99.0)
    assert loo_f1 is not None and loo_auc is not None
    # Held-out F1 must not EXCEED the in-sample fit (the whole point of LOO honesty).
    assert loo_f1 <= f1_insample + 1e-9
    assert loo_auc > 0.75          # cleanly separable synthetic ⇒ callable


def test_loo_returns_none_when_too_few():
    assert C._loo_f1_auc(_synth_cells(n_pos=2, n_neg=1)[:3], 99.0) == (None, None, 3)


def test_separable_cells_are_callable_via_fit_math():
    """The pooled-fit statistic separates the two synthetic classes (AUC high)."""
    cells = _synth_cells(seed=5)
    t = C._neg_t_pix(cells, 99.0)
    auc = C._roc_auc(C._ring_frac(cells, t), np.array([c[0] for c in cells]))
    assert auc > 0.9
