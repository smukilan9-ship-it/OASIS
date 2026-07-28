"""The membrane pixel threshold must not depend on how many cells are positive.

`membrane_pos_frac` asks what fraction of a ring is stained, which needs a definition of
"stained" that exists on an unlabelled image at apply time. The obvious choice -- a high
percentile of the image's pooled ring pixels -- has a feedback defect: the stained pixels
are IN the pooled distribution, so the more positive cells an image has, the higher the
percentile climbs, and the stricter "positive" becomes. A slide with more signal would be
called less positive.

`auto_pixel_threshold` uses median + k*MAD instead. Both statistics describe the bulk of
the distribution, which is ring background, and neither is moved by the stained tail.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.quant.cell_expansion import auto_pixel_threshold   # noqa: E402

BG_OD, BG_SD, STAIN_OD = 0.05, 0.012, 0.35


def _field(frac_positive, n_cells=200, ring_px=200, seed=0):
    """One image's ring pixels: negatives are all background, positives carry a stained arc."""
    rng = np.random.default_rng(seed)
    n_pos = int(n_cells * frac_positive)
    cells = [rng.normal(BG_OD, BG_SD, ring_px) for _ in range(n_cells - n_pos)]
    cells += [np.r_[rng.normal(BG_OD, BG_SD, int(ring_px * 0.7)),
                    rng.normal(STAIN_OD, 0.05, ring_px - int(ring_px * 0.7))]
              for _ in range(n_pos)]
    return cells


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.3, 0.5, 0.8])
def test_threshold_separates_background_from_stain(frac):
    t = auto_pixel_threshold(_field(frac))
    assert BG_OD + 2 * BG_SD < t < STAIN_OD - 4 * 0.05, (
        f"threshold {t:.4f} does not sit between background and stain at {frac:.0%} positive")


def test_threshold_is_stable_across_positivity():
    """The whole point: an image with more positive cells gets the same definition."""
    ts = [auto_pixel_threshold(_field(f)) for f in (0.0, 0.1, 0.3, 0.5, 0.8)]
    assert max(ts) / min(ts) < 1.5, f"threshold drifted with positivity: {ts}"


def test_percentile_rule_would_have_failed_this():
    """Documents the rejected alternative, so nobody re-adopts it as 'simpler'."""
    p99 = [float(np.percentile(np.concatenate(_field(f)), 99))
           for f in (0.0, 0.1, 0.3, 0.5, 0.8)]
    assert max(p99) / min(p99) > 4, "the percentile rule was supposed to be unstable here"
    # And it lands inside the stained population, so almost nothing would be called.
    assert max(p99) > STAIN_OD


def test_no_measurable_rings_calls_nothing_positive():
    assert auto_pixel_threshold([]) == float("inf")
    assert auto_pixel_threshold([None, np.array([])]) == float("inf")


def test_flat_channel_degrades_rather_than_dividing_by_zero():
    assert auto_pixel_threshold([np.full(50, 0.2)]) == pytest.approx(0.2)
