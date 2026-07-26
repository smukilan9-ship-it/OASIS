"""
Tier 1 — ring connectivity (quant.cell_expansion.ring_connectivity).

The point of the feature is that it sees something completeness cannot: a contiguous
membrane arc and the same number of scattered speckle pixels have identical
`membrane_pos_frac`. Synthetic rings; no images needed.
"""
import numpy as np
import pytest

from oasis.quant.cell_expansion import ring_connectivity

CENTRE = (0.0, 0.0)


def _ring(n=720, radius=10.0):
    """Evenly sampled ring pixels and their angles."""
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xy = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    return xy, ang


def _vals(ang, positive_mask, hi=0.8, lo=0.02):
    v = np.full(ang.shape, lo)
    v[positive_mask] = hi
    return v


@pytest.mark.parametrize("coverage,min_ratio", [(0.1, 20), (0.2, 10), (0.3, 8), (0.5, 4)])
def test_contiguous_arc_beats_scattered_speckle_at_equal_completeness(coverage, min_ratio):
    """The premise of the whole feature. If this fails, connectivity is redundant.

    Both rings have identical `membrane_pos_frac`, so completeness cannot separate them.
    Measured discrimination at 720 ring pixels / 72 sectors:

        coverage   arc      speckle   ratio
        0.10       0.097    0.000       inf
        0.20       0.194    0.014      14x
        0.30       0.306    0.028      11x
        0.50       0.500    0.111     4.5x

    Discrimination is strongest at the low coverage typical of real membranous staining
    and narrows as coverage approaches 50 %, where each sector becomes a coin flip against
    the 0.5 sector threshold and runs start appearing in noise. The low-coverage regime is
    the one this feature is for; the 50 % row is recorded so the ceiling is not a surprise.
    """
    xy, ang = _ring()
    n = len(ang)
    k = int(n * coverage)
    arc = np.zeros(n, bool); arc[:k] = True                 # one clean contiguous arc
    rng = np.random.default_rng(0)
    speckle = np.zeros(n, bool)
    speckle[rng.choice(n, k, replace=False)] = True         # same pixel count, scattered

    assert arc.mean() == pytest.approx(speckle.mean())      # completeness sees no difference

    c_arc, arcs_arc, _ = ring_connectivity(xy, _vals(ang, arc), CENTRE, 0.4)
    c_spk, arcs_spk, _ = ring_connectivity(xy, _vals(ang, speckle), CENTRE, 0.4)

    assert c_arc == pytest.approx(coverage, abs=0.02)       # arc length is recovered
    assert c_spk == 0.0 or c_arc / c_spk >= min_ratio       # and it dominates the speckle
    assert arcs_arc == 1                                    # the arc is read as one arc


def test_full_ring_scores_one():
    xy, ang = _ring()
    c, arcs, covered = ring_connectivity(xy, _vals(ang, np.ones(len(ang), bool)), CENTRE, 0.4)
    assert c == pytest.approx(1.0)
    assert arcs == 1
    assert covered == pytest.approx(1.0)


def test_unstained_ring_scores_zero():
    xy, ang = _ring()
    c, arcs, covered = ring_connectivity(xy, _vals(ang, np.zeros(len(ang), bool)), CENTRE, 0.4)
    assert (c, arcs, covered) == (0.0, 0, 0.0)


def test_arc_spanning_the_wraparound_is_not_split():
    """An arc across the -pi/+pi seam is one arc, not two. A naive linear scan reports
    two half-length runs and halves the score for a perfectly good membrane."""
    xy, ang = _ring()
    n = len(ang)
    wrapped = np.zeros(n, bool)
    wrapped[: n // 8] = True
    wrapped[-n // 8:] = True                 # 25 % of the ring, straddling the seam
    c, arcs, _ = ring_connectivity(xy, _vals(ang, wrapped), CENTRE, 0.4)
    assert arcs == 1
    assert c == pytest.approx(0.25, abs=0.05)


def test_connectivity_grows_with_arc_length():
    xy, ang = _ring()
    n = len(ang)
    scores = []
    for frac in (0.1, 0.25, 0.5, 0.75):
        m = np.zeros(n, bool); m[: int(n * frac)] = True
        scores.append(ring_connectivity(xy, _vals(ang, m), CENTRE, 0.4)[0])
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


def test_dab_dominance_gate_rejects_counterstain():
    """A ring can be 'above threshold' purely from dark haematoxylin. The gate is what
    stops counterstain being read as a complete membrane."""
    xy, ang = _ring()
    m = np.ones(len(ang), bool)
    vals = _vals(ang, m, hi=0.5)
    hema = np.full(ang.shape, 0.9)           # counterstain darker than the DAB
    c_gated, _, _ = ring_connectivity(xy, vals, CENTRE, 0.4, hvals=hema)
    c_ungated, _, _ = ring_connectivity(xy, vals, CENTRE, 0.4, hvals=hema,
                                        dab_dominance_gate=False)
    assert c_gated == 0.0
    assert c_ungated == pytest.approx(1.0)


def test_empty_sectors_do_not_sever_an_arc():
    """Thin or Voronoi-clipped rings leave angular gaps with no pixels at all. Absence of
    pixels is not evidence of absent stain; treating it as unstained would break a
    continuous arc into fragments."""
    xy, ang = _ring()
    n = len(ang)
    keep = np.ones(n, bool)
    keep[n // 4: n // 4 + 3] = False          # a few missing pixels inside the arc
    m = np.zeros(n, bool); m[: n // 2] = True
    c, arcs, _ = ring_connectivity(xy[keep], _vals(ang, m)[keep], CENTRE, 0.4)
    assert arcs == 1
    assert c > 0.4


def test_degenerate_inputs_are_survivable():
    assert ring_connectivity(None, None, CENTRE, 0.4) == (0.0, 0, 0.0)
    assert ring_connectivity(np.zeros((0, 2)), np.zeros(0), CENTRE, 0.4) == (0.0, 0, 0.0)
    # mismatched lengths must not raise
    assert ring_connectivity(np.zeros((5, 2)), np.zeros(3), CENTRE, 0.4) == (0.0, 0, 0.0)


def test_covered_frac_separates_coverage_from_contiguity():
    """Equal coverage, different contiguity: connectivity must diverge while covered_frac
    agrees. That is precisely the axis completeness cannot see."""
    xy, ang = _ring()
    n = len(ang)
    arc = np.zeros(n, bool); arc[: n // 5] = True              # one arc, 20 % of the ring
    # Broken into sector-scale fragments — 10-pixel blocks match the 72-sector binning,
    # so the fragmentation is visible at the resolution the feature actually works at.
    broken = np.zeros(n, bool)
    for start in range(0, n, 50):
        broken[start:start + 10] = True                        # also ~20 % coverage
    c_arc, arcs_arc, cov_arc = ring_connectivity(xy, _vals(ang, arc), CENTRE, 0.4)
    c_brk, arcs_brk, cov_brk = ring_connectivity(xy, _vals(ang, broken), CENTRE, 0.4)

    assert cov_arc == pytest.approx(cov_brk, abs=0.05)         # same coverage
    assert c_arc > 3 * c_brk                                   # very different contiguity
    assert arcs_arc == 1 and arcs_brk > 5


def test_speckle_finer_than_a_sector_is_smoothed_into_coverage():
    """A documented limitation, not a bug. Sector binning is a low-pass filter: speckle
    finer than one sector raises every sector's stained fraction uniformly and reads as
    coverage rather than fragmentation. It buys robustness to single-pixel noise at the
    cost of blindness below sector scale — relevant when choosing `n_sectors`."""
    xy, ang = _ring()
    alternating = np.zeros(len(ang), bool)
    alternating[::2] = True                                    # 50 %, every other pixel
    c, arcs, covered = ring_connectivity(xy, _vals(ang, alternating), CENTRE, 0.4)
    assert covered > 0.9        # reads as nearly complete coverage...
    assert c > 0.5              # ...and as a long contiguous arc, though it is neither
