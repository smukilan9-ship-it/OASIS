"""The assessability gate and the effect size that replaces the p-value as the headline.

The gate is EVIDENCE, not a preference: validation/validate_min_cells.py measured it at 150
reps x 199 permutations on two substrates, and the rule reproduces the detectable set with
zero mismatches. These tests exist so a later edit cannot quietly move it.
"""
import numpy as np
import pytest

from oasis.spatial.spatial_stats import (ASSESSABILITY_MIN_MAJOR, ASSESSABILITY_MIN_MINOR,
                                         _MDE_COUNT_GRID, assessable_counts, contact_effect,
                                         detection_limit, minimum_detectable_enrichment)


def test_registration_only_call_is_unchanged():
    """Existing callers pass tre alone; those values must not move."""
    assert minimum_detectable_enrichment(0.0) == 1.50
    assert minimum_detectable_enrichment(10.0) == 2.13
    assert minimum_detectable_enrichment(20.0) == 2.70
    assert minimum_detectable_enrichment(None) is None


def test_gate_reproduces_the_measured_detectable_set():
    """The rule must agree with the simulation on every combination that was run.

    Detectable in the sweep == present in _MDE_COUNT_GRID for that (n_a, n_b).
    """
    n_a_levels = (20, 100, 300)
    n_b_levels = (5, 10, 20, 40, 80, 160, 320)
    for a in n_a_levels:
        for b in n_b_levels:
            measured = b in _MDE_COUNT_GRID[a]
            assert assessable_counts(a, b) is measured, (
                f"gate disagrees with the measurement at n_a={a}, n_b={b}")


def test_the_scenario_that_motivated_the_gate():
    """20 A cells and 5 B cells: a p-value there is not reportable."""
    assert assessable_counts(20, 5) is False
    d = detection_limit(0.0, 20, 5)
    assert d["assessable"] is False
    assert d["mde"] is None
    assert d["limited_by"] == "cell counts"


def test_gate_is_symmetric_in_the_two_markers():
    """Which marker is rare should not change whether the pair can be assessed."""
    for a, b in ((100, 20), (20, 100), (300, 40), (40, 300), (10, 500), (500, 10)):
        assert assessable_counts(a, b) == assessable_counts(b, a)


def test_gate_boundaries():
    assert assessable_counts(ASSESSABILITY_MIN_MAJOR, ASSESSABILITY_MIN_MINOR) is True
    assert assessable_counts(ASSESSABILITY_MIN_MAJOR, ASSESSABILITY_MIN_MINOR - 1) is False
    assert assessable_counts(ASSESSABILITY_MIN_MAJOR - 1, ASSESSABILITY_MIN_MINOR) is False
    for bad in ((None, 100), (100, None), (float("nan"), 100), (-5, 100), (0, 100)):
        assert assessable_counts(*bad) is False


def test_mde_matches_the_grid_at_measured_points():
    for a, row in _MDE_COUNT_GRID.items():
        for b, expected in row.items():
            got = minimum_detectable_enrichment(0.0, a, b)
            assert got == pytest.approx(expected, abs=0.01), f"n_a={a} n_b={b}"


def test_more_cells_never_makes_detection_harder():
    """MDE must be monotone in both counts — otherwise the interpolation is wrong."""
    for a in (100, 300):
        vals = [minimum_detectable_enrichment(0.0, a, b) for b in (20, 40, 80, 160, 320)]
        assert all(x >= y for x, y in zip(vals, vals[1:])), vals
    for b in (40, 160):
        vals = [minimum_detectable_enrichment(0.0, a, b) for a in (100, 300)]
        assert vals[0] >= vals[1], vals


def test_the_binding_constraint_is_reported():
    """Whichever of registration or counts limits harder is the one named."""
    plenty = detection_limit(20.0, 300, 320)          # bad registration, many cells
    assert plenty["limited_by"] == "registration error"
    few = detection_limit(0.0, 100, 20)               # perfect registration, few cells
    assert few["limited_by"] == "cell counts"
    assert few["mde"] > plenty["mde"] or few["mde"] > 4.0


def test_below_the_gate_no_sensitivity_is_quoted():
    """Fail closed: never return a number that implies the pair could have found something."""
    assert minimum_detectable_enrichment(0.0, 20, 5) is None
    assert minimum_detectable_enrichment(0.0, 300, 10) is None


def _fake_result(obs_scale):
    """K curves for a pattern with `obs_scale` times the null's contact-band pairs."""
    r = np.arange(0.0, 51.0, 2.0)
    null = np.pi * r ** 2
    return {"radii_um": r.tolist(), "K_observed": (null * obs_scale).tolist(),
            "null_mean_K": null.tolist(), "null_lower_K": (null * 0.8).tolist(),
            "null_upper_K": (null * 1.2).tolist()}


def test_effect_reports_enrichment_and_neighbours():
    e = contact_effect(_fake_result(2.5), n_a=300, n_b=320, tissue_area_um2=1.0e6)
    assert e["enrichment"] == pytest.approx(2.5, abs=0.01)
    assert e["chance_range"] == [pytest.approx(0.8, abs=0.01), pytest.approx(1.2, abs=0.01)]
    assert e["outside_chance_range"] is True
    # lambda_B * K(rmax), with lambda_B = 320 / 1e6 um^2 and K = 2.5 * pi * 20^2
    assert e["neighbours_observed"] == pytest.approx(
        (320 / 1.0e6) * 2.5 * np.pi * 20 ** 2, rel=0.02)
    assert e["neighbours_observed"] > e["neighbours_expected"]


def test_effect_flag_is_descriptive_not_a_verdict():
    """A pattern inside the chance range must not be flagged, whatever its ratio."""
    e = contact_effect(_fake_result(1.1), n_a=300, n_b=320, tissue_area_um2=1.0e6)
    assert e["enrichment"] == pytest.approx(1.1, abs=0.01)
    assert e["outside_chance_range"] is False
    assert "significant" not in e and "informative" not in e


def test_effect_returns_none_when_the_band_is_undefined():
    assert contact_effect({"radii_um": [], "K_observed": [], "null_mean_K": []},
                          10, 10, 1.0) is None
