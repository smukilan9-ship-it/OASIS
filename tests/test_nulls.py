"""
Tier 1 — null-model calibration & power (spatial_stats.cross_k_all_nulls).

The scientifically load-bearing claim: the reweighted inhomogeneous null does NOT
report the false ASSOCIATION that the homogeneous-CSR baseline reports when two
populations merely share a tissue compartment. Fast deterministic smoke tests +
opt-in slow rate calibration (`pytest -m slow`).
"""
import numpy as np
import pytest
from oasis.spatial import spatial_stats as ss
from oasis.spatial import spatial

R = np.arange(0, 101, 4.0)


def _poisson(n, w, rng):
    return np.column_stack([rng.uniform(0, w, n), rng.uniform(0, w, n)])


def _shared_compartment(seed, w=1400.0):
    """A and B independent, but both live only in the SAME few blobs (shared tissue
    preference). This is the exact confounder the reweighted null must correct."""
    rng = np.random.default_rng(seed)
    centers = np.array([[420, 480], [980, 940], [500, 1050]])
    def draw(n):
        c = centers[rng.integers(0, len(centers), n)]
        return np.clip(c + rng.normal(0, 120, (n, 2)), 0, w)
    return draw(450), draw(450), w


def _verdict(a, b, w, n_perm=199):
    return ss.cross_k_all_nulls(a, b, R, w * w, 1.0, n_perm=n_perm, seed=0)


# ── Fast deterministic smoke tests ───────────────────────────────────────────
def test_attraction_is_robust(attracted_pair):
    """Genuine cell-scale clustering ⇒ significant association under the primary null."""
    o = _verdict(attracted_pair["a"], attracted_pair["b"], attracted_pair["w"])
    assert o["robustness"]["verdict"] == "robust"
    assert o["global"]["direction"] == "association"


def test_independent_csr_not_robust():
    """Independent CSR (fixed seed) ⇒ no robust association (false-positive control)."""
    rng = np.random.default_rng(0); w = 1200.0
    o = _verdict(_poisson(400, w, rng), _poisson(400, w, rng), w)
    assert o["robustness"]["verdict"] != "robust"


def test_reweighted_corrects_shared_preference():
    """Shared-compartment pattern: the homogeneous CSR baseline reports a FALSE
    association; the reweighted primary must NOT report that same association."""
    a, b, w = _shared_compartment(0)
    o = _verdict(a, b, w, n_perm=299)
    hom = o["nulls"]["homogeneous"]["global"]
    rw = o["nulls"]["reweighted"]["global"]
    assert hom["significant"] and hom["direction"] == "association", \
        "CSR baseline should exhibit the shared-preference association bias"
    assert not (rw["significant"] and rw["direction"] == "association"), \
        "reweighted null must not inherit the CSR false association"


def test_architecture_verdict_names_dense_tissue_not_unreliable():
    v = ss.architecture_scale_verdict(35.0, bandwidth_um=75.0)
    assert v["status"] == "dense_tissue"
    assert v["ok"] is False


def test_bandwidth_marker_absent_below_floor():
    # <5 positives → no spatial arrangement to test → marker_absent (Q3), not a
    # 'dense' or generic underpowered verdict.
    pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    r = spatial.precheck_bandwidth_within_window(
        {"A": pts, "B": pts.copy()}, ["A", "B"], pixel_size_um=1.0,
        window=None, bandwidth_um=75.0)

    assert r["worst_status"] == "marker_absent"
    assert r["per_image"]["A"]["power"] == "absent"
    assert "A" in r["absent_markers"]


def test_bandwidth_sparse_marker_runs_not_failclosed():
    # One marker adequate (≥30), the other sparse (5≤n<30): architecture comes from
    # the all-cell support and the pair is flagged underpowered_sparse_marker — it is
    # NOT force-failed, so segregation can still be reported (Q3).
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 500, (40, 2))          # adequate
    b = rng.uniform(0, 500, (15, 2))          # sparse
    support = rng.uniform(0, 500, (800, 2))   # dense all-cell field
    r = spatial.precheck_bandwidth_within_window(
        {"A": a, "B": b}, ["A", "B"], pixel_size_um=1.0,
        window=None, bandwidth_um=75.0, support=support)

    assert r["worst_status"] == "underpowered_sparse_marker"
    assert r["per_image"]["B"]["power"] == "sparse"
    assert r["per_image"]["A"]["power"] == "adequate"
    assert r["tissue_scale_source"] == "all_cell_support"


# ── Opt-in slow rate calibration (pytest -m slow) ────────────────────────────
@pytest.mark.slow
def test_power_rate_high():
    rng_seeds = range(8)
    def run(s):
        rng = np.random.default_rng(s); w = 1200.0
        a = _poisson(250, w, rng); i = rng.integers(0, len(a), 350)
        b = np.clip(a[i] + rng.normal(0, 6, (350, 2)), 0, w)
        return _verdict(a, b, w)["robustness"]["verdict"] == "robust"
    rate = np.mean([run(s) for s in rng_seeds])
    assert rate >= 0.8, f"power too low ({rate:.2f}); the primary null should detect real clustering"


@pytest.mark.slow
def test_type_i_error_controlled():
    """Independent CSR type-I rate. The reweighted primary is KNOWN to be mildly
    anti-conservative (~10% vs 5% target on synthetic); flag gross breakage (>25%)."""
    def run(s):
        rng = np.random.default_rng(1000 + s); w = 1200.0
        return _verdict(_poisson(400, w, rng), _poisson(400, w, rng), w
                        )["robustness"]["verdict"] == "robust"
    rate = np.mean([run(s) for s in range(16)])
    assert rate <= 0.25, f"type-I error inflated ({rate:.2f}) — null calibration regressed"


def test_marginal_architecture_does_not_route_to_the_reweighted_primary():
    """`caution` must NOT count as a valid 75 um bandwidth.

    It used to. `valid = worst in ("ok", "caution")` sent architecture only marginally
    coarser than the bandwidth to the reweighted primary with a "treat with care" string
    attached. Measured in validate_saturated_marker_null.py, the levels that land in
    `caution` are exactly where that null's FALSE cell-scale-engagement rate reaches
    0.17-0.25 against a nominal 0.05, while the dense morphology-conditioned null stays at
    0.00 on the same draws. A warning string does not undo a 5x inflated false-positive
    rate on the tab's headline claim.

    The two failure modes compound, which is how it stayed hidden: saturating a marker
    makes its pattern resemble the whole cell population, which RAISES the estimated
    architecture scale out of `dense_tissue` and into `caution` — so the worst input is the
    one routed to the weaker null.
    """
    import inspect

    from oasis.spatial import spatial as sp

    src = inspect.getsource(sp)
    assert 'valid = worst == "ok"' in src, (
        "the bandwidth pre-flight accepts something other than 'ok' as valid again")
    assert 'valid = worst in ("ok", "caution")' not in src, (
        "'caution' routes to the reweighted primary again")


def test_the_architecture_verdict_still_distinguishes_three_regimes():
    """Routing changed; the CLASSIFIER did not, and `caution` must stay its own state.

    Collapsing it into `dense_tissue` would lose the distinction the reason string needs
    to explain itself, and would hide how close a pair sits to the usable regime.
    """
    from oasis.spatial.spatial_stats import (architecture_scale_verdict,
                                             _REWEIGHT_BANDWIDTH_UM)

    bw = _REWEIGHT_BANDWIDTH_UM
    assert architecture_scale_verdict(3.0 * bw)["status"] == "ok"
    assert architecture_scale_verdict(1.2 * bw)["status"] == "caution"
    assert architecture_scale_verdict(0.4 * bw)["status"] == "dense_tissue"
    assert architecture_scale_verdict(None)["status"] == "unknown"
    # `ok` is the only status that reports ok=True, which is what the router now keys on.
    assert architecture_scale_verdict(1.2 * bw)["ok"] is False
