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


def test_no_valid_primary_null_fails_closed():
    """A pair with no valid primary null must be withheld, not answered by the rejected one.

    _build_precheck_null_plan documents this outcome — "none (fail-closed) ... dense fallback
    gates fail; no robust primary null -> run withheld" — but the association loop had no
    `continue`, so a pair whose 75 um pre-flight had just been declared INVALID, and whose
    dense fallback was then unavailable, fell through and ran on the REWEIGHTED primary: the
    null the pre-flight had rejected for that tissue.

    Measured end to end on LL477 without a support CSV, that printed
    "primary(reweighted) SIGNIFICANT association (p=0.01) ROBUSTNESS=robust" on tissue where
    the reweighted null's size is 0.24. Section 15.3's routing change makes this reachable
    more often, since `caution` pairs now arrive here too.
    """
    import inspect

    from oasis.spatial import spatial as sp

    src = inspect.getsource(sp.run_spatial_association)
    assert 'error="no_valid_primary_null"' in src, (
        "the fail-closed branch for a missing primary null is gone")
    # it must come BEFORE the statistic runs, or it withholds nothing
    i_block = src.index('no_valid_primary_null')
    i_stat = src.index("cross_k_all_nulls(")
    assert i_block < i_stat, (
        "the fail-closed check must precede cross_k_all_nulls, otherwise the rejected null "
        "has already produced a verdict")


def test_absence_is_a_finding_and_not_the_same_state_as_a_missing_null():
    """Two states were collapsed into one silent dead end, and they mean different things.

    `marker_absent` (<5 positives) means there is no spatial arrangement to test — that IS
    the result, and an abundance/absence answer with the counts and the region examined is
    useful. Reporting it as "no association" would be wrong, because nothing was measured.
    "No valid primary null" means the architecture is fine but no null could be built, so
    nothing can be said either way. That is a genuine failure.

    An absence result must therefore still carry the window and the counts, and skip only
    the statistic.
    """
    import inspect

    from oasis.spatial import spatial as sp

    src = inspect.getsource(sp.run_spatial_association)
    assert 'finding="absence"' in src, "absence no longer reports itself as a finding"
    assert '"marker_absent"' in src, "absence is no longer distinguished from a missing null"
    # the absence payload must carry the evidence, not just a message
    for key in ("positives_total", "positives_in_window", "analysis_window"):
        assert key in src, f"absence result does not report {key}"
    # and it must never be presented as a measured null result
    assert '"statistics_valid": False' in src


def test_absent_and_sparse_markers_are_named_with_their_counts():
    """"CD8: 3 positive" beats "(CD8, TIM-3)".

    Naming the marker without its count says which one to look at but not whether it is 4
    cells or 0 — and those call for different next steps (re-threshold vs. this region has
    no signal). The count was already in per_image and simply was not surfaced.
    """
    from oasis.spatial.spatial import _named_counts

    per_image = {"CD8": {"n": 3}, "TIM-3": {"n": 1}}
    assert _named_counts(["CD8", "TIM-3"], per_image) == "CD8: 3 positive, TIM-3: 1 positive"
    assert _named_counts(["TIM-3"], per_image) == "TIM-3: 1 positive"
    assert _named_counts([], per_image) == "no marker"


# ── `caution` must reach the null it was routed to (research/ihc.md §15.3) ──────────────
def _plan_for(worst, tmp_path, *, n_support=800, dense_auto_null=True, n_pos=200):
    """Run the pre-flight null plan for one bandwidth verdict, with every gate satisfiable."""
    import shapely
    from oasis.spatial.spatial import _build_precheck_null_plan

    rng = np.random.default_rng(0)
    window = shapely.geometry.box(0, 0, 1000, 1000)
    pts = lambda n: rng.uniform(50, 950, (n, 2))
    csv = tmp_path / "support.tsv"          # QuPath's detection export is TAB delimited
    sup = rng.uniform(50, 950, (n_support, 2))
    csv.write_text("Centroid X µm\tCentroid Y µm\n"
                   + "\n".join(f"{x}\t{y}" for x, y in sup), encoding="utf-8")
    return _build_precheck_null_plan(
        {"valid": worst == "ok", "worst_status": worst},
        {"CD8": pts(n_pos), "CD45": pts(n_pos)}, ["CD8", "CD45"], window, 1.0,
        str(csv), dense_auto_null, True, 30, 500)


def test_caution_routes_to_the_dense_null_in_both_the_plan_and_the_run(tmp_path):
    """Marginal architecture takes the dense null, and the run must agree with the screen.

    Real failure, measured on serial HyReCo CD8/CD45: all three pairs came back with an
    architecture scale of 96-109 um, which is `caution` (between 75 and 150 um). The
    pre-flight planned `dense_morphology` and the badge said so; the association loop asked
    for the dense fallback only on `dense_tissue_bandwidth_invalid` or
    `underpowered_sparse_marker`, so it recorded `dense_null_status: not_evaluated` and
    withheld every pair. The gates never failed — they never ran.

    The routing itself is not a new decision (§15.3, validate_saturated_marker_null.py): in
    the two grid cells that land in `caution` the reweighted null's false engagement rate is
    0.17 and 0.25 against a nominal 0.05, and dense_morphology's is 0.00 in both.
    """
    import inspect

    from oasis.spatial.spatial import _DENSE_FALLBACK_STATUSES, run_spatial_association

    assert "caution" in _DENSE_FALLBACK_STATUSES

    plan = _plan_for("caution", tmp_path)
    assert plan["primary_null"] == "dense_morphology"
    assert plan["fail_closed"] is False
    assert not plan["dense"]["failed_gates"]

    # The run must ask for the fallback on the SAME set the plan promised it on, or the
    # screen and the worker disagree again. One list, read by both.
    src = inspect.getsource(run_spatial_association)
    assert "_pf_worst in _DENSE_FALLBACK_STATUSES" in src, (
        "the association loop no longer reads the shared eligibility list")


def test_caution_is_not_described_as_dense_tissue(tmp_path):
    """A 97 um field is marginally COARSER than the bandwidth, not inside it.

    Both verdicts take the dense null, for opposite reasons. Telling an operator their
    97 um tissue is "dense/fine" is a claim they can check against the image and disbelieve.
    """
    caution = _plan_for("caution", tmp_path)
    dense = _plan_for("dense_tissue_bandwidth_invalid", tmp_path)

    assert "marginally coarser" in caution["reason"]
    assert "dense/fine" not in caution["reason"]
    assert "dense/fine" in dense["reason"]


def test_an_unrecognised_bandwidth_verdict_fails_closed(tmp_path):
    """Only a verdict the run would route to the dense null may plan it.

    Without this the plan reaches the dense gates for ANY status that is not `ok`,
    `marker_absent` or `architecture_not_estimable` — including one added later that the
    association loop knows nothing about, which is how `caution` diverged in the first place.
    """
    plan = _plan_for("some_status_added_later", tmp_path)
    assert plan["primary_null"] == "none"
    assert plan["fail_closed"] is True


def test_caution_still_fails_closed_when_the_support_null_is_unavailable(tmp_path):
    """Routing `caution` to the dense null must not weaken the gates it now depends on."""
    plan = _plan_for("caution", tmp_path, n_support=10)
    assert plan["primary_null"] == "none"
    assert plan["fail_closed"] is True
    assert "min_support" in plan["dense"]["failed_gates"]

    off = _plan_for("caution", tmp_path, dense_auto_null=False)
    assert off["fail_closed"] is True
