"""
Tier 1 — null-model calibration & power (spatial_stats.cross_k_all_nulls).

The scientifically load-bearing claim: the reweighted inhomogeneous null does NOT
report the false ASSOCIATION that the homogeneous-CSR baseline reports when two
populations merely share a tissue compartment. Fast deterministic smoke tests +
opt-in slow rate calibration (`pytest -m slow`).
"""
import numpy as np
import pytest
import spatial_stats as ss

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
