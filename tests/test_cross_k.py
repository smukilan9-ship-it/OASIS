"""
Tier 1 — cross-type Ripley's K correctness (spatial_stats.cross_k_function).
Pure, fast, deterministic; no Monte-Carlo, no data. Known-answer geometry.
"""
import numpy as np
import spatial_stats as ss

R = np.arange(0, 101, 2.0)
W = 1200.0


def _L_at(out, r_um):
    rr = np.asarray(out["radii_um"]); L = np.asarray(out["L_minus_r"])
    return float(L[np.argmin(np.abs(rr - r_um))])


def test_k_is_zero_at_zero_and_monotone(csr_pair):
    out = ss.cross_k_function(csr_pair["a"], csr_pair["b"], R, csr_pair["area"], 1.0)
    K = np.asarray(out["K_observed"])
    assert K[0] == 0.0
    assert np.all(np.diff(K) >= -1e-6), "K(r) must be non-decreasing in r"


def test_csr_association_curve_near_zero(csr_pair):
    """Independent CSR ⇒ L(r)-r ≈ 0 at the interaction scale."""
    out = ss.cross_k_function(csr_pair["a"], csr_pair["b"], R, csr_pair["area"], 1.0)
    assert abs(_L_at(out, 20.0)) < 8.0


def test_attraction_gives_positive_L(attracted_pair):
    """B clustered around A ⇒ L(r)-r strongly positive at short range."""
    out = ss.cross_k_function(attracted_pair["a"], attracted_pair["b"], R,
                              attracted_pair["area"], 1.0)
    assert _L_at(out, 20.0) > 12.0


def test_pixel_size_scales_radii():
    """radii_um must scale with pixel_size_um (px → µm conversion is applied)."""
    a = np.array([[10.0, 10.0], [50.0, 50.0]]); b = np.array([[12.0, 12.0]])
    o1 = ss.cross_k_function(a, b, R, W * W, 1.0)
    o2 = ss.cross_k_function(a, b, R, W * W, 0.5)
    assert np.allclose(np.asarray(o2["radii_um"]), 0.5 * np.asarray(o1["radii_um"]))
