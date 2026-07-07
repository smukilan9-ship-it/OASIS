"""
validate_dclf.py — calibration + power check for the global DCLF envelope test
in spatial_stats.cross_k_null (the `global.global_p_dclf` field).

The DCLF test must satisfy two properties:
  1. CALIBRATION — under complete spatial randomness (independent A and B) the
     global p-value is ~Uniform(0,1); in particular P(p <= 0.05) ~= 0.05. This
     is what makes it an honest single test (unlike ORing per-radius decisions,
     whose family-wise false-positive rate is far above 0.05).
  2. POWER — on a clustered (associated) pattern the global p is small, and the
     one-sided association p is the small one; on a segregated pattern the
     one-sided segregation p is the small one.

Run:  .venv/bin/python validation/validate_dclf.py
"""

import os, sys, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_stats import cross_k_null   # noqa: E402

W = H = 1000.0
AREA = W * H
# Radii span the DCLF band (10–50 µm) with pixel_size=1 → r in px.
RADII = np.arange(0.0, 80.0, 2.0)


def _csr(rng, n):
    return rng.uniform([0, 0], [W, H], (n, 2))


def check_calibration(n_real=400, n_perm=199):
    print("\n" + "=" * 70)
    print("CHECK 1 — DCLF calibration under CSR (global p ~ Uniform)")
    print("=" * 70)
    rng = np.random.default_rng(20260614)
    ps = np.empty(n_real)
    for i in range(n_real):
        A = _csr(rng, 200)
        B = _csr(rng, 150)
        res = cross_k_null(A, B, RADII, AREA, 1.0, n_perm=n_perm,
                           seed=int(rng.integers(1 << 30)), tissue_polygon=None)
        ps[i] = res["global"]["global_p_dclf"]

    rate05 = float(np.mean(ps <= 0.05))
    rate10 = float(np.mean(ps <= 0.10))
    rate50 = float(np.mean(ps <= 0.50))
    # Binomial SE of the 0.05 rate at n_real ~ sqrt(.05*.95/n_real)
    se = (0.05 * 0.95 / n_real) ** 0.5
    print(f"  {n_real} CSR realizations, n_perm={n_perm}")
    print(f"  mean p            = {ps.mean():.3f}   (expect ~0.50)")
    print(f"  P(p <= 0.05)      = {rate05:.3f}   (expect ~0.05, ±{2*se:.3f})")
    print(f"  P(p <= 0.10)      = {rate10:.3f}   (expect ~0.10)")
    print(f"  P(p <= 0.50)      = {rate50:.3f}   (expect ~0.50)")
    ok = abs(rate05 - 0.05) <= 0.03 and abs(ps.mean() - 0.5) <= 0.06
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (false-positive rate controlled)")
    return ok


def check_power():
    print("\n" + "=" * 70)
    print("CHECK 2 — DCLF power on clustered / segregated patterns")
    print("=" * 70)
    rng = np.random.default_rng(1)

    # Associated: B placed near random A points
    A = _csr(rng, 250)
    B = A[rng.integers(0, len(A), 250)] + rng.normal(0, 8.0, (250, 2))
    r_as = cross_k_null(A, B, RADII, AREA, 1.0, n_perm=499, seed=0)["global"]
    print(f"  ASSOCIATION: p_dclf={r_as['global_p_dclf']:.4f}  "
          f"p_assoc={r_as['global_p_association']:.4f}  "
          f"p_seg={r_as['global_p_segregation']:.4f}  dir={r_as['direction']}")

    # Segregated: B kept away from A (A left half, B right half)
    A2 = rng.uniform([0, 0], [450, H], (250, 2))
    B2 = rng.uniform([550, 0], [W, H], (250, 2))
    r_sg = cross_k_null(A2, B2, RADII, AREA, 1.0, n_perm=499, seed=0)["global"]
    print(f"  SEGREGATION: p_dclf={r_sg['global_p_dclf']:.4f}  "
          f"p_assoc={r_sg['global_p_association']:.4f}  "
          f"p_seg={r_sg['global_p_segregation']:.4f}  dir={r_sg['direction']}")

    ok = (r_as["significant"] and r_as["direction"] == "association"
          and r_sg["significant"] and r_sg["direction"] == "segregation")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(clustered → association, separated → segregation, both p<0.05)")
    return ok


if __name__ == "__main__":
    r1 = check_calibration()
    r2 = check_power()
    print("\n" + "=" * 70)
    print(f"  {'PASS' if r1 else 'FAIL'}  calibration (uniform p under CSR)")
    print(f"  {'PASS' if r2 else 'FAIL'}  power + directionality")
    print("=" * 70)
    sys.exit(0 if (r1 and r2) else 1)
