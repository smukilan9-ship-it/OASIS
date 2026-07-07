"""
validate_cross_k.py — correctness validation for spatial_stats.cross_k_function /
cross_k_null. Reference-free, rigorous checks (no R/spatstat needed):

  Check A  Exact brute-force O(N*M) pair counting vs our cKDTree estimator
           on a fixed, saved point pattern  → must match to ~float epsilon.
  Check B  Closed form: for two INDEPENDENT Poisson patterns the cross-type
           K(r) = pi*r^2. We verify convergence across many realizations.
           Reported BOTH for the plain estimator (edge-biased, as in the
           pipeline) and a toroidal/periodic estimator (edge-free) so the
           known boundary bias is isolated and quantified.
  Check C  L-function identity L(r)=sqrt(K/pi) and L-r sign behaviour on a
           constructed attraction pattern.
  Check D  Null-model calibration: per-r one-sided p-values from cross_k_null
           are ~Uniform under CSR (type-I error controlled). This is what
           actually matters for the pipeline, because the MC null uses the
           SAME (uncorrected) estimator as the observed curve, so the edge
           bias cancels in the significance test.

Run:  .venv/bin/python validation/validate_cross_k.py
"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_stats import cross_k_function, cross_k_null   # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────────
# Reference implementations
# ──────────────────────────────────────────────────────────────────────────────

def brute_force_cross_k(A, B, radii, area):
    """Naive O(N*M) cross-type Ripley's K (plain estimator, no edge correction)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    na, nb = len(A), len(B)
    # full pairwise distance matrix
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))   # (na, nb)
    out = np.empty(len(radii))
    for i, r in enumerate(radii):
        out[i] = (area / (na * nb)) * np.count_nonzero(d <= r)
    return out


def torus_cross_k(A, B, radii, W, H):
    """
    Edge-corrected cross-K via the minimum-image (toroidal) distance on a
    W x H periodic window. For independent uniform patterns this estimator is
    unbiased, so E[K(r)] = pi*r^2 exactly (no boundary deficit).
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    na, nb = len(A), len(B)
    dx = np.abs(A[:, None, 0] - B[None, :, 0]); dx = np.minimum(dx, W - dx)
    dy = np.abs(A[:, None, 1] - B[None, :, 1]); dy = np.minimum(dy, H - dy)
    d = np.sqrt(dx ** 2 + dy ** 2)
    area = W * H
    return np.array([(area / (na * nb)) * np.count_nonzero(d <= r) for r in radii])


# ──────────────────────────────────────────────────────────────────────────────
# Check A — brute force vs cKDTree on a fixed, saved pattern
# ──────────────────────────────────────────────────────────────────────────────

def check_A():
    print("\n" + "=" * 70)
    print("CHECK A — exact brute-force O(N*M) vs cKDTree estimator (fixed pattern)")
    print("=" * 70)
    W, H = 1000.0, 800.0
    na, nb = 200, 100
    rng = np.random.default_rng(42)
    A = rng.uniform([0, 0], [W, H], (na, 2))
    B = rng.uniform([0, 0], [W, H], (nb, 2))
    area = W * H

    # Save the exact coordinates so the test is reproducible
    csv_path = os.path.join(OUT_DIR, "fixed_pattern.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "x", "y"])
        for x, y in A:
            w.writerow(["A", f"{x:.10f}", f"{y:.10f}"])
        for x, y in B:
            w.writerow(["B", f"{x:.10f}", f"{y:.10f}"])
    print(f"  Saved fixed pattern ({na} A + {nb} B, window {W}x{H}) -> {csv_path}")

    radii = np.arange(0.0, 200.0 + 5.0, 5.0)        # px (pixel_size = 1)

    ours = cross_k_function(A, B, radii, area, pixel_size_um=1.0)
    k_ours = np.array(ours["K_observed"])           # µm^2 == px^2 at s=1
    k_ref  = brute_force_cross_k(A, B, radii, area)

    abs_diff = np.abs(k_ours - k_ref)
    denom    = np.where(np.abs(k_ref) > 0, np.abs(k_ref), 1.0)
    rel_err  = abs_diff / denom
    print(f"  radii evaluated: {len(radii)}  (0..200 px, step 5)")
    print(f"  max  abs diff : {abs_diff.max():.3e}  px^2")
    print(f"  mean abs diff : {abs_diff.mean():.3e}  px^2")
    print(f"  max  rel error: {rel_err.max():.3e}")

    # Also validate the L-function identity L = sqrt(K/pi) and L-r
    l_minus_r = np.array(ours["L_minus_r"])
    l_check   = np.sqrt(np.clip(k_ref, 0, None) / np.pi) - radii
    l_diff    = np.abs(l_minus_r - l_check).max()
    print(f"  L-r identity max abs diff: {l_diff:.3e} px")

    ok = abs_diff.max() < 1e-6 and l_diff < 1e-6
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(brute force and cKDTree agree to float precision)")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Check B — independent Poisson: K(r) -> pi*r^2
# ──────────────────────────────────────────────────────────────────────────────

def check_B(n_real=400):
    print("\n" + "=" * 70)
    print("CHECK B — independent Poisson convergence to K(r) = pi*r^2")
    print("=" * 70)
    W, H = 1000.0, 1000.0
    na, nb = 300, 300
    area = W * H
    radii = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    theory = np.pi * radii ** 2

    plain_acc = np.zeros((n_real, len(radii)))
    torus_acc = np.zeros((n_real, len(radii)))
    rng = np.random.default_rng(7)
    for i in range(n_real):
        A = rng.uniform([0, 0], [W, H], (na, 2))
        B = rng.uniform([0, 0], [W, H], (nb, 2))
        plain_acc[i] = np.array(
            cross_k_function(A, B, radii, area, 1.0)["K_observed"])
        torus_acc[i] = torus_cross_k(A, B, radii, W, H)

    plain_mean = plain_acc.mean(0)
    torus_mean = torus_acc.mean(0)
    se = plain_acc.std(0) / np.sqrt(n_real)

    print(f"  {n_real} realizations, window {W:.0f}x{H:.0f}, "
          f"{na} A + {nb} B per realization\n")
    print(f"  {'r':>6} {'pi*r^2':>12} {'plain mean':>12} {'torus mean':>12} "
          f"{'plain%dev':>10} {'torus%dev':>10}")
    for j, r in enumerate(radii):
        pdev = 100 * (plain_mean[j] - theory[j]) / theory[j]
        tdev = 100 * (torus_mean[j] - theory[j]) / theory[j]
        print(f"  {r:>6.0f} {theory[j]:>12.1f} {plain_mean[j]:>12.1f} "
              f"{torus_mean[j]:>12.1f} {pdev:>9.2f}% {tdev:>9.2f}%")

    # Torus (edge-free) estimator must match pi*r^2 within a few SE.
    z_torus = np.abs(torus_mean - theory) / (torus_acc.std(0) / np.sqrt(n_real))
    torus_ok = np.all(z_torus < 4.0)
    print(f"\n  Torus estimator vs pi*r^2: max |z| = {z_torus.max():.2f} "
          f"-> {'PASS (matches within MC error)' if torus_ok else 'FAIL'}")
    print("  Plain estimator is biased LOW at large r (boundary deficit) — this")
    print("  is the KNOWN, expected behaviour of the uncorrected estimator and")
    print("  is exactly why the pipeline compares against a SIMULATED null,")
    print("  not the theoretical pi*r^2 line (see Check D).")
    return torus_ok


# ──────────────────────────────────────────────────────────────────────────────
# Check C — attraction pattern: L-r should be strongly positive at short range
# ──────────────────────────────────────────────────────────────────────────────

def check_C():
    print("\n" + "=" * 70)
    print("CHECK C — constructed attraction pattern: L-r > 0 at short range")
    print("=" * 70)
    W, H = 1000.0, 1000.0
    area = W * H
    rng = np.random.default_rng(11)
    A = rng.uniform([0, 0], [W, H], (250, 2))
    # B placed near random A points (sigma 8 px) -> strong short-range attraction
    B = A[rng.integers(0, len(A), 250)] + rng.normal(0, 8.0, (250, 2))
    radii = np.arange(0.0, 102.0, 2.0)
    res = cross_k_function(A, B, radii, area, 1.0)
    lmr = np.array(res["L_minus_r"])
    g = np.array([np.nan if v is None else v for v in res["g_observed"]])

    i10 = np.argmin(np.abs(radii - 10))
    i50 = np.argmin(np.abs(radii - 50))
    print(f"  L-r at r=10: {lmr[i10]:+.2f} px   L-r at r=50: {lmr[i50]:+.2f} px")
    print(f"  g(r) at r=4..10: {np.round(g[2:6], 2)}  (CSR -> ~1; attraction -> >>1)")
    ok = lmr[i10] > 5.0 and g[2] > 2.0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(attraction correctly produces large positive L-r and g>>1)")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Check D — null calibration (type-I error of the per-r test under CSR)
# ──────────────────────────────────────────────────────────────────────────────

def check_D(n_real=300, n_perm=199):
    print("\n" + "=" * 70)
    print("CHECK D — per-r p-value calibration under CSR (type-I error control)")
    print("=" * 70)
    W, H = 1000.0, 1000.0
    area = W * H
    na, nb = 200, 150
    radii = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Track p-values at each test radius across independent CSR realizations.
    pvals = np.zeros((n_real, len(radii)))
    rng = np.random.default_rng(2024)
    for i in range(n_real):
        A = rng.uniform([0, 0], [W, H], (na, 2))
        B = rng.uniform([0, 0], [W, H], (nb, 2))
        # Use a child seed per realization for the internal null permutations.
        res = cross_k_null(A, B, radii, area, 1.0,
                           n_perm=n_perm, seed=int(rng.integers(1 << 30)),
                           tissue_polygon=None)
        pvals[i] = np.array(res["p_values"])

    print(f"  {n_real} CSR realizations, n_perm={n_perm}, "
          f"window {W:.0f}x{H:.0f}, {na} A + {nb} B\n")
    print(f"  {'r':>6} {'mean p':>9} {'P(p<=0.05)':>12}  (expect ~0.05)")
    rates = []
    for j, r in enumerate(radii):
        rate = np.mean(pvals[:, j] <= 0.05)
        rates.append(rate)
        print(f"  {r:>6.0f} {pvals[:, j].mean():>9.3f} {rate:>12.3f}")
    rates = np.array(rates)
    # Binomial SE on the 0.05 rate at this n_real is ~0.013; allow up to ~0.10.
    ok = np.all(rates <= 0.10)
    print(f"\n  Mean p across all r should be ~0.5: "
          f"{pvals.mean():.3f}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(per-r false-positive rate controlled near nominal 0.05)")
    print("  NOTE: the GLOBAL 'significant' flag scans many radii (r<=50um) and")
    print("  is therefore an OR over correlated per-r tests — its family-wise")
    print("  type-I error is higher than 0.05 (multiple-comparison scan). The")
    print("  per-r p-values above are the calibrated quantity.")
    return ok


if __name__ == "__main__":
    results = {
        "A_bruteforce_vs_kdtree": check_A(),
        "B_poisson_pi_r2":        check_B(),
        "C_attraction_signature": check_C(),
        "D_null_calibration":     check_D(),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 70)
    sys.exit(0 if all(results.values()) else 1)
