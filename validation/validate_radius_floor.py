"""
validate_radius_floor.py — what residual registration error really does to the test.

CLAIM UNDER TEST
    Serial sections always deform. The historical ≤5 µm landmark-certification gate
    withholds the whole spatial analysis from a pair with as little as ~2 µm RMS of
    pervasive elastic deformation. Is that gate protecting the reader from a wrong
    conclusion, or is it withholding a valid one?

    Three things are measured on OASIS's own DCLF test (cross_k_all_nulls), with the
    B points displaced by Gaussian error ε and then clipped to the analysis window,
    exactly as run_spatial_association does:

      A. SIZE      Under independence, the false-positive rate must stay at the nominal
                   ~5% as ε grows. If it does, registration error cannot invent a finding.
      B. POWER     Under a weak true association, detection must degrade gracefully with
                   ε rather than collapse — the cost of error is sensitivity.
      C. BAND      Raising the DCLF band floor to k·ε must not IMPROVE power. If it does
                   not, clipping the band is pure loss and the floor must be applied as a
                   reporting boundary rather than as a gate on the statistic.

WHAT THE RESULT MEANS
    If A holds, a DEFORMED pair may be analysed: its error can only weaken a real
    association, never manufacture one. The pair's TRE then determines the smallest
    inter-cell distance it can RESOLVE (spatial_stats.registration_radius_floor), which
    bounds what may be CLAIMED — contact scale (~10–20 µm) versus neighbourhood scale —
    not whether the pair runs.

CRITICAL SCOPE LIMIT
    A and B hold because the transform is LANDMARK-driven, and anatomical landmarks are
    uncorrelated with where the stained cells sit. An INTENSITY-driven non-rigid
    registration optimises on a signal correlated with cell density and could pull A-rich
    tissue onto B-rich tissue, manufacturing exactly the association under test. Nothing
    here licenses one.

    Also note the window clipping. Points displaced outside the analysis window MUST be
    dropped (run_spatial_association does this). Retaining them while holding the area
    fixed corrupts the density bookkeeping and inflates size to ~0.42 at ε = 20 µm — an
    artefact of the bookkeeping, not a property of registration error.

Run:  python validation/validate_radius_floor.py           (~3 min)
      python validation/validate_radius_floor.py --quick   (~40 s, fewer repeats)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (cross_k_all_nulls, registration_radius_floor,
                           _RADIUS_FLOOR_FACTOR)

SIDE_PX = 2000.0
PIXEL_SIZE_UM = 0.5
AREA_PX = SIDE_PX ** 2
RADII_PX = np.arange(2.0, 101.0, 2.0) / PIXEL_SIZE_UM

DCLF_RMIN_UM, DCLF_RMAX_UM = 10.0, 50.0
EPS_UM = (0.0, 5.0, 8.0, 12.0, 20.0)
ALPHA = 0.05

# Size may wander this far from nominal on a finite number of repeats before we call it
# inflation. With 60 repeats the binomial SE at p=0.05 is ~0.028, so 0.15 is ~3.5 SE.
SIZE_CEILING = 0.15


def _test(a, b, rmin, seed, n_perm):
    r = cross_k_all_nulls(a, b, RADII_PX, AREA_PX, PIXEL_SIZE_UM,
                          n_perm=n_perm, seed=seed,
                          dclf_rmin_um=rmin, dclf_rmax_um=DCLF_RMAX_UM,
                          nulls=("homogeneous",))
    g = r["global"]
    return bool(g["significant"]), g["direction"]


def _displace_and_clip(points, eps_um, seed):
    """Registration error, then drop points pushed out of the analysis window —
    exactly what run_spatial_association's filter_points_in_polygon does."""
    if eps_um <= 0:
        return points
    g = np.random.default_rng(seed)
    p = points + g.normal(0, eps_um / PIXEL_SIZE_UM, points.shape)
    return p[(p > 0).all(1) & (p < SIDE_PX).all(1)]


def _independent(seed, n_a=400, n_b=600):
    g = np.random.default_rng(seed)
    return g.uniform(0, SIDE_PX, (n_a, 2)), g.uniform(0, SIDE_PX, (n_b, 2))


def _weak_association(seed, frac=0.12, cluster_um=15.0, n_a=300, n_b=400):
    """Only `frac` of B cells are recruited to an A cell. Deliberately weak: a saturated
    association detects at power 1.0 everywhere and discriminates nothing."""
    g = np.random.default_rng(seed)
    a = g.uniform(0, SIDE_PX, (n_a, 2))
    n_c = int(frac * n_b)
    recruited = a[g.integers(0, n_a, n_c)] + g.normal(0, cluster_um / PIXEL_SIZE_UM, (n_c, 2))
    b = np.vstack([recruited, g.uniform(0, SIDE_PX, (n_b - n_c, 2))])
    return a, b[(b > 0).all(1) & (b < SIDE_PX).all(1)]


def check_size(n_rep, n_perm):
    """(A) Registration error must not inflate the false-positive rate."""
    print("\n[A] SIZE — can registration error invent a finding?")
    print(f"    Independent A/B, DCLF band {DCLF_RMIN_UM:.0f}–{DCLF_RMAX_UM:.0f} µm, "
          f"α={ALPHA}. Target ≈{ALPHA:.2f}.\n")
    print(f"    {'ε (µm)':>8} | {'false-positive rate':>20} | {'of which association':>21}")
    ok = True
    for eps in EPS_UM:
        rej, assoc = 0, 0
        for k in range(n_rep):
            a, b = _independent(1000 + k)
            sig, direction = _test(a, _displace_and_clip(b, eps, 7000 + k),
                                   DCLF_RMIN_UM, k + 1, n_perm)
            rej += sig
            assoc += (sig and direction == "association")
        rate = rej / n_rep
        ok = ok and rate <= SIZE_CEILING
        flag = "" if rate <= SIZE_CEILING else "   <-- INFLATED"
        print(f"    {eps:>8.0f} | {rate:>20.3f} | {assoc:>21}{flag}")
    print(f"\n    {'PASS' if ok else 'FAIL'} — the test is "
          f"{'correctly sized at every ε; error cannot manufacture a finding'
             if ok else 'NOT correctly sized: error inflates type-I error'}.")
    return ok


def check_power(n_rep, n_perm):
    """(B) Error must cost sensitivity, gracefully — not validity."""
    print("\n[B] POWER — what does registration error cost?")
    print("    Weak true association (12% of B recruited at 15 µm), correct direction.\n")
    print(f"    {'ε (µm)':>8} | {'detection rate':>16}")
    rates = []
    for eps in EPS_UM:
        hit = 0
        for k in range(n_rep):
            a, b = _weak_association(3000 + k)
            sig, direction = _test(a, _displace_and_clip(b, eps, 9000 + k),
                                   DCLF_RMIN_UM, k + 1, n_perm)
            hit += (sig and direction == "association")
        rates.append(hit / n_rep)
        print(f"    {eps:>8.0f} | {rates[-1]:>16.2f}")
    # Graceful = monotone-ish decline, never a collapse to zero.
    ok = rates[0] > 0.2 and rates[-1] > 0.5 * rates[0]
    print(f"\n    {'PASS' if ok else 'FAIL'} — detection {rates[0]:.2f} → {rates[-1]:.2f} "
          f"as ε goes {EPS_UM[0]:.0f} → {EPS_UM[-1]:.0f} µm. Error costs "
          f"{'sensitivity, not validity' if ok else 'more than sensitivity'}.")
    return ok


def check_band_floor(n_rep, n_perm):
    """(C) Clipping the DCLF band to k·ε must not improve power — else it is pure loss."""
    print("\n[C] BAND — does raising the DCLF floor to k·ε buy anything?")
    print("    Same weak association. If clipping does not raise power, do not clip.\n")
    print(f"    {'ε (µm)':>8} | {'floor=10 µm':>13} | {'floor=2ε':>10} | {'floor=3ε':>10}")
    clipping_helps = False
    for eps in EPS_UM:
        row = []
        for floor in (DCLF_RMIN_UM, max(DCLF_RMIN_UM, 2 * eps), max(DCLF_RMIN_UM, 3 * eps)):
            if floor >= DCLF_RMAX_UM:
                row.append(None)
                continue
            hit = 0
            for k in range(n_rep):
                a, b = _weak_association(3000 + k)
                sig, direction = _test(a, _displace_and_clip(b, eps, 9000 + k),
                                       floor, k + 1, n_perm)
                hit += (sig and direction == "association")
            row.append(hit / n_rep)
        base = row[0]
        for alt in row[1:]:
            if alt is not None and base is not None and alt > base + 0.10:
                clipping_helps = True
        cells = ["n/a" if v is None else f"{v:.2f}" for v in row]
        print(f"    {eps:>8.0f} | {cells[0]:>13} | {cells[1]:>10} | {cells[2]:>10}")
    ok = not clipping_helps
    print(f"\n    {'PASS' if ok else 'FAIL'} — clipping the band "
          f"{'never improves power, so the floor is applied as a REPORTING boundary and '
             'the DCLF band is left alone'
             if ok else 'DOES improve power; reconsider clipping the band'}.")
    return ok


def main():
    quick = "--quick" in sys.argv
    n_rep = 20 if quick else 60
    n_perm = 99 if quick else 199

    print("=" * 78)
    print("Registration error: size, power, and the radius floor")
    print("=" * 78)
    print(f"Field {SIDE_PX*PIXEL_SIZE_UM:.0f}×{SIDE_PX*PIXEL_SIZE_UM:.0f} µm at "
          f"{PIXEL_SIZE_UM} µm/px · {n_rep} repeats × {n_perm} permutations"
          f"{'  [QUICK]' if quick else ''}")
    print(f"Reporting floor shipped in spatial_stats: r_min = {_RADIUS_FLOOR_FACTOR} × TRE "
          f"(e.g. TRE 8 µm → resolves ≥ {registration_radius_floor(8.0):.0f} µm)")

    results = {
        "size preserved (error cannot invent a finding)": check_size(n_rep, n_perm),
        "power degrades gracefully (error costs sensitivity)": check_power(n_rep, n_perm),
        "band clipping buys nothing (floor is a reporting boundary)": check_band_floor(n_rep, n_perm),
    }

    print("\n" + "=" * 78)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    every = all(results.values())
    print("=" * 78)
    print("RESULT:", "PASS" if every else "FAIL")
    if every:
        print("\nA deformed serial-section pair may be analysed. Its registration error")
        print("weakens a true association and cannot manufacture one, so a significant")
        print("result stands; a null result may simply be under-powered. The error sets")
        print(f"the smallest resolvable inter-cell distance (~{_RADIUS_FLOOR_FACTOR}×TRE),")
        print("which bounds what may be CLAIMED — contact scale vs neighbourhood scale —")
        print("not whether the pair runs.")
        print("\nHolds ONLY for landmark-driven, cell-blind transforms.")
    return 0 if every else 1


if __name__ == "__main__":
    sys.exit(main())
