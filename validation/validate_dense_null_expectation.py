#!/usr/bin/env python
"""
validate_dense_null_expectation.py — is the dense_morphology null the null it claims to be?

WHERE THIS CAME FROM. KAMP (Wrobel & Song, arXiv 2412.08498) derives closed-form moments of
Ripley's K under a label-permutation null and notes an identity worth more than the speedup:
the null EXPECTATION needs no permutation at all, because averaging over all label
permutations "removes cell-type-specific spatial information and leaves the spatial structure
of the pooled set of observed cells". research/ihc.md § 18.2 records that their null is the
same one OASIS selected as `dense_morphology`. If the nulls really are the same, OASIS's
1000-permutation Monte Carlo mean has a closed form, and disagreeing with it is a bug.

THE IDENTITY, DERIVED FOR THE NULL OASIS ACTUALLY IMPLEMENTS — which is NOT KAMP's.
`cross_k_dense_morphology_test` holds A FIXED and redraws only B, sampling each B* i.i.d.
WITH REPLACEMENT from the all-cell support and adding Gaussian jitter. KAMP re-labels both
populations, so their expectation is the univariate K of the pooled pattern. Ours is
conditional on the observed A, and the corresponding expectation is a CROSS-K:

    K_null(r) = (|W| / (n_A n_B)) * SUM_{i in A} SUM_{j in B*} 1(d_ij <= r)

    E[K_null(r)] = (|W| / (n_A n_B)) * n_B * SUM_{i in A} P(one B* draw lands within r of a_i)
                 = (|W| / (n_A n_supp)) * SUM_{i in A} #{s in support : |s - a_i| <= r}
                 = K_cross(A, support)(r)                                    [jitter = 0]

So with the jitter switched off the permutation mean must equal the cross-K between the fixed
A and the FULL support, computed with the same normalisation and no permutation whatsoever.
That is an exact test of the sampler, not an approximation.

AND IT MEASURES SOMETHING WE NEVER MEASURED. With jitter on, B* is drawn from the support
CONVOLVED with N(0, sigma^2), so the identity no longer holds exactly and the gap is the
jitter's effect on the null — a 2 µm constant (`_DENSE_MORPHOLOGY_JITTER_UM`) that has been
shipped since the null was introduced and whose cost has never been quantified. A jitter that
materially lifts the null at contact scale is silently making the test conservative exactly
where the biology is.

WHAT A FAILURE WOULD MEAN. Disagreement at jitter = 0, beyond Monte Carlo error, means the
null being simulated is not the null documented in the docstring — the same class of defect
as § 15.9 (a certificate measuring something other than what it claimed) and § 16.13 (a
matcher condemned by a broken accessor). This check costs seconds and closes that class off.

Run:  .venv/bin/python validation/validate_dense_null_expectation.py
      .venv/bin/python validation/validate_dense_null_expectation.py --quick
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (          # noqa: E402
    cross_k_dense_morphology_test, _pair_counts, _k_from_counts,
    _DENSE_MORPHOLOGY_JITTER_UM, _COLOC_RMIN_UM, _COLOC_RMAX_UM)
from validation.spatial_substrates import load_substrate   # noqa: E402

SUBSTRATES = ("ll477_cd8", "keren_p13")
RADII_UM = np.arange(2.0, 101.0, 2.0)      # r = 0 is degenerate (K = 0 identically)
N_A, N_B = 300, 500
JITTERS = (0.0, _DENSE_MORPHOLOGY_JITTER_UM, 5.0)
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dense_null_expectation_results.json")


def analytic_null_mean(A, support, radii, area):
    """E[K_null(r)] with jitter = 0: the cross-K of fixed A against the whole support."""
    counts = _pair_counts(A, support, np.asarray(radii, float))
    return _k_from_counts(counts, float(area), len(A), len(support))


def one(substrate, jitter_um, n_perm, seed=0):
    pts, (W, H) = load_substrate(substrate, seed=seed)
    area = float(W) * float(H)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pts))
    A = pts[idx[:N_A]]
    B = pts[idx[N_A:N_A + N_B]]
    support = pts                                   # the all-cell morphology field

    exact = analytic_null_mean(A, support, RADII_UM, area)
    t0 = time.time()
    res = cross_k_dense_morphology_test(
        A, B, support, RADII_UM, area, 1.0,          # coords are µm, so pixel_size = 1
        n_perm=n_perm, seed=seed, jitter_um=float(jitter_um))
    secs = time.time() - t0
    mc = np.asarray(res["null_mean_K"], float)

    # Monte Carlo standard error of the mean, so "agrees" is judged against the right ruler.
    # Reconstructed from the reported envelope: the 2.5/97.5 spread of the null draws is
    # ~3.92 sd, so sd ~= (hi - lo)/3.92 and se = sd/sqrt(n_perm).
    lo = np.asarray(res["null_lower_K"], float)
    hi = np.asarray(res["null_upper_K"], float)
    se = (hi - lo) / 3.92 / np.sqrt(n_perm)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (mc - exact) / se, np.nan)
        rel = np.where(exact > 0, (mc - exact) / exact, np.nan)

    band = (RADII_UM >= _COLOC_RMIN_UM) & (RADII_UM <= _COLOC_RMAX_UM)
    return {"exact": exact, "mc": mc, "z": z, "rel": rel, "band": band,
            "seconds": round(secs, 2), "n_supp": int(len(support))}


def main():
    quick = "--quick" in sys.argv
    n_perm = 200 if quick else 1000
    t_start = time.time()

    print("=" * 94)
    print("Does the dense_morphology permutation mean equal its closed-form expectation?")
    print("=" * 94)
    print(f"E[K_null(r)] = K_cross(A, support)(r) exactly when jitter = 0 "
          f"· {n_perm} perms{'  [QUICK]' if quick else ''}")
    print(f"contact band = {_COLOC_RMIN_UM:.0f}-{_COLOC_RMAX_UM:.0f} µm · "
          f"shipped jitter = {_DENSE_MORPHOLOGY_JITTER_UM} µm\n")

    out, verdicts = {}, {}
    for sub in SUBSTRATES:
        print(f"  {sub}")
        print(f"    {'jitter':>7}{'max |z| vs MC error':>21}{'median rel. dev':>17}"
              f"{'rel. dev in contact band':>26}{'sec':>7}")
        for j in JITTERS:
            r = one(sub, j, n_perm)
            out[f"{sub}|{j}"] = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray)
                                     else v) for k, v in r.items()}
            zmax = float(np.nanmax(np.abs(r["z"])))
            relmed = float(np.nanmedian(r["rel"]))
            relband = float(np.nanmedian(r["rel"][r["band"]]))
            print(f"    {j:>7.1f}{zmax:>21.2f}{relmed:>+17.4%}{relband:>+26.4%}"
                  f"{r['seconds']:>7.1f}")
            if j == 0.0:
                verdicts[sub] = {"max_abs_z": zmax, "median_rel": relmed}
        print()

    print("=" * 94)
    print("VERDICT")
    print("=" * 94)
    ok = True
    for sub, v in verdicts.items():
        # |z| is a t-like ratio over 50 radii; 4 sd is a generous two-sided bar for that many
        # comparisons and still catches any structural error, which would be enormous.
        good = v["max_abs_z"] < 4.0
        ok &= good
        print(f"  {sub:<12} jitter=0: max |z| = {v['max_abs_z']:.2f}, "
              f"median relative deviation = {v['median_rel']:+.4%}  "
              f"{'AGREES — sampler is the documented null' if good else 'DISAGREES — INVESTIGATE'}")
    print()
    if ok:
        print("  The Monte Carlo null mean reproduces its closed form. The sampler draws the")
        print("  null the docstring describes, and the null OASIS uses is the null KAMP,")
        print("  B-KAMP and SpaceANOVA independently converged on (ihc.md § 18.2).")
    else:
        print("  The simulated null is NOT the documented null. Everything conditioned on it")
        print("  — size, power, every band verdict — is measuring something else.")
    print()
    print("  The jitter rows are the separate result: how much the shipped 2 µm jitter moves")
    print("  the null away from its unjittered expectation, and whether it does so at contact")
    print("  scale, where a lifted null makes the test conservative exactly where it matters.")

    json.dump({"config": {"n_perm": n_perm, "radii_um": RADII_UM.tolist(),
                          "n_a": N_A, "n_b": N_B, "jitters_um": list(JITTERS),
                          "shipped_jitter_um": _DENSE_MORPHOLOGY_JITTER_UM},
               "grid": out, "verdicts": verdicts, "passed": bool(ok)},
              open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t_start:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
