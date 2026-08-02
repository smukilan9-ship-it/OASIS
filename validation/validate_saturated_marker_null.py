#!/usr/bin/env python
"""
validate_saturated_marker_null.py — what a saturated positivity call does to the statistic.

THE SITUATION IT MODELS. On LL477 the TIM-3 threshold marks 11,584 of 12,224 cells positive
(~95 %) because the cutoff sits below the image's own background DAB OD. Deciding TIM-3
callability is not OASIS's job. Deciding what OASIS does when a saturated marker reaches the
spatial statistic IS, because the pipeline cannot tell a saturated call from a real one and
will happily report a verdict either way.

THE PREDICTION UNDER TEST. Cross-K is linear in the B point set, so a contaminated marker is
a mixture,

    K_obs(r) = w * K_A,Btrue(r) + (1 - w) * K_A,Bfalse(r),    w = n_true / n_called

and B_false is drawn from ALL cells. Cells co-occur with cells, so K_A,Bfalse sits at or
above the independence expectation wherever tissue is dense. The prediction is therefore
asymmetric and specific: saturation should bias TOWARD a false attraction claim and AWAY
from detecting segregation. A false attraction claim at 10-20 um is a false cell-scale
ENGAGEMENT claim, which is the headline the Spatial Association tab exists to make -- so
this is the dangerous direction, not a nuisance.

WHAT IS MEASURED. A is held at ~3 % of cells (LL477's real CD8 positive rate). B is swept
from 5 % to 95 % of cells, selected INDEPENDENTLY of A every time, so the truth is always
"no association" and any claim is a false positive. Three things per level:

    size            rate of a significant ATTRACTION claim (truth: none)
    pre-flight      what architecture_scale_verdict says, i.e. which null production would
                    actually pick -- a null that is never selected cannot cause harm, and
                    the reweighted null's behaviour outside its regime is not a live defect
    both nulls      reweighted (used only when architecture is coarse) and dense_morphology
                    (the fallback, which conditions on all-cell morphology and should be
                    immune to saturation almost by construction: when B is 95 % of cells,
                    B and the support are nearly the same set)

Run:  python validation/validate_saturated_marker_null.py           (~10 min)
      python validation/validate_saturated_marker_null.py --quick   (~3 min)
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (cross_k_all_nulls, estimate_architecture_scale,
                                         architecture_scale_verdict, _BAND_STATISTIC)
from validation.spatial_substrates import load_substrate

RADII_UM = np.arange(0.0, 101.0, 2.0)
SUBSTRATES = ("ll477_cd8", "keren_p13")
NULLS = ("reweighted", "dense_morphology")
A_FRAC = 0.03                       # LL477's measured CD8 positive rate
B_FRACS = (0.05, 0.20, 0.50, 0.80, 0.95)
SIZE_MAX = 0.15

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "saturated_marker_null_results.json")


def one(pts, W, H, b_frac, seed, n_perm):
    """A and B chosen INDEPENDENTLY, so the truth is always 'no association'."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    idx = rng.permutation(n)
    n_a = max(int(A_FRAC * n), 30)
    A = pts[idx[:n_a]]
    # B is an independent draw over ALL cells, exactly what a threshold that marks 95 % of
    # cells positive produces: it carries no information about A.
    B = pts[rng.choice(n, size=max(int(b_frac * n), 30), replace=False)]
    r = cross_k_all_nulls(A, B, RADII_UM, W * H, 1.0, n_perm=n_perm, seed=seed,
                          nulls=NULLS, morphology_support=pts,
                          registration_radius_floor_um=None)
    out = {}
    for nl in NULLS:
        nul = r["nulls"].get(nl)
        if not nul:
            continue
        b = nul.get(_BAND_STATISTIC) or nul["bands"]
        out[nl] = {k: (b[k]["direction"] == "association" and b[k]["significant"])
                   for k in ("colocalization", "coinfiltration")}
    # which null production would choose for this B
    sc = estimate_architecture_scale(B, 1.0)
    s_um = sc.get("scale_um") if isinstance(sc, dict) else sc
    out["_route"] = architecture_scale_verdict(s_um)["status"]
    return out


def main():
    quick = "--quick" in sys.argv
    n_rep = 12 if quick else 40
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 88)
    print("Saturated marker — does a positivity call that marks 95 % of cells invent a claim?")
    print("=" * 88)
    print(f"A held at {A_FRAC:.0%} of cells (LL477's CD8 rate); B swept, always INDEPENDENT "
          f"of A\n{n_rep} reps x {n_perm} perms · verdict statistic = {_BAND_STATISTIC}"
          f"{'  [QUICK]' if quick else ''}\n")

    res = {}
    for sub in SUBSTRATES:
        pts, (W, H) = load_substrate(sub)
        print(f"  {sub}  ({len(pts)} cells)")
        print(f"    {'B frac':>7}{'route':>16}" +
              "".join(f"{nl[:12] + ' coloc':>20}" for nl in NULLS))
        for bf in B_FRACS:
            acc = {nl: {"colocalization": 0, "coinfiltration": 0} for nl in NULLS}
            routes = {}
            for k in range(n_rep):
                o = one(pts, W, H, bf, 5000 + k, n_perm)
                routes[o["_route"]] = routes.get(o["_route"], 0) + 1
                for nl in NULLS:
                    if nl in o:
                        for band in ("colocalization", "coinfiltration"):
                            acc[nl][band] += bool(o[nl][band])
            route = max(routes, key=routes.get)
            for nl in NULLS:
                for band in acc[nl]:
                    acc[nl][band] /= float(n_rep)
            res[(sub, bf)] = {"route": route,
                              **{nl: acc[nl] for nl in NULLS}}
            cells = "".join(f"{acc[nl]['colocalization']:>20.2f}" for nl in NULLS)
            print(f"    {bf:>7.0%}{route:>16}{cells}", flush=True)
        print()

    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    worst = {}
    for nl in NULLS:
        vals = [res[k][nl]["colocalization"] for k in res]
        worst[nl] = max(vals)
        print(f"  {nl:<18} worst false-ENGAGEMENT rate across all saturation levels: "
              f"{max(vals):.2f}  {'OK' if max(vals) <= SIZE_MAX else '<-- INFLATED'}")
    routes = {res[k]["route"] for k in res}
    print(f"\n  production routing at every level tested: {sorted(routes)}")
    if routes == {"dense_tissue"}:
        note = ("Every level routes to the dense fallback, so the reweighted column is "
                "what production AVOIDS here, not what it uses.")
    else:
        note = ("Some levels route to the reweighted primary; its column is then live "
                "behaviour, not a hypothetical.")
    print(f"  {note}")

    dense_ok = worst.get("dense_morphology", 1.0) <= SIZE_MAX
    print(f"\n  {'PASS' if dense_ok else 'FAIL'} — the null production actually selects "
          f"{'holds' if dense_ok else 'DOES NOT hold'} under saturation.")
    if dense_ok:
        print("  Saturation costs POWER (the true signal is diluted by the false positives)")
        print("  but does not MANUFACTURE a cell-scale engagement claim.")

    json.dump({"config": {"n_rep": n_rep, "n_perm": n_perm, "a_frac": A_FRAC,
                          "b_fracs": list(B_FRACS), "statistic": _BAND_STATISTIC},
               "grid": {f"{a}|{b}": v for (a, b), v in res.items()},
               "worst_false_engagement": worst, "routes": sorted(routes),
               "dense_holds": bool(dense_ok)},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
