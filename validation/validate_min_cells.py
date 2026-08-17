#!/usr/bin/env python
"""
validate_min_cells.py — how few cells can a cross-type association survive?

WHY THIS EXISTS. `minimum_detectable_enrichment` answers "how well is this pair registered"
and nothing else. The other thing that decides whether a null — or a finding — can be
believed is how many cells there are. A pair with 5 B cells and 20 A cells will still return
a p-value, and nothing in the pipeline currently stops it: cross_k_function short-circuits
only at n == 0. This sweep measures what actually happens down there, so the gate can be set
from evidence instead of a round number.

THREE THINGS ARE MEASURED, and they answer different worries.

  SIZE — the false-positive rate on an INDEPENDENT pair. If size stays at the nominal 0.05
  as n falls, then a significant result at n_b = 5 is not a lie; the test is still correctly
  sized and the problem is elsewhere. If size inflates, small-n findings are manufactured and
  the gate has to be hard. This is the crux, and it has to be measured before anything else
  is worth saying.

  POWER — against known enrichments, giving the minimum detectable enrichment at each n.
  A pair that cannot detect 5x is not a pair that found nothing; it is a pair that could not
  have found anything.

  PRECISION — the spread of the estimated enrichment. Even a correctly-sized test is useless
  to report if, under independence, chance alone throws up estimates from 0.3x to 6x: an
  observed "4x enriched" then carries no information. This is the number that answers the
  objection directly, and it is why an effect size without an interval is not a result.

Enrichment is estimated as the contact-band annulus increment over its null mean:

    Ehat = [K_obs(hi) - K_obs(lo)] / [K_null(hi) - K_null(lo)]

which is the fold-excess of cross-type pairs separated by 10-20 um — the same quantity
`target_enrichment` plants, so the estimator and the truth are on one scale.

Truth is imposed by THINNING real segmented cells (spatial_substrates), never by inventing
coordinates, and is run on OASIS's own cohort tissue and on an independent modality (Keren
TNBC) so a conclusion that holds on one and not the other is not a conclusion.

Run:  python validation/validate_min_cells.py            (~15 min)
      python validation/validate_min_cells.py --quick    (~3 min)
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (cross_k_all_nulls, _BAND_STATISTIC,
                                         _COLOC_RMIN_UM, _COLOC_RMAX_UM)
from validation.spatial_substrates import load_substrate, impose_band_association

RADII_UM = np.arange(0.0, 101.0, 2.0)
SUBSTRATES = ("ll477_cd8", "keren_p13")
NULL = "dense_morphology"
CONTACT = (_COLOC_RMIN_UM, _COLOC_RMAX_UM)

N_A = (20, 100, 300)
N_B = (5, 10, 20, 40, 80, 160, 320)
ENRICHMENTS = (1.5, 2.0, 3.0, 5.0)
TARGET_POWER = 0.8
NOMINAL_SIZE = 0.05

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "min_cells_results.json")

_CACHE = {}


def substrate(name):
    """Load once — the CSV read dominates a single trial otherwise."""
    if name not in _CACHE:
        _CACHE[name] = load_substrate(name, seed=0)
    return _CACHE[name]


def _enrichment_hat(res):
    """Observed contact-band pair excess over its null mean. None when undefined."""
    r = np.asarray(res["radii_um"], float)
    k_obs = np.asarray(res["K_observed"], float)
    k_nul = np.asarray(res["null_mean_K"], float)
    i_lo = int(np.argmin(np.abs(r - CONTACT[0])))
    i_hi = int(np.argmin(np.abs(r - CONTACT[1])))
    if i_hi <= i_lo:
        return None
    den = k_nul[i_hi] - k_nul[i_lo]
    if not np.isfinite(den) or den <= 0:
        return None
    val = (k_obs[i_hi] - k_obs[i_lo]) / den
    return float(val) if np.isfinite(val) else None


def one(sub, n_a, n_b, enrichment, seed, n_perm):
    """One trial. Returns (significant_association, enrichment_hat) or None if unachievable.

    NOTE there is deliberately no `len(B) < 30` guard here, unlike validate_detectable_effect
    — small B is the entire subject of this sweep, and guarding it away would silently delete
    the rows the gate has to be read from.
    """
    pts, (W, H) = substrate(sub)
    rng = np.random.default_rng(seed)
    try:
        if enrichment is None:                       # size: independent pair
            A, B = impose_band_association(pts, CONTACT, n_a=n_a, n_b=n_b,
                                           boost=0.0, rng=rng)
        else:
            A, B = impose_band_association(pts, CONTACT, n_a=n_a, n_b=n_b,
                                           target_enrichment=enrichment, rng=rng)
    except ValueError:
        return None                                  # above the geometric enrichment cap
    if len(A) == 0 or len(B) == 0:
        return None
    res = cross_k_all_nulls(A, B, RADII_UM, W * H, 1.0, n_perm=n_perm, seed=seed,
                            nulls=(NULL,), morphology_support=pts,
                            registration_radius_floor_um=None)
    nres = res["nulls"][NULL]
    band = nres[_BAND_STATISTIC]["colocalization"]
    hit = bool(band["direction"] == "association" and band["significant"])
    return hit, _enrichment_hat(nres)


def cell(sub, n_a, n_b, enrichment, n_rep, n_perm, seed0):
    hits = n = 0
    ests = []
    for k in range(n_rep):
        o = one(sub, n_a, n_b, enrichment, seed0 + k, n_perm)
        if o is None:
            continue
        hit, est = o
        hits += hit
        n += 1
        if est is not None:
            ests.append(est)
    if n == 0:
        return None
    out = {"rate": round(hits / n, 3), "n_trials": n}
    if ests:
        a = np.asarray(ests, float)
        out.update({"e_median": round(float(np.median(a)), 2),
                    "e_lo": round(float(np.percentile(a, 2.5)), 2),
                    "e_hi": round(float(np.percentile(a, 97.5)), 2)})
    return out


def min_detectable(levels, powers, target):
    """Smallest enrichment reaching `target` power, linearly interpolated.

    Same rule as validate_detectable_effect so the two curves compose.
    """
    for i in range(1, len(levels)):
        a, b = powers[i - 1], powers[i]
        if a is None or b is None:
            continue
        if a < target <= b:
            f = 0.0 if b <= a else (target - a) / (b - a)
            return round(levels[i - 1] + f * (levels[i] - levels[i - 1]), 2)
    if powers and powers[0] is not None and powers[0] >= target:
        return levels[0]
    return None


def main():
    quick = "--quick" in sys.argv
    # 150 reps, not 60: the size row is the load-bearing one and 60 reps resolves a 0.05
    # rate only to ±0.03, which cannot distinguish "correctly sized" from "twice nominal".
    n_rep = 20 if quick else 150
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 100)
    print("Minimum cell counts for a cross-type association — size, power, and precision")
    print("=" * 100)
    print(f"null={NULL}  statistic={_BAND_STATISTIC}  band={CONTACT[0]:.0f}-{CONTACT[1]:.0f}µm  "
          f"{n_rep} reps x {n_perm} perms{'  [QUICK]' if quick else ''}")
    print("registration error is held at 0 µm — this sweep isolates the count axis\n")

    size, power, mde = {}, {}, {}

    for sub in SUBSTRATES:
        print(f"\n{'=' * 100}\n  {sub}\n{'=' * 100}")

        print(f"\n  SIZE — false-positive rate on INDEPENDENT pairs (nominal {NOMINAL_SIZE})")
        print(f"    {'n_A':>5}" + "".join(f"{f'n_B={b}':>12}" for b in N_B))
        for n_a in N_A:
            cells = []
            for n_b in N_B:
                c = cell(sub, n_a, n_b, None, n_rep, n_perm, 7000)
                size[(sub, n_a, n_b)] = c
                cells.append("     n/a" if c is None else f"{c['rate']:.3f}")
            print(f"    {n_a:>5}" + "".join(f"{c:>12}" for c in cells), flush=True)

        print(f"\n  PRECISION — enrichment estimated on INDEPENDENT pairs (truth = 1.0x)")
        print("    what chance alone can produce; a wide interval means an observed ratio "
              "says nothing")
        print(f"    {'n_A':>5}" + "".join(f"{f'n_B={b}':>18}" for b in N_B))
        for n_a in N_A:
            cells = []
            for n_b in N_B:
                c = size.get((sub, n_a, n_b))
                cells.append("           n/a" if not c or "e_lo" not in c
                             else f"{c['e_lo']:.2f}-{c['e_hi']:.2f}x")
            print(f"    {n_a:>5}" + "".join(f"{c:>18}" for c in cells), flush=True)

        print(f"\n  POWER, and the minimum detectable enrichment at {TARGET_POWER:.0%}")
        for n_a in N_A:
            print(f"\n    n_A = {n_a}")
            print(f"      {'n_B':>5}" + "".join(f"{e:>9.1f}x" for e in ENRICHMENTS)
                  + f"{'min detectable':>17}")
            for n_b in N_B:
                ps = []
                for enr in ENRICHMENTS:
                    c = cell(sub, n_a, n_b, enr, n_rep, n_perm, 4000)
                    power[(sub, n_a, n_b, enr)] = c
                    ps.append(None if c is None else c["rate"])
                m = min_detectable(list(ENRICHMENTS), ps, TARGET_POWER)
                mde[(sub, n_a, n_b)] = m
                cs = "".join(f"{('  n/a' if p is None else f'{p:.2f}'):>10}" for p in ps)
                print(f"      {n_b:>5}{cs}"
                      f"{('  not reached' if m is None else f'{m:.2f}x'):>17}", flush=True)

    print(f"\n{'=' * 100}\n  WHERE THE GATE GOES\n{'=' * 100}")
    print(f"    {'n_B':>5}" + "".join(f"{f'{s} nA={a}':>26}"
                                      for s in SUBSTRATES for a in N_A))
    for n_b in N_B:
        row = ""
        for s in SUBSTRATES:
            for a in N_A:
                m = mde.get((s, a, n_b))
                z = size.get((s, a, n_b))
                sz = "" if not z else f" size {z['rate']:.2f}"
                row += f"{('not reached' if m is None else f'{m:.2f}x') + sz:>26}"
        print(f"    {n_b:>5}{row}")

    def key(d):
        return {"|".join(str(x) for x in k): v for k, v in d.items()}

    json.dump({"config": {"n_rep": n_rep, "n_perm": n_perm, "n_a": list(N_A),
                          "n_b": list(N_B), "enrichments": list(ENRICHMENTS),
                          "target_power": TARGET_POWER, "nominal_size": NOMINAL_SIZE,
                          "null": NULL, "statistic": _BAND_STATISTIC,
                          "band_um": list(CONTACT), "substrates": list(SUBSTRATES),
                          "registration_error_um": 0.0},
               "size": key(size), "power": key(power),
               "min_detectable_enrichment": key(mde)},
              open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
