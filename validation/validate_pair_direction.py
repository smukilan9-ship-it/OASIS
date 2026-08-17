#!/usr/bin/env python
"""
validate_pair_direction.py — with unequal counts, which marker should be A?

Cross-type K is symmetric in theory: the same ordered pairs are counted either way. The TEST
is not, because the null holds A fixed and randomises B (`_null_k_homogeneous` and friends
take `tree_a` plus a count `n_b`). So the choice of which marker occupies which slot changes
what is being resampled, and with very unequal counts that should matter.

The intuition to check: putting the SPARSE marker in A leaves the abundant marker to be
resampled, and a null built from many random points is tighter than one built from few. A
tighter null should mean more power. The opposite arrangement resamples a handful of points
per permutation, so the null itself is noisy and the observed curve has less to beat.

Both orderings are run on the SAME planted pattern, so nothing differs except the slot
assignment. Size is measured the same way on independent pairs, because a power gain bought
with an inflated false-positive rate is not a gain.

Run:  python validation/validate_pair_direction.py           (~10 min)
      python validation/validate_pair_direction.py --quick   (~2 min)
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
# (sparse marker count, abundant marker count) — the shape of the one-sided pool.
COUNT_PAIRS = ((60, 300), (60, 500), (80, 300), (100, 500))
ENRICHMENTS = (2.0, 3.0)

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pair_direction_results.json")
_CACHE = {}


def substrate(name):
    if name not in _CACHE:
        _CACHE[name] = load_substrate(name, seed=0)
    return _CACHE[name]


def hit(a_pts, b_pts, support, area, seed, n_perm):
    """Is a contact-band association called, with a_pts in the A slot?"""
    r = cross_k_all_nulls(a_pts, b_pts, RADII_UM, area, 1.0, n_perm=n_perm, seed=seed,
                          nulls=(NULL,), morphology_support=support,
                          registration_radius_floor_um=None)
    b = r["nulls"][NULL][_BAND_STATISTIC]["colocalization"]
    return bool(b["direction"] == "association" and b["significant"])


def trial(sub, n_sparse, n_abundant, enrichment, seed, n_perm):
    """Plant once, then read the same pattern both ways round."""
    pts, (W, H) = substrate(sub)
    rng = np.random.default_rng(seed)
    try:
        if enrichment is None:
            sparse, abundant = impose_band_association(
                pts, CONTACT, n_a=n_sparse, n_b=n_abundant, boost=0.0, rng=rng)
        else:
            sparse, abundant = impose_band_association(
                pts, CONTACT, n_a=n_sparse, n_b=n_abundant,
                target_enrichment=enrichment, rng=rng)
    except ValueError:
        return None
    if len(sparse) == 0 or len(abundant) == 0:
        return None
    area = W * H
    return (hit(sparse, abundant, pts, area, seed, n_perm),      # sparse in A
            hit(abundant, sparse, pts, area, seed, n_perm))      # abundant in A


def cell(sub, ns, na, enr, n_rep, n_perm, seed0):
    s_first = a_first = n = 0
    for k in range(n_rep):
        o = trial(sub, ns, na, enr, seed0 + k, n_perm)
        if o is None:
            continue
        s_first += o[0]
        a_first += o[1]
        n += 1
    if n == 0:
        return None
    return {"sparse_in_A": round(s_first / n, 3), "abundant_in_A": round(a_first / n, 3),
            "n_trials": n}


def main():
    quick = "--quick" in sys.argv
    n_rep = 25 if quick else 100
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 92)
    print("Which marker belongs in the A slot when the counts are unequal")
    print("=" * 92)
    print(f"null={NULL}  statistic={_BAND_STATISTIC}  band={CONTACT[0]:.0f}-{CONTACT[1]:.0f}µm  "
          f"{n_rep} reps x {n_perm} perms{'  [QUICK]' if quick else ''}")
    print("both orderings read the SAME planted pattern; only the slot assignment differs\n")

    out = {}
    for sub in SUBSTRATES:
        print(f"\n  {sub}")
        print(f"    {'sparse':>7}{'abundant':>10}{'truth':>8}"
              f"{'sparse in A':>14}{'abundant in A':>15}{'  gain':>8}")
        for ns, na in COUNT_PAIRS:
            for enr in (None,) + ENRICHMENTS:
                c = cell(sub, ns, na, enr, n_rep, n_perm, 5000 if enr else 8000)
                out[f"{sub}|{ns}|{na}|{enr}"] = c
                if c is None:
                    continue
                label = "size" if enr is None else f"{enr:.1f}x"
                gain = c["sparse_in_A"] - c["abundant_in_A"]
                print(f"    {ns:>7}{na:>10}{label:>8}"
                      f"{c['sparse_in_A']:>14.2f}{c['abundant_in_A']:>15.2f}"
                      f"{gain:>+8.2f}", flush=True)

    powers = [(v["sparse_in_A"], v["abundant_in_A"])
              for k, v in out.items() if v and not k.endswith("|None")]
    sizes = [(v["sparse_in_A"], v["abundant_in_A"])
             for k, v in out.items() if v and k.endswith("|None")]
    print(f"\n{'=' * 92}")
    if powers:
        s = float(np.mean([p[0] for p in powers]))
        a = float(np.mean([p[1] for p in powers]))
        print(f"  mean POWER  sparse in A {s:.3f}   abundant in A {a:.3f}   "
              f"difference {s - a:+.3f}")
    if sizes:
        s = float(np.mean([p[0] for p in sizes]))
        a = float(np.mean([p[1] for p in sizes]))
        print(f"  mean SIZE   sparse in A {s:.3f}   abundant in A {a:.3f}   (nominal 0.05)")
    json.dump({"config": {"n_rep": n_rep, "n_perm": n_perm, "null": NULL,
                          "statistic": _BAND_STATISTIC, "band_um": list(CONTACT),
                          "count_pairs": [list(c) for c in COUNT_PAIRS],
                          "enrichments": list(ENRICHMENTS)},
               "results": out}, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
