#!/usr/bin/env python
"""
validate_detectable_effect.py — what is the SMALLEST association a given pair could find?

WHY NOT JUST REPORT POWER. Power is not a property of a pair. It is a property of a pair AND
an effect size, and quoting it without the effect size is actively misleading: the same
registration error that gives 0.28 power against a 2.0x enrichment gives 1.00 against 2.5x
(validate_band_power_curve.py). "Power 0.28" reads as "this test is useless" when it means
"this would miss a weak association and catch a strong one" — the opposite conclusion for a
researcher deciding whether to trust a null result.

WHAT TO REPORT INSTEAD. The minimum detectable enrichment: the effect size at which this
pair's registration error still yields 80 % power. That is a single number, it is the thing a
reader actually needs to interpret an absence, and it turns "no association detected" into
the honest and useful "no association of 2.4x or more was detectable here".

SO THIS SWEEPS BOTH AXES. Registration error and effect size, jointly, on real tissue with
the corrected statistic. Existing sweeps varied one and fixed the other, which is why the
minimum detectable effect could not be read off them.

WHY REGISTRATION ERROR COSTS POWER AT ALL, since the mechanism is the point. Error displaces
each B point relative to A by a distance uncorrelated with the biology, so a genuine excess
concentrated at one radius is smeared across a range about that radius. The excess is not
destroyed, it is spread — so the deviation from the null at any single radius shrinks, and a
test that ranks the observed curve against null curves has less to rank on. Nothing recovers
it: the information about which A cell a B cell was near is gone once the two are misplaced
relative to each other. This is also why error cannot INVENT a finding (validated
separately): smearing moves the observed curve TOWARD the null, never away from it.

Run:  python validation/validate_detectable_effect.py           (~20 min)
      python validation/validate_detectable_effect.py --quick   (~5 min)
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
from validation.validate_radius_floor_localisation import _displace_smooth

RADII_UM = np.arange(0.0, 101.0, 2.0)
SUBSTRATES = ("ll477_cd8", "keren_p13")
NULL = "dense_morphology"
CONTACT = (_COLOC_RMIN_UM, _COLOC_RMAX_UM)
EPS_UM = (0.0, 5.0, 10.0, 15.0, 20.0)
ENRICHMENTS = (1.5, 2.0, 2.5, 3.0, 4.0)
TARGET_POWER = 0.8

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "detectable_effect_results.json")


def one(substrate, eps_um, enrichment, seed, n_perm):
    pts, (W, H) = load_substrate(substrate, seed=seed)
    try:
        A, B = impose_band_association(pts, CONTACT, rng=np.random.default_rng(seed),
                                       target_enrichment=enrichment)
    except ValueError:
        return None                       # enrichment above the geometric cap
    if eps_um > 0:
        from validation import validate_radius_floor_localisation as L
        px = L.PIXEL_SIZE_UM
        B = _displace_smooth(B / px, eps_um, np.random.default_rng(90000 + seed)) * px
        B = B[(B[:, 0] > 0) & (B[:, 0] < W) & (B[:, 1] > 0) & (B[:, 1] < H)]
    if len(B) < 30:
        return None
    r = cross_k_all_nulls(A, B, RADII_UM, W * H, 1.0, n_perm=n_perm, seed=seed,
                          nulls=(NULL,), morphology_support=pts,
                          registration_radius_floor_um=None)
    b = r["nulls"][NULL][_BAND_STATISTIC]["colocalization"]
    return bool(b["direction"] == "association" and b["significant"])


def min_detectable(levels, powers, target):
    """Smallest enrichment reaching `target` power, linearly interpolated."""
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
    n_rep = 12 if quick else 30
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 92)
    print("Minimum detectable enrichment — jointly over registration error and effect size")
    print("=" * 92)
    print(f"null={NULL} statistic={_BAND_STATISTIC} · {n_rep} reps x {n_perm} perms · "
          f"target power {TARGET_POWER:.0%}{'  [QUICK]' if quick else ''}\n")

    grid, mde = {}, {}
    for sub in SUBSTRATES:
        print(f"  {sub} — power")
        print(f"    {'eps µm':>8}" + "".join(f"{e:>8.2f}x" for e in ENRICHMENTS)
              + f"{'min detectable':>16}")
        for eps in EPS_UM:
            powers = []
            for enr in ENRICHMENTS:
                hits = n = 0
                for k in range(n_rep):
                    o = one(sub, eps, enr, 4000 + k, n_perm)
                    if o is None:
                        continue
                    hits += o
                    n += 1
                powers.append(round(hits / n, 3) if n else None)
                grid[(sub, eps, enr)] = powers[-1]
            m = min_detectable(list(ENRICHMENTS), powers, TARGET_POWER)
            mde[(sub, eps)] = m
            cells = "".join(f"{('  n/a' if p is None else f'{p:.2f}'):>9}" for p in powers)
            print(f"    {eps:>8.0f}{cells}"
                  f"{('  not reached' if m is None else f'{m:.2f}x'):>16}", flush=True)
        print()

    print("=" * 92)
    print("WHAT TO SURFACE ON A RESULT")
    print("=" * 92)
    print(f"    {'eps µm':>8}" + "".join(f"{s:>16}" for s in SUBSTRATES))
    for eps in EPS_UM:
        row = "".join(
            f"{('not reached' if mde[(s, eps)] is None else f'{mde[(s, eps)]:.2f}x'):>16}"
            for s in SUBSTRATES)
        print(f"    {eps:>8.0f}{row}")
    print()
    print("  Read this as: at that registration error, an association weaker than the stated")
    print("  enrichment would probably have been missed. A null result should be reported")
    print("  against that number, never as an absence.")

    json.dump({"config": {"n_rep": n_rep, "n_perm": n_perm, "eps_um": list(EPS_UM),
                          "enrichments": list(ENRICHMENTS), "target_power": TARGET_POWER,
                          "null": NULL, "statistic": _BAND_STATISTIC},
               "power": {f"{a}|{b}|{c}": v for (a, b, c), v in grid.items()},
               "min_detectable_enrichment": {f"{a}|{b}": v for (a, b), v in mde.items()}},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
