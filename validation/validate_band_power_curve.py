#!/usr/bin/env python
"""
validate_band_power_curve.py — power of the shipped band test versus effect size.

WHY. validate_band_decomposition.py reports the shipped combination
(dense_morphology + bands_ring) at power 0.38, measured at ONE effect size: a 2.0x per-band
enrichment, chosen because it is the largest value achievable for every truth on every
substrate once the two annuli are pinned to the reported bands. A single power number is
not interpretable on its own -- 0.38 is either fine or disqualifying depending entirely on
how big the effect had to be to get it, and a reader cannot tell which from one point.

So this sweeps the effect size and reports the curve. The useful output is not a pass/fail
but two numbers a reader can hold onto: the enrichment at which power reaches 0.8, and the
power at the smallest enrichment anyone would call biologically interesting.

WHAT IT IS NOT. Not a claim about what enrichment real CD8/TIM-3 tissue carries -- nothing
here measures that. It characterises the TEST, so that a null result from it can be read
correctly: "no association detected" means something different at power 0.9 than at 0.3, and
without this curve there is no way to say which regime a given run was in.

ENRICHMENT IS CAPPED BY GEOMETRY, not by choice. A set already holding a fraction p of
candidate cells cannot be over-represented beyond 1/p (see spatial_substrates.solve_boost),
and with the contact band at 10-20 um and the regional band at 30-40 um those annuli hold
roughly 0.18 and 0.20 of candidates each. Levels above the cap are reported as unreachable
rather than silently clipped to something smaller.

Run:  python validation/validate_band_power_curve.py           (~15 min)
      python validation/validate_band_power_curve.py --quick   (~4 min)
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
REGIONAL = (30.0, 40.0)
ENRICHMENTS = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
TARGET_POWER = 0.8

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "band_power_curve_results.json")


def one(substrate, truth, enrichment, seed, n_perm):
    pts, (W, H) = load_substrate(substrate, seed=seed)
    ann = CONTACT if truth == "contact" else REGIONAL
    try:
        A, B = impose_band_association(pts, ann, rng=np.random.default_rng(seed),
                                       target_enrichment=enrichment)
    except ValueError:
        return None                      # enrichment unreachable for this geometry
    r = cross_k_all_nulls(A, B, RADII_UM, W * H, 1.0, n_perm=n_perm, seed=seed,
                          nulls=(NULL,), morphology_support=pts,
                          registration_radius_floor_um=None)
    b = r["nulls"][NULL][_BAND_STATISTIC]
    band = "colocalization" if truth == "contact" else "coinfiltration"
    return bool(b[band]["direction"] == "association" and b[band]["significant"])


def interpolate_threshold(levels, powers, target):
    """Smallest enrichment reaching `target` power, linearly interpolated between levels."""
    for i in range(1, len(levels)):
        if powers[i - 1] is None or powers[i] is None:
            continue
        if powers[i - 1] < target <= powers[i]:
            span = powers[i] - powers[i - 1]
            f = 0.0 if span <= 0 else (target - powers[i - 1]) / span
            return round(levels[i - 1] + f * (levels[i] - levels[i - 1]), 2)
    if powers and powers[-1] is not None and powers[-1] >= target:
        return levels[-1]
    return None


def main():
    quick = "--quick" in sys.argv
    n_rep = 15 if quick else 40
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 88)
    print(f"Power vs effect size — null={NULL}, statistic={_BAND_STATISTIC}")
    print("=" * 88)
    print(f"{n_rep} reps x {n_perm} perms · contact band "
          f"{CONTACT[0]:.0f}-{CONTACT[1]:.0f} µm, regional {REGIONAL[0]:.0f}-"
          f"{REGIONAL[1]:.0f} µm{'  [QUICK]' if quick else ''}\n")

    res, thresholds = {}, {}
    for sub in SUBSTRATES:
        for truth in ("contact", "regional"):
            powers = []
            for e in ENRICHMENTS:
                hits = n = 0
                for k in range(n_rep):
                    o = one(sub, truth, e, 4000 + k, n_perm)
                    if o is None:
                        continue
                    hits += o
                    n += 1
                powers.append(round(hits / n, 3) if n else None)
                res[(sub, truth, e)] = {"power": powers[-1], "n": n}
            thresholds[(sub, truth)] = interpolate_threshold(
                list(ENRICHMENTS), powers, TARGET_POWER)
            cells = "".join(f"{('  n/a' if p is None else f'{p:.2f}'):>8}" for p in powers)
            print(f"  {sub:<12}{truth:<10}{cells}")
    print(f"  {'':22}" + "".join(f"{e:>8.2f}" for e in ENRICHMENTS) + "   <- enrichment")

    print("\n" + "=" * 88)
    print("READING")
    print("=" * 88)
    print(f"  enrichment needed for power {TARGET_POWER:.0%}:")
    for (sub, truth), thr in thresholds.items():
        txt = f"{thr:.2f}x" if thr else f"not reached at {max(ENRICHMENTS):.1f}x"
        print(f"    {sub:<12}{truth:<10}{txt}")
    at2 = {k: v["power"] for k, v in res.items() if k[2] == 2.0}
    print(f"\n  at the 2.0x used in validate_band_decomposition.py: "
          + ", ".join(f"{k[0]}/{k[1]} {v:.2f}" for k, v in at2.items() if v is not None))
    print("\n  A null result from this test should be read against these numbers: at a")
    print("  modest enrichment the test is not sensitive, so 'no association detected' is")
    print("  weak evidence of absence unless the effect it could have found is stated.")

    json.dump({"config": {"n_rep": n_rep, "n_perm": n_perm, "null": NULL,
                          "statistic": _BAND_STATISTIC,
                          "enrichments": list(ENRICHMENTS),
                          "target_power": TARGET_POWER},
               "curve": {f"{a}|{b}|{c}": v for (a, b, c), v in res.items()},
               "enrichment_for_target_power": {f"{a}|{b}": v
                                               for (a, b), v in thresholds.items()}},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
