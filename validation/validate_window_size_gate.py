#!/usr/bin/env python
"""
validate_window_size_gate.py — how small an analysis window still supports a claim?

THE GATE UNDER TEST. `CERTIFICATION_GATES["min_roi_frac"] = 0.10` rejects any certified
window smaller than 10 % of the field, and it is doing real work: on a 10X frame
(~1.56 mm²) that is ~156,000 um², and a measured run had three regions at 228,443 /
165,322 / 121,895 um² — the third was thrown out as NOT_CERTIFIABLE despite having the BEST
registration of the three (7.2 um against 7.9 and 8.4). It was rejected on area alone.

That produces a squeeze the operator feels directly: a bigger window carries more
deformation and fails the error gate, a smaller window registers better but fails this area
gate — and by the time it passes both it may hold too few positive cells to test anything.

WHERE 0.10 CAME FROM. Nothing measured. It is shared with the LOCALLY_CERTIFIED hull check,
where it means "is this sub-window a usable fraction of the field", and was reused here.

WHAT ACTUALLY MATTERS is not the window's share of the frame — a 10 % window on a 4X field
is four times the tissue of a 10 % window at 20X — but whether the STATISTIC still behaves
inside it. Two things can break as a window shrinks:

  SIZE   the cross-K null must stay correctly sized. Fewer points and a smaller window mean
         edge effects and a coarser permutation distribution; if size inflates, small
         windows manufacture findings and the gate is protecting something real.
  POWER  a real association must still be detectable. If power collapses, the window is not
         wrong, it is useless — a different failure, and one the operator should be told
         about rather than have silently withheld.

So this sweeps window area (as a fraction of the field) on real tissue and measures both,
plus the cell counts that come with it, and reports where each one actually breaks.

WHAT IT CANNOT SETTLE. The gate also guards against certifying a transform over a window far
larger than the landmarks that produced it (research/ihc.md § 15.9) — that is a REGISTRATION
question, not a statistical one, and it is handled separately by the hull check. This script
speaks only to whether the statistic survives a small window.

Run:  python validation/validate_window_size_gate.py           (~12 min)
      python validation/validate_window_size_gate.py --quick   (~3 min)
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
from oasis.spatial.serial_registration import CERTIFICATION_GATES
from validation.spatial_substrates import load_substrate, impose_band_association

RADII_UM = np.arange(0.0, 101.0, 2.0)
SUBSTRATES = ("ll477_cd8", "keren_p13")
NULL = "dense_morphology"
CONTACT = (_COLOC_RMIN_UM, _COLOC_RMAX_UM)
TARGET_ENRICHMENT = 2.0
# fraction of the FIELD the analysis window covers
FRACS = (0.02, 0.04, 0.06, 0.10, 0.15, 0.25, 0.40, 0.70)
ALPHA_MAX = 0.15
POWER_MIN = 0.50

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "window_size_gate_results.json")


def crop(pts, W, H, frac):
    """Centred square window covering `frac` of the field, and the points inside it."""
    side = float(np.sqrt(frac * W * H))
    x0, y0 = (W - side) / 2.0, (H - side) / 2.0
    m = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + side)
         & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + side))
    return pts[m], side * side


def one(substrate, frac, truth, seed, n_perm):
    pts, (W, H) = load_substrate(substrate, seed=seed)
    sub, area = crop(pts, W, H, frac)
    if len(sub) < 40:
        return None
    # Association is imposed INSIDE the window, so the truth is not clipped by the crop —
    # cropping an already-associated pattern would confound window size with a truncated
    # truth, and it is the window that is under test.
    kw = dict(boost=0.0) if truth == "independent" else dict(
        target_enrichment=TARGET_ENRICHMENT)
    n_a = max(int(0.06 * len(sub)), 15)
    n_b = max(int(0.10 * len(sub)), 20)
    try:
        A, B = impose_band_association(sub, CONTACT, n_a=n_a, n_b=n_b,
                                       rng=np.random.default_rng(seed), **kw)
    except ValueError:
        # Enrichment is capped at 1/p by geometry, and in a SMALL window of dense tissue the
        # 10-20 um annulus can already hold over half the candidates — so 2.0x is
        # unreachable there. Reported as unreachable rather than silently retried at a
        # weaker effect, which would confound window size with effect size, the one
        # confound this sweep exists to avoid.
        return None
    if len(A) < 10 or len(B) < 10:
        return None
    r = cross_k_all_nulls(A, B, RADII_UM, area, 1.0, n_perm=n_perm, seed=seed,
                          nulls=(NULL,), morphology_support=sub,
                          registration_radius_floor_um=None)
    b = r["nulls"][NULL][_BAND_STATISTIC]["colocalization"]
    return {"hit": bool(b["direction"] == "association" and b["significant"]),
            "n_a": len(A), "n_b": len(B), "area_um2": area}


def main():
    quick = "--quick" in sys.argv
    n_rep = 15 if quick else 40
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 92)
    print("Analysis-window size — where does the statistic actually break?")
    print("=" * 92)
    print(f"null={NULL} statistic={_BAND_STATISTIC} · {n_rep} reps x {n_perm} perms · "
          f"shipped gate min_roi_frac={CERTIFICATION_GATES['min_roi_frac']}"
          f"{'  [QUICK]' if quick else ''}\n")

    res = {}
    for sub in SUBSTRATES:
        print(f"  {sub}")
        print(f"    {'frac':>6}{'area µm²':>11}{'n_a':>6}{'n_b':>6}"
              f"{'SIZE':>8}{'POWER':>8}")
        for f in FRACS:
            row = {}
            for truth in ("independent", "contact"):
                hits = n = 0
                meta = {}
                for k in range(n_rep):
                    o = one(sub, f, truth, 4000 + k, n_perm)
                    if o is None:
                        continue
                    hits += o["hit"]
                    n += 1
                    meta = o
                row[truth] = (hits / n) if n else None
                row.update({k: meta.get(k) for k in ("n_a", "n_b", "area_um2")})
            res[(sub, f)] = row
            fmt = lambda v: "  n/a" if v is None else f"{v:.2f}"      # noqa: E731
            print(f"    {f:>6.2f}{(row.get('area_um2') or 0):>11.0f}"
                  f"{(row.get('n_a') or 0):>6}{(row.get('n_b') or 0):>6}"
                  f"{fmt(row['independent']):>8}{fmt(row['contact']):>8}", flush=True)
        print()

    print("=" * 92)
    print("WHERE EACH ONE BREAKS")
    print("=" * 92)
    size_break, power_break = {}, {}
    for sub in SUBSTRATES:
        sb = next((f for f in FRACS
                   if (res[(sub, f)]["independent"] or 0) > ALPHA_MAX), None)
        pb = next((f for f in reversed(FRACS)
                   if (res[(sub, f)]["contact"] or 0) < POWER_MIN), None)
        size_break[sub], power_break[sub] = sb, pb
        print(f"  {sub:<12} size inflates at frac "
              f"{('never in range' if sb is None else f'{sb:.2f}')}"
              f"   ·   power falls below {POWER_MIN:.2f} at or under frac "
              f"{('never in range' if pb is None else f'{pb:.2f}')}")

    shipped = CERTIFICATION_GATES["min_roi_frac"]
    worst_size = [v for v in size_break.values() if v is not None]
    worst_power = [v for v in power_break.values() if v is not None]
    print()
    if not worst_size:
        print(f"  SIZE never inflates, down to frac {min(FRACS):.2f}. The gate is NOT")
        print(f"  protecting validity — a small window does not manufacture a finding.")
    else:
        print(f"  SIZE inflates from frac {max(worst_size):.2f} down; a gate at or above "
              f"that IS protecting validity.")
    if worst_power:
        print(f"  POWER is the binding constraint: it falls below {POWER_MIN:.2f} at frac "
              f"{max(worst_power):.2f} and smaller.")
        implied = max(worst_power)
    else:
        implied = min(FRACS)
        print(f"  POWER holds at every tested fraction.")
    print(f"\n  shipped min_roi_frac = {shipped:.2f}; evidence supports "
          f"{'keeping it' if abs(implied - shipped) < 0.03 else f'~{implied:.2f}'}")
    print("  A window that is too small to detect anything should be reported as")
    print("  UNDERPOWERED with its counts, not silently dropped as NOT_CERTIFIABLE —")
    print("  those are different statements and only one of them is true.")

    json.dump({"config": {"n_rep": n_rep, "n_perm": n_perm, "fracs": list(FRACS),
                          "null": NULL, "statistic": _BAND_STATISTIC,
                          "shipped_min_roi_frac": shipped},
               "grid": {f"{a}|{b}": v for (a, b), v in res.items()},
               "size_break": size_break, "power_break": power_break,
               "implied_min_frac": implied},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
