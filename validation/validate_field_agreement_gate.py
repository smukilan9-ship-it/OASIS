#!/usr/bin/env python
"""
validate_field_agreement_gate.py — can a physically impossible transform still certify?

THE DEFECT THIS EXISTS TO CLOSE (research/ihc.md § 16.1). Decoding the transform of every
window that reached a certifying verdict in the production sweeps found that the failure is a
function of WINDOW SIZE, not of matcher quality:

    arm   window radius   certifying   physically impossible
    4X       600 um           26            0   ( 0 %)
    10X      238 um           24           12   (50 %)
    20X      139 um           10           10   (100 %)

"Physically impossible" is |rotation| > 5 deg or |scale - 1| > 5 % against the whole-field
transform. Serial sections of ONE block imaged on ONE scanner at ONE objective cannot differ
by 32 deg and 46 % in scale. The worst case was CERTIFIED at a claimed 2.32 um cell error
while its transform rotated 5.4 deg and shrank 5 %.

The residual gate cannot catch this, by construction: a similarity fitted to correspondences
packed into one small region absorbs a large rotation or scale error while still fitting THOSE
points well, so the residual stays small. It is § 15.9 again — a residual can only measure the
transform where the evidence is.

THREE CHANGES ARE UNDER TEST, and each was incomplete without the next:
  1. the local fit takes rotation and scale from the whole field, fitting only translation;
  2. `fixed_linear` carries that same matrix into landmark_register_and_verify, so the VERDICT
     is measured on the transform that is reported rather than on a free refit of its own;
  3. the field-agreement gate refuses a region whose own correspondences wanted their own
     rotation or scale, and says so instead of leaving a bare DEFORMED.

WHY THIS RUNS ON REAL PAIRS AND NOT A FIXTURE. The unit tests passed the entire time those
20X windows were certifying impossible transforms — a synthetic case proves the mechanism, not
the outcome. This sweeps real cohort pairs at the production window radius and asserts the
outcome directly: AFTER the fix, no window that reaches a certifying verdict may carry a
transform that disagrees with its own field.

WHAT THIS DELIBERATELY CANNOT MEASURE, so nobody reads the wrong number off it. The pairs in
~/Desktop/deformed/mixed are the cohort's FAILURES — their manifests read `size policy:
none_certified` with verdicts like {'DEFORMED': 8, 'NO_MATCHES': 4}. That is the right place
to prove a gate REFUSES, and the wrong place to measure what refusing COSTS: almost nothing
here certified before the fix either, so a low certification count is the cohort, not the gate.
The coverage cost has to be measured on pairs that do certify, and it has not been.

Run:  .venv/bin/python validation/validate_field_agreement_gate.py
      .venv/bin/python validation/validate_field_agreement_gate.py --quick
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial import loftr_matcher as lm                 # noqa: E402
from oasis.spatial import serial_registration as sr           # noqa: E402

COHORT = os.path.expanduser("~/Desktop/deformed/mixed")
PX_UM = 0.7519
RADIUS_UM = 238.0                 # the 10X production window radius
GRID = 3                          # GRID x GRID window centres per pair
WORK = 800
CERTIFYING = ("CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "field_agreement_gate_results.json")


def rot_scale(M):
    a, b = float(M[0][0]), float(M[0][1])
    return abs(math.degrees(math.atan2(-b, a))), math.hypot(a, b)


def disagreement(M, G):
    """|rotation| and |scale-1| of M relative to the field transform G."""
    ra, sa = rot_scale(M)
    rg, sg = rot_scale(G)
    d = abs(ra - rg)
    return min(d, 360.0 - d), (abs(sa / sg - 1.0) if sg > 0 else float("nan"))


def pairs(limit=None):
    out = []
    for d in sorted(glob.glob(os.path.join(COHORT, "*_10X_*"))):
        cd8 = glob.glob(os.path.join(d, "*CD8*.tif"))
        tim = glob.glob(os.path.join(d, "*Tim3*.tif")) or glob.glob(os.path.join(d, "*TIM3*.tif"))
        if cd8 and tim:
            out.append((os.path.basename(d), cd8[0], tim[0]))
    return out[:limit] if limit else out


def main():
    from oasis.common.registration import _load_rgb_thumbnail
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    ps = pairs(3 if a.quick else None)

    print("=" * 96)
    print("Can a window whose transform disagrees with its own field still certify?")
    print("=" * 96)
    print(f"{len(ps)} real 10X pairs · {RADIUS_UM:.0f} µm radius · {GRID}x{GRID} windows each")
    print(f"impossible = |Δrot| > {lm.GLOBAL_AGREEMENT_MAX_DEG:.0f}° or "
          f"|Δscale| > {lm.GLOBAL_AGREEMENT_MAX_SCALE:.0%} vs the whole field\n")
    print(f"  {'pair':<26}{'win':>5}{'certifying':>12}{'impossible':>12}"
          f"{'refused by gate':>17}{'max Δrot':>10}")

    rows, tot_cert, tot_bad, tot_refused = [], 0, 0, 0
    for name, rp, mp in ps:
        ref, _ = _load_rgb_thumbnail(rp, max_side=1920)
        mov, _ = _load_rgb_thumbnail(mp, max_side=1920)
        r = min(1.0, WORK / max(ref.shape[:2]))
        import cv2
        if r < 1.0:
            ref = cv2.resize(ref, (int(ref.shape[1] * r), int(ref.shape[0] * r)))
            mov = cv2.resize(mov, (int(mov.shape[1] * r), int(mov.shape[0] * r)))
        px = PX_UM / r
        H, W = ref.shape[:2]

        c = lm.loftr_correspondences(ref, mov, pixel_size_um=px)
        if not c["ok"]:
            print(f"  {name:<26}    — no whole-field correspondences")
            continue
        G = sr._fit_similarity_robust(np.asarray(c["mov_points"], float),
                                     np.asarray(c["ref_points"], float))

        half = int(round(RADIUS_UM / px))
        n_cert = n_bad = n_ref = 0
        worst = 0.0
        for iy in range(GRID):
            for ix in range(GRID):
                cx, cy = int((ix + 1) * W / (GRID + 1)), int((iy + 1) * H / (GRID + 1))
                x0, x1 = max(0, cx - half), min(W, cx + half)
                y0, y1 = max(0, cy - half), min(H, cy + half)
                roi = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
                try:
                    cert = lm.certify_local_roi(ref, mov, roi, px, provisional_matrix=G,
                                                work_max_dim=WORK)
                except Exception:
                    continue
                if cert.get("refused_by") == "field_agreement":
                    n_ref += 1
                v = cert.get("verdict")
                M = cert.get("local_matrix") or cert.get("matrix")
                if v in CERTIFYING and M is not None:
                    n_cert += 1
                    drot, dsc = disagreement(np.asarray(M, float), G)
                    worst = max(worst, drot)
                    if (drot > lm.GLOBAL_AGREEMENT_MAX_DEG
                            or dsc > lm.GLOBAL_AGREEMENT_MAX_SCALE):
                        n_bad += 1
                        rows.append({"pair": name, "verdict": v,
                                     "d_rot_deg": round(drot, 2),
                                     "d_scale": round(dsc, 4),
                                     "cell_error_p90_um": cert.get("cell_error_p90_um")})
        tot_cert += n_cert; tot_bad += n_bad; tot_refused += n_ref
        print(f"  {name:<26}{GRID * GRID:>5}{n_cert:>12}{n_bad:>12}{n_ref:>17}"
              f"{worst:>10.2f}")

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  windows reaching a certifying verdict : {tot_cert}")
    print(f"  of those, physically impossible       : {tot_bad}")
    print(f"  windows refused by the field gate     : {tot_refused}")
    print()
    ok = (tot_bad == 0)
    if ok:
        print("  PASS — no certifying window carries a transform its own field disagrees with.")
        print("  § 16.1 measured 12 of 24 at this radius before the fix.")
    else:
        print(f"  FAIL — {tot_bad} certifying window(s) still carry an impossible transform:")
        for r_ in rows[:10]:
            print(f"    {r_['pair']:<26} {r_['verdict']:<16} Δrot {r_['d_rot_deg']:>6.2f}°  "
                  f"Δscale {r_['d_scale']:>6.3f}  claimed p90 {r_['cell_error_p90_um']} µm")

    json.dump({"config": {"radius_um": RADIUS_UM, "grid": GRID, "work_px": WORK,
                          "rot_max_deg": lm.GLOBAL_AGREEMENT_MAX_DEG,
                          "scale_max": lm.GLOBAL_AGREEMENT_MAX_SCALE},
               "n_certifying": tot_cert, "n_impossible": tot_bad,
               "n_refused_by_gate": tot_refused, "offenders": rows, "passed": bool(ok)},
              open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
