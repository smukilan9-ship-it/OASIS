"""
Does a LOCAL rigid fit certify where the GLOBAL fit fails? The premise of the draw-your-own-
ROI feature, tested on the exact ANHIR pairs that failed globally.

THE CLAIM. Serial-section deformation is a SMOOTH field. A single similarity transform cannot
absorb it over a whole slide (mammary: 335-713 um realized error). But over a small enough
region the field is locally near-affine, so a similarity fit ON THAT REGION should leave only
a few um -- certifiable for cell-level colocalization. If true, letting a user carve a region
and fitting it locally rescues tissue that a global fit rejects. If false (the field is rough
even locally), no ROI helps and we must say so.

THE TEST. For each pair, for shrinking region radii, drop a window at many landmark-centred
locations. Fit the similarity on PS landmarks INSIDE the window; measure realized error at the
independent JB landmarks inside the SAME window. This is annotator-independent (JB never
touches the fit) and correspondence-source-independent (landmarks isolate deformation from any
matcher noise). Report the realized p90 as the region shrinks.

  global p90  >> local p90   -> local rigidity is real; the feature works. Report the region
                               size at which local error drops below the cell scale.
  local p90 ~ global p90     -> deformation is rough; shrinking the ROI does not help.

Run:  .venv/bin/python validation/validate_local_roi_rescue.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from oasis.spatial import serial_registration as sr  # noqa: E402

NATIVE_PX_100 = {"lung-lesion": 0.174, "lung-lobes": 1.274, "mammary-gland": 2.294}
LANDMARK_SCALE_PC = {"lung-lesion": 50, "lung-lobes": 100, "mammary-gland": 100}
PAIRS = [
    ("lung-lesion_3", "29-041-Izd2-w35-He-les3", "29-041-Izd2-w35-proSPC-4-les3"),
    ("mammary-gland_1", "s1_37-HE_A4926-4L", "s1_40-PR_A4926-4L"),
    ("mammary-gland_2", "s2_63-HE_A4926-4L", "s2_68-ER-A4962-4L"),
]
RADIUS_FRACS = [1.0, 0.5, 0.35, 0.25, 0.15, 0.10]   # of tissue diagonal
MIN_IN_WINDOW = 8
N_CENTERS = 40
CELL_SCALE_UM = 10.0                                 # "certifiable-ish" yardstick for the sweep


def _prefix(t):
    for p in NATIVE_PX_100:
        if t.startswith(p):
            return p
    raise KeyError(t)


def px_100(t):
    return NATIVE_PX_100[_prefix(t)]


def load_xy(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = [h.strip().lower() for h in rows[0]]
    xi = hdr.index("x") if "x" in hdr else -2
    yi = hdr.index("y") if "y" in hdr else -1
    pts = []
    for r in rows[1:]:
        try:
            pts.append([float(r[xi]), float(r[yi])])
        except (ValueError, IndexError):
            pass
    return np.array(pts, float)


def find_lm(tissue, user, base):
    for root in (os.path.join(HERE, "public_landmarks", "annotations"),
                 os.path.expanduser("~/oasis_validation_datasets/CIMA_ANHIR/inputs/annotations")):
        hits = glob.glob(os.path.join(root, tissue, f"user-{user}_scale-*", base + ".csv"))
        if hits:
            return hits[0]
    return None


def realized(M, mov, ref, px):
    return np.linalg.norm(sr._apply_affine(mov, M) - ref, axis=1) * px


def one_pair(tissue, fixed, moving):
    print(f"\n{'-'*74}\n{tissue}")
    lp = {(u, k): find_lm(tissue, u, b) for u in ("PS", "JB")
          for k, b in (("ref", fixed), ("mov", moving))}
    if any(v is None for v in lp.values()):
        print("  missing annotations -- skipped"); return None
    P = {k: load_xy(v) for k, v in lp.items()}
    n = min(len(v) for v in P.values())
    # work in 100pc landmark space; px at that scale
    px = px_100(tissue) * (100.0 / LANDMARK_SCALE_PC[_prefix(tissue)])
    ps_r, ps_m = P[("PS", "ref")][:n], P[("PS", "mov")][:n]
    jb_r, jb_m = P[("JB", "ref")][:n], P[("JB", "mov")][:n]
    # drop discordant (annotation-only decision)
    fr, fm = sr.fle_from_repeat(ps_r, jb_r, px), sr.fle_from_repeat(ps_m, jb_m, px)
    if fr["fle_um"] is None or fm["fle_um"] is None:
        print("  FLE undetermined -- skipped"); return None
    keep = np.array(fr["concordant"]) & np.array(fm["concordant"])
    ps_r, ps_m, jb_r, jb_m = ps_r[keep], ps_m[keep], jb_r[keep], jb_m[keep]
    fr = sr.fle_from_repeat(ps_r, jb_r, px); fm = sr.fle_from_repeat(ps_m, jb_m, px)
    fle = float(np.sqrt(np.mean([fr["fle_um"] ** 2, fm["fle_um"] ** 2])))
    diag = float(np.hypot(np.ptp(ps_r[:, 0]), np.ptp(ps_r[:, 1])))

    print(f"  n={keep.sum()} concordant   px={px:.3f} um   FLE floor {fle:.1f} um/coord   "
          f"tissue diag {diag*px/1000:.1f} mm")
    rng = np.random.default_rng(0)
    print(f"  {'region':>8}  {'radius':>8}  {'windows':>7}  {'med in-win':>10}  "
          f"{'realized p90 (local fit)':>24}")
    out = {}
    for frac in RADIUS_FRACS:
        R = frac * diag / 2.0
        p90s, counts = [], []
        centers = ps_r if frac < 1.0 else ps_r[:1]     # global: one window = everything
        idx = rng.choice(len(centers), min(N_CENTERS, len(centers)), replace=False)
        for ci in idx:
            c = centers[ci]
            d = np.linalg.norm(ps_r - c, axis=1)
            sel = d <= R if frac < 1.0 else np.ones(len(ps_r), bool)
            if sel.sum() < MIN_IN_WINDOW:
                continue
            M = sr._fit_similarity_robust(ps_m[sel], ps_r[sel])
            re = realized(M, jb_m[sel], jb_r[sel], px)   # JB held out, same window
            p90s.append(float(np.percentile(re, 90))); counts.append(int(sel.sum()))
            if frac == 1.0:
                break
        if not p90s:
            print(f"  {frac:7.0%}  {R*px/1000:6.2f}mm  too few landmarks per window")
            continue
        med_p90 = float(np.median(p90s))
        out[frac] = med_p90
        flag = "  <- below cell scale" if med_p90 <= CELL_SCALE_UM else ""
        print(f"  {frac:7.0%}  {R*px/1000:6.2f}mm  {len(p90s):7d}  {int(np.median(counts)):10d}  "
              f"{med_p90:20.1f} um{flag}")
    g = out.get(1.0, float('nan'))
    best = min((v for k, v in out.items() if k < 1.0), default=float('nan'))
    print(f"  => global {g:.0f} um  vs  best-local {best:.0f} um   "
          f"({'RESCUED' if best <= CELL_SCALE_UM else 'reduced' if best < 0.6*g else 'NOT rescued'})")
    return dict(tissue=tissue, glob=g, best_local=best, fle=fle)


def main():
    print("=" * 74)
    print("Local-rigid ROI rescue: does shrinking the region certify what global rejects?")
    print("=" * 74)
    res = [r for r in (one_pair(*p) for p in PAIRS) if r]
    print("\n" + "=" * 74)
    for r in res:
        verdict = ("RESCUED" if r["best_local"] <= CELL_SCALE_UM
                   else "reduced" if r["best_local"] < 0.6 * r["glob"] else "NOT rescued")
        print(f"  {r['tissue']:16s} global {r['glob']:6.0f} um -> best-local {r['best_local']:6.0f} um   {verdict}")
    print("\nIf mammary is RESCUED at small radius, the draw-your-own-ROI + local-rigid-fit")
    print("feature is scientifically justified. If NOT rescued, deformation is rough even")
    print("locally and no ROI can certify that tissue.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
