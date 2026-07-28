#!/usr/bin/env python3
"""Is a certifying ROI a NEIGHBOURHOOD, or a single lucky window?

A local certification says: inside this ROI, the fitted transform is accurate to the stated
error. Nothing in that statement is checkable from the ROI itself -- the fit and the residual
come from the same correspondences. What IS checkable is whether the claim survives moving
the window. Serial-section deformation is smooth, so a genuine local alignment must certify
across a patch of overlapping windows, and those windows' independently-fitted transforms
must place the same point in nearly the same place. A fit to noise cannot do either.

Two prior sweeps could not answer this, both by construction:

  * the coarse tiling (`batch.py`) SKIPS any candidate within 1.8R of a region it already
    kept, so its returned regions are disjoint by definition;
  * the exhaustive search (`exhaustive.py`) BREAKS on the first window that certifies.

So both reported "one region" for reasons that have nothing to do with the tissue. This
harness removes both: it sweeps the full grid at one radius with 50%-overlapping windows and
no early exit, and reports

    n_certifying        how many windows certify
    n_with_a_neighbour  how many of those have a certifying window adjacent to them
    disagree_um         spread between the certifying windows' transforms, evaluated at a
                        common set of probe points -- the max over probes for each pair of
                        windows, then the median and max over pairs

Read the last two together. Contiguity without agreement means the windows certify next to
each other while describing different alignments, which is worse than not certifying.

Usage:
  .venv/bin/python validation/roi_certification_neighbourhood.py \\
      --root ~/Desktop/"Region of interest" \\
      --pairs LL477_Liver_4X_3,LL479_Liver_10X_2 --radius-um 600 --arm identity

  --arm register_similarity uses that provisional transform to locate the moving patch, the
  same choice the exhaustive search recorded per pair. It does not change the final fit,
  which is always recomputed locally.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                # noqa: E402
from oasis.common.registration import _load_rgb_thumbnail  # noqa: E402
from oasis.spatial import serial_registration as sr        # noqa: E402
from oasis.spatial import loftr_matcher as lm              # noqa: E402

# µm/px at full resolution for this cohort's objectives, measured from its own scale bars.
PX_AT_OBJECTIVE = {"4X": 1.8798, "10X": 0.7519, "20X": 0.37595}
PASSING = ("CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED")


def circle(cx, cy, r, k=40):
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]


def apply_similarity(M, pts):
    """mov->ref similarity in the codebase convention (2x3 or 3x3)."""
    M = np.asarray(M, float)
    return np.asarray(pts, float) @ M[:2, :2].T + M[:2, 2]


def objective_of(name, default="10X"):
    for k in ("20X", "10X", "4X"):
        if k in name:
            return k
    return default


def index_pairs(root):
    """{folder basename: path} for every folder holding TIFFs."""
    out = {}
    for base, _dirs, files in os.walk(root):
        if any(f.lower().endswith(".tif") for f in files):
            out[os.path.basename(base)] = base
    return out


def sweep_pair(src, objective, radius_um, arm, ref_marker="CD8", mov_marker="Tim3"):
    """Full-grid sweep at one radius. Returns the record described in the module docstring."""
    files = sorted(glob.glob(os.path.join(src, "*.tif")))
    a = [f for f in files if ref_marker in os.path.basename(f)]
    b = [f for f in files if mov_marker in os.path.basename(f)]
    if not a or not b:
        return None
    ref, ref_scale = _load_rgb_thumbnail(a[0], max_side=1920)
    mov, _ = _load_rgb_thumbnail(b[0], max_side=1920)
    px = PX_AT_OBJECTIVE[objective] / max(ref_scale, 1e-9)
    H, W = ref.shape[:2]

    # Sweep the tissue, not the slide background: an ROI centred on empty glass fails for a
    # reason that says nothing about registration and would dilute every ratio here.
    grey = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    tissue = cv2.morphologyEx((grey < 235).astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((15, 15), np.uint8))
    ys, xs = np.where(tissue > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    if arm == "register_similarity":
        M = np.asarray(sr.register_similarity(ref, mov, px)["matrix"], float)
    else:
        M = np.array([[1., 0, 0], [0, 1., 0]])

    R = radius_um / px
    step = R / 2.0                      # 50% overlap: adjacent windows share most of their tissue
    cys = np.arange(max(y0 + R, R), min(y1 - R, H - R) + 1e-6, step)
    cxs = np.arange(max(x0 + R, R), min(x1 - R, W - R) + 1e-6, step)

    tally, hits = Counter(), []
    for cy in cys:
        for cx in cxs:
            c = lm.certify_local_roi(ref, mov, circle(cx, cy, R), px,
                                     provisional_matrix=M, fle_fast=True,
                                     work_max_dim=800, loftr_kw={"local_k": 8})
            tally[c.get("verdict")] += 1
            if c.get("verdict") in PASSING:
                hits.append({"cx": float(cx), "cy": float(cy), "verdict": c.get("verdict"),
                             "n_correspondences": c.get("n_correspondences"),
                             "deformation_rms_um": c.get("deformation_rms_um"),
                             "cell_error_p90_um": c.get("cell_error_p90_um"),
                             "matrix": c.get("matrix")})
    lm.clear_loftr_caches()

    rec = {"objective": objective, "radius_um": float(radius_um), "arm": arm,
           "grid": [len(cxs), len(cys)], "n_windows": int(sum(tally.values())),
           "n_certifying": len(hits), "tally": dict(tally), "windows": hits}

    if len(hits) >= 2:
        P = np.array([[h["cx"], h["cy"]] for h in hits])
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        # 8-neighbourhood on a step-spaced grid: the diagonal is step*sqrt(2) ~ 1.41*step.
        rec["n_with_a_neighbour"] = int(((d <= step * 1.45).sum(1) > 0).sum())
        rec["nearest_neighbour_um"] = round(float(d.min() * px), 1)
        mapped = [apply_similarity(h["matrix"], P) for h in hits]
        pairwise = [float((np.linalg.norm(mapped[i] - mapped[j], axis=1) * px).max())
                    for i in range(len(hits)) for j in range(i + 1, len(hits))]
        rec["disagree_um_median"] = round(float(np.median(pairwise)), 2)
        rec["disagree_um_max"] = round(float(max(pairwise)), 2)
    else:
        # One window certifying cannot corroborate itself; that is the finding, not a gap.
        rec["n_with_a_neighbour"] = 0
        rec["nearest_neighbour_um"] = None
        rec["disagree_um_median"] = rec["disagree_um_max"] = None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/Desktop/Region of interest")
    ap.add_argument("--pairs", required=True, help="comma-separated folder names")
    ap.add_argument("--radius-um", type=float, required=True)
    ap.add_argument("--arm", default="identity",
                    choices=("identity", "register_similarity"))
    ap.add_argument("--ref-marker", default="CD8")
    ap.add_argument("--mov-marker", default="Tim3")
    ap.add_argument("--out", default="validation/roi_certification_neighbourhood_results.json")
    args = ap.parse_args()

    index = index_pairs(os.path.expanduser(args.root))
    out, t0 = [], time.time()
    print(f"{'pair':<24}{'obj':>4}{'R µm':>7}{'certify':>10}{'nbr':>5}{'dis med':>9}{'dis max':>9}")
    for pair in [p for p in args.pairs.split(",") if p]:
        src = index.get(pair)
        if not src:
            print(f"  ! {pair}: not found under {args.root}")
            continue
        rec = sweep_pair(src, objective_of(pair), args.radius_um, args.arm,
                         args.ref_marker, args.mov_marker)
        if rec is None:
            print(f"  ! {pair}: no {args.ref_marker}/{args.mov_marker} pair in the folder")
            continue
        rec["pair"] = pair
        out.append(rec)
        med = rec["disagree_um_median"]
        mx = rec["disagree_um_max"]
        print(f"{pair:<24}{rec['objective']:>4}{rec['radius_um']:>7.0f}"
              f"{str(rec['n_certifying']) + '/' + str(rec['n_windows']):>10}"
              f"{rec['n_with_a_neighbour']:>5}"
              f"{('-' if med is None else format(med, '.1f')):>9}"
              f"{('-' if mx is None else format(mx, '.1f')):>9}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))

    tc = sum(r["n_certifying"] for r in out)
    tw = sum(r["n_windows"] for r in out)
    tn = sum(r["n_with_a_neighbour"] for r in out)
    print(f"\n  {tc}/{tw} windows certify; {tn} of those {tc} have a certifying neighbour "
          f"({time.time() - t0:.0f}s)")
    print(f"  Written: {args.out}")


if __name__ == "__main__":
    main()
