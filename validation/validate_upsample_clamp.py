#!/usr/bin/env python3
"""
validate_upsample_clamp.py — does feeding coarse images at native size cost real cells?

THE CLAIM UNDER TEST. `segment.preferred_downsample` clamped at 1.0 and never enlarged an
image. On a 10x slide at 0.7519 µm/px the unclamped factor is 0.665, so nuclei reach InstanSeg
at two-thirds the size it was trained on. The clamp's original rationale was an LL477
observation: enlarging yields 6719 nuclei against QuPath's 4796, versus 5380 clamped, so the
extra objects were read as the model splitting enlarged nuclei.

WHY THAT RATIONALE COULD NOT SETTLE IT. QuPath applies the same clamp, so agreeing with it
cannot validate it. And the validated corpus cannot either: DeepLIIF is 0.25 µm/px, FINER than
the model, so it always downsamples and never exercises the clamp. The coarse regime was
never scored against ground truth.

THE TEST. Resample DeepLIIF panels to a coarse pixel size to simulate a 10x acquisition, then
segment the coarse image several ways and score every mode against the SAME IF-derived ground
truth, in the original full-resolution coordinate space:

  native        the untouched panel                  -- upper bound: what the fine image gets
  clamped       coarse panel, ds forced to 1.0       -- the historical behaviour
  upsampled     coarse panel, ds = 0.665             -- enlarged to the model's 0.5 µm/px
  wrong_px      coarse panel mislabelled 1.0 µm/px   -- isolates a pixel-size error from the
                                                        clamp, since both give ds 1.0

Matching tolerance is fixed in MICRONS so no mode is advantaged by the space it ran in.

CAVEAT, stated up front: resampling a 40x panel approximates a 10x acquisition, it does not
reproduce 10x optics. This measures the resolution effect, not a real 10x scan.

Run:  .venv/bin/python validation/validate_upsample_clamp.py
      .venv/bin/python validation/validate_upsample_clamp.py --n 100 --coarse-px 0.7519
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.common.paths import default_model_dir
from oasis.quant import segment as sg

DEFAULT_SRC = os.path.expanduser(
    "~/oasis_validation_datasets/DeepLIIF/inputs/DeepLIIF_Testing_Set")
PANEL = 512
NATIVE_PX = 0.25          # DeepLIIF is 40x
TOL_UM = 3.75             # 15 px at 0.25 µm/px, the tolerance the DeepLIIF harness uses


def gt_cells(mask_rgb):
    """DeepLIIF SegMask -> cell centroids. red = positive, blue = negative."""
    r, g, b = (mask_rgb[..., i].astype(int) for i in range(3))
    out = []
    for m in [(r > 90) & (r > b + 25) & (r >= g), (b > 90) & (b > r + 25) & (b >= g)]:
        L, n = ndimage.label(m)
        for i in range(1, n + 1):
            comp = L == i
            if comp.sum() >= 20:
                ys, xs = np.where(comp)
                out.append((float(xs.mean()), float(ys.mean())))
    return out


def match(gt_xy, pred_xy, tol_px):
    """Greedy one-to-one nearest-neighbour matching; returns the number of GT cells matched."""
    if not len(gt_xy) or not len(pred_xy):
        return 0
    tree = cKDTree(pred_xy)
    d, idx = tree.query(gt_xy, k=min(8, len(pred_xy)))
    d, idx = np.atleast_2d(d.T).T, np.atleast_2d(idx.T).T
    used, n = set(), 0
    for gi in np.argsort(d[:, 0]):
        for dd, pi in zip(d[gi], idx[gi]):
            if dd <= tol_px and pi not in used:
                used.add(int(pi))
                n += 1
                break
    return n


def load_panels(src, n):
    files = sorted(glob.glob(os.path.join(src, "*.png")))
    if not files:
        raise SystemExit(f"no panels under {src}")
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]
    panels = []
    for f in files:
        im = np.asarray(Image.open(f).convert("RGB"))
        npan = im.shape[1] // PANEL
        g = gt_cells(im[:, (npan - 1) * PANEL:npan * PANEL])
        if len(g) >= 5:
            panels.append({"ihc": im[:, 0:PANEL], "gt": np.array(g)})
    return panels


def run_mode(panels, model_dir, px, side, clamp, device="cpu"):
    """Segment every panel under one mode and score it against the ground truth."""
    tol_px = TOL_UM / NATIVE_PX          # scoring always happens in original 512-space
    scale_back = 1.0 if side is None else PANEL / side
    G = P = M = 0
    for p in panels:
        rgb = p["ihc"] if side is None else \
            cv2.resize(p["ihc"], (side, side), interpolation=cv2.INTER_AREA)
        res = sg.segment_image("", model_dir, px, device=device, rgb=rgb,
                               allow_upsample=not clamp)
        pred = (np.array([r["centroid_px"] for r in res["records"]]) * scale_back
                if res["records"] else np.empty((0, 2)))
        G += len(p["gt"])
        P += len(pred)
        M += match(p["gt"], pred, tol_px)
    rec = M / G if G else 0.0
    prec = M / P if P else 0.0
    return {"recall": rec, "precision": prec,
            "f1": (2 * rec * prec / (rec + prec)) if (rec + prec) else 0.0,
            "gt": G, "pred": P, "matched": M,
            "ds": sg.preferred_downsample(px, 0.5, allow_upsample=not clamp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--coarse-px", type=float, default=0.7519,
                    help="TRUE pixel size to simulate (default: the LL477 10x calibration)")
    ap.add_argument("--wrong-px", type=float, default=1.00,
                    help="the pixel size the filename map would assign instead, so the "
                         "mislabelling can be scored separately from the clamp "
                         "(10x: 1.00 vs 0.7519; 20x: 0.50 vs 0.376)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    model_dir = default_model_dir()
    panels = load_panels(a.src, a.n)
    side = int(round(PANEL * NATIVE_PX / a.coarse_px))
    print(f"panels {len(panels)}   GT cells {sum(len(p['gt']) for p in panels)}")
    print(f"coarse simulation: {PANEL}px @{NATIVE_PX} -> {side}px @{a.coarse_px} µm/px\n")

    modes = [
        ("native",    None, NATIVE_PX,   True),
        ("clamped",   side, a.coarse_px, True),
        ("upsampled", side, a.coarse_px, False),
        ("wrong_px",  side, a.wrong_px,  True),
    ]
    out = {}
    for name, sd, px, clamp in modes:
        r = run_mode(panels, model_dir, px, sd, clamp, a.device)
        out[name] = r
        print(f"{name:<11} ds={r['ds']:.3f}  recall {r['recall']:.3f}  "
              f"precision {r['precision']:.3f}  F1 {r['f1']:.3f}  "
              f"detections {r['pred']} (of {r['gt']} GT)")

    c, u, wp = out["clamped"], out["upsampled"], out["wrong_px"]
    dP, dM = u["pred"] - c["pred"], u["matched"] - c["matched"]
    if dP > 0:
        print(f"\nunclamping: +{dP} detections containing +{dM} real cells "
              f"({dM / dP * 100:.0f}% useful), recovering "
              f"{dM / (c['gt'] - c['matched']) * 100:.0f}% of what the clamped run missed")
        ok = u["recall"] > c["recall"] and u["f1"] > c["f1"]
        print(f"VERDICT: {'clamp costs real recall' if ok else 'clamp is not costing recall'}")
    else:
        # This is the regime where the image is FINER than the model (20x and up): the clamp
        # never engages, so allow_upsample is a no-op and the only question is the pixel size.
        print(f"\nclamp does not engage at {a.coarse_px} um/px (ds {c['ds']:.3f} >= 1) — "
              f"allow_upsample is a NO-OP here, identical results either way")
        ok = True

    # The pixel-size arm is meaningful whenever it lands on a different downsample.
    if abs(wp["ds"] - c["ds"]) > 1e-6:
        dP2, dM2 = wp["pred"] - c["pred"], wp["matched"] - c["matched"]
        useful = (dM2 / dP2 * 100) if dP2 else float("nan")
        print(f"\nmislabelling {a.coarse_px} um/px as {a.wrong_px} changes the downsample "
              f"{c['ds']:.3f} -> {wp['ds']:.3f}:")
        print(f"  recall {c['recall']:.3f} -> {wp['recall']:.3f}   "
              f"precision {c['precision']:.3f} -> {wp['precision']:.3f}   "
              f"F1 {c['f1']:.3f} -> {wp['f1']:.3f}")
        print(f"  {dP2:+} detections carrying {dM2:+} real cells "
              f"({useful:.0f}% useful)")
        print(f"  VERDICT: the mislabelled pixel size is "
              f"{'WORSE' if wp['f1'] < c['f1'] else 'not worse'} than the calibrated one")
    else:
        print("\nwrong_px is identical to the calibrated run: a pixel-size error of this size "
              "does not change segmentation, only the micron-denominated measurements.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
