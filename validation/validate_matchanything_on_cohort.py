#!/usr/bin/env python
"""
Does MatchAnything actually beat LoFTR on OUR slides — and does it survive small windows?

WHY THIS EXISTS. research/ihc.md § 16.7 recommends MatchAnything (arXiv 2501.07556) on the
strength of published ANHIR cross-stain results that beat the challenge winners. § 16.10
immediately warns that a leaderboard win is grounds to TEST a matcher here, not to expect it
to work here — "Mismatched" (arXiv 2408.16445) measured SP-LG-GIM dropping 0.560 -> 0.161 mAA
out of domain. This script is that test. It reuses `validate_matchers_on_cohort.py` verbatim
for tests A and B so the numbers land in the same table as LoFTR, DISK, DeDoDe and KeyNet.

THREE TESTS, because the first two cannot answer the question that matters.

  A. SYNTHETIC WARP — exact truth on this tissue (inherited).
  B. REAL PAIR, WHOLE FRAME — coverage and a blunder proxy (inherited). This is the test
     that killed DISK+LightGlue.
  C. SMALL WINDOWS — new, and the decisive one.

WHY C IS DECISIVE. § 16.1 decoded the transform of every window that reached a certifying
verdict in the production sweeps, and found the failure is a function of WINDOW SIZE, not of
average matcher quality:

    arm   window radius   certifying   physically impossible
    4X        600 um          26            0   ( 0 %)
    10X       238 um          24           12   (50 %)
    20X       139 um          10           10   (100 %)

At 600 um the shipped matcher on these same slides is flawless; at 139 um not one window is
physically possible. So a matcher that wins on the whole 800 px frame has told us nothing
about the regime the app actually certifies in. Test C crops the same window radii the
production arms used, runs each matcher INSIDE the crop, and asks the one question the
certificate cannot ask itself: does the fitted rotation and scale agree with the whole-field
transform? Serial sections of one block imaged on one scanner differ by one placement
rotation and one scale. A window that disagrees is wrong no matter how small its residual.

WHY TWO MATCHANYTHING ARMS. Every other arm is fed hematoxylin+CLAHE (`sparse_matcher._prep`)
so that a matcher comparison is a matcher comparison and not a preprocessing one. But
MatchAnything's whole claim is modality invariance — it is trained on ~800M synthetically
cross-modalised pairs precisely so it does not need the channel separated for it. Feeding it
our preprocessing could be removing the signal it was built to use. So it runs both ways,
and the difference is itself a result.

Run:  .venv/bin/python validation/validate_matchanything_on_cohort.py            (A+B+C)
      .venv/bin/python validation/validate_matchanything_on_cohort.py --stage ab (quick)
Needs: transformers>=4.56 + torchvision. Weights zju-community/matchanything_eloftr
       (Apache-2.0, 16.0M params) download once to the HF cache.
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial import serial_registration as sr            # noqa: E402
from oasis.spatial import loftr_matcher as lm                  # noqa: E402
from validation import validate_matchers_on_cohort as C        # noqa: E402

HF_MODEL = "zju-community/matchanything_eloftr"
SIDE = 832                     # the model's native square input
CONF = 0.2                     # HF tutorial's default post-processing threshold
# The production window radii, in um, from roi_production_arm_*.json are 600 / 238 / 139.
# 600 um does not fit inside this cohort's 800x600 working frame (it would need a 666 px
# crop against a 600 px height), so 400 um stands in as the large-window control. The two
# radii that actually fail in production, 238 and 139, are the production values exactly.
WINDOW_RADII_UM = (400.0, 238.0, 139.0)
GRID = 3                       # GRID x GRID window centres per radius
# § 16.1's own criterion for "this transform is not tissue", calibrated on the 4X arm where
# correspondences are plentiful and NOTHING exceeded 3.1 deg or 2 % scale.
ROT_MAX_DEG, SCALE_MAX = 5.0, 0.05

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "matchanything_cohort_results.json")
_MA = {}


def _square(img):
    """Pad to square with white, then ISOTROPIC resize to SIDE. Returns img, scale, (ox, oy).

    Done by hand rather than left to the processor because the processor stretches to a
    832x832 square, and an anisotropic resize turns a rotation into a non-similarity —
    which is the one thing this whole harness is trying to measure.
    """
    import cv2
    h, w = img.shape[:2]
    side = max(h, w)
    can = np.full((side, side, 3), 255, np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    can[oy:oy + h, ox:ox + w] = img
    return (cv2.resize(can, (SIDE, SIDE), interpolation=cv2.INTER_AREA),
            side / float(SIDE), (ox, oy))


def _matchanything(a_rgb, b_rgb, px, preprocess):
    """Correspondences from MatchAnything-ELoFTR. `preprocess` mirrors every other arm.

    DO NOT USE `post_process_keypoint_matching`. Measured: it returns **int32 coordinates
    snapped to the 1/8 coarse grid** — 8 px in the 832 px model frame, ~7.7 px in ours —
    discarding the fine refinement that is Efficient LoFTR's headline contribution. Through
    it, a 3 deg rotation of an image against ITSELF reads as 22 px median error, 8 % RANSAC
    inliers and a recovered rotation of +5.4 deg; the same call reading the raw float
    keypoints gives 3.0 px, 46 % inliers and +2.964 deg. The first set of numbers is an
    artefact of the port and would have condemned a working matcher.

    The raw fields are `keypoints` (B, 2, N, 2) normalised to [0, 1], `matches` (B, 2, N)
    with -1 for unmatched, and `matching_scores` (B, 2, N).
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    if "m" not in _MA:
        _MA["p"] = AutoImageProcessor.from_pretrained(HF_MODEL)
        _MA["m"] = AutoModelForKeypointMatching.from_pretrained(HF_MODEL).eval()
    proc, model = _MA["p"], _MA["m"]

    if preprocess:
        a = np.repeat(C._gray(a_rgb)[:, :, None], 3, axis=2)
        b = np.repeat(C._gray(b_rgb)[:, :, None], 3, axis=2)
    else:
        a, b = a_rgb, b_rgb
    A, sa, (oax, oay) = _square(a)
    B, sb, (obx, oby) = _square(b)
    inputs = proc([[A, B]], return_tensors="pt", do_resize=False)
    with torch.no_grad():
        out = model(**inputs)
    kp = out["keypoints"][0].cpu().numpy()             # (2, N, 2), normalised
    mt = out["matches"][0].cpu().numpy()               # (2, N)
    sc = out["matching_scores"][0].cpu().numpy()       # (2, N)
    i0 = np.nonzero((mt[0] > -1) & (sc[0] >= CONF))[0]
    if len(i0) < 6:
        return None, None
    j0 = mt[0][i0]
    p = kp[0][i0] * SIDE * sa - np.array([oax, oay])
    q = kp[1][j0] * SIDE * sb - np.array([obx, oby])
    return p.astype(float), q.astype(float)


def rot_scale(M):
    """Rotation in degrees and isotropic scale of a 2x3 similarity."""
    a, b = float(M[0][0]), float(M[0][1])
    return abs(math.degrees(math.atan2(-b, a))), math.hypot(a, b)


# ── C. the small-window test ─────────────────────────────────────────────────────────
def test_windows(ref, mov, px_work, arms):
    """Crop the production window radii and ask whether each window's fit is tissue."""
    H, W = ref.shape[:2]
    out = {}
    for tag, fn in arms:
        # whole-field reference transform for this arm: the fit it makes with everything
        try:
            p, q = fn(ref, mov, px_work)
        except Exception as e:
            out[tag] = {"error": f"{type(e).__name__}: {str(e)[:70]}"}
            continue
        if p is None or len(p) < 6:
            out[tag] = {"error": "no whole-field fit"}
            continue
        Mg = sr._fit_similarity_robust(q, p)
        rg, sg = rot_scale(Mg)
        per_radius = {}
        for R_um in WINDOW_RADII_UM:
            half = int(round(R_um / px_work))
            if 2 * half >= min(H, W):
                per_radius[str(R_um)] = {"skipped": "window larger than frame"}
                continue
            rows = []
            for iy in range(GRID):
                for ix in range(GRID):
                    cx = int((ix + 1) * W / (GRID + 1))
                    cy = int((iy + 1) * H / (GRID + 1))
                    x0, x1 = max(0, cx - half), min(W, cx + half)
                    y0, y1 = max(0, cy - half), min(H, cy + half)
                    ca, cb = ref[y0:y1, x0:x1], mov[y0:y1, x0:x1]
                    try:
                        wp, wq = fn(ca, cb, px_work)
                    except Exception:
                        wp = None
                    if wp is None or len(wp) < 6:
                        rows.append({"n": 0 if wp is None else len(wp), "fit": False})
                        continue
                    Mw = sr._fit_similarity_robust(wq, wp)
                    rw, sw = rot_scale(Mw)
                    resid = np.linalg.norm(sr._apply_affine(wq, Mw) - wp, axis=1) * px_work
                    d_rot = abs(rw - rg)
                    d_sc = abs(sw / sg - 1.0) if sg > 0 else float("nan")
                    rows.append({
                        "n": int(len(wp)), "fit": True,
                        "rot_deg": round(rw, 2), "scale": round(sw, 4),
                        "d_rot_vs_global_deg": round(d_rot, 2),
                        "d_scale_vs_global": round(d_sc, 4),
                        "resid_med_um": round(float(np.median(resid)), 3),
                        "implausible": bool(d_rot > ROT_MAX_DEG or d_sc > SCALE_MAX)})
            fitted = [r for r in rows if r["fit"]]
            bad = [r for r in fitted if r["implausible"]]
            per_radius[str(R_um)] = {
                "windows": len(rows), "fitted": len(fitted), "implausible": len(bad),
                "implausible_frac": round(len(bad) / len(fitted), 3) if fitted else None,
                "n_median": int(np.median([r["n"] for r in fitted])) if fitted else 0,
                "d_rot_max_deg": round(max((r["d_rot_vs_global_deg"] for r in fitted),
                                           default=0.0), 2),
                "d_scale_max": round(max((r["d_scale_vs_global"] for r in fitted),
                                         default=0.0), 4),
                "resid_med_um_median": round(float(np.median(
                    [r["resid_med_um"] for r in fitted])), 3) if fitted else None,
                "rows": rows}
        out[tag] = {"global_rot_deg": round(rg, 2), "global_scale": round(sg, 4),
                    "global_n": int(len(p)), "per_radius": per_radius}
    return out


def main():
    from oasis.common.registration import _load_rgb_thumbnail
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="abc", choices=["ab", "c", "abc"])
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    t_start = time.time()

    arms = [("loftr", lambda a, b, px: C._loftr(a, b, px)),
            ("matchanything_hema", lambda a, b, px: _matchanything(a, b, px, True)),
            ("matchanything_rgb", lambda a, b, px: _matchanything(a, b, px, False))]
    C.ARMS = arms                                    # tests A and B iterate C.ARMS

    print("=" * 100)
    print("MatchAnything-ELoFTR against the shipped matcher, on the LL477 cohort")
    print("=" * 100)
    print(f"model {HF_MODEL} · conf {CONF} · work {C.WORK} px · "
          f"gross > {C.GROSS_UM} µm · stage {args.stage}\n")

    results = {}
    for cd8, tim3, tag in C.PAIRS:
        rp, mp = os.path.join(C.CD8, cd8), os.path.join(C.TIM3, tim3)
        if not (os.path.exists(rp) and os.path.exists(mp)):
            print(f"skip {tag}: missing file")
            continue
        ref, _ = _load_rgb_thumbnail(rp, max_side=1920)
        mov, _ = _load_rgb_thumbnail(mp, max_side=1920)
        refw, r = C._resize(ref, C.WORK)
        movw, _ = C._resize(mov, C.WORK)
        pxw = C.PX / r
        print(f"\n=== {tag}  ({cd8} ↔ {tim3})   {refw.shape[1]}x{refw.shape[0]} @ "
              f"{pxw:.2f} µm/px ===")
        rec = {}

        if args.stage in ("ab", "abc"):
            syn = C.test_synthetic(refw, pxw)
            real = C.test_real(refw, movw, pxw)
            rec["synthetic"], rec["real"] = syn, real
            print(f"  {'matcher':<21}{'A: warp n':>10}{'err med':>9}{'gross':>8}"
                  f"   |{'B: REAL n':>10}{'resid med':>11}{'max/med':>9}{'gross':>8}{'sec':>7}")
            for t, _ in arms:
                s, b = syn.get(t, {}), real.get(t, {})
                if s.get("error") or b.get("error"):
                    print(f"  {t:<21}  {s.get('error') or b.get('error')}")
                    continue
                print(f"  {t:<21}{s.get('n', 0):>10}{s.get('err_med_um', '--'):>9}"
                      f"{s.get('gross_frac', '--'):>8}   |{b.get('n', 0):>10}"
                      f"{b.get('resid_med_um', '--'):>11}{b.get('max_over_med', '--'):>9}"
                      f"{b.get('gross_frac', '--'):>8}{b.get('seconds', '--'):>7}")

        if args.stage in ("c", "abc"):
            win = test_windows(refw, movw, pxw, arms)
            rec["windows"] = win
            print(f"\n  C: small windows — does the window's fit agree with the field?")
            print(f"  {'matcher':<21}{'radius µm':>10}{'fitted':>8}{'n med':>7}"
                  f"{'implausible':>13}{'Δrot max':>10}{'Δscale max':>12}{'resid med':>11}")
            for t, _ in arms:
                w = win.get(t, {})
                if w.get("error"):
                    print(f"  {t:<21}  {w['error']}")
                    continue
                for R in WINDOW_RADII_UM:
                    d = w["per_radius"].get(str(R), {})
                    if d.get("skipped"):
                        print(f"  {t:<21}{R:>10.0f}   {d['skipped']}")
                        continue
                    fr = d.get("implausible_frac")
                    bad = "{}/{}".format(d["implausible"], d["fitted"])
                    pct = "--" if fr is None else "{:.0f}%".format(100 * fr)
                    res = d["resid_med_um_median"]
                    res = "--" if res is None else "{:.2f}".format(res)
                    print(f"  {t:<21}{R:>10.0f}{d['fitted']:>4}/{d['windows']:<3}"
                          f"{d['n_median']:>7}{bad:>8}{pct:>5}"
                          f"{d['d_rot_max_deg']:>10.1f}{d['d_scale_max']:>12.3f}{res:>11}")
        results[tag] = rec
        lm.clear_loftr_caches()

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    summary = {}
    for t, _ in arms:
        ns = [results[k].get("real", {}).get(t, {}).get("n", 0) for k in results
              if "real" in results[k]]
        gs = [results[k]["real"][t].get("gross_frac") for k in results
              if "real" in results[k] and t in results[k]["real"]]
        gs = [g for g in gs if g is not None]
        imp = {}
        for R in WINDOW_RADII_UM:
            f, b = 0, 0
            for k in results:
                d = results[k].get("windows", {}).get(t, {}).get("per_radius", {}).get(str(R))
                if d and not d.get("skipped"):
                    f += d["fitted"]; b += d["implausible"]
            imp[str(R)] = {"fitted": f, "implausible": b,
                           "frac": round(b / f, 3) if f else None}
        summary[t] = {"real_n_per_pair": ns, "min_n": int(min(ns)) if ns else 0,
                      "gross_frac_max": max(gs) if gs else None, "window_implausible": imp}
        line = (f"  {t:<21} whole-frame matches {str(ns):<20}"
                + (f" worst gross {100 * max(gs):5.1f}%" if gs else ""))
        for R in WINDOW_RADII_UM:
            d = imp[str(R)]
            if d["frac"] is not None:
                line += f" | {R:.0f}µm {100 * d['frac']:3.0f}% bad"
        print(line)

    with open(args.out, "w") as f:
        json.dump({"config": {"model": HF_MODEL, "conf": CONF, "work_px": C.WORK,
                              "gross_um": C.GROSS_UM, "radii_um": list(WINDOW_RADII_UM),
                              "grid": GRID, "rot_max_deg": ROT_MAX_DEG,
                              "scale_max": SCALE_MAX},
                   "pairs": results, "summary": summary}, f, indent=2)
    print(f"\nWrote {args.out}   ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
