#!/usr/bin/env python
"""
validate_roi_coverage.py — how much of each field can actually be certified?

The question this answers: for every CD8/TIM-3 pair that carries enough cells, how large a
window does the production certifier accept, and does that window still hold 50 positives of
both markers? Coverage is the whole ballgame — a pair with 473 CD8 and 154 TIM-3 needs 32 %
of the field to survive the count floor, and one with 103/55 needs 91 %.

Nothing here is operator-driven. Correspondences come from LoFTR (`loftr_correspondences`),
their localisation error from `loftr_fle`, and the verdict from `landmark_register_and_verify`
— the same certifier the UI calls, with the same gates. The certifier finds its own
LOCALLY_CERTIFIED hull, so no ROI has to be drawn: the hull IS the measurement.

`correspondences_for_certification` is deliberately NOT used. Its docstring records a measured
negative result on exactly this stain pair: lumen centroids cannot be matched across CD8 and
TIM-3 by appearance.

Arm `rigid` is today's behaviour. Arm `nonrigid` is § 23.7 Route 1: warp the moving image
first, then certify a similarity on correspondences taken from the warped pair. The warp is
never certified — only the similarity that survives it is, which is what keeps § 3.5's budget
intact and what separates Route 1 from the Route 3 that failed in § 23.10.

Run:  python validation/validate_roi_coverage.py --arm rigid
      python validation/validate_roi_coverage.py --arm rigid --limit 3     (smoke test)
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cohort paths are this machine's, so they are named by environment variable and the run
# fails with a clear message rather than hunting for someone else's Desktop.
MANIFEST = os.path.expanduser(os.environ.get("OASIS_PAIR_MANIFEST", "~/oasis_pair_manifest.json"))
IMG_ROOT = os.path.expanduser(os.environ.get("OASIS_10X_ANALYSED", "~/Desktop/10x analyzed"))
MIN_POSITIVE = 50
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "roi_coverage_results.json")


def _hull_frac(poly, wh):
    if not poly:
        return None
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))   # shoelace
    return float(area / (wh[0] * wh[1]))


def find_image(stem):
    """Images live one folder deep under the cohort dir, named after the folder."""
    for marker in ("cd8", "tim3"):
        p = os.path.join(IMG_ROOT, marker, stem, stem + ".tif")
        if os.path.exists(p):
            return p
    return None


def certify_one(ref_path, mov_path, pixel_size_um, arm):
    from oasis.spatial import loftr_matcher as lm
    from oasis.spatial.serial_registration import (landmark_register_and_verify,
                                                   _load_rgb_thumbnail)
    ref_rgb, ref_s = _load_rgb_thumbnail(ref_path, max_side=1920)
    mov_rgb, mov_s = _load_rgb_thumbnail(mov_path, max_side=1920)
    if ref_rgb is None or mov_rgb is None:
        return {"error": "image load failed"}
    px_t = pixel_size_um / max(ref_s, 1e-9)
    H, W = ref_rgb.shape[:2]

    if arm == "nonrigid":
        mov_rgb = _valis_warp(ref_path, mov_path, mov_rgb)
        if mov_rgb is None:
            return {"error": "non-rigid warp unavailable"}

    t0 = time.time()
    c = lm.loftr_correspondences(ref_rgb, mov_rgb, px_t)
    n = int(c.get("n") or 0)
    if n < 6:
        return {"verdict": "NOT_CERTIFIABLE", "n_corr": n, "coverage": 0.0,
                "reason": f"only {n} LoFTR correspondences", "secs": round(time.time() - t0, 1)}
    ref_pts = np.asarray(c["ref_points"], float)
    mov_pts = np.asarray(c["mov_points"], float)
    fle = lm.loftr_fle(ref_rgb, mov_rgb, ref_pts, mov_pts, px_t)
    fle_um = float(fle.get("fle_um") or 0.0) or None

    out = landmark_register_and_verify(ref_pts, mov_pts, px_t, image_wh=(W, H),
                                       fle_um=fle_um,
                                       landmarks_are_model_selected=True)
    verdict = out.get("verdict")
    # CERTIFIED means the whole field holds; a local hull covers only its own area.
    cov = 1.0 if verdict == "CERTIFIED" else (_hull_frac(out.get("roi_polygon"), (W, H)) or 0.0)
    if verdict in ("DEFORMED", "NOT_CERTIFIABLE"):
        cov = 0.0
    return {"verdict": verdict, "n_corr": n, "fle_um": fle_um, "coverage": round(cov, 4),
            "cell_error_p90_um": out.get("cell_error_p90_um"),
            "min_interpretable_radius_um": out.get("min_interpretable_radius_um"),
            "reason": out.get("reason"), "secs": round(time.time() - t0, 1)}


def _valis_warp(ref_path, mov_path, mov_rgb):
    """Route 1's alignment step. Returns None when the VALIS runtime is not available."""
    raise NotImplementedError(
        "non-rigid arm needs the VALIS runtime; run the rigid arm first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("rigid", "nonrigid"), default="rigid")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N pairs")
    a = ap.parse_args()

    if not os.path.exists(MANIFEST):
        print(f"No pair manifest at {MANIFEST}.\n"
              f"Set OASIS_PAIR_MANIFEST to a JSON file with an 'all_pairs' list "
              f"(file_a, file_b, px, pos_a, pos_b, base, passes).")
        return 2
    man = json.load(open(MANIFEST, encoding="utf-8"))
    pairs = [r for r in man["all_pairs"] if r["passes"]]
    pairs.sort(key=lambda r: -min(r["pos_a"], r["pos_b"]))
    if a.limit:
        pairs = pairs[:a.limit]

    print("=" * 100)
    print(f"Certified coverage, arm = {a.arm}")
    print("=" * 100)
    print(f"{len(pairs)} pairs that clear {MIN_POSITIVE} positives on the full field.")
    print("'needed' is the coverage at which the ROI still holds 50 of both markers.\n")
    print(f"{'pair':<24}{'CD8+':>6}{'TIM3+':>7}{'needed':>8}{'verdict':>19}"
          f"{'coverage':>10}{'usable':>8}")
    print("-" * 100)

    rows, t0 = [], time.time()
    for r in pairs:
        stem_a = r["file_a"][:-4]
        stem_b = r["file_b"][:-4]
        pa, pb = find_image(stem_a), find_image(stem_b)
        if not pa or not pb:
            print(f"{r['base']:<24}  image not found"); continue
        res = certify_one(pa, pb, r["px"], a.arm)
        need = MIN_POSITIVE / min(r["pos_a"], r["pos_b"])
        cov = res.get("coverage") or 0.0
        usable = bool(r["pos_a"] * cov > MIN_POSITIVE and r["pos_b"] * cov > MIN_POSITIVE)
        rows.append({**r, **res, "needed": round(need, 3), "usable": usable})
        print(f"{r['base']:<24}{r['pos_a']:>6}{r['pos_b']:>7}{need*100:>7.0f}%"
              f"{str(res.get('verdict') or res.get('error'))[:18]:>19}"
              f"{cov*100:>9.1f}%{('YES' if usable else '-'):>8}", flush=True)

    print("-" * 100)
    if rows:
        cov = [x["coverage"] or 0.0 for x in rows]
        print(f"median certified coverage {np.median(cov)*100:.1f}%   "
              f"max {max(cov)*100:.1f}%   "
              f"usable pairs {sum(x['usable'] for x in rows)} of {len(rows)}")
        from collections import Counter
        print("verdicts: " + ", ".join(f"{k} {v}" for k, v in
                                       Counter(x.get("verdict") for x in rows).items()))
    prev = {}
    if os.path.exists(OUT_JSON):
        prev = json.load(open(OUT_JSON, encoding="utf-8"))
    prev[a.arm] = rows
    json.dump(prev, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print(f"\nWrote {OUT_JSON}   ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
