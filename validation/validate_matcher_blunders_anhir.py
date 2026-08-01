#!/usr/bin/env python
"""
Is LoFTR the right matcher — measured by BLUNDER RATE, not by localisation accuracy.

WHY THIS EXISTS, AND WHY THE EARLIER ANSWER WAS WRONG. Phase A
(validate_loftr_fle_groundtruth.py) measured LoFTR's localisation error against a known warp
at 0.22 um and reported ZERO gross errors across thirteen warps. That was used to argue no
better matcher could help. The argument does not hold: Phase A matches a section against a
RESAMPLED COPY OF ITSELF, where a blunder is close to impossible by construction. It measures
the localisation floor and is silent about the failure that actually hurts.

On real serial pairs the blunders are there. Residuals from LL477 regions, after the shipped
filters:

    n=39   median 6.55 um   p90 55.62   max 57.86   -> max/median 8.8
    n=8    median 5.16 um   p90 15.80   max 34.55   -> max/median 6.7

For isotropic noise max/median at n~40 is about 2.5. Roughly a tenth of the surviving
correspondences are grossly wrong, and because the gate reads a p90, those few set the
reported cell error — 58 um on a region whose typical correspondence is good to 6.5 um.
`_fit_similarity_robust` cannot save it (Huber down-weights, never rejects) and the
local-smoothness filter demonstrably misses them.

So the question is not "which matcher localises best" — they are all sub-pixel — it is
"which matcher produces the fewest confidently-wrong matches on cross-stain serial sections".

THE ARMS. Every matcher is handed the SAME input OASIS would give it: the hematoxylin channel,
CLAHE-equalised. This compares matchers, not preprocessing.

    loftr_raw        LoFTR, no OASIS filters      — the matcher alone
    loftr_oasis      LoFTR + cycle/scale/local    — what ships
    disk_lightglue   DISK detector + LightGlue    — learned sparse, SOTA-competitive
    sift_lightglue   SIFT + LightGlue             — classical detector, learned matcher

The module docstring for loftr_matcher justifies detector-free over "SIFT with mutual-NN +
Lowe", which returned zero matches. That is a fair verdict on 1999 nearest-neighbour matching
and NOT on a modern learned matcher fed the same detector, which is why sift_lightglue is
here.

THE MEASUREMENTS, per pair per arm:
  n              correspondences returned (coverage — a matcher that finds nothing is useless)
  blunder_frac   fraction whose residual to a robust similarity exceeds 5x the median
  max_over_med   the ratio that exposed the problem on LL477
  tre_med/p90    realized error at EXPERT landmarks the matcher never saw — the only
                 non-circular accuracy measure, since every arm's own residual is measured on
                 points it selected itself

Run:  .venv/bin/python validation/validate_matcher_blunders_anhir.py [--limit N]
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import serial_registration as sr          # noqa: E402
from oasis.spatial import loftr_matcher as lm                # noqa: E402

ROOT = os.path.expanduser("~/oasis_validation_datasets/ANHIR_medium/images")
WORK_MAX = 1024
BLUNDER_MULT = 5.0        # residual > 5x the median is not noise at any plausible tail
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "matcher_blunders_anhir_results.json")


def load_landmarks(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    hdr, out = rows[0], []
    xi = hdr.index("X") if "X" in hdr else 1
    yi = hdr.index("Y") if "Y" in hdr else 2
    for r in rows[1:]:
        try:
            out.append((float(r[xi]), float(r[yi])))
        except (ValueError, IndexError):
            continue
    return np.asarray(out, float)


def load_image(path):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = min(1.0, WORK_MAX / float(max(W, H)))
    if s < 1.0:
        im = im.resize((max(int(W * s), 8), max(int(H * s), 8)), Image.LANCZOS)
    return np.asarray(im), s


def prep_gray(rgb):
    """Exactly what loftr_matcher._prep does before the transformer sees an image, so every
    arm is compared on the same input rather than on a preprocessing difference."""
    import cv2
    from oasis.common.registration import extract_hematoxylin
    h = extract_hematoxylin(rgb).astype(np.float32)
    h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(h)


# ── arms ─────────────────────────────────────────────────────────────────────────────
def _loftr(ref_rgb, mov_rgb, filtered):
    kw = {} if filtered else dict(scales=(0.75,), local_k=0, scale_tol_stride=0.0)
    c = lm.loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um=1.0, **kw)
    if not c["ok"]:
        return None, None
    return np.asarray(c["ref_points"], float), np.asarray(c["mov_points"], float)


def _sparse_lightglue(ref_gray, mov_gray, detector):
    """DISK or SIFT keypoints matched by LightGlue. Returns (ref_pts, mov_pts)."""
    import torch
    import kornia.feature as KF
    ta = torch.from_numpy(ref_gray).float()[None, None] / 255.0
    tb = torch.from_numpy(mov_gray).float()[None, None] / 255.0
    with torch.inference_mode():
        if detector == "disk":
            d = KF.DISK.from_pretrained("depth").eval()
            fa = d(ta.repeat(1, 3, 1, 1), 2048, pad_if_not_divisible=True)[0]
            fb = d(tb.repeat(1, 3, 1, 1), 2048, pad_if_not_divisible=True)[0]
            ka, da_ = fa.keypoints, fa.descriptors
            kb, db_ = fb.keypoints, fb.descriptors
            lafa = KF.laf_from_center_scale_ori(ka[None])
            lafb = KF.laf_from_center_scale_ori(kb[None])
        else:                                     # sift
            d = KF.SIFTFeature(2048).eval()
            lafa, _, da_ = d(ta)
            lafb, _, db_ = d(tb)
            ka = KF.get_laf_center(lafa)[0]
            kb = KF.get_laf_center(lafb)[0]
            da_, db_ = da_[0], db_[0]
        matcher = KF.LightGlueMatcher(detector).eval()
        hw1 = torch.tensor(ref_gray.shape)
        hw2 = torch.tensor(mov_gray.shape)
        _dists, idxs = matcher(da_, db_, lafa, lafb, hw1=hw1, hw2=hw2)
    if idxs is None or len(idxs) < 6:
        return None, None
    i = idxs.cpu().numpy()
    return ka.cpu().numpy()[i[:, 0]], kb.cpu().numpy()[i[:, 1]]


ARMS = [
    ("loftr_oasis",    lambda rr, mr, rg, mg: _loftr(rr, mr, True)),
    ("loftr_raw",      lambda rr, mr, rg, mg: _loftr(rr, mr, False)),
    ("disk_lightglue", lambda rr, mr, rg, mg: _sparse_lightglue(rg, mg, "disk")),
    ("sift_lightglue", lambda rr, mr, rg, mg: _sparse_lightglue(rg, mg, "sift")),
]


def score(ref_pts, mov_pts, src_lm, dst_lm):
    """Blunder contamination from the arm's own residuals, accuracy from EXPERT landmarks."""
    if ref_pts is None or len(ref_pts) < 6:
        return None
    M = sr._fit_similarity_robust(mov_pts, ref_pts)       # mov -> ref
    resid = np.linalg.norm(sr._apply_affine(mov_pts, M) - ref_pts, axis=1)
    med = float(np.median(resid))
    tre = np.linalg.norm(sr._apply_affine(src_lm, M) - dst_lm, axis=1)
    return {"n": int(len(ref_pts)),
            "resid_med": round(med, 3),
            "max_over_med": round(float(resid.max() / max(med, 1e-9)), 2),
            "blunder_frac": round(float((resid > BLUNDER_MULT * med).mean()), 4),
            "tre_med": round(float(np.median(tre)), 2),
            "tre_p90": round(float(np.percentile(tre, 90)), 2)}


def run(limit, seed=0):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "dataset_medium.csv")))
            if r["status"] == "training"]
    np.random.default_rng(seed).shuffle(rows)
    out = []
    print(f"{'pair':<34}" + "".join(f"{t.split('_')[0][:5] + '·' + k:>12}"
                                    for t, _ in ARMS for k in ("n", "blnd")))
    for r in rows[:limit]:
        sp, tp = os.path.join(ROOT, r["Source image"]), os.path.join(ROOT, r["Target image"])
        slp, tlp = os.path.join(ROOT, r["Source landmarks"]), os.path.join(ROOT, r["Target landmarks"])
        if not all(os.path.exists(p) for p in (sp, tp, slp, tlp)):
            continue
        try:
            mov_rgb, s_mov = load_image(sp)
            ref_rgb, s_ref = load_image(tp)
            src_lm = load_landmarks(slp) * s_mov
            dst_lm = load_landmarks(tlp) * s_ref
        except Exception as e:
            print(f"{r['Source image'][:32]:<34} LOAD ERROR {e}")
            continue
        n_lm = min(len(src_lm), len(dst_lm))
        if n_lm < 8:
            continue
        src_lm, dst_lm = src_lm[:n_lm], dst_lm[:n_lm]
        ref_g, mov_g = prep_gray(ref_rgb), prep_gray(mov_rgb)

        res, cells = {}, ""
        for tag, fn in ARMS:
            try:
                t0 = time.time()
                rp, mp = fn(ref_rgb, mov_rgb, ref_g, mov_g)
                s = score(rp, mp, src_lm, dst_lm)
                if s:
                    s["seconds"] = round(time.time() - t0, 1)
                res[tag] = s
            except Exception as e:
                res[tag] = None
                print(f"    {tag} failed: {type(e).__name__}: {e}")
            s = res[tag]
            cells += (f"{s['n']:>12}{s['blunder_frac']:>12.3f}" if s
                      else f"{'--':>12}{'--':>12}")
        lm.clear_loftr_caches()
        name = f"{r['Source image'].split('/')[0]} {os.path.basename(sp)[:12]}"
        print(f"{name[:32]:<34}{cells}")
        out.append({"pair": name, **res})
    return out


def summarise(out):
    if not out:
        print("\nno pairs completed")
        return {}
    print(f"\n=== {len(out)} ANHIR training pairs; expert landmarks held out from every arm ===")
    print(f"{'arm':<18}{'pairs':>6}{'n med':>8}{'blunder%':>10}{'max/med':>9}"
          f"{'TRE med':>9}{'TRE p90':>9}{'sec':>7}")
    summary = {}
    for tag, _ in ARMS:
        v = [o[tag] for o in out if o.get(tag)]
        if not v:
            print(f"{tag:<18}{0:>6}   (no pair completed)")
            continue
        g = lambda k: float(np.median([x[k] for x in v]))   # noqa: E731
        summary[tag] = {"pairs": len(v), "n_median": g("n"),
                        "blunder_frac_median": round(g("blunder_frac"), 4),
                        "max_over_med_median": round(g("max_over_med"), 2),
                        "tre_med_median": round(g("tre_med"), 2),
                        "tre_p90_median": round(g("tre_p90"), 2),
                        "seconds_median": round(g("seconds"), 1)}
        s = summary[tag]
        print(f"{tag:<18}{len(v):>6}{s['n_median']:>8.0f}{100*s['blunder_frac_median']:>9.1f}%"
              f"{s['max_over_med_median']:>9.2f}{s['tre_med_median']:>9.2f}"
              f"{s['tre_p90_median']:>9.2f}{s['seconds_median']:>7.1f}")

    print("\n" + "-" * 78)
    print("READING IT")
    print("-" * 78)
    base = summary.get("loftr_oasis")
    if base:
        for tag in ("loftr_raw", "disk_lightglue", "sift_lightglue"):
            s = summary.get(tag)
            if not s:
                continue
            better_tre = s["tre_med_median"] <= base["tre_med_median"] * 1.05
            fewer_blunders = s["blunder_frac_median"] < base["blunder_frac_median"]
            call = ("fewer blunders AND no worse at expert landmarks — a real alternative"
                    if better_tre and fewer_blunders else
                    "fewer blunders but worse expert TRE" if fewer_blunders else
                    "no blunder advantage")
            print(f"  {tag:<16} {s['n_median']/max(base['n_median'],1):5.2f}x n, "
                  f"blunders {100*s['blunder_frac_median']:.1f}% vs "
                  f"{100*base['blunder_frac_median']:.1f}%, TRE {s['tre_med_median']} vs "
                  f"{base['tre_med_median']}  -> {call}")
    print("\n  Swapping the matcher is a bigger change than any filter tweak: LoFTR is")
    print("  detector-free and the three model-free filters are written against its funnel.")
    print("  Only a large, consistent blunder advantage justifies it.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    rows = run(a.limit)
    s = summarise(rows)
    with open(a.out, "w") as f:
        json.dump({"pairs": rows, "summary": s}, f, indent=2)
    print(f"\nWrote {a.out}")
