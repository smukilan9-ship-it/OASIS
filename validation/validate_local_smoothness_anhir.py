"""Does the local-smoothness filter help or harm, measured against EXPERT landmarks?

WHY THIS TEST AND NOT A RESIDUAL. The filter was added because LoFTR's cycle+scale
survivors still contain gross mismatches that wreck the certification gate. But "residuals
got smaller" is not evidence: the filter selects the points the residual is then measured
on, so it is guaranteed to shrink its own residual. The only honest question is whether the
resulting TRANSFORM lands expert-annotated anatomy closer to where the expert put it.

ANHIR training pairs give exactly that. LoFTR never sees the landmarks, so the entire expert
set is held out from both arms, and the comparison is like-for-like:

    arm A (before)  loftr_correspondences(..., local_k=0)   cycle + scale only
    arm B (after)   loftr_correspondences(..., local_k=8)   + local smoothness

Fit a similarity from each arm's correspondences, apply it to the expert SOURCE landmarks,
and measure the distance to the expert TARGET landmarks. Lower is better; the paired
per-pair delta is the result. A filter that merely flattered its own residual would show no
improvement here — or a regression.

Reported in PIXELS at the working scale, not µm: ANHIR's per-tissue µm/px is not needed for
a paired A/B on identical images, and asserting one we have not verified would be the kind
of unchecked bookkeeping the sibling ANHIR harness exists to avoid.

Run:  .venv/bin/python validation/validate_local_smoothness_anhir.py [--limit N]
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import serial_registration as sr          # noqa: E402
from oasis.spatial import loftr_matcher as lm                # noqa: E402

ROOT = os.path.expanduser("~/oasis_validation_datasets/ANHIR_medium/images")
WORK_MAX = 1024          # long side LoFTR runs at; both arms identical


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


def tre(M, src_lm, dst_lm):
    """Realized error at the expert landmarks under transform M (source -> target)."""
    return np.linalg.norm(sr._apply_affine(src_lm, M) - dst_lm, axis=1)


def run(limit, seed_shuffle=0):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "dataset_medium.csv")))
            if r["status"] == "training"]
    rng = np.random.default_rng(seed_shuffle)
    rng.shuffle(rows)                     # avoid testing only COAD_*, which sorts first
    out = []
    print(f"{'pair':<44}{'nA':>5}{'nB':>5}{'drop':>7}"
          f"{'medA':>9}{'medB':>9}{'p90A':>9}{'p90B':>9}")
    for r in rows[:limit]:
        sp = os.path.join(ROOT, r["Source image"])
        tp = os.path.join(ROOT, r["Target image"])
        slp = os.path.join(ROOT, r["Source landmarks"])
        tlp = os.path.join(ROOT, r["Target landmarks"])
        if not all(os.path.exists(p) for p in (sp, tp, slp, tlp)):
            continue
        try:
            mov_rgb, s_mov = load_image(sp)
            ref_rgb, s_ref = load_image(tp)
            src_lm = load_landmarks(slp) * s_mov          # into working pixels
            dst_lm = load_landmarks(tlp) * s_ref
        except Exception as e:
            print(f"{r['Source image'][:42]:<44}  LOAD ERROR {e}")
            continue
        n_lm = min(len(src_lm), len(dst_lm))
        if n_lm < 8:
            continue
        src_lm, dst_lm = src_lm[:n_lm], dst_lm[:n_lm]

        res = {}
        for tag, k in (("A", 0), ("B", 8)):
            c = lm.loftr_correspondences(ref_rgb, mov_rgb, pixel_size_um=1.0, local_k=k)
            if not c["ok"]:
                res[tag] = None
                continue
            rp = np.asarray(c["ref_points"], float)      # in ref (target) pixels
            mp = np.asarray(c["mov_points"], float)      # in mov (source) pixels
            M = sr._fit_similarity_robust(mp, rp)        # source -> target
            e = tre(M, src_lm, dst_lm)
            res[tag] = {"n": c["n"], "drop": c["local_drop_frac"],
                        "med": float(np.median(e)), "p90": float(np.percentile(e, 90))}
        lm.clear_loftr_caches()
        a, b = res.get("A"), res.get("B")
        if not a or not b:
            print(f"{(r['Source image'].split('/')[0] + ' ' + os.path.basename(sp))[:42]:<44}"
                  f"  one arm found no usable correspondences — skipped")
            continue
        name = f"{r['Source image'].split('/')[0]} {os.path.basename(sp)}→{os.path.basename(tp)}"
        print(f"{name[:42]:<44}{a['n']:>5}{b['n']:>5}{b['drop']:>7.2f}"
              f"{a['med']:>9.1f}{b['med']:>9.1f}{a['p90']:>9.1f}{b['p90']:>9.1f}")
        out.append({"pair": name, "A": a, "B": b})
    return out


def summarise(out):
    if not out:
        print("\nno pairs completed")
        return
    dmed = np.array([o["B"]["med"] - o["A"]["med"] for o in out])
    dp90 = np.array([o["B"]["p90"] - o["A"]["p90"] for o in out])
    print(f"\n=== {len(out)} ANHIR training pairs, expert landmarks held out from both arms ===")
    print(f"median TRE   A {np.median([o['A']['med'] for o in out]):8.2f} px   "
          f"B {np.median([o['B']['med'] for o in out]):8.2f} px")
    print(f"p90 TRE      A {np.median([o['A']['p90'] for o in out]):8.2f} px   "
          f"B {np.median([o['B']['p90'] for o in out]):8.2f} px")
    print(f"pairs improved (median TRE): {(dmed < 0).sum()}/{len(out)}   "
          f"worsened: {(dmed > 0).sum()}")
    print(f"paired delta median TRE: median {np.median(dmed):+.2f} px, "
          f"mean {dmed.mean():+.2f} px")
    print(f"paired delta p90 TRE:    median {np.median(dp90):+.2f} px, "
          f"mean {dp90.mean():+.2f} px")
    try:
        from scipy.stats import wilcoxon
        if len(dmed) >= 6 and np.any(dmed != 0):
            print(f"Wilcoxon signed-rank on median TRE: p={wilcoxon(dmed).pvalue:.4g}")
    except Exception:
        pass
    print(f"mean fraction dropped by the filter: {np.mean([o['B']['drop'] for o in out]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    summarise(run(args.limit))
