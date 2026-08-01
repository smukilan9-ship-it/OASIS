#!/usr/bin/env python
"""
Is LL477's 8-22 % blunder rate the matcher, or the slides?

THE OPEN QUESTION. validate_matchers_on_cohort.py established that LoFTR is the only matcher
that works on H-DAB serial sections — DISK and SIFT return nothing, DeDoDe returns thousands
of matches of which 37-67 % are gross, KeyNet's are 100 % gross. But LoFTR itself carries
8-22 % gross correspondences on the real LL477 pairs, against 0 % on a synthetic warp of the
same tissue. Something about matching two DIFFERENT physical sections produces them, and two
explanations were left open:

  (a) the matcher cannot bridge a stain difference on serial sections, full stop; or
  (b) LL477's slides are the problem — its TIM-3 has no callable signal at all
      (research/ihc.md § 12: 4-58 positive cells at a defensible threshold, reported as
      11,584), and a slide that faint may also be a poor structural match.

HyReCo separates them. It is professionally prepared, expert-annotated, published serial
sections of the same modality. If LoFTR's blunder rate here is ~1 %, the fault is LL477's
staining and no amount of model work fixes it. If it is also 8-22 %, it is inherent to
cross-stain serial matching and blunder rejection is the only remedy.

METHOD. Fields are 1920x1440 at exactly 0.7519 um/px — LL477's frame, so the comparison is
like-for-like — centred on each expert landmark, so both stains' fields cover the same
anatomy. Blunders are measured the way they were on LL477: residual against a robust
similarity, gross above 15 um, plus the max/median ratio that exposed the problem (isotropic
noise gives roughly 2.5 at these counts).

WHAT THIS IS NOT. A field contains at most one landmark, so nothing here is scored against
the experts — that is validate_hyreco_landmark_tre.py's job at section scale. This measures
the matcher's self-consistency on good slides against the same measurement on bad ones.

Run:  python validation/validate_hyreco_field_blunders.py
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import serial_registration as sr        # noqa: E402
from oasis.spatial import loftr_matcher as lm              # noqa: E402

FIELDS = "/Volumes/Expansion/oasis_datasets/HyReCo/_rendered/fields"
PX = 0.7519
GROSS_UM = 15.0
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hyreco_field_blunders_results.json")
# LL477, from validate_matchers_on_cohort.py — the thing being compared against.
LL477_GROSS = (0.083, 0.221, 0.103)


def field(case, stain, i):
    p = f"{FIELDS}/{case}/{case}_{stain}_lm{i:02d}.png"
    if not os.path.exists(p) or os.path.basename(p).startswith("._"):
        return None
    import cv2
    im = cv2.imread(p)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def run(cases, pairs):
    rows = []
    print(f"{'case':>5} {'pair':>11} {'lm':>3} {'n':>6} {'med um':>8} {'max/med':>8} "
          f"{'gross':>7}")
    for case in cases:
        idxs = sorted({int(re.search(r"_lm(\d+)\.png$", f).group(1))
                       for f in glob.glob(f"{FIELDS}/{case}/*_lm*.png")
                       if not os.path.basename(f).startswith("._")})
        for ref_s, mov_s in pairs:
            for i in idxs:
                a, b = field(case, ref_s, i), field(case, mov_s, i)
                if a is None or b is None:
                    continue
                c = lm.loftr_correspondences(a, b, pixel_size_um=PX)
                if not c["ok"]:
                    print(f"{case:>5} {ref_s + '/' + mov_s:>11} {i:>3} {'--':>6}  "
                          f"{c.get('msg', '')[:44]}")
                    rows.append({"case": case, "pair": f"{ref_s}/{mov_s}", "lm": i, "n": 0})
                    lm.clear_loftr_caches()
                    continue
                p = np.asarray(c["ref_points"], float)
                q = np.asarray(c["mov_points"], float)
                M = sr._fit_similarity_robust(q, p)
                keep = sr.reject_local_residual_outliers(p, q, M)
                if keep.sum() >= 6:
                    p, q = p[keep], q[keep]
                    M = sr._fit_similarity_robust(q, p)
                r = np.linalg.norm(sr._apply_affine(q, M) - p, axis=1) * PX
                med = float(np.median(r))
                row = {"case": case, "pair": f"{ref_s}/{mov_s}", "lm": i, "n": int(len(p)),
                       "resid_med_um": round(med, 3),
                       "max_over_med": round(float(r.max() / max(med, 1e-9)), 2),
                       "gross_frac": round(float((r > GROSS_UM).mean()), 4)}
                rows.append(row)
                print(f"{case:>5} {ref_s + '/' + mov_s:>11} {i:>3} {row['n']:>6} "
                      f"{row['resid_med_um']:>8.2f} {row['max_over_med']:>8.2f} "
                      f"{100 * row['gross_frac']:>6.1f}%")
                lm.clear_loftr_caches()
    return rows


def summarise(rows):
    ok = [r for r in rows if r.get("n", 0) >= 6]
    print("\n" + "=" * 78)
    print("HyReCo (published, expert-annotated) vs LL477 (our slides)")
    print("=" * 78)
    if not ok:
        print("  no field produced correspondences")
        return {}
    g = np.array([r["gross_frac"] for r in ok])
    n = np.array([r["n"] for r in ok], float)
    mm = np.array([r["max_over_med"] for r in ok])
    md = np.array([r["resid_med_um"] for r in ok])
    print(f"  fields with correspondences : {len(ok)}/{len(rows)}")
    print(f"  correspondences   median    : {np.median(n):.0f}")
    print(f"  residual median   median    : {np.median(md):.2f} µm")
    print(f"  max/median        median    : {np.median(mm):.2f}   (isotropic noise ~2.5)")
    print(f"  GROSS FRACTION    median    : {100 * np.median(g):.1f}%   mean {100 * g.mean():.1f}%")
    print(f"  LL477 for comparison        : {100 * np.mean(LL477_GROSS):.1f}% "
          f"(8.3 / 22.1 / 10.3 on its three pairs)")
    print()
    if np.median(g) < 0.5 * np.mean(LL477_GROSS):
        print("  Materially cleaner on published slides. That points at LL477's staining")
        print("  rather than at the matcher — and no model work fixes a slide.")
    elif np.median(g) > 1.5 * np.mean(LL477_GROSS):
        print("  WORSE than LL477. Something about this rendering or these sections is harder;")
        print("  read the per-field rows before concluding anything about the matcher.")
    else:
        print("  Comparable to LL477. The blunders are inherent to cross-stain serial matching,")
        print("  not an artefact of our slides, so rejection is the remedy and not better slides.")
    return {"n_fields": len(ok), "gross_median": float(np.median(g)),
            "gross_mean": float(g.mean()), "n_median": float(np.median(n)),
            "resid_med_median": float(np.median(md)),
            "max_over_med_median": float(np.median(mm)),
            "ll477_gross_mean": float(np.mean(LL477_GROSS))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["611", "679"])
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    rows = run(a.cases, [("CD8", "HE"), ("CD8", "CD45")])
    s = summarise(rows)
    json.dump({"fields": rows, "summary": s}, open(a.out, "w"), indent=2)
    print(f"\nWrote {a.out}")
