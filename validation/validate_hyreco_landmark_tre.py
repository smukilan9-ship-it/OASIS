#!/usr/bin/env python
"""
Does OASIS's predicted cell error match the truth? — first external test on serial H-DAB.

WHAT HAS NEVER BEEN CHECKED. The certification gate outputs a cell-error p90 and a verdict.
Every number behind it — the FLE, the deformation term, the Fitzpatrick-West bound — has been
calibrated on ANHIR (lung/mammary/kidney, largely cross-stain) or measured against synthetic
warps of a single section. research/registration.md § 11.6 names the gap explicitly: no
expert landmarks on serial H-DAB, so the predicted error has never been compared with a
realized one on the modality OASIS actually ships for.

HyReCo closes it. Nine cases, five stains on CONSECUTIVE sections, 11-19 landmarks per
section placed by hand and verified by two researchers. LoFTR never sees them, so the entire
expert set is held out from the transform under test.

THE MEASUREMENT, per stain pair:

  1. Register the two sections the way OASIS does — register_similarity for the provisional,
     then certify_local_roi over the tissue, which fits its own local similarity from LoFTR
     correspondences and returns a verdict plus a predicted cell-error p90.
  2. Push the MOVING section's expert landmarks through that transform.
  3. Compare with the REFERENCE section's expert landmarks. That distance is the realized TRE.
  4. Set predicted against realized.

A gate that is honest predicts at or above what actually happens. Predicting BELOW the
realized error is the dangerous direction — it would mean certifications claim a precision
the registration does not have.

SCALE, AND WHY IT IS THE SECTION AND NOT A FIELD. The landmarks are 11-19 points spread over
a whole 23 x 53 mm section, roughly 3 mm apart, so a 1.4 mm field contains zero or one of them
and cannot be scored. This measures registration at the SECTION scale. It does not validate
the 450 um ROI certification, and nothing here should be read as though it does — see
validation/datasets/hyreco_render.py.

Run:  python validation/validate_hyreco_landmark_tre.py [--cases 611 679]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import serial_registration as sr        # noqa: E402
from oasis.spatial import loftr_matcher as lm              # noqa: E402

REND = "/Volumes/Expansion/oasis_datasets/HyReCo/_rendered/section"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hyreco_landmark_tre_results.json")


def load(case, stain):
    """Rendered section + its expert landmarks in that render's pixel frame + its µm/px."""
    import cv2
    p = f"{REND}/{case}/{case}_{stain}.png"
    if not os.path.exists(p):
        return None
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    lm_px = np.loadtxt(f"{REND}/{case}/{case}_{stain}_landmarks_px.csv",
                       delimiter=",", skiprows=1).reshape(-1, 2)
    # µm/px is recoverable from the render: the section renderer scaled the level-0 slide by a
    # constant, and it recorded the landmarks in the SAME frame, so the two are consistent.
    import openslide
    src = f"/Volumes/Expansion/oasis_datasets/HyReCo/HyReCo/{stain}/{case}.tif"
    s = openslide.OpenSlide(src)
    um_px = float(s.properties[openslide.PROPERTY_NAME_MPP_X]) * (s.dimensions[0] / img.shape[1])
    s.close()
    return img, lm_px, um_px


def run(cases, pairs):
    rows = []
    print(f"{'case':>5} {'pair':>12} {'n':>4} {'verdict':>18} {'predicted':>10} "
          f"{'REALIZED median TRE':>19} {'honest?':>8}")
    print(f"{'':>5} {'':>12} {'':>4} {'':>18} {'p90 µm':>10} "
          f"{'provis.  certif.':>19}")
    for case in cases:
        for ref_stain, mov_stain in pairs:
            a = load(case, ref_stain)
            b = load(case, mov_stain)
            if a is None or b is None:
                print(f"{case:>5} {ref_stain + '->' + mov_stain:>12}  (slide not downloaded)")
                continue
            ref, ref_lm, ref_um = a
            mov, mov_lm, mov_um = b
            n = min(len(ref_lm), len(mov_lm))
            ref_lm, mov_lm = ref_lm[:n], mov_lm[:n]

            # TWO transforms, measured separately, because they fail differently.
            #
            # register_similarity is multi-resolution mutual information and always returns
            # something, so it can be scored on every pair. certify_local_roi runs LoFTR, and
            # at ~13 um/px on a whole section it frequently finds nothing — in which case
            # `local_matrix` is absent and falling back to the provisional would report the
            # PROVISIONAL's error under the certification's name. A first pass did exactly
            # that and printed a 10,672 um "TRE" for a registration that never happened.
            M0 = np.asarray(sr.register_similarity(ref, mov, ref_um)["matrix"], float)
            H, W = ref.shape[:2]
            roi = np.array([[W * .02, H * .02], [W * .98, H * .02],
                            [W * .98, H * .98], [W * .02, H * .98]], float)
            cert = lm.certify_local_roi(ref, mov, roi, ref_um, provisional_matrix=M0,
                                        work_max_dim=1024)

            def _tre(M):
                # THE EXPERT SET IS HELD OUT: no landmark informed either transform.
                d = np.linalg.norm(sr._apply_affine(mov_lm, np.asarray(M, float)) - ref_lm,
                                   axis=1) * ref_um
                return float(np.median(d)), float(np.percentile(d, 90))

            g_med, g_p90 = _tre(M0)
            has_cert = cert.get("local_matrix") is not None
            c_med, c_p90 = _tre(cert["local_matrix"]) if has_cert else (None, None)
            pred = cert.get("cell_error_p90_um")
            honest = (pred is not None and c_p90 is not None and pred >= c_p90)
            rows.append({"case": case, "ref": ref_stain, "mov": mov_stain, "n": int(n),
                         "verdict": cert.get("verdict"), "predicted_p90_um": pred,
                         "provisional_median_um": round(g_med, 2),
                         "provisional_p90_um": round(g_p90, 2),
                         "certified_median_um": None if c_med is None else round(c_med, 2),
                         "certified_p90_um": None if c_p90 is None else round(c_p90, 2),
                         "n_correspondences": cert.get("n_correspondences"),
                         "um_px": round(ref_um, 3),
                         "conservative": bool(honest) if pred is not None else None})
            print(f"{case:>5} {ref_stain + '->' + mov_stain:>12} {n:>4} "
                  f"{str(cert.get('verdict')):>18} "
                  f"{('--' if pred is None else f'{pred:.1f}'):>10} "
                  f"{g_med:>8.1f} {('--' if c_med is None else f'{c_med:.1f}'):>9} "
                  f"{('n/a' if pred is None else ('yes' if honest else 'NO')):>8}")
            lm.clear_loftr_caches()
    return rows


def summarise(rows):
    ok = [r for r in rows if r["predicted_p90_um"] is not None
          and r["certified_p90_um"] is not None]
    print("\n" + "=" * 78)
    print("PREDICTED vs REALIZED — expert landmarks, held out from every transform")
    print("=" * 78)
    if not ok:
        print("  no pair produced a prediction")
        return {}
    p = np.array([r["predicted_p90_um"] for r in ok])
    q = np.array([r["certified_p90_um"] for r in ok])
    m = np.array([r["certified_median_um"] for r in ok])
    gp = np.array([r["provisional_p90_um"] for r in rows])
    print(f"  provisional-only TRE p90 over all {len(rows)} pairs: "
          f"median {np.median(gp):.1f} µm  (register_similarity alone, no LoFTR)")
    cons = int(sum(r["conservative"] for r in ok))
    print(f"  pairs                       : {len(ok)}")
    print(f"  realized TRE  median        : {np.median(m):.1f} µm")
    print(f"  realized TRE  p90           : {np.median(q):.1f} µm")
    print(f"  predicted cell-error p90    : {np.median(p):.1f} µm")
    print(f"  predicted / realized ratio  : {np.median(p / np.maximum(q, 1e-9)):.2f}")
    print(f"  CONSERVATIVE (pred >= real) : {cons}/{len(ok)}")
    print()
    if cons == len(ok):
        print("  The gate never under-predicted. That is the safe direction: a certification")
        print("  claims no more precision than the registration delivers.")
    else:
        print(f"  {len(ok) - cons} pair(s) were predicted BETTER than they actually are. That is")
        print("  the dangerous direction and needs explaining before any certified result on")
        print("  this modality is trusted.")
    return {"n": len(ok), "realized_med": float(np.median(m)),
            "realized_p90": float(np.median(q)), "predicted_p90": float(np.median(p)),
            "ratio": float(np.median(p / np.maximum(q, 1e-9))), "conservative": cons}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["611", "679"])
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    PAIRS = [("CD8", "HE"), ("CD8", "CD45"), ("HE", "CD45")]
    rows = run(a.cases, PAIRS)
    s = summarise(rows)
    json.dump({"pairs": rows, "summary": s}, open(a.out, "w"), indent=2)
    print(f"\nWrote {a.out}")
