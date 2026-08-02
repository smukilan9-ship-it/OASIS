#!/usr/bin/env python
"""
validate_ll477_reanalysis.py — what the session's fixes do to the real shipped result.

THE SHIPPED RESULT. The only full spatial run in the repo, on LL477 CD8 vs TIM-3
(~/Desktop/ihc_spatial_results), reports:

    primary null       dense_morphology        (correctly routed — architecture 71.8 um)
    colocalization     none          p = 0.101
    co-infiltration    ATTRACTION    p = 0.022     <-- the claim under test
    robustness         csr_only

That co-infiltration claim was computed on `bands`, the DCLF on L-r. § 13.2 established
that L inherits K's cumulative memory, so the 20-50 um band carries everything below 20 um
and fires on a truth that lives entirely at contact scale. Over 50 draws on three
substrates: a contact-only truth produced a co-infiltration claim at 0.98.

So the shipped claim is exactly the kind this session found unreliable, on the exact
statistic found unreliable. This re-runs it.

WHY REPRODUCE RATHER THAN RE-RUN THE PIPELINE. A full pipeline run would also re-segment and
re-threshold, mixing several changes together. Everything needed to isolate the STATISTIC is
already persisted: the certified local matrix, the certified ROI polygon, both detection
tables and both pixel sizes. Reconstructing from those changes exactly one thing.

The reconstruction is only trustworthy if it reproduces the run it claims to re-analyse, so
the point counts are checked against the shipped n_a = 38 and n_b = 34 before anything is
concluded. If they do not match, the script says so and stops.

Run:  python validation/validate_ll477_reanalysis.py
"""
import csv
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import cross_k_all_nulls, _BAND_STATISTIC

RESULTS = "/Users/mukilan/Desktop/ihc_spatial_results/spatial_association_results.json"
DET_DIR = "/Users/mukilan/Desktop/ihc_spatial_results/LL477_CD8_x10_3__roi0"
DET_A = f"{DET_DIR}/LL477_CD8_x10_3.tif - LL477_CD8_x10_3.tif #1_detections.csv"
DET_B = f"{DET_DIR}/LL477_Tim3_10X_3.tif - LL477_Tim3_10X_3.tif #1_detections.csv"
IMG_A = "/Users/mukilan/Desktop/cd8_input/LL477_CD8_x10_3.tif"
IMG_B = "/Users/mukilan/Desktop/tim3 input/LL477_Tim3_10X_3.tif"
EXPECT_N_A, EXPECT_N_B = 38, 34
EXPECT_AREA_UM2 = 243200.46
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ll477_reanalysis_results.json")


def load_positive(path, pixel_size_um):
    """Positive-cell centroids in that image's PIXEL frame (the matrix/ROI frame)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    xy = [(float(r["Centroid X µm"]) / pixel_size_um,
           float(r["Centroid Y µm"]) / pixel_size_um)
          for r in rows if "Positive" in (r.get("Classification") or "")]
    allxy = [(float(r["Centroid X µm"]) / pixel_size_um,
              float(r["Centroid Y µm"]) / pixel_size_um) for r in rows]
    return np.asarray(xy, float), np.asarray(allxy, float)


def in_polygon(pts, poly):
    poly = np.asarray(poly, float)
    inside = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        c = False
        for a in range(len(poly)):
            b = (a - 1) % len(poly)
            xi, yi = poly[a]
            xj, yj = poly[b]
            if (yi > p[1]) != (yj > p[1]) and \
               p[0] < (xj - xi) * (p[1] - yi) / ((yj - yi) or 1e-9) + xi:
                c = not c
        inside[i] = c
    return inside


def main():
    rec = json.load(open(RESULTS))
    rec = rec[0] if isinstance(rec, list) else rec
    v = rec["spatial_association"]["association"]["CD8__TIM-3"]
    cert = v["certification"]
    params = rec["resolved_params"]
    px_a = float(params["pixel_size_a_um"])
    px_b = float(params["pixel_size_b_um"])
    M = np.asarray(cert["matrix"], float)
    roi = np.asarray(cert["roi_polygon"], float)

    print("=" * 88)
    print("LL477 CD8 vs TIM-3 — re-analysis under this session's fixes")
    print("=" * 88)
    print(f"  shipped: primary={v['primary_null']}  "
          f"coloc={v['interaction']['colocalization']['verdict']} "
          f"(p={v['interaction']['colocalization']['global_p_dclf']})  "
          f"coinfil={v['interaction']['coinfiltration']['verdict']} "
          f"(p={v['interaction']['coinfiltration']['global_p_dclf']})")
    print(f"  certification: {cert['verdict']} via {cert['method']}, "
          f"pixel sizes {px_a:.4f} / {px_b:.4f} µm\n")

    A_all, A_support = load_positive(DET_A, px_a)
    B_all, B_support = load_positive(DET_B, px_b)
    B_reg = (np.c_[B_all, np.ones(len(B_all))] @ M.T) if len(B_all) else B_all
    B_sup_reg = np.c_[B_support, np.ones(len(B_support))] @ M.T

    # THE ANALYSIS WINDOW IS NOT THE ROI. The run recorded
    # tissue_mask_method="otsu_intersection_certified_roi": both tissue masks, B's mapped
    # through the transform, intersected with each other AND with the certified ROI. Using
    # the ROI alone gave n_a=80 / n_b=53 against the shipped 38 / 34 -- the window is what
    # differs, not the positivity call (Classification reproduces exactly at DAB>0.2 and
    # DAB>0.1). Rebuilt here with the pipeline's own helpers rather than approximated.
    from shapely.geometry import Polygon
    from oasis.spatial.spatial_stats import (estimate_tissue_polygon, transform_polygon,
                                             intersection_window, filter_points_in_polygon)
    # Signature is (area_px, polygon), and production passes the REFERENCE pixel size for
    # BOTH images (spatial.py:485-492) — not each image's own. Matching that exactly,
    # because a window rebuilt differently is a different analysis.
    _, poly_a = estimate_tissue_polygon(IMG_A, px_a)
    _, poly_b = estimate_tissue_polygon(IMG_B, px_a)
    # transform_centroids also reads scale_ref/scale_mov (it was built for thumbnail-space
    # transforms). The stored certification matrix is already FULL-RESOLUTION -- applying it
    # directly to full-res points reproduces n_a -- so both scales are 1.0 here. Without
    # them transform_polygon swallows a KeyError and silently returns the UNtransformed
    # polygon, which is how the first attempt got n_b=15 against a shipped 34.
    poly_b_in_a = transform_polygon(poly_b, {"matrix": M, "scale_ref": 1.0,
                                             "scale_mov": 1.0})
    window, area_px, iou, frac_a = intersection_window(poly_a, poly_b_in_a)
    if window is not None:
        window = window.intersection(Polygon(roi))
    if window is None or window.is_empty:
        print("  could not rebuild the analysis window")
        return 1
    area_px = float(window.area)

    A, _ = filter_points_in_polygon(A_all, window)
    B, _ = filter_points_in_polygon(B_reg, window)
    sup_a, _ = filter_points_in_polygon(A_support, window)
    sup_b, _ = filter_points_in_polygon(B_sup_reg, window)
    support = np.vstack([sup_a, sup_b])

    area_um2 = area_px * px_a * px_a
    print(f"  reconstructed n_a={len(A)} (shipped {EXPECT_N_A}), "
          f"n_b={len(B)} (shipped {EXPECT_N_B}), support={len(support)}")
    print(f"  window area {area_um2:.0f} µm² (shipped {EXPECT_AREA_UM2:.0f}), "
          f"IoU {iou:.3f} (shipped {v['intersection_overlap_iou']:.3f})")
    faithful = (abs(len(A) - EXPECT_N_A) <= 2 and abs(len(B) - EXPECT_N_B) <= 2
                and abs(area_um2 - EXPECT_AREA_UM2) / EXPECT_AREA_UM2 < 0.05)
    if not faithful:
        print("\n  RECONSTRUCTION DOES NOT MATCH THE SHIPPED RUN — stopping. Any comparison")
        print("  from here would be between two different analyses, not before and after.")
        json.dump({"faithful": False, "n_a": len(A), "n_b": len(B)},
                  open(OUT_JSON, "w"), indent=2)
        return 1
    print("  reconstruction matches the shipped run.\n")

    area = area_px
    radii_px = np.arange(0.0, 101.0, 2.0) / px_a
    r = cross_k_all_nulls(A, B, radii_px, area, px_a, n_perm=999, seed=0,
                          nulls=("dense_morphology", "homogeneous"),
                          morphology_support=support,
                          registration_radius_floor_um=None)
    nul = r["nulls"]["dense_morphology"]

    print("  RE-RUN under the same null, comparing band statistics:")
    print(f"    {'statistic':<16}{'colocalization':>24}{'co-infiltration':>24}")
    out = {}
    for st in ("bands", "bands_pcf", "bands_annulus", "bands_ring"):
        b = nul.get(st)
        if not b:
            continue
        f = lambda k: f"{b[k]['direction']} p={b[k]['global_p_dclf']}"      # noqa: E731
        mark = "  <- verdict reads this" if st == _BAND_STATISTIC else ""
        print(f"    {st:<16}{f('colocalization'):>24}{f('coinfiltration'):>24}{mark}")
        out[st] = {k: {"direction": b[k]["direction"], "p": b[k]["global_p_dclf"],
                       "significant": b[k]["significant"]}
                   for k in ("colocalization", "coinfiltration")}

    print("\n" + "=" * 88)
    print("WHAT CHANGES")
    print("=" * 88)
    old = out.get("bands", {}).get("coinfiltration", {})
    new = out.get(_BAND_STATISTIC, {}).get("coinfiltration", {})
    print(f"  co-infiltration on `bands`      : {old.get('direction')} "
          f"(p={old.get('p')})   <- what shipped")
    print(f"  co-infiltration on `{_BAND_STATISTIC}` : {new.get('direction')} "
          f"(p={new.get('p')})   <- what ships now")
    if old.get("significant") and not new.get("significant"):
        note = ("The shipped co-infiltration claim DOES NOT SURVIVE the corrected "
                "statistic. It was the cumulative-K artefact.")
    elif old.get("significant") and new.get("significant"):
        note = ("The claim SURVIVES the corrected statistic, so it was not an artefact of "
                "the cumulative L-r — it is a real regional-scale finding.")
    elif not old.get("significant") and new.get("significant"):
        note = ("The corrected statistic finds a claim the shipped one MISSED — consistent "
                "with L-r going blind to an upper-band signal when the lower band is "
                "depleted (§ 13.2, regional truth detected at only 0.06).")
    else:
        note = "Neither statistic claims co-infiltration here."
    print(f"\n  {note}")

    json.dump({"faithful": True, "n_a": len(A), "n_b": len(B),
               "shipped": {"coloc": v["interaction"]["colocalization"],
                           "coinfil": v["interaction"]["coinfiltration"]},
               "rerun": out, "verdict_statistic": _BAND_STATISTIC, "note": note},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n  Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
