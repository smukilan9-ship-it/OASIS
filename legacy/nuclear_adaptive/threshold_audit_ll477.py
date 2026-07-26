"""
threshold_audit_ll477.py — does the adaptive threshold land where the trusted fixed one is?

The pipeline ships fixed per-stain cutoffs (config `stain_thresholds`): CD8 0.2, TIM-3 0.1.
Those are the trusted operating points. The adaptive classifier is meant to replace them
per-image, so the question that matters is not "does it produce a number" but "does it
produce a number near the one we already trust, on real slides from this lab".

Runs every real image in ~/Desktop/cd8_input and ~/Desktop/tim3 input (scale-bar images
excluded — they are photographs of a ruler, not tissue) and reports, per image, the adaptive
threshold under three ways of getting per-cell DAB:

  A  fixed QuPath stain vectors          — what production does today
  B  per-image Macenko stain vectors     — cell_expansion's estimator, which exists because
                                           fixed vectors mis-deconvolve slides whose white
                                           balance differs from QuPath's convention
  C  percentile-normalised image, then fixed vectors
                                         — "is thresholding a normalised image better?"

Nuclear DAB mean is the signal in all three, because classify_nuclear is a nuclear tool.
CD8 and TIM-3 are membranous markers, so a nuclear cut is not the production call for them —
that is the ring/completeness path. This audit is about whether the ADAPTIVE MACHINERY finds
the right operating point on this tissue, which is what was asked; the membranous caveat is
reported alongside rather than silently ignored.

Run:  .venv/bin/python validation/threshold_audit_ll477.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.common.paths import default_model_dir

MODEL_DIR = default_model_dir()

# Scale-bar calibration measured for this cohort: bars are uniformly 133 px / 0.7519 µm.
PX_UM = 0.7518796992481203

FOLDERS = [
    ("CD8", os.path.expanduser("~/Desktop/cd8_input"), 0.2),
    ("TIM-3", os.path.expanduser("~/Desktop/tim3 input"), 0.1),
]


def real_images(folder):
    """Tissue images only — anything with 'scale' in the name is a ruler photograph."""
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        if "scale" in name.lower():
            continue
        out.append(os.path.join(folder, name))
    return out


def dab_variants(rgb):
    """Per-pixel DAB OD three ways. Returns {variant: (hem_od, dab_od)}."""
    from oasis.quant import segment as sg
    from oasis.quant import cell_expansion as ce

    out = {}

    # A — production: fixed QuPath vectors, fixed white point.
    out["A_fixed"] = sg._od_channels(rgb)

    # B — per-image Macenko vectors. Falls back to fixed when the estimate is degenerate,
    # which is itself a result worth seeing.
    try:
        vecs = ce._estimate_stain_vectors(rgb, sg.QUPATH_WHITE)
        if vecs:
            out["B_macenko"] = ce._od_channels(rgb, vecs, sg.QUPATH_WHITE)[:2]
        else:
            out["B_macenko"] = None      # estimator declined
    except Exception as e:
        out["B_macenko"] = None
        out["_b_error"] = str(e)

    # C — normalise the image first (the same 0.1/99.9 percentile stretch the segmenter
    # feeds the model), then deconvolve with fixed vectors. Per-channel percentile
    # rescaling is effectively a white balance, which is the interesting part: it should
    # help exactly where fixed vectors struggle.
    #
    # _apply_norm returns the MODEL's layout, (1, 3, H, W) float in ~0..1 — not an image.
    # Deconvolution needs HxWx3 in 0..255, so convert back explicitly. (Passing the model
    # tensor straight through silently produced (1, 3, H)-shaped "OD" and an unrelated
    # IndexError three call-frames later.)
    try:
        ranges = sg.norm_range(rgb)
        if sg.low_contrast(ranges):
            out["C_normalised"] = None   # stretching noise would manufacture signal
        else:
            norm = np.asarray(sg._apply_norm(rgb, ranges))
            if norm.ndim == 4:
                norm = norm[0]
            if norm.ndim == 3 and norm.shape[0] == 3:
                norm = norm.transpose(1, 2, 0)
            norm8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
            assert norm8.shape == rgb.shape[:2] + (3,), f"bad normalised shape {norm8.shape}"
            out["C_normalised"] = sg._od_channels(norm8)
    except Exception as e:
        out["C_normalised"] = None
        out["_c_error"] = str(e)

    return out


def audit_image(path, expected, model):
    from oasis.quant import segment as sg
    from oasis.quant.cell_expansion import _load_rgb_full
    from oasis.quant.nuclear_classify import classify_nuclear

    rgb = _load_rgb_full(path)
    labels = sg.segment_labels(rgb, model, "cpu")
    n_cells = int(labels.max())

    row = {"image": os.path.basename(path), "shape": list(rgb.shape[:2]),
           "n_cells": n_cells, "expected_fixed": expected, "variants": {}}
    if n_cells == 0:
        row["error"] = "no cells segmented"
        return row

    for name, chans in dab_variants(rgb).items():
        if name.startswith("_") or chans is None:
            if not name.startswith("_"):
                row["variants"][name] = {"available": False,
                                         "reason": "estimator declined / low contrast"}
            continue
        hem, dab = chans
        recs = sg._measure(labels, hem, dab, PX_UM)
        vals = np.array([r["measurements"]["DAB: Mean"] for r in recs], dtype=float)

        # ashman_min=2.0 is the shipped default; 1.25 was the operating point that won on
        # DeepLIIF. Report both so the choice is visible rather than assumed.
        res = {}
        for tag, amin in (("ashman2.0", 2.0), ("ashman1.25", 1.25)):
            c = classify_nuclear(vals, fixed_threshold=expected, ashman_min=amin,
                                 allow_fixed_fallback=False)
            res[tag] = {"threshold": c["threshold"], "method": c["method"],
                        "sep": round(float(c["separability"]), 3),
                        "abstain": bool(c["abstain"]),
                        "pos_frac": (float(np.mean(vals > c["threshold"]))
                                     if c["threshold"] is not None else None)}
        row["variants"][name] = {
            "available": True,
            "dab_median": round(float(np.median(vals)), 4),
            "dab_p90": round(float(np.percentile(vals, 90)), 4),
            "dab_p99": round(float(np.percentile(vals, 99)), 4),
            "otsu": round(float(sg.otsu_threshold(vals) or 0.0), 4),
            "pos_frac_at_fixed": round(float(np.mean(vals > expected)), 4),
            **res,
        }
    return row


def main():
    from oasis.quant import segment as sg
    model = sg.load_model(MODEL_DIR, "cpu")

    report = []
    for stain, folder, expected in FOLDERS:
        imgs = real_images(folder)
        print(f"\n{'=' * 78}\n{stain}  ({len(imgs)} tissue images, fixed cutoff {expected})"
              f"\n{folder}\n{'=' * 78}")
        for p in imgs:
            row = audit_image(p, expected, model)
            row["stain"] = stain
            report.append(row)

            print(f"\n{row['image']}   {row['shape'][0]}x{row['shape'][1]}   "
                  f"{row['n_cells']} cells")
            if "error" in row:
                print(f"   ERROR: {row['error']}")
                continue
            for vname, v in row["variants"].items():
                if not v.get("available"):
                    print(f"   {vname:14s} unavailable ({v['reason']})")
                    continue
                a2, a125 = v["ashman2.0"], v["ashman1.25"]
                print(f"   {vname:14s} med {v['dab_median']:.3f}  p90 {v['dab_p90']:.3f}  "
                      f"otsu {v['otsu']:.3f}  pos@fixed {v['pos_frac_at_fixed']:.3f}")
                for tag, r in (("D>=2.00", a2), ("D>=1.25", a125)):
                    thr = "abstain" if r["threshold"] is None else f"{r['threshold']:.4f}"
                    pf = "-" if r["pos_frac"] is None else f"{r['pos_frac']:.3f}"
                    print(f"      {tag}: thr {thr:>8s}  sep {r['sep']:.2f}  "
                          f"method {r['method']:<8s} pos {pf}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "threshold_audit_ll477_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
