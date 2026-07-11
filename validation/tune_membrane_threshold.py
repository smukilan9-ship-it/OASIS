#!/usr/bin/env python3
"""
tune_membrane_threshold.py
Fit the membrane-completeness classification cutoffs for a faint membranous
marker (TIM-3) against HAND-LABELLED cells, and report whether the labelled
positives and negatives actually separate.

WHY THIS EXISTS
---------------
A membranous stain sits on a thin arc of the cell membrane, not across the whole
cytoplasmic ring. Classifying on the ring MEAN dilutes a faint arc below
threshold (lost positives) and cannot be rescued by lowering the threshold —
diffuse background then crosses too (false positives). The fix is to classify on
membrane COMPLETENESS: the fraction of ring pixels above a pixel-level OD
threshold (`membrane_pos_frac`) and/or the ring's 90th-percentile OD
(`cytoplasm_dab_p90`). Those features keep a faint concentrated arc separable
from diffuse background — but only with cutoffs fit to ground truth. This script
fits them.

WHAT YOU PROVIDE
----------------
  --image      the TIM-3 image (same one the GeoJSON was segmented from)
  --labelled   a GeoJSON of detections in which a SUBSET of cells are hand-
               classified positive / negative. It should ALSO contain the full
               detection set with QuPath "DAB: Mean" per nucleus, because the OD
               channel is calibrated to QuPath's scale on >=50 nuclei (the same
               parity gate the production pipeline uses).

Label cells with class names distinct from any auto-classification so they are
unambiguous — e.g. export the raw post-segmentation objects and mark cells
"membrane_pos" / "membrane_neg" (pass via --pos-label / --neg-label). Defaults
are "Positive" / "Negative".

WHAT IT DOES
------------
  1. Re-measures the cytoplasmic ring on every cell (calibrated OD), keeping the
     per-pixel ring values for the labelled cells.
  2. Anchors the pixel threshold t_pix to the negative cells' ring background
     (99th percentile of pooled negative ring pixels) unless --pix-thr is given.
  3. Builds ROC curves for membrane_pos_frac and for p90 vs the labels; reports
     AUC and the operating points at (a) max Youden's J and (b) a target
     sensitivity (default 0.95 — losing positives is the failure we care about).
  4. Optionally writes the fitted cutoffs to a YAML/JSON for the pipeline config.

If the AUC is high and there is a clear gap, you have data-backed cutoffs. If it
is near 0.5, no cutoff exists — the membrane signal is too faint to call, which
is a finding, not a tuning failure.

Usage:
  python3 validation/tune_membrane_threshold.py \
      --image path/to/tim3.tif --labelled path/to/labelled.geojson \
      --pixel-size 0.5 --expansion 2.0 --target-sensitivity 0.95 \
      --out validation/membrane_cutoffs.yaml
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.quant.cell_expansion import measure_cytoplasm_dab


def _classification_name(props):
    cls = props.get("classification") or props.get("class")
    if isinstance(cls, dict):
        return str(cls.get("name", ""))
    return str(cls) if cls else ""


def _load_labels(geojson_path, pos_label, neg_label):
    """Return a list aligned 1:1 with features: 1 (pos), 0 (neg), or None."""
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [])
    pos_l, neg_l = pos_label.lower(), neg_label.lower()
    labels = []
    for feat in features:
        name = _classification_name(feat.get("properties", {})).lower()
        if name == pos_l:
            labels.append(1)
        elif name == neg_l:
            labels.append(0)
        else:
            labels.append(None)
    return labels


def _roc(scores, labels):
    """ROC from scores (higher => more positive) and binary labels (1/0).
    Returns sorted arrays (thresholds, tpr, fpr) and AUC. Pure numpy."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    if P == 0 or N == 0:
        return None
    # Candidate thresholds: each unique score (a cell is positive iff score >= t).
    thr = np.unique(scores)
    thr = np.concatenate([[np.nextafter(thr[0], -np.inf)], thr])  # include "all positive"
    tpr, fpr = [], []
    for t in thr:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        tpr.append(tp / P)
        fpr.append(fp / N)
    tpr, fpr = np.array(tpr), np.array(fpr)
    order = np.argsort(fpr)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2 renamed it
    auc = float(_trap(tpr[order], fpr[order]))
    return {"thr": thr, "tpr": tpr, "fpr": fpr, "auc": auc, "P": P, "N": N}


def _operating_points(roc, target_sens):
    youden = roc["tpr"] - roc["fpr"]
    j = int(np.argmax(youden))
    best_j = {"thr": float(roc["thr"][j]), "sens": float(roc["tpr"][j]),
              "spec": float(1 - roc["fpr"][j]), "youden": float(youden[j])}
    # Smallest threshold (most permissive => highest sens) whose sensitivity still
    # meets the target; among those, the one with best specificity.
    meets = np.where(roc["tpr"] >= target_sens)[0]
    if len(meets):
        k = meets[np.argmin(roc["fpr"][meets])]
        at_target = {"thr": float(roc["thr"][k]), "sens": float(roc["tpr"][k]),
                     "spec": float(1 - roc["fpr"][k])}
    else:
        at_target = None
    return best_j, at_target


def _summ(name, vals_pos, vals_neg):
    def s(a):
        a = np.asarray(a, float)
        return (f"n={len(a)} median={np.median(a):.4f} "
                f"mean={a.mean():.4f} [p10={np.percentile(a,10):.4f}, "
                f"p90={np.percentile(a,90):.4f}]")
    print(f"  {name}:")
    print(f"    positives  {s(vals_pos)}")
    print(f"    negatives  {s(vals_neg)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--labelled", required=True, help="GeoJSON with hand-labelled cells")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="µm/px (default: 0.5 with a warning)")
    ap.add_argument("--expansion", type=float, default=2.0, help="ring width µm")
    ap.add_argument("--pos-label", default="Positive")
    ap.add_argument("--neg-label", default="Negative")
    ap.add_argument("--pix-thr", type=float, default=None,
                    help="pixel OD threshold for pos-fraction (default: anchored "
                         "to 99th pct of negative ring pixels)")
    ap.add_argument("--neg-pixel-percentile", type=float, default=99.0)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--out", default=None, help="write fitted cutoffs (.yaml/.json)")
    args = ap.parse_args()

    px = args.pixel_size
    if px is None:
        print("WARNING: --pixel-size not given; assuming 0.5 µm/px. The ring width "
              "in pixels depends on this — pass the real value for a faithful fit.")
        px = 0.5

    labels = _load_labels(args.labelled, args.pos_label, args.neg_label)
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    print(f"Labelled cells: {n_pos} positive ('{args.pos_label}'), "
          f"{n_neg} negative ('{args.neg_label}')")
    if n_pos < 10 or n_neg < 10:
        print("WARNING: fewer than 10 labelled cells in a class — the fit will be "
              "noisy. Aim for ~50+ per class, including faint/borderline cases.")
    if n_pos == 0 or n_neg == 0:
        sys.exit("ERROR: need both positive and negative labelled cells. Check "
                 "--pos-label / --neg-label match the class names in the GeoJSON.")

    print("Measuring cytoplasmic ring on all detections (calibrated OD)...")
    results = measure_cytoplasm_dab(
        args.image, args.labelled, px,
        expansion_um=args.expansion, keep_ring_values=True,
    )

    # Gather labelled cells that yielded a ring measurement.
    pos_frac_raw, p90_raw, ring_vals, ring_h = {}, {}, {}, {}
    used_pos, used_neg = [], []
    for idx, (lab, res) in enumerate(zip(labels, results)):
        if lab is None or not res:
            continue
        vals = res.get("ring_values")
        if not vals:
            continue
        ring_vals[idx] = np.asarray(vals, float)
        hv = res.get("ring_h_values")
        ring_h[idx] = np.asarray(hv, float) if hv else None
        p90_raw[idx] = res.get("cytoplasm_dab_p90")
        (used_pos if lab == 1 else used_neg).append(idx)

    if not used_pos or not used_neg:
        sys.exit("ERROR: no labelled cells produced a ring measurement (all "
                 "degenerate?). Check geometry / image path.")

    # --- Anchor the pixel threshold to negative-cell background ---
    if args.pix_thr is not None:
        t_pix = float(args.pix_thr)
        print(f"Pixel threshold t_pix = {t_pix:.4f} OD (user-specified)")
    else:
        neg_pixels = np.concatenate([ring_vals[i] for i in used_neg])
        t_pix = float(np.percentile(neg_pixels, args.neg_pixel_percentile))
        print(f"Pixel threshold t_pix = {t_pix:.4f} OD "
              f"(= {args.neg_pixel_percentile:.0f}th pct of {len(neg_pixels)} "
              f"negative ring pixels)")

    # --- Compute per-cell scores ---
    # Apply the SAME DAB-dominance gate as production (cell_expansion): a positive
    # membrane pixel must exceed t_pix AND be more DAB than hematoxylin, so the
    # fitted cutoffs transfer to the pipeline unchanged.
    for idx, vals in ring_vals.items():
        pos = vals > t_pix
        hv = ring_h.get(idx)
        if hv is not None and len(hv) == len(vals):
            pos = pos & (vals > hv)
        pos_frac_raw[idx] = float(pos.mean())

    idxs = used_pos + used_neg
    y = np.array([1] * len(used_pos) + [0] * len(used_neg))
    frac_scores = np.array([pos_frac_raw[i] for i in idxs])
    p90_scores = np.array([p90_raw[i] if p90_raw[i] is not None else 0.0 for i in idxs])

    print("\nPer-cell feature distributions (labelled cells):")
    _summ("membrane_pos_frac", frac_scores[y == 1], frac_scores[y == 0])
    _summ("cytoplasm_dab_p90", p90_scores[y == 1], p90_scores[y == 0])

    fitted = {"membrane_pix_thr": round(t_pix, 5)}
    for name, scores, cfg_key in [
        ("membrane_pos_frac", frac_scores, "membrane_frac_min"),
        ("cytoplasm_dab_p90", p90_scores, "membrane_p90_thr"),
    ]:
        roc = _roc(scores, y)
        print(f"\n=== {name} ===")
        if roc is None:
            print("  cannot build ROC (one class empty)")
            continue
        print(f"  AUC = {roc['auc']:.3f}  (0.5 = no separation, 1.0 = perfect)")
        best_j, at_t = _operating_points(roc, args.target_sensitivity)
        print(f"  max-Youden cutoff {name} >= {best_j['thr']:.4f}: "
              f"sens={best_j['sens']:.2f} spec={best_j['spec']:.2f}")
        if at_t:
            print(f"  @>= {args.target_sensitivity:.2f} sensitivity: "
                  f"cutoff {name} >= {at_t['thr']:.4f} "
                  f"sens={at_t['sens']:.2f} spec={at_t['spec']:.2f}")
        else:
            print(f"  no cutoff reaches {args.target_sensitivity:.2f} sensitivity")
        if roc["auc"] < 0.65:
            print(f"  ** WARNING: AUC {roc['auc']:.3f} is low — this feature does "
                  f"NOT separate the labelled classes. The membrane signal may be "
                  f"too faint to call, or labels/threshold need review.")
        # Prefer the target-sensitivity cutoff (priority = recover positives);
        # fall back to Youden if the target is unreachable.
        chosen = at_t["thr"] if at_t else best_j["thr"]
        fitted[cfg_key] = round(float(chosen), 5)

    print("\n--- Fitted cutoffs (membrane completeness classifier) ---")
    print("  Add to config to enable; p90 guard is optional (drop membrane_p90_thr")
    print("  to classify on fraction alone):")
    for k in ("membrane_pix_thr", "membrane_frac_min", "membrane_p90_thr"):
        if k in fitted:
            print(f"  {k}: {fitted[k]}")
    print("\n  Validate on held-out labelled cells before trusting these in "
          "production — fitting and reporting on the same cells overstates accuracy.")

    if args.out:
        if args.out.endswith((".yaml", ".yml")):
            with open(args.out, "w") as f:
                for k, v in fitted.items():
                    f.write(f"{k}: {v}\n")
        else:
            with open(args.out, "w") as f:
                json.dump(fitted, f, indent=2)
        print(f"\nWrote fitted cutoffs to {args.out}")


if __name__ == "__main__":
    main()
