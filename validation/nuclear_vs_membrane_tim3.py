#!/usr/bin/env python3
"""Does measuring the NUCLEUS work for a membranous marker, if the staining is good?

The question is not rhetorical. Measuring nuclear DAB for a surface marker reads the wrong
compartment, but strong membrane staining bleeds over the nuclear mask, so nuclear OD can
track the truth by side effect. If it tracked it well enough, the whole membranous path --
the ring, the Voronoi clipping, the completeness features, the labelling burden -- would be
machinery for nothing. Nobody had measured it.

Five arms, all on the SAME hand-labelled cells and the SAME folds:

    1. nuclear cutoff        one threshold on nuclear DAB OD          (what you get with
                                                                       Membranous off)
    2. ring-mean cutoff      one threshold on mean ring DAB OD        (the pre-2026-07 rule)
    3. completeness cutoff   one threshold on membrane_pos_frac       (the ring rule)
    4. nuclear classifier    6 nuclear features, logistic
    5. membrane classifier   9 ring features, logistic

Held out by IMAGE, never by cell: cells within a slide share a staining run and an
illumination field, so a cell-wise split scores a model that has already seen the slide.
Single-feature arms need no fitting for AUC, but their F1 does -- the threshold is chosen
on the training images and applied to the held-out one, so every arm is scored the same way.

Ground truth: 281 positive / 318 negative cells hand-labelled across four CRC-ICM TIM-3
fields (validation/make_tim3_label_tool.py). One of the four, 92290_TIM3_IM, is visibly
faint and is the known failure case; it is reported separately rather than averaged away.

Usage:
  .venv/bin/python validation/nuclear_vs_membrane_tim3.py \
      --labels ~/Downloads --images ~/oasis_validation_datasets/TIM3_CRC_ICM/inputs/labeling/_seg_in
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.quant import classifier as CL          # noqa: E402
from oasis.webui import calibration               # noqa: E402

POS_LABEL, NEG_LABEL = "membrane_pos", "membrane_neg"


def read_labels(geojson_path):
    """(pos_idx, neg_idx) by feature order."""
    gj = json.load(open(geojson_path))
    pos, neg = [], []
    for i, ft in enumerate(gj.get("features", [])):
        cl = (ft.get("properties", {}) or {}).get("classification") or {}
        name = cl.get("name") if isinstance(cl, dict) else cl
        if name == POS_LABEL:
            pos.append(i)
        elif name == NEG_LABEL:
            neg.append(i)
    return pos, neg


def roc_auc(score, y):
    return CL.roc_auc(np.asarray(score, float), np.asarray(y, int))


def best_f1_threshold(score, y):
    """The threshold maximising F1 on this (training) set."""
    score = np.asarray(score, float)
    y = np.asarray(y, int)
    finite = np.isfinite(score)
    if not finite.any():
        return 0.0
    best_t, best = float(np.nanmin(score)), -1.0
    for t in np.unique(score[finite]):
        f = CL.prf1((score > t).astype(int), y)["f1"]
        if f > best:
            best, best_t = f, float(t)
    return best_t


def loio_single_feature(score, y, ids):
    """Leave-one-image-out for a one-number rule.

    AUC needs no training, so the pooled AUC is the full-sample AUC. F1 does: the cutoff is
    chosen on the training images only, which is what makes it comparable to the classifier
    arms rather than an optimistic in-sample best.
    """
    score, y, ids = np.asarray(score, float), np.asarray(y, int), np.asarray(ids)
    pred = np.zeros(len(y), int)
    folds = []
    for img in dict.fromkeys(ids.tolist()):
        te, tr = ids == img, ids != img
        t = best_f1_threshold(score[tr], y[tr])
        pred[te] = (score[te] > t).astype(int)
        folds.append({"image": str(img), "n": int(te.sum()),
                      "n_positive": int(y[te].sum()), "threshold": round(t, 5),
                      "auc": (round(a, 4) if (a := roc_auc(score[te], y[te])) is not None else None),
                      **CL.prf1(pred[te], y[te])})
    pooled = CL.prf1(pred, y)
    return {"ok": True, "pooled": {**pooled, "auc": round(roc_auc(score, y), 4)},
            "folds": folds}


def measure_ring_values(lab_dir, img_dir, pixel_size):
    """Per-labelled-cell (image, label, ring OD values, ring H values).

    `calibration._measure_labeled` keeps the per-pixel ring array, so a completeness
    fraction can be recomputed at any pixel threshold without re-measuring the image.
    """
    out = []
    for gj in sorted(lab_dir.glob("*_labelled*.geojson")):
        stem = gj.name.split("_labelled")[0]
        img = next((p for p in img_dir.glob(f"{stem}.*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")), None)
        if img is None:
            continue
        pos, neg = read_labels(gj)
        if not pos or not neg:
            continue
        for lab, rv, rh in calibration._measure_labeled(str(img), str(gj), pixel_size, pos, neg):
            out.append((stem, lab, rv, rh))
    return out


def loio_fitted_pix_thr(labelled, neg_pct=99.0):
    """Completeness with the pixel threshold fitted on the TRAINING images' negatives."""
    ids = np.array([r[0] for r in labelled])
    y = np.array([r[1] for r in labelled], int)
    cells = [(r[1], r[2], r[3]) for r in labelled]
    pred = np.zeros(len(y), int)
    frac_held = np.full(len(y), np.nan)
    folds = []
    for img in dict.fromkeys(ids.tolist()):
        te, tr = ids == img, ids != img
        train = [c for c, keep in zip(cells, tr) if keep]
        test = [c for c, keep in zip(cells, te) if keep]
        t_pix = calibration._neg_t_pix(train, neg_pct)
        cut, _ = calibration._best_f1_cut(calibration._ring_frac(train, t_pix), y[tr])
        f_te = calibration._ring_frac(test, t_pix)
        frac_held[te] = f_te
        pred[te] = (f_te > cut).astype(int)
        folds.append({"image": str(img), "n": int(te.sum()),
                      "n_positive": int(y[te].sum()),
                      "pix_thr": round(float(t_pix), 5), "frac_cut": round(float(cut), 5),
                      "auc": (round(a, 4) if (a := roc_auc(f_te, y[te])) is not None else None),
                      **CL.prf1(pred[te], y[te])})
    return {"ok": True,
            "pooled": {**CL.prf1(pred, y), "auc": round(roc_auc(frac_held, y), 4)},
            "folds": folds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="~/Downloads")
    ap.add_argument("--images",
                    default="~/oasis_validation_datasets/TIM3_CRC_ICM/inputs/labeling/_seg_in")
    ap.add_argument("--pixel-size", type=float, default=0.5)
    ap.add_argument("--out", default="validation/nuclear_vs_membrane_tim3_results.json")
    args = ap.parse_args()

    lab_dir = Path(os.path.expanduser(args.labels))
    img_dir = Path(os.path.expanduser(args.images))

    cells_all, ids_all = [], []
    per_image = []
    for gj in sorted(lab_dir.glob("*_labelled*.geojson")):
        stem = gj.name.split("_labelled")[0]
        img = next((p for p in img_dir.glob(f"{stem}.*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")), None)
        if img is None:
            print(f"  ! no image for {stem}, skipping")
            continue
        pos, neg = read_labels(gj)
        if not pos or not neg:
            print(f"  ! {stem} has only one class, skipping")
            continue
        print(f"\n== {stem}: {len(pos)} positive / {len(neg)} negative")
        # kind='membrane' returns the nuclear measurements too, so one measurement pass
        # feeds both arms and neither can be scored on a different set of cells.
        cells = calibration.cells_for_classifier(str(img), str(gj), args.pixel_size,
                                                 pos, neg, "membrane")
        cells_all.extend(cells)
        ids_all.extend([stem] * len(cells))
        per_image.append({"image": stem, "n_pos": len(pos), "n_neg": len(neg),
                          "n_measured": len(cells)})

    if len(set(ids_all)) < 3:
        print("Need at least 3 labelled images."), sys.exit(1)

    y = np.array([c["label"] for c in cells_all], int)
    ids = np.array(ids_all)
    Xn, names_n = CL.extract_features(cells_all, "nuclear")
    Xm, names_m = CL.extract_features(cells_all, "membrane")

    def col(names, X, nm):
        return X[:, names.index(nm)]

    arms = {
        "1. nuclear cutoff (Membranous OFF)":
            loio_single_feature(col(names_n, Xn, "dab_mean"), y, ids),
        "2. ring-mean cutoff":
            loio_single_feature(col(names_m, Xm, "ring_mean"), y, ids),
        "3. completeness, auto pixel threshold":
            loio_single_feature(col(names_m, Xm, "membrane_pos_frac"), y, ids),
        "4. nuclear classifier (6 features)":
            CL.leave_one_image_out(Xn, y, ids, "nuclear", names_n),
        "5. membrane classifier (9 features)":
            CL.leave_one_image_out(Xm, y, ids, "membrane", names_m),
    }

    # What does the self-calibrating pixel threshold COST? Arm 3 derives "stained" from the
    # image's own ring pixels, which is the only thing possible on an unlabelled slide.
    # This arm derives it the way the original tuner did -- the 99th percentile of the
    # NEGATIVE cells' ring pixels -- fitted on the training images only. It cannot ship
    # (it needs labels for the image being scored's cohort), but the gap between the two is
    # the price of being able to run at all, and it should be stated rather than assumed.
    labelled = measure_ring_values(lab_dir, img_dir, args.pixel_size)
    if labelled:
        arms["6. completeness, threshold fitted on labels"] = loio_fitted_pix_thr(labelled)

    print(f"\n{'='*78}\n  HELD-OUT BY IMAGE — {len(y)} cells, {int(y.sum())} positive, "
          f"{len(set(ids_all))} images\n{'='*78}")
    print(f"  {'arm':38} {'AUC':>6} {'F1':>6} {'prec':>6} {'rec':>6}")
    for name, r in arms.items():
        if not r.get("ok"):
            print(f"  {name:38} FAILED: {r.get('reason')}")
            continue
        p = r["pooled"]
        print(f"  {name:38} {p['auc']:>6.3f} {p['f1']:>6.3f} "
              f"{p['precision']:>6.3f} {p['recall']:>6.3f}")

    print(f"\n  Per-image held-out F1 (the spread is the finding):")
    imgs = list(dict.fromkeys(ids_all))
    print(f"  {'arm':38} " + " ".join(f"{i[:11]:>12}" for i in imgs))
    for name, r in arms.items():
        if not r.get("ok"):
            continue
        by = {f["image"]: f for f in r.get("folds", [])}
        print(f"  {name:38} " + " ".join(
            f"{by[i]['f1']:>12.3f}" if i in by and "f1" in by[i] else f"{'—':>12}"
            for i in imgs))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_cells": int(len(y)), "n_positive": int(y.sum()),
         "images": per_image, "pixel_size_um": args.pixel_size,
         "arms": arms}, indent=2))
    print(f"\n  Written: {out}")


if __name__ == "__main__":
    main()
