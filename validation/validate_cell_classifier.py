"""
validate_cell_classifier.py — does the per-cohort classifier beat the fixed cutoff?

Ground truth is DeepLIIF's SegMask panel: co-registered immunofluorescence, thresholded by
the DeepLIIF authors into red (positive) / blue (negative) cells. It is not hand-labelled,
which is the point — it gives per-cell labels across hundreds of images with no annotator
in the loop, and enough images to hold out whole slides.

Each testing-set tile is a 6-panel strip; panel 0 is the IHC the pipeline sees and panel 5
is the mask it must never see. We segment panel 0 with the real InstanSeg path, measure
both stain channels, match each detected nucleus to the nearest ground-truth cell, and then
ask three questions in order:

  1. What does the shipped **fixed cutoff** score on these cells? (the baseline to beat)
  2. What does the classifier score under **leave-one-image-out**? (the honest estimate)
  3. Is the difference worth the cost of labelling?

Reporting leave-one-image-out rather than a pooled cell split is the whole point: cells
within a tile share a staining run and illumination, so a cell-wise split scores a model
whose own tile is still in training.

Run:  .venv/bin/python validation/validate_cell_classifier.py --images 80
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.common.paths import default_model_dir          # noqa: E402
from oasis.quant import classifier as C                    # noqa: E402

DATA = os.path.expanduser(
    "~/oasis_validation_datasets/DeepLIIF/inputs/DeepLIIF_Testing_Set")
PANEL = 512
FIXED_CUTOFF = 0.20          # the shipped nuclear default (research/ihc.md § 11.2)
MATCH_TOL_PX = 12.0          # centroid match radius between detection and ground truth


def gt_cells(mask_rgb):
    """Ground-truth centroids and labels from the SegMask panel: red=positive, blue=negative."""
    from scipy import ndimage
    out = []
    r = (mask_rgb[:, :, 0] > 150) & (mask_rgb[:, :, 1] < 100) & (mask_rgb[:, :, 2] < 100)
    b = (mask_rgb[:, :, 2] > 150) & (mask_rgb[:, :, 0] < 100) & (mask_rgb[:, :, 1] < 100)
    for mask, label in ((r, 1), (b, 0)):
        lab, n = ndimage.label(mask)
        if n == 0:
            continue
        for cy, cx in ndimage.center_of_mass(mask, lab, range(1, n + 1)):
            if np.isfinite(cx) and np.isfinite(cy):
                out.append(((float(cx), float(cy)), label))
    return out


def cells_for_image(path, model):
    """Segment panel 0, measure it, and pair each detection with its ground-truth label."""
    from PIL import Image
    from scipy.spatial import cKDTree
    from oasis.quant import segment as sg

    strip = np.asarray(Image.open(path).convert("RGB"))
    if strip.shape[1] < 6 * PANEL:
        return []
    ihc = strip[:, :PANEL]
    truth = gt_cells(strip[:, 5 * PANEL:6 * PANEL])
    if len(truth) < 5:
        return []

    labels = sg.segment_labels(ihc, model, "cpu")
    if int(labels.max()) == 0:
        return []
    hem, dab = sg._od_channels(ihc)
    recs = sg._measure(labels, hem, dab, 1.0)
    if not recs:
        return []

    gt_xy = np.array([t[0] for t in truth])
    gt_y = np.array([t[1] for t in truth])
    tree = cKDTree(gt_xy)

    cells = []
    for r in recs:
        m = r["measurements"]
        cx, cy = r["centroid_px"]
        d, j = tree.query([cx, cy])
        if d > MATCH_TOL_PX:
            continue                       # unmatched detection — no label, so no opinion
        cells.append({
            "centroid": (cx, cy),
            "dab_mean": m.get("DAB: Mean"),
            "dab_p90": m.get("DAB: Max"),
            "hema_mean": m.get("Hematoxylin: Mean"),
            "area_px": m.get("Area µm^2", m.get("Area")),
            "label": int(gt_y[j]),
        })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=80,
                    help="how many tiles to use (deterministic sample of the testing set)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None,
                    help="npz to write/reuse extracted features (skips segmentation)")
    ap.add_argument("--drift", type=float, default=0.0,
                    help="simulate batch staining variation: add a per-image DAB offset "
                         "drawn from N(0, drift) OD. This is the condition the classifier "
                         "is supposed to survive and a fixed cutoff is not.")
    args = ap.parse_args()

    import glob
    from oasis.quant import segment as sg

    # Segmentation dominates the runtime, so a cached feature matrix lets the *analysis*
    # be re-run under different conditions (see --drift) without re-segmenting.
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        return analyse(z["X"], z["y"], list(z["ids"]), z["dab"], args)

    files = sorted(glob.glob(os.path.join(DATA, "*.png")))
    if not files:
        print(f"No DeepLIIF tiles at {DATA}")
        return 2
    # Deterministic, evenly spread across the set so both tissue types are represented.
    if args.images < len(files):
        step = len(files) / args.images
        files = [files[int(i * step)] for i in range(args.images)]

    print(f"Loading segmenter…")
    model = sg.load_model(default_model_dir(), "cpu")

    X_rows, y_rows, ids, dab_rows = [], [], [], []
    t0 = time.time()
    for k, path in enumerate(files, 1):
        stem = os.path.basename(path)[:-4]
        try:
            cells = cells_for_image(path, model)
        except Exception as e:
            print(f"  [{k}/{len(files)}] {stem}: FAILED ({e})")
            continue
        if len(cells) < 10:
            continue
        Xi, names = C.extract_features(cells, "nuclear")
        X_rows.append(Xi)
        y_rows.append(np.array([c["label"] for c in cells]))
        dab_rows.append(np.array([c["dab_mean"] or np.nan for c in cells]))
        ids += [stem] * len(cells)
        if k % 10 == 0:
            print(f"  [{k}/{len(files)}] {len(ids)} labelled cells so far "
                  f"({time.time() - t0:.0f}s)")

    if not X_rows:
        print("No usable images.")
        return 1

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    dab = np.concatenate(dab_rows)
    if args.cache:
        np.savez_compressed(args.cache, X=X, y=y, ids=np.array(ids), dab=dab)
        print(f"cached features -> {args.cache}")
    return analyse(X, y, ids, dab, args)


def analyse(X, y, ids, dab, args):
    """Everything downstream of feature extraction, so it can be re-run from cache."""
    X = np.array(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    dab = np.asarray(dab, dtype=np.float64)
    names = C.feature_names("nuclear")

    if args.drift > 0:
        # A per-image additive OD offset is what a different staining run, antibody lot or
        # scanner actually does to this measurement. The fixed cutoff has no way to know;
        # the classifier has `dab_minus_local_bg`, which is measured against each cell's own
        # neighbours and is therefore invariant to it. This is the honest test of the claim.
        rng = np.random.default_rng(0)
        offsets = {img: rng.normal(0.0, args.drift) for img in dict.fromkeys(ids)}
        shift = np.array([offsets[i] for i in ids])
        dab = dab + shift
        for j, nm in enumerate(names):
            if nm in ("dab_mean", "dab_p90", "hema_mean"):
                X[:, j] = X[:, j] + shift
        print(f"simulated staining drift: per-image DAB offset ~ N(0, {args.drift} OD)\n")

    n_img = len(set(ids))
    print(f"\n{len(y):,} matched cells across {n_img} images "
          f"({y.mean() * 100:.1f}% positive)\n")

    # ── 1. the baseline the classifier has to beat ──
    base = C.prf1((dab > FIXED_CUTOFF).astype(int), y)
    base_auc = C.roc_auc(np.nan_to_num(dab), y)
    print(f"FIXED CUTOFF ({FIXED_CUTOFF} OD)")
    print(f"  F1 {base['f1']:.3f}  precision {base['precision']:.3f}  "
          f"recall {base['recall']:.3f}  AUC {base_auc:.3f}")

    # Best cutoff achievable in hindsight — the ceiling for any single-threshold rule.
    grid = np.linspace(0.02, 0.6, 59)
    f1s = [C.prf1((dab > t).astype(int), y)["f1"] for t in grid]
    bi = int(np.argmax(f1s))
    print(f"  best possible single cutoff: {grid[bi]:.2f} OD → F1 {f1s[bi]:.3f} "
          f"(hindsight only — not available at run time)")

    # ── 2. the honest estimate ──
    r = C.leave_one_image_out(X, y, ids, "nuclear", names)
    if not r["ok"]:
        print(f"\nLOIO failed: {r['reason']}")
        return 1
    p = r["pooled"]
    print(f"\nCLASSIFIER — leave-one-image-out ({r['n_images']} folds)")
    print(f"  F1 {p['f1']:.3f}  precision {p['precision']:.3f}  "
          f"recall {p['recall']:.3f}  AUC {p['auc']:.3f}")
    print(f"  per-fold F1: min {r['fold_f1_min']:.3f}  mean {r['fold_f1_mean']:.3f}  "
          f"max {r['fold_f1_max']:.3f}  sd {r['fold_f1_std']:.3f}")

    worst = sorted([f for f in r["folds"] if "f1" in f], key=lambda f: f["f1"])[:3]
    print("  worst folds: " + ", ".join(f"{f['image'][:22]} F1 {f['f1']:.2f}" for f in worst))

    model_all = C.fit(X, y, "nuclear", names)
    print("\n  what the model leans on (standardised weights):")
    for c in model_all.coefficient_report()[:5]:
        print(f"    {c['feature']:22s} {c['weight']:+.3f}")

    delta = p["f1"] - base["f1"]
    print(f"\nVERDICT: classifier F1 {p['f1']:.3f} vs fixed {base['f1']:.3f} "
          f"({delta:+.3f})")
    print("  " + ("classifier wins" if delta > 0.02 else
                  "no material gain — the fixed cutoff is doing the job here"))

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "cell_classifier_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_cells": int(len(y)), "n_images": n_img,
                   "positive_rate": round(float(y.mean()), 4),
                   "fixed": {**base, "auc": round(base_auc, 4),
                             "cutoff": FIXED_CUTOFF},
                   "best_possible_cutoff": {"cutoff": round(float(grid[bi]), 3),
                                            "f1": round(float(f1s[bi]), 4)},
                   "classifier_loio": r,
                   "coefficients": model_all.coefficient_report()}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
