"""
validate_segmentation.py — HARNESS for validating cell detection, positive/
negative classification, and the cytoplasm-ring membrane measurement against
MANUAL ground truth. It cannot run without human-provided annotation; run with
no arguments to print exactly what you must produce.

What it computes when given data:
  1. DETECTION agreement — match manual centroids to pipeline centroids within a
     tolerance (µm) by global nearest-neighbour; report precision / recall / F1.
  2. CLASSIFICATION agreement — on matched cells, positive/negative confusion
     matrix, sensitivity, specificity, accuracy, Cohen's kappa.
  3. COUNT agreement — total / positive counts, manual vs pipeline.
  4. (optional) CYTOPLASM vs QuPath detectionsToCells — Bland-Altman of per-cell
     DAB OD (mean bias, SD, 95% limits of agreement) when a QuPath cell-expansion
     export is supplied.

Self-test:  python validation/validate_segmentation.py --selftest
  Builds synthetic ground truth + a perturbed "pipeline" output and confirms the
  metric math is correct (no biological claim — it proves the harness works).

────────────────────────────────────────────────────────────────────────────────
WHAT THE USER MUST PRODUCE (manual ground truth — only a human can do this)
────────────────────────────────────────────────────────────────────────────────
Pick ≥3 representative images per marker (CD8 and TIM-3), ideally whole small
ROIs (e.g. 1000×1000 px) rather than whole slides, spanning sparse and dense
infiltrate.

In QuPath, on EACH chosen image/ROI:
  1. Manually mark every cell (Points tool or manually corrected detections) and
     classify each as "Positive" or "Negative" by eye (a pathologist/trained
     annotator). This is the ground truth — do NOT derive it from the pipeline.
  2. Export the manual annotation as GeoJSON:
       File ▸ Export objects as GeoJSON  →  <image>_manual.geojson
     OR a CSV with columns: x,y,label   (label ∈ {positive,negative}; x,y in the
     SAME full-resolution pixel coordinates as the image).
  3. Run the pipeline on the same image; its detections GeoJSON is
       <output_dir>/<image>_detections.geojson
  4. (optional, for the cytoplasm Bland-Altman) In QuPath run
     `detectionsToCells` (cell expansion) on the SAME detections, measure
     "Cell: DAB OD mean", and export a CSV with columns: x,y,cell_dab  (the
     pipeline's matching per-cell value is read from its GeoJSON
     measurements["Cell: DAB OD mean"]).

Then run, per image:
  python validation/validate_segmentation.py \
      --manual  <image>_manual.geojson \
      --pipeline <output_dir>/<image>_detections.geojson \
      --pixel-size 0.5 --tolerance-um 5 \
      [--qupath-cells <image>_qupath_cells.csv] \
      --out validation/seg_report_<image>.txt

Aggregate the per-image precision/recall/F1 and the classification kappa across
your chosen images to report a single, data-backed agreement number — this
REPLACES the unsupported "~90% agreement" claim.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _label_is_positive(name):
    return "positive" in str(name).lower()


def load_geojson_cells(path):
    """Return (centroids Nx2 float, labels list[bool positive]) from a QuPath
    GeoJSON of detections/annotations (Point or Polygon)."""
    with open(path) as f:
        gj = json.load(f)
    cents, labs = [], []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {}) or {}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if not coords:
            continue
        if gtype == "Point":
            xy = coords[:2]
        elif gtype == "Polygon":
            ring = np.asarray(coords[0], float)[:, :2]
            xy = ring.mean(axis=0)
        elif gtype == "MultiPolygon":
            ring = np.asarray(coords[0][0], float)[:, :2]
            xy = ring.mean(axis=0)
        else:
            continue
        props = feat.get("properties", {}) or {}
        cls = props.get("classification", {})
        name = cls.get("name", "") if isinstance(cls, dict) else cls
        cents.append([float(xy[0]), float(xy[1])])
        labs.append(_label_is_positive(name))
    return np.asarray(cents, float).reshape(-1, 2), labs


def load_csv_cells(path):
    """Return (centroids, labels) from a CSV with columns x,y,label."""
    cents, labs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            cents.append([float(row["x"]), float(row["y"])])
            labs.append(_label_is_positive(row.get("label", "")))
    return np.asarray(cents, float).reshape(-1, 2), labs


def load_cells(path):
    return (load_csv_cells if path.lower().endswith(".csv")
            else load_geojson_cells)(path)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def match_centroids(gt_xy, pred_xy, tol_px):
    """Global nearest-neighbour matching within tol_px (greedy by distance).
    Returns (matched pairs [(i_gt, j_pred)], n_tp, n_fp, n_fn)."""
    from scipy.spatial import cKDTree
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return [], 0, len(pred_xy), len(gt_xy)
    tree = cKDTree(pred_xy)
    d, j = tree.query(gt_xy, k=1)
    cand = sorted((float(d[i]), int(i), int(j[i])) for i in range(len(gt_xy)))
    used_gt, used_pred, pairs = set(), set(), []
    for dist, i, jj in cand:
        if dist > tol_px or i in used_gt or jj in used_pred:
            continue
        used_gt.add(i); used_pred.add(jj); pairs.append((i, jj))
    n_tp = len(pairs)
    return pairs, n_tp, len(pred_xy) - n_tp, len(gt_xy) - n_tp


def _kappa(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    if n == 0:
        return None
    po = (tp + tn) / n
    p_pos = ((tp + fn) / n) * ((tp + fp) / n)
    p_neg = ((tn + fp) / n) * ((tn + fn) / n)
    pe = p_pos + p_neg
    return None if pe >= 1.0 else (po - pe) / (1.0 - pe)


def detection_and_classification(gt_xy, gt_lab, pred_xy, pred_lab, tol_px):
    pairs, n_tp, n_fp, n_fn = match_centroids(gt_xy, pred_xy, tol_px)
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    rec  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # classification confusion on matched cells (positive = "disease" class)
    tp = tn = fp = fn = 0
    for i, j in pairs:
        g, p = gt_lab[i], pred_lab[j]
        if g and p:       tp += 1
        elif not g and not p: tn += 1
        elif p and not g: fp += 1
        else:             fn += 1
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    acc  = (tp + tn) / len(pairs) if pairs else None
    return {
        "detection": {"n_gt": len(gt_xy), "n_pred": len(pred_xy),
                      "tp": n_tp, "fp": n_fp, "fn": n_fn,
                      "precision": prec, "recall": rec, "f1": f1},
        "classification": {"matched": len(pairs), "tp": tp, "tn": tn,
                           "fp": fp, "fn": fn, "sensitivity": sens,
                           "specificity": spec, "accuracy": acc,
                           "cohens_kappa": _kappa(tp, tn, fp, fn)},
        "counts": {"gt_total": len(gt_xy), "pred_total": len(pred_xy),
                   "gt_positive": int(sum(gt_lab)),
                   "pred_positive": int(sum(pred_lab))},
    }


def bland_altman(a, b):
    """Bland-Altman of two per-cell measurements (a=pipeline, b=reference)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) == 0:
        return None
    diff = a - b
    bias = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if len(a) > 1 else 0.0
    return {"n": int(len(a)), "mean_bias": bias, "sd_diff": sd,
            "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
            "mean_pipeline": float(np.mean(a)), "mean_reference": float(np.mean(b))}


def cytoplasm_vs_qupath(pipeline_geojson, qupath_cells_csv, tol_px,
                        meas_key="Cell: DAB OD mean"):
    """Bland-Altman: pipeline cytoplasm/cell OD vs QuPath detectionsToCells OD,
    matched by centroid within tol_px."""
    from scipy.spatial import cKDTree
    with open(pipeline_geojson) as f:
        gj = json.load(f)
    pc, pv = [], []
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        meas = props.get("measurements", {}) or {}
        val = meas.get(meas_key)
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates")
        if val is None or not coords:
            continue
        if geom.get("type") == "Point":
            xy = coords[:2]
        else:
            ring = np.asarray(coords[0] if geom["type"] == "Polygon" else coords[0][0],
                              float)[:, :2]
            xy = ring.mean(axis=0)
        pc.append([float(xy[0]), float(xy[1])]); pv.append(float(val))
    qc, qv = [], []
    with open(qupath_cells_csv) as f:
        for row in csv.DictReader(f):
            qc.append([float(row["x"]), float(row["y"])]); qv.append(float(row["cell_dab"]))
    if not pc or not qc:
        return None
    pc, qc = np.asarray(pc, float), np.asarray(qc, float)
    tree = cKDTree(pc)
    d, j = tree.query(qc, k=1)
    a, b = [], []
    used = set()
    for i in range(len(qc)):
        if d[i] <= tol_px and int(j[i]) not in used:
            used.add(int(j[i])); a.append(pv[int(j[i])]); b.append(qv[i])
    return bland_altman(a, b)


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def format_report(res, ba=None, pixel_size=None, tol_um=None):
    L = []
    d, c, n = res["detection"], res["classification"], res["counts"]
    L.append("=" * 64)
    L.append("SEGMENTATION VALIDATION REPORT")
    L.append("=" * 64)
    if pixel_size is not None:
        L.append(f"pixel size {pixel_size} µm/px   match tolerance {tol_um} µm")
    L.append("\nDETECTION (manual vs pipeline, centroid matching)")
    L.append(f"  manual={d['n_gt']}  pipeline={d['n_pred']}  "
             f"TP={d['tp']} FP={d['fp']} FN={d['fn']}")
    L.append(f"  precision={d['precision']:.3f}  recall={d['recall']:.3f}  "
             f"F1={d['f1']:.3f}")
    L.append("\nCLASSIFICATION (positive vs negative, on matched cells)")
    L.append(f"  matched={c['matched']}  TP={c['tp']} TN={c['tn']} "
             f"FP={c['fp']} FN={c['fn']}")
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    L.append(f"  sensitivity={fmt(c['sensitivity'])}  "
             f"specificity={fmt(c['specificity'])}  accuracy={fmt(c['accuracy'])}  "
             f"kappa={fmt(c['cohens_kappa'])}")
    L.append("\nCOUNTS")
    L.append(f"  total: manual={n['gt_total']} pipeline={n['pred_total']}")
    L.append(f"  positive: manual={n['gt_positive']} pipeline={n['pred_positive']}")
    if ba:
        L.append("\nCYTOPLASM vs QuPath detectionsToCells (Bland-Altman, per-cell OD)")
        L.append(f"  n={ba['n']}  mean bias={ba['mean_bias']:+.4f}  "
                 f"SD={ba['sd_diff']:.4f}")
        L.append(f"  95% limits of agreement: [{ba['loa_lower']:+.4f}, "
                 f"{ba['loa_upper']:+.4f}]")
        L.append(f"  mean pipeline={ba['mean_pipeline']:.4f}  "
                 f"mean QuPath={ba['mean_reference']:.4f}")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (proves the harness math without any biological claim)
# ──────────────────────────────────────────────────────────────────────────────

def selftest():
    print("SELF-TEST — synthetic ground truth + perturbed pipeline output\n")
    rng = np.random.default_rng(0)
    n = 200
    gt_xy = rng.uniform(0, 1000, (n, 2))
    gt_lab = (rng.random(n) < 0.4).tolist()           # 40% positive

    # pipeline: drop 10 (FN), add 8 spurious (FP), jitter, flip 12 labels
    keep = rng.permutation(n)[:n - 10]
    pred_xy = gt_xy[keep] + rng.normal(0, 1.0, (len(keep), 2))
    pred_lab = [gt_lab[i] for i in keep]
    flip = rng.permutation(len(keep))[:12]
    for k in flip:
        pred_lab[k] = not pred_lab[k]
    spurious = rng.uniform(0, 1000, (8, 2))
    pred_xy = np.vstack([pred_xy, spurious])
    pred_lab = pred_lab + [bool(x) for x in (rng.random(8) < 0.4)]

    res = detection_and_classification(gt_xy, gt_lab, pred_xy, pred_lab, tol_px=5.0)
    print(format_report(res, pixel_size=1.0, tol_um=5.0))

    d, c = res["detection"], res["classification"]
    # Hard invariants the matcher must satisfy exactly, plus sane-range checks
    # (σ=1px jitter / random spurious points can shift a match or two, so we do
    # NOT assert exact TP/FP/FN — only that the metrics are correct and bounded).
    inv = (d["tp"] + d["fn"] == n) and (d["tp"] + d["fp"] == len(pred_xy))
    ranges = (d["precision"] > 0.90 and d["recall"] > 0.90 and d["f1"] > 0.90
              and 6 <= d["fn"] <= 14 and 6 <= d["fp"] <= 12)
    label_ok = (c["fp"] + c["fn"]) == 12          # 12 deliberate flips, exact
    # Bland-Altman self-check: a known constant bias must be recovered.
    ba = bland_altman(np.arange(50) + 0.2, np.arange(50))
    ba_ok = abs(ba["mean_bias"] - 0.2) < 1e-9 and ba["sd_diff"] < 1e-9
    ok = inv and ranges and label_ok and ba_ok
    print(f"\n  matcher invariants (TP+FN=n_gt, TP+FP=n_pred): {inv}")
    print(f"  detection metrics in range (P,R,F1>0.9; FN~10, FP~8): {ranges} "
          f"[TP={d['tp']} FP={d['fp']} FN={d['fn']}]")
    print(f"  label disagreements recovered: {c['fp']+c['fn']} (expect 12) -> "
          f"{label_ok}")
    print(f"  Bland-Altman bias recovered: {ba['mean_bias']:.4f} (expect 0.2000) "
          f"-> {ba_ok}")
    print(f"\n  SELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", help="manual ground-truth GeoJSON or CSV (x,y,label)")
    ap.add_argument("--pipeline", help="pipeline *_detections.geojson")
    ap.add_argument("--pixel-size", type=float, default=0.5)
    ap.add_argument("--tolerance-um", type=float, default=5.0)
    ap.add_argument("--qupath-cells", help="QuPath detectionsToCells CSV (x,y,cell_dab)")
    ap.add_argument("--out", help="write the report to this path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if not args.manual or not args.pipeline:
        print(__doc__)
        print("\nNO DATA PROVIDED. This is a harness — supply --manual and "
              "--pipeline (see the instructions above), or run --selftest to "
              "verify the metric math.")
        sys.exit(2)

    tol_px = args.tolerance_um / args.pixel_size
    gt_xy, gt_lab = load_cells(args.manual)
    pred_xy, pred_lab = load_cells(args.pipeline)
    res = detection_and_classification(gt_xy, gt_lab, pred_xy, pred_lab, tol_px)
    ba = (cytoplasm_vs_qupath(args.pipeline, args.qupath_cells, tol_px)
          if args.qupath_cells else None)
    report = format_report(res, ba, args.pixel_size, args.tolerance_um)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\n  report saved to {args.out}")


if __name__ == "__main__":
    main()
