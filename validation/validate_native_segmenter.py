"""
validate_native_segmenter.py — does the in-process InstanSeg reproduce QuPath?

This is the gate that decides whether QuPath can be dropped. Every validated number in research/ihc.md
(det-F1 0.807 vs StarDist, class-F1 0.81 on DeepLIIF, the HNSCC membrane results) was measured
through QuPath. Replacing the segmenter invalidates all of them unless the replacement agrees.

Three checks, in increasing strength:

  A. MODEL FIDELITY — run the TorchScript bundle on the reference tensor shipped inside the model
     directory and compare to its shipped reference output. This isolates "can we run the model
     at all" from "did we reproduce QuPath's wrapper", and it is exact: any mismatch means the
     preprocessing or the load is wrong, with no tissue-variability excuse.

  B. DECONVOLUTION PARITY — for a real image with an existing QuPath export, compare our DAB
     optical density inside QuPath's OWN nucleus polygons against QuPath's exported "DAB: Mean".
     Using QuPath's polygons isolates the colour maths from the segmentation, so a failure here
     is unambiguous.

  C. END-TO-END PARITY — segment the same image ourselves and compare object counts, centroid
     matching (greedy nearest-neighbour within a tolerance) and per-matched-cell DAB agreement.
     This is the one that can fail for legitimate reasons (tile seams, resampling interpolation),
     so it reports distributions rather than asserting equality.

Run:  .venv/bin/python validation/validate_native_segmenter.py
      .venv/bin/python validation/validate_native_segmenter.py --image X --geojson Y --px 0.75
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.common.paths import default_model_dir

# Vendored at models/ so a fresh clone can run this without QuPath ever being installed.
MODEL_DIR = default_model_dir()

# A real QuPath export produced by the pipeline, with its source image and the pixel size the
# run actually used (from the summary JSON beside it).
DEFAULT_IMAGE = os.path.expanduser("~/Desktop/cd8_input/LL477_CD8_x10_3.tif")
DEFAULT_GEOJSON = os.path.expanduser(
    "~/Desktop/ihc_spatial_results/LL477_CD8_x10_3__roi0/"
    "LL477_CD8_x10_3.tif - LL477_CD8_x10_3.tif #1_detections.geojson")
DEFAULT_PX = 0.7518796992481203
DEFAULT_DAB_THR = 0.2


def check_a_model_fidelity():
    """Exact reproduction of the model's own reference output."""
    import torch
    from oasis.quant import segment as sg

    x = np.load(os.path.join(MODEL_DIR, "test-input.npy"))
    expected = np.load(os.path.join(MODEL_DIR, "test-output_instance_segmentation.npy")).squeeze()
    model = sg.load_model(MODEL_DIR, "cpu")
    # feed through the module's own normalisation so this tests OUR preprocessing, not a
    # bespoke one written for the test
    rgb = x[0].transpose(1, 2, 0)
    with torch.no_grad():
        y = model(torch.from_numpy(sg._normalize(rgb)))
    y = (y[0] if isinstance(y, (tuple, list)) else y).cpu().numpy().squeeze()

    n_got = len(np.unique(y)) - 1
    n_exp = len(np.unique(expected)) - 1
    exact = bool(np.array_equal(y, expected))
    fg_y, fg_e = y > 0, expected > 0
    iou = float((fg_y & fg_e).sum()) / max(float((fg_y | fg_e).sum()), 1.0)
    print(f"  labels {n_got} (expected {n_exp}) | exact match {exact} | foreground IoU {iou:.4f}")
    return {"pass": exact, "n_labels": n_got, "n_expected": n_exp, "exact": exact, "iou": iou}


def _qupath_features(geojson_path):
    with open(geojson_path) as f:
        return json.load(f)["features"]


def _feature_polygon(feat):
    g = feat.get("geometry") or {}
    if g.get("type") != "Polygon":
        return None
    ring = g["coordinates"][0]
    return np.asarray(ring, dtype=np.float64)


def check_b_deconvolution(image_path, geojson_path, limit=1500):
    """Our DAB OD vs QuPath's 'DAB: Mean', measured inside QuPath's own polygons."""
    import matplotlib.path as mpath
    from oasis.quant import segment as sg
    from oasis.quant.cell_expansion import _load_rgb_full

    rgb = _load_rgb_full(image_path)
    _hem, dab = sg._od_channels(rgb)
    ours, theirs = [], []
    for feat in _qupath_features(geojson_path):
        qm = (feat.get("properties", {}).get("measurements", {}) or {}).get("DAB: Mean")
        if not isinstance(qm, (int, float)):
            continue
        p = _feature_polygon(feat)
        if p is None:
            continue
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, dab.shape[1]), min(y1, dab.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = mpath.Path(p).contains_points(
            np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(yy.shape)
        if inside.sum() < 3:
            continue
        ours.append(float(dab[y0:y1, x0:x1][inside].mean()))
        theirs.append(float(qm))
        if len(ours) >= limit:
            break
    a, b = np.asarray(ours), np.asarray(theirs)
    corr = float(np.corrcoef(a, b)[0, 1])
    slope, intercept = np.polyfit(a, b, 1)
    mae = float(np.mean(np.abs(a - b)))
    ok = corr >= 0.99 and abs(slope - 1.0) <= 0.05 and mae <= 0.01
    print(f"  n={len(a)} corr={corr:.4f} slope={slope:.4f} intercept={intercept:.4f} "
          f"MAE={mae:.4f} OD -> {'PASS' if ok else 'FAIL'}")
    return {"pass": ok, "n": len(a), "corr": corr, "slope": float(slope),
            "intercept": float(intercept), "mae_od": mae}


def _greedy_match(ref_xy, our_xy, tol_px):
    """Greedy nearest-neighbour matching within tol_px. Returns (idx_ref, idx_our) pairs."""
    from scipy.spatial import cKDTree
    if len(ref_xy) == 0 or len(our_xy) == 0:
        return np.empty((0,), int), np.empty((0,), int)
    tree = cKDTree(our_xy)
    dist, idx = tree.query(ref_xy, k=1, distance_upper_bound=tol_px)
    used, pairs = set(), []
    for i in np.argsort(dist):
        d, j = dist[i], idx[i]
        if not np.isfinite(d) or j in used:
            continue
        used.add(j)
        pairs.append((i, j))
    if not pairs:
        return np.empty((0,), int), np.empty((0,), int)
    p = np.asarray(pairs)
    return p[:, 0], p[:, 1]


def check_c_end_to_end(image_path, geojson_path, px_um, dab_thr, device="cpu", tol_um=5.0):
    """Full native segmentation vs the QuPath export: counts, centroid match rate, DAB agreement."""
    from oasis.quant import segment as sg

    feats = _qupath_features(geojson_path)
    q_xy, q_dab, q_cls = [], [], []
    for feat in feats:
        p = _feature_polygon(feat)
        if p is None:
            continue
        q_xy.append(p[:-1].mean(0))
        meas = feat.get("properties", {}).get("measurements", {}) or {}
        q_dab.append(meas.get("DAB: Mean"))
        cls = (feat.get("properties", {}).get("classification") or {}).get("name", "")
        q_cls.append(cls)
    q_xy = np.asarray(q_xy)

    t0 = time.time()
    res = sg.segment_image(image_path, MODEL_DIR, px_um, device=device, dab_threshold=dab_thr)
    secs = time.time() - t0
    o_xy = np.asarray([r["centroid_px"] for r in res["records"]])
    o_dab = np.asarray([r["measurements"]["DAB: Mean"] for r in res["records"]])
    o_cls = [r["classification"] for r in res["records"]]

    tol_px = tol_um / px_um
    mi, mj = _greedy_match(q_xy, o_xy, tol_px)
    n_q, n_o, n_m = len(q_xy), len(o_xy), len(mi)
    recall = n_m / n_q if n_q else 0.0
    precision = n_m / n_o if n_o else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    count_ratio = n_o / n_q if n_q else 0.0

    dab_mae = dab_corr = None
    cls_agree = None
    if n_m:
        qa = np.asarray([q_dab[i] for i in mi], dtype=np.float64)
        oa = o_dab[mj]
        good = np.isfinite(qa) & np.isfinite(oa)
        if good.sum() > 10:
            dab_mae = float(np.mean(np.abs(qa[good] - oa[good])))
            dab_corr = float(np.corrcoef(qa[good], oa[good])[0, 1])
        cls_agree = float(np.mean([q_cls[i] == o_cls[j] for i, j in zip(mi, mj)]))

    print(f"  QuPath {n_q} cells | native {n_o} cells (ratio {count_ratio:.3f}) in {secs:.0f}s "
          f"| downsample {res['downsample']:.3f}")
    print(f"  centroid match @{tol_um}µm: recall {recall:.3f} precision {precision:.3f} "
          f"det-F1 {f1:.3f}")
    if dab_mae is not None:
        print(f"  matched-cell DAB: corr {dab_corr:.4f} MAE {dab_mae:.4f} OD | "
              f"class agreement {cls_agree:.3f}")
    return {"n_qupath": n_q, "n_native": n_o, "count_ratio": count_ratio,
            "det_f1": f1, "recall": recall, "precision": precision,
            "dab_mae_od": dab_mae, "dab_corr": dab_corr, "class_agreement": cls_agree,
            "downsample": res["downsample"], "secs": round(secs, 1)}


def check_d_tiling(image_path, px_um, device="cpu", tile=256, tol_px=3.0):
    """Does tiling change the answer?

    The DeepLIIF parity gate could not test this: its 512 px panels at 0.25 µm/px resample to
    256 px, which is ONE tile. So the seam logic — context padding plus centroid-in-core
    ownership — was never exercised by the numbers that cleared QuPath.

    Here the SAME image is segmented twice: once whole (a single tile, no seams anywhere) and
    once with a deliberately small tile so seams cut through the middle of the tissue. A correct
    implementation gives nearly the same objects. Two failure modes are checked by name:

      DUPLICATES  — an object emitted by two tiles appears as two detections a few pixels apart.
                    Counted directly as near-coincident pairs in the tiled result.
      SEAM LOSS   — an object dropped by both tiles. Would show as whole-only objects whose
                    centroids sit on a seam line, so mismatches are reported by distance to the
                    nearest seam.
    """
    import numpy as np
    from oasis.quant import segment as sg
    from oasis.quant.cell_expansion import _load_rgb_full
    from scipy.spatial import cKDTree

    rgb = _load_rgb_full(os.path.expanduser(image_path))
    h, w = rgb.shape[:2]
    model = sg.load_model(MODEL_DIR, device)

    whole = sg.segment_labels(rgb, model, device=device, tile=max(h, w) + 64)
    tiled = sg.segment_labels(rgb, model, device=device, tile=tile)

    def _centroids(lab):
        from scipy import ndimage
        n = int(lab.max())
        if n == 0:
            return np.zeros((0, 2))
        c = ndimage.center_of_mass(lab > 0, lab, range(1, n + 1))
        return np.asarray([(x, y) for y, x in c])

    cw, ct = _centroids(whole), _centroids(tiled)

    # duplicates: near-coincident pairs within the TILED result
    dups = 0
    if len(ct) > 1:
        tree = cKDTree(ct)
        dups = sum(1 for pair in tree.query_pairs(r=tol_px))

    mi, mj = _greedy_match(cw, ct, tol_px)
    recall = len(mi) / len(cw) if len(cw) else 0.0
    precision = len(mi) / len(ct) if len(ct) else 0.0

    # how far are the UNMATCHED whole-image objects from a seam? if seam loss is the cause,
    # they cluster at distance ~0; if it is ordinary edge-context variation, they scatter.
    seam_dist = None
    unmatched = np.setdiff1d(np.arange(len(cw)), mi)
    if len(unmatched):
        xs = cw[unmatched, 0] % tile
        ys = cw[unmatched, 1] % tile
        d = np.minimum(np.minimum(xs, tile - xs), np.minimum(ys, tile - ys))
        seam_dist = {"median_px": float(np.median(d)),
                     "frac_within_8px_of_seam": float(np.mean(d <= 8))}
    # baseline: what fraction of ALL objects lie within 8 px of a seam? if the unmatched are
    # not enriched above this, seams are not the explanation.
    xs, ys = cw[:, 0] % tile, cw[:, 1] % tile
    dall = np.minimum(np.minimum(xs, tile - xs), np.minimum(ys, tile - ys))
    base_frac = float(np.mean(dall <= 8)) if len(cw) else 0.0

    n_tiles = len(sg._tile_grid(h, w, tile))
    ok = dups == 0 and recall >= 0.97 and precision >= 0.97
    print(f"  whole-image {len(cw)} objects (1 tile) vs tiled {len(ct)} ({n_tiles} tiles of {tile})")
    print(f"  agreement: recall {recall:.4f} precision {precision:.4f} | "
          f"duplicate pairs in tiled result: {dups}")
    if seam_dist:
        print(f"  unmatched whole-image objects: {len(unmatched)} | "
              f"{seam_dist['frac_within_8px_of_seam']:.3f} within 8px of a seam "
              f"(baseline for all objects: {base_frac:.3f})")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return {"pass": ok, "n_whole": len(cw), "n_tiled": len(ct), "n_tiles": n_tiles,
            "recall": recall, "precision": precision, "duplicate_pairs": dups,
            "unmatched_seam_profile": seam_dist, "seam_baseline_frac": base_frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--geojson", default=DEFAULT_GEOJSON)
    ap.add_argument("--px", type=float, default=DEFAULT_PX)
    ap.add_argument("--dab-threshold", type=float, default=DEFAULT_DAB_THR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--skip-tiling", action="store_true")
    args = ap.parse_args()

    report = {}
    print("A. model fidelity (TorchScript vs the model's shipped reference output)")
    report["model_fidelity"] = check_a_model_fidelity()

    have_ref = os.path.exists(args.image) and os.path.exists(args.geojson)
    if have_ref:
        print("\nB. colour deconvolution parity (our DAB OD inside QuPath's own polygons)")
        report["deconvolution"] = check_b_deconvolution(args.image, args.geojson)
        if not args.skip_e2e:
            print("\nC. end-to-end parity (native segmentation vs the QuPath export)")
            report["end_to_end"] = check_c_end_to_end(
                args.image, args.geojson, args.px, args.dab_threshold, args.device)
    else:
        print("\nB/C skipped — no reference QuPath export at the given paths.")

    if os.path.exists(args.image) and not args.skip_tiling:
        print("\nD. tiling / seam handling (whole-image vs forced small tiles)")
        report["tiling"] = check_d_tiling(args.image, args.px, args.device)

    ok = (report["model_fidelity"]["pass"]
          and report.get("deconvolution", {"pass": True})["pass"]
          and report.get("tiling", {"pass": True})["pass"])
    print(f"\n##METRICS## {json.dumps(report, default=str)}")
    out = os.environ.get("OASIS_REPORT_DIR")
    if out:
        with open(os.path.join(out, "native_segmenter_parity.json"), "w") as f:
            json.dump(report, f, indent=2)
    print(f"\n{'PASS' if ok else 'FAIL'} (A, B, D are assertive; C is reported)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
