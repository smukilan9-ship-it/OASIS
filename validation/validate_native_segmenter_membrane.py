"""
validate_native_segmenter_membrane.py — does the membrane/cytoplasm path survive the swap?

`cell_expansion.measure_cytoplasm_dab` is the membranous (CD8/TIM-3) measurement path. It has a
hard dependency on the segmenter's output that the nuclear parity gate does not exercise: it
recomputes DAB with per-image Macenko stain vectors and then CALIBRATES that channel against the
"DAB: Mean" the GeoJSON carries, refusing to run with fewer than 50 references and failing a
parity gate (r >= 0.90, slope > 0, MAE <= 0.015 OD) if the fit is poor. In the QuPath era that
reference came from QuPath.

So there are two questions, and they are different:

  1. Does the path RUN on native GeoJSON at all — is the anchor satisfied and does the internal
     parity gate pass? (A hard failure here blocks the swap outright.)
  2. Do the membrane MEASUREMENTS agree cell-for-cell between a QuPath-derived and a
     native-derived GeoJSON for the same image?

Note what question 2 can and cannot show. The two runs segment the same tissue slightly
differently, so a matched cell's ring is not pixel-identical; some spread is expected and is not
evidence of a defect. What would be evidence of a defect is a systematic shift — a different
calibration slope, or a membrane-positive fraction that moves the population.

The HNSCC membranous validation (validate_membrane_cd8_hnscc.py, held-out F1 0.76 / AUC 0.89) is
deliberately NOT re-run here: it drives the method from EXPERT nuclear masks, not from our
segmenter, so it is unaffected by this change.

Run:  .venv/bin/python validation/validate_native_segmenter_membrane.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.common.paths import default_model_dir

# Vendored at models/ so a fresh clone can run this without QuPath ever being installed.
MODEL_DIR = default_model_dir()
IMAGE = os.path.expanduser("~/Desktop/cd8_input/LL477_CD8_x10_3.tif")
QUPATH_GEOJSON = os.path.expanduser(
    "~/Desktop/ihc_spatial_results/LL477_CD8_x10_3__roi0/"
    "LL477_CD8_x10_3.tif - LL477_CD8_x10_3.tif #1_detections.geojson")
PX = 0.7518796992481203
EXPANSION_UM = 2.0


def _run_membrane(image_path, geojson_path, px, membrane_pix_thr):
    from oasis.quant.cell_expansion import measure_cytoplasm_dab
    return measure_cytoplasm_dab(image_path, geojson_path, px,
                                 expansion_um=EXPANSION_UM,
                                 membrane_pix_thr=membrane_pix_thr)


def _summary(rows, label):
    cyt = np.asarray([r["cytoplasm_dab_mean"] for r in rows
                      if r and r.get("cytoplasm_dab_mean") is not None])
    frac = np.asarray([r["membrane_pos_frac"] for r in rows
                       if r and r.get("membrane_pos_frac") is not None])
    print(f"  {label}: {len(cyt)} measured cells | ring DAB median {np.median(cyt):.4f} "
          f"p90 {np.percentile(cyt, 90):.4f}")
    if len(frac):
        print(f"    membrane_pos_frac: median {np.median(frac):.3f} "
              f"mean {frac.mean():.3f} | frac>0.5: {np.mean(frac > 0.5):.3f}")
    return {"n": int(len(cyt)), "ring_median": float(np.median(cyt)),
            "ring_p90": float(np.percentile(cyt, 90)),
            "memfrac_median": float(np.median(frac)) if len(frac) else None,
            "memfrac_mean": float(frac.mean()) if len(frac) else None,
            "memfrac_gt50": float(np.mean(frac > 0.5)) if len(frac) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--geojson", default=QUPATH_GEOJSON)
    ap.add_argument("--px", type=float, default=PX)
    ap.add_argument("--membrane-thr", type=float, default=0.15)
    args = ap.parse_args()

    from oasis.quant import segment as sg
    from validation.validate_native_segmenter import _greedy_match, _feature_polygon

    report = {}

    # 1. produce a native GeoJSON for the same image
    print("Segmenting natively and writing GeoJSON")
    res = sg.segment_image(args.image, MODEL_DIR, args.px, dab_threshold=0.2)
    native_geojson = os.path.join(
        os.environ.get("OASIS_REPORT_DIR", "/tmp"), "native_membrane_test.geojson")
    sg.write_geojson(res, native_geojson, os.path.basename(args.image))
    print(f"  {len(res['records'])} cells -> {native_geojson}")

    # 2. does the path RUN on native output? (the blocking question)
    print("\n1. membrane path on NATIVE GeoJSON (anchor + internal parity gate)")
    try:
        native_rows = _run_membrane(args.image, native_geojson, args.px, args.membrane_thr)
        report["native_runs"] = True
        report["native"] = _summary(native_rows, "native")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        report["native_runs"] = False
        report["native_error"] = str(e)
        print("\n##METRICS## " + json.dumps(report, default=str))
        print("\nFAIL — the membrane path cannot consume native segmenter output")
        return 1

    print("\n2. membrane path on QuPath GeoJSON (control)")
    qupath_rows = _run_membrane(args.image, args.geojson, args.px, args.membrane_thr)
    report["qupath"] = _summary(qupath_rows, "QuPath")

    # 3. cell-for-cell agreement on matched cells
    print("\n3. matched-cell agreement")
    q_xy, q_val, q_frac = [], [], []
    for row in qupath_rows:
        if row and row.get("centroid") and row.get("cytoplasm_dab_mean") is not None:
            q_xy.append(row["centroid"])
            q_val.append(row["cytoplasm_dab_mean"])
            q_frac.append(row.get("membrane_pos_frac"))
    n_xy, n_val, n_frac = [], [], []
    for row in native_rows:
        if row and row.get("centroid") and row.get("cytoplasm_dab_mean") is not None:
            n_xy.append(row["centroid"])
            n_val.append(row["cytoplasm_dab_mean"])
            n_frac.append(row.get("membrane_pos_frac"))
    q_xy, n_xy = np.asarray(q_xy), np.asarray(n_xy)
    mi, mj = _greedy_match(q_xy, n_xy, 5.0 / args.px)
    qa = np.asarray([q_val[i] for i in mi])
    na = np.asarray([n_val[j] for j in mj])
    corr = float(np.corrcoef(qa, na)[0, 1]) if len(qa) > 10 else None
    mae = float(np.mean(np.abs(qa - na))) if len(qa) else None
    slope = float(np.polyfit(qa, na, 1)[0]) if len(qa) > 10 else None
    print(f"  matched {len(mi)} cells | ring DAB corr {corr:.4f} slope {slope:.4f} "
          f"MAE {mae:.4f} OD")
    fa = np.asarray([q_frac[i] for i in mi if q_frac[i] is not None], dtype=float)
    fb = np.asarray([n_frac[j] for i, j in zip(mi, mj) if q_frac[i] is not None], dtype=float)
    frac_corr = frac_mae = None
    if len(fa) > 10:
        frac_corr = float(np.corrcoef(fa, fb)[0, 1])
        frac_mae = float(np.mean(np.abs(fa - fb)))
        print(f"  membrane_pos_frac: corr {frac_corr:.4f} MAE {frac_mae:.4f}")
    report["matched"] = {"n": int(len(mi)), "ring_corr": corr, "ring_slope": slope,
                         "ring_mae_od": mae, "memfrac_corr": frac_corr,
                         "memfrac_mae": frac_mae}

    # The population-level check is the one that matters for a marker call: the median ring OD
    # and the membrane-positive rate must not shift materially.
    pop_shift = abs(report["native"]["ring_median"] - report["qupath"]["ring_median"])
    frac_shift = (abs(report["native"]["memfrac_gt50"] - report["qupath"]["memfrac_gt50"])
                  if report["native"]["memfrac_gt50"] is not None else 0.0)
    ok = report["native_runs"] and pop_shift <= 0.01 and frac_shift <= 0.05
    print(f"\n  population shift: ring-DAB median Δ {pop_shift:.4f} OD (limit 0.01), "
          f"membrane-positive rate Δ {frac_shift:.3f} (limit 0.05)")
    report["pass"] = ok

    print(f"\n##METRICS## {json.dumps(report, default=str)}")
    out = os.environ.get("OASIS_REPORT_DIR")
    if out:
        with open(os.path.join(out, "native_segmenter_membrane.json"), "w") as f:
            json.dump(report, f, indent=2)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
