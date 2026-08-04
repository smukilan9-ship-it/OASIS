"""
validate_native_segmenter_deepliif.py — the parity GATE for dropping QuPath.

`validate_native_segmenter.py` shows the native segmenter reproduces the model exactly and
QuPath's colour deconvolution to 0.003 OD. Neither proves the thing that actually matters: that
the numbers research/ihc.md publishes still hold. Those were all measured through QuPath.

This runs the native segmenter over the same 598 DeepLIIF images, with the same pixel size
(0.25 µm) and DAB threshold (0.2) as the recorded QuPath run, writes GeoJSON in the same shape,
and scores it with the SAME scorer (`deepliif_pipeline_validation.score`) against the same
IF-derived ground truth. The comparison is therefore like-for-like: only the segmenter changed.

Recorded QuPath baseline (validation RESULTS.md, raw InstanSeg / fixed 0.2):
    seg recall 0.752 | seg precision 0.871 | class-only F1 0.809 | class acc 0.928 | e2e F1 0.666

Run:  .venv/bin/python validation/validate_native_segmenter_deepliif.py --n 598
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.expanduser(
    "~/oasis_validation_datasets/DeepLIIF/_generated_outputs/pipeline_validation")
from oasis.common.paths import default_model_dir

# Vendored at models/ so a fresh clone can run this without QuPath ever being installed.
MODEL_DIR = default_model_dir()

PIXEL_SIZE_UM = 0.25       # the pixel size the recorded QuPath run used
DAB_THRESHOLD = 0.2

# What the QuPath-era run measured, for a like-for-like diff.
BASELINE = {"seg_recall": 0.752, "seg_precision": 0.871, "class_f1": 0.809,
            "class_acc": 0.928, "e2e_f1": 0.666}

# Tolerance: a real segmenter swap will not be bit-identical. The gate is that the published
# figures survive, not that nothing moved.
TOL = 0.03


def run_native(cond="raw_instanseg", n=None, device="cpu", out_name="_native_out"):
    """Segment every prepared input with the in-process segmenter, writing QuPath-shaped
    GeoJSON/CSV/summary into <cond>/<out_name>/."""
    import glob
    import numpy as np
    from PIL import Image
    from oasis.quant import segment as sg

    src = os.path.join(DATA, cond, "changed_inputs")
    dst = os.path.join(DATA, cond, out_name)
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.png")))
    if n:
        files = files[:n]
    t0 = time.time()
    total_cells = 0
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        rgb = np.asarray(Image.open(f).convert("RGB"))
        res = sg.segment_image(f, MODEL_DIR, PIXEL_SIZE_UM, device=device,
                               dab_threshold=DAB_THRESHOLD, rgb=rgb)
        total_cells += len(res["records"])
        sg.write_geojson(res, os.path.join(dst, stem + "_detections.geojson"), stem)
        sg.write_detections_csv(res, os.path.join(dst, stem + "_detections.csv"), stem)
        sg.write_summary(res, os.path.join(dst, stem + "_summary.json"), stem)
        if i % 50 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i}/{len(files)} images | {total_cells} cells | "
                  f"{el:.0f}s ({el / i:.2f}s/img)", flush=True)
    return {"images": len(files), "cells": total_cells, "secs": round(time.time() - t0, 1),
            "out_dir": dst}


def score_dir(cond, out_name, adaptive=False):
    """Score a directory of GeoJSON with the project's own DeepLIIF scorer, unmodified."""
    import validation.deepliif_pipeline_validation as dv
    # the harness resolves paths from its module-level BASE and always reads <cond>/_pipeline_out;
    # point both at the directory under test so the SAME scorer runs on both segmenters
    orig_base = dv.BASE
    dv.BASE = DATA
    try:
        # the harness always reads <cond>/_pipeline_out, so swap that name to the directory
        # under test rather than editing the scorer
        target = os.path.join(DATA, cond, "_pipeline_out")
        backup = os.path.join(DATA, cond, "_pipeline_out__qupath_backup")
        swapped = False
        if out_name != "_pipeline_out":
            if os.path.islink(target):
                os.unlink(target)
            elif os.path.isdir(target):
                os.rename(target, backup)
                swapped = True
            os.symlink(os.path.join(DATA, cond, out_name), target)
        try:
            dv.score(cond, adaptive=adaptive)          # prints, and writes f1/metrics.json
            with open(os.path.join(DATA, cond, "f1", "metrics.json"), encoding="utf-8") as f:
                m = json.load(f)
            m.pop("per_image", None)                   # 598 entries, not wanted in the report
            return m
        finally:
            if out_name != "_pipeline_out":
                if os.path.islink(target):
                    os.unlink(target)
                if swapped and os.path.isdir(backup):
                    os.rename(backup, target)
    finally:
        dv.BASE = orig_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="raw_instanseg")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-run", action="store_true",
                    help="score an existing _native_out without re-segmenting")
    args = ap.parse_args()

    report = {"baseline_qupath": BASELINE, "pixel_size_um": PIXEL_SIZE_UM,
              "dab_threshold": DAB_THRESHOLD, "cond": args.cond}

    if not args.skip_run:
        print(f"Segmenting {args.cond} with the native in-process InstanSeg "
              f"(px {PIXEL_SIZE_UM}, DAB {DAB_THRESHOLD})")
        report["run"] = run_native(args.cond, args.n, args.device)
        print(f"  wrote {report['run']['images']} images -> {report['run']['out_dir']}")

    print("\nScoring NATIVE segmenter against DeepLIIF IF ground truth")
    report["native"] = score_dir(args.cond, "_native_out")

    print("\nScoring the recorded QuPath outputs with the same scorer (control)")
    report["qupath"] = score_dir(args.cond, "_pipeline_out")

    # The gate: every published figure must survive the segmenter swap within TOL.
    nat, qup = report["native"], report["qupath"]
    checks = [
        ("detection recall", qup["detection_recall"], nat["detection_recall"]),
        ("detection precision", qup["detection_precision"], nat["detection_precision"]),
        ("classification-only F1", qup["classification_only"]["f1"],
         nat["classification_only"]["f1"]),
        ("classification accuracy", qup["classification_only"]["accuracy"],
         nat["classification_only"]["accuracy"]),
        ("end-to-end F1", qup["end_to_end"]["f1"], nat["end_to_end"]["f1"]),
    ]
    print(f"\nParity vs QuPath (tolerance ±{TOL}):")
    ok = True
    deltas = {}
    for name, q, n in checks:
        d = n - q
        deltas[name] = round(d, 4)
        passed = abs(d) <= TOL
        ok = ok and passed
        print(f"  {name:26s} QuPath {q:.3f} -> native {n:.3f} "
              f"({d:+.3f}) {'ok' if passed else 'OUT OF TOLERANCE'}")
    report["deltas"] = deltas
    report["pass"] = ok

    print(f"\n##METRICS## {json.dumps(report, default=str)}")
    out = os.environ.get("OASIS_REPORT_DIR")
    if out:
        with open(os.path.join(out, "native_segmenter_deepliif.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(f"\n{'PASS' if ok else 'FAIL'} — "
          f"{'QuPath can be removed' if ok else 'QuPath must stay until this passes'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
