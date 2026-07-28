"""
reclassify.py — apply a DAB optical-density cutoff to an already-segmented image.

Segmentation is expensive; deciding where "positive" starts is not. This module is the
seam between the two. It reads the per-cell `DAB: Mean` a completed segmentation already
wrote into the GeoJSON and re-derives the classification from a cutoff, so an operator can
move the cutoff and see the consequence without paying for segmentation again.

Two things this is deliberately careful about.

**One cutoff per cohort.** `apply_threshold` records both the value used and the cohort
default it came from. When the two differ the image carries `threshold_override`, and
every downstream reader can see that this image was measured on a different scale from
its neighbours. A per-image exception is a legitimate answer to a one-off bad stain, but
it breaks the comparability that makes a cohort a cohort, so it is written down rather
than absorbed. (research/ihc.md § 11.4.)

**Nuclear only.** Membranous markers are called on ring completeness against calibrated
cutoffs, not on a nuclear OD cut — see `cell_expansion.py`. `is_membrane_result` detects
those images so callers can show their distribution without offering a control that would
not apply. Retuning a membrane cutoff belongs in the Calibrate tab, where it is fitted
against labelled cells instead of guessed from a slider.
"""
import csv
import glob
import json
import os

import numpy as np

# What the segmenter writes per cell. The second spelling is QuPath's own export name;
# both have appeared in GeoJSONs this pipeline has produced.
DAB_KEYS = ("DAB: Mean", "Nucleus: DAB OD mean")


def read_dab_values(geojson_path):
    """Per-cell nuclear DAB OD, in GeoJSON feature order. NaN where a cell has none."""
    with open(geojson_path) as f:
        gj = json.load(f)
    out = []
    for ft in gj.get("features", []):
        m = (ft.get("properties", {}) or {}).get("measurements", {}) or {}
        v = next((m[k] for k in DAB_KEYS if isinstance(m.get(k), (int, float))), None)
        out.append(float(v) if v is not None else np.nan)
    return np.asarray(out, dtype=float)


def is_membrane_result(summary):
    """True when positivity came from the cytoplasm-ring path, not a nuclear OD cut."""
    return (summary or {}).get("measurement_compartment") == "cytoplasm"


def histogram(values, bins=60, lo=0.0, hi=None):
    """Binned DAB distribution for the review plot.

    The upper edge defaults to the 99.5th percentile rather than the maximum: a handful of
    saturated or debris cells otherwise compress the entire distribution into the first two
    bins, which is exactly the range the operator needs to see. Counts above the last edge
    are folded into `overflow` so nothing is silently dropped from the total.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"edges": [], "counts": [], "overflow": 0, "n": 0}
    if hi is None:
        hi = float(np.percentile(v, 99.5))
    if not (hi > lo):                       # degenerate (all-identical) distribution
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, int(bins) + 1)
    counts, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
    overflow = int((v > hi).sum())
    # np.clip folded the overflow into the top bin; undo that so the bin is honest.
    if overflow:
        counts[-1] = max(0, int(counts[-1]) - overflow)
    return {"edges": [round(float(e), 5) for e in edges],
            "counts": [int(c) for c in counts],
            "overflow": overflow,
            "n": int(v.size)}


def positive_fraction(values, threshold):
    """Fraction of measurable cells a cutoff would call positive."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.mean(v > float(threshold)))


def _write_csv_classification(output_dir, img_stem, features):
    """Keep the detections CSV's Classification column in step with the GeoJSON.

    Best-effort: a mismatched row count means the CSV was written by a different run, and
    rewriting it from this one would corrupt it, so it is left alone.
    """
    matches = glob.glob(os.path.join(output_dir, f"{img_stem}*_detections.csv"))
    if not matches:
        return
    try:
        with open(matches[0], newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if not rows or len(rows) - 1 != len(features):
            return
        ci = rows[0].index("Classification")
        for row, feat in zip(rows[1:], features):
            cls = ((feat.get("properties", {}) or {}).get("classification", {}) or {})
            row[ci] = cls.get("name", "")
        with open(matches[0], "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)
    except (OSError, ValueError):
        return


def apply_threshold(geojson_path, summary_path, threshold, *, cohort_threshold=None,
                    output_dir=None, img_stem=None):
    """Reclassify one image at `threshold` and write the result back.

    Updates the GeoJSON classifications, the summary JSON counts, and (when `output_dir`
    and `img_stem` are given) the detections CSV. Returns a dict describing what happened.

    `cohort_threshold` is the run-wide default. When it differs from `threshold` the
    summary gains `threshold_override`, which is how the report tells the reader this
    image was not measured on the cohort's scale.
    """
    threshold = float(threshold)
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [])

    values = read_dab_values(geojson_path)
    pos = 0
    for ft, v in zip(features, values):
        is_pos = bool(np.isfinite(v) and v > threshold)
        props = ft.setdefault("properties", {})
        props["classification"] = {"name": "Positive" if is_pos else "Negative",
                                   "color": [255, 0, 0] if is_pos else [0, 200, 0]}
        pos += int(is_pos)
    with open(geojson_path, "w") as f:
        json.dump(gj, f)

    if output_dir and img_stem:
        _write_csv_classification(output_dir, img_stem, features)

    total = len(features)
    summary = {}
    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f) or {}
        except (OSError, ValueError):
            summary = {}

    summary["dab_threshold"] = round(threshold, 5)
    summary["positive_cells"] = pos
    summary["negative_cells"] = total - pos
    summary["total_cells"] = total
    summary["positivity_pct"] = round(pos * 100.0 / total, 2) if total else 0.0
    if cohort_threshold is not None:
        summary["cohort_threshold"] = round(float(cohort_threshold), 5)
        if abs(float(cohort_threshold) - threshold) > 1e-9:
            summary["threshold_override"] = round(threshold, 5)
        else:
            summary.pop("threshold_override", None)
    if summary_path:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    return {"threshold": round(threshold, 5),
            "cohort_threshold": (round(float(cohort_threshold), 5)
                                 if cohort_threshold is not None else None),
            "override": (cohort_threshold is not None
                         and abs(float(cohort_threshold) - threshold) > 1e-9),
            "total_cells": total,
            "positive_cells": pos,
            "positivity_pct": summary["positivity_pct"]}


def apply_classifier(geojson_path, summary_path, model, cells, *, fixed_threshold,
                     abstain_band=0.10, output_dir=None, img_stem=None):
    """Classify one image with a fitted per-cohort model, or refuse and fall back.

    The model writes the *same* `classification` property the threshold path writes, so
    Quant, Spatial and batch need no changes — it is a third way to fill one contract.

    Two refusals are possible and both are recorded rather than hidden:

    **The image is out of range.** A model fitted on well-stained slides will render a
    confident verdict on a faint one unless something stops it. When the applicability gate
    rejects the image, the fixed cohort cutoff is used instead and the summary says so. On
    faint tissue a fixed cutoff is safer, not more accurate — it under-calls, where a
    trained rule finds the best split of noise and manufactures positives.

    **A cell is on the boundary.** Cells inside `abstain_band` around p=0.5 are counted and
    reported. They are still given a call so downstream counts stay complete, but the count
    of them is the honest signal that the model was unsure.
    """
    import numpy as np
    from oasis.quant import classifier as CL

    X, _names = CL.extract_features(cells, model.kind)
    ok, reason = model.applicable(X)
    if not ok:
        res = apply_threshold(geojson_path, summary_path, fixed_threshold,
                              cohort_threshold=fixed_threshold,
                              output_dir=output_dir, img_stem=img_stem)
        # `apply_threshold` re-calls every cell on NUCLEAR DAB (DAB_KEYS is nuclear), so
        # after a refusal this image was not scored on the ring at all. The summary was
        # already stamped `cytoplasm` by the ring pass that ran before this, and left that
        # way it reported a compartment and a ring rule for calls that came from a nuclear
        # cutoff — the two columns a reader looks at, both wrong, on exactly the faint
        # slides where it matters most. Correct the record to what actually happened.
        _stamp(summary_path, {"classifier_name": getattr(model, "name", None),
                              "classifier_fingerprint": model.fingerprint(),
                              "classifier_applied": False,
                              "classifier_refused_reason": reason,
                              "staining_quality": "low",
                              "measurement_compartment": "nucleus",
                              "membrane_classifier": None,
                              "membrane_pix_thr": None,
                              "membrane_frac_min": None,
                              "positivity_method": "fixed cutoff (classifier refused)"})
        return {**res, "applied": False, "reason": reason}

    labels, abstained = model.predict(X, abstain_band=abstain_band)
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [])
    pos = 0
    for ft, is_pos in zip(features, labels):
        p = bool(is_pos)
        props = ft.setdefault("properties", {})
        props["classification"] = {"name": "Positive" if p else "Negative",
                                   "color": [255, 0, 0] if p else [0, 200, 0]}
        pos += int(p)
    with open(geojson_path, "w") as f:
        json.dump(gj, f)
    if output_dir and img_stem:
        _write_csv_classification(output_dir, img_stem, features)

    total = len(features)
    _stamp(summary_path, {
        "positive_cells": pos, "negative_cells": total - pos, "total_cells": total,
        "positivity_pct": round(pos * 100.0 / total, 2) if total else 0.0,
        "classifier_name": getattr(model, "name", None),
        "classifier_fingerprint": model.fingerprint(),
        "classifier_applied": True,
        "classifier_abstained": int(np.asarray(abstained).sum()),
        "positivity_method": "trained classifier",
        # The cutoff no longer describes how cells were called; leaving a stale one in the
        # summary would let a reader attribute these counts to a threshold.
        "threshold_override": None,
    })
    return {"applied": True, "total_cells": total, "positive_cells": pos,
            "abstained": int(np.asarray(abstained).sum())}


def _stamp(summary_path, fields):
    """Merge provenance into the summary JSON, dropping keys explicitly set to None."""
    if not summary_path:
        return
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f) or {}
        except (OSError, ValueError):
            summary = {}
    for k, v in fields.items():
        if v is None:
            summary.pop(k, None)
        else:
            summary[k] = v
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
