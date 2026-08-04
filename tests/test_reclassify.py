"""
Tier 1 — applying a DAB cutoff to an already-segmented image (quant.reclassify).
Synthetic GeoJSON/summary on disk; no images and no segmenter needed.
"""
import json

import numpy as np
import pytest

from oasis.quant import reclassify as R


def _write_case(tmp_path, values, stem="slide_a", compartment=None):
    """A minimal GeoJSON + summary pair shaped like the ones the pipeline writes."""
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [float(i), 0.0]},
         "properties": {"measurements": ({"DAB: Mean": float(v)}
                                         if v is not None else {})}}
        for i, v in enumerate(values)]}
    geo = tmp_path / f"{stem}_detections.geojson"
    geo.write_text(json.dumps(gj), encoding="utf-8")
    summ = tmp_path / f"{stem}_summary.json"
    payload = {"total_cells": len(values)}
    if compartment:
        payload["measurement_compartment"] = compartment
    summ.write_text(json.dumps(payload), encoding="utf-8")
    return str(geo), str(summ)


def test_read_dab_values_preserves_order_and_marks_missing(tmp_path):
    geo, _ = _write_case(tmp_path, [0.05, None, 0.42])
    vals = R.read_dab_values(geo)
    assert vals[0] == pytest.approx(0.05)
    assert np.isnan(vals[1])
    assert vals[2] == pytest.approx(0.42)


def test_threshold_is_strict_greater_than(tmp_path):
    """A cell sitting exactly on the cutoff is negative, matching `dab > threshold`
    everywhere else in the pipeline. Off-by-one here would silently shift every count."""
    geo, summ = _write_case(tmp_path, [0.20, 0.2000001])
    res = R.apply_threshold(geo, summ, 0.20)
    assert res["positive_cells"] == 1


def test_unmeasurable_cells_are_negative_not_positive(tmp_path):
    """A cell with no DAB measurement must never be called positive — failing closed is
    the whole point of the fixed cutoff on faint tissue."""
    geo, summ = _write_case(tmp_path, [None, None, 0.9])
    res = R.apply_threshold(geo, summ, 0.2)
    assert res["positive_cells"] == 1
    assert res["total_cells"] == 3


def test_classification_and_summary_are_written_back(tmp_path):
    geo, summ = _write_case(tmp_path, [0.05, 0.30, 0.40])
    R.apply_threshold(geo, summ, 0.2)
    names = [f["properties"]["classification"]["name"]
             for f in json.loads(open(geo, encoding="utf-8").read())["features"]]
    assert names == ["Negative", "Positive", "Positive"]
    s = json.loads(open(summ, encoding="utf-8").read())
    assert (s["positive_cells"], s["negative_cells"], s["total_cells"]) == (2, 1, 3)
    assert s["positivity_pct"] == pytest.approx(66.67, abs=0.01)
    assert s["dab_threshold"] == pytest.approx(0.2)


def test_reclassifying_twice_is_idempotent(tmp_path):
    """The operator will move the slider repeatedly; each pass must re-derive from the
    measurements, never accumulate on the previous classification."""
    geo, summ = _write_case(tmp_path, [0.05, 0.30, 0.40])
    R.apply_threshold(geo, summ, 0.35)
    first = R.apply_threshold(geo, summ, 0.2)
    second = R.apply_threshold(geo, summ, 0.2)
    assert first["positive_cells"] == second["positive_cells"] == 2


def test_matching_the_cohort_default_records_no_override(tmp_path):
    geo, summ = _write_case(tmp_path, [0.05, 0.30])
    res = R.apply_threshold(geo, summ, 0.2, cohort_threshold=0.2)
    assert res["override"] is False
    assert "threshold_override" not in json.loads(open(summ, encoding="utf-8").read())


def test_departing_from_the_cohort_default_is_recorded(tmp_path):
    """The override has to survive into the summary, because that is what the report
    reads to tell the user this image was not measured on the cohort's scale."""
    geo, summ = _write_case(tmp_path, [0.05, 0.30])
    res = R.apply_threshold(geo, summ, 0.12, cohort_threshold=0.2)
    assert res["override"] is True
    s = json.loads(open(summ, encoding="utf-8").read())
    assert s["threshold_override"] == pytest.approx(0.12)
    assert s["cohort_threshold"] == pytest.approx(0.2)


def test_override_is_cleared_when_the_image_returns_to_the_cohort_value(tmp_path):
    """Undo must actually undo. A stale threshold_override would mark a conforming image
    as an exception forever."""
    geo, summ = _write_case(tmp_path, [0.05, 0.30])
    R.apply_threshold(geo, summ, 0.12, cohort_threshold=0.2)
    R.apply_threshold(geo, summ, 0.2, cohort_threshold=0.2)
    assert "threshold_override" not in json.loads(open(summ, encoding="utf-8").read())


def test_membrane_results_are_detected(tmp_path):
    _, summ_m = _write_case(tmp_path, [0.1], stem="m", compartment="cytoplasm")
    _, summ_n = _write_case(tmp_path, [0.1], stem="n")
    assert R.is_membrane_result(json.loads(open(summ_m, encoding="utf-8").read())) is True
    assert R.is_membrane_result(json.loads(open(summ_n, encoding="utf-8").read())) is False


def test_histogram_counts_every_measurable_cell():
    vals = np.r_[np.full(100, 0.02), np.full(5, 0.9)]
    h = R.histogram(vals, bins=20)
    assert h["n"] == 105
    assert sum(h["counts"]) + h["overflow"] == 105


def test_histogram_range_is_not_dragged_out_by_a_few_saturated_cells():
    """With a rare-positive marker the interesting structure is near zero; a single
    outlier at 3.0 must not push every real cell into the first bin."""
    vals = np.r_[np.random.default_rng(0).uniform(0, 0.3, 1000), [3.0]]
    h = R.histogram(vals, bins=20)
    assert h["edges"][-1] < 1.0
    assert h["overflow"] >= 1


def test_histogram_survives_a_degenerate_distribution():
    h = R.histogram(np.full(50, 0.1), bins=10)
    assert h["n"] == 50
    assert sum(h["counts"]) + h["overflow"] == 50


def test_histogram_of_nothing_measurable():
    h = R.histogram(np.array([np.nan, np.nan]), bins=10)
    assert h == {"edges": [], "counts": [], "overflow": 0, "n": 0}


def test_positive_fraction_ignores_unmeasurable_cells():
    assert R.positive_fraction([0.1, 0.3, np.nan], 0.2) == pytest.approx(0.5)
