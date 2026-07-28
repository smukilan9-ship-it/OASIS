"""results.csv must be filled in by the metrics the pipeline actually produces.

This exists because it did not. `results_table.COLUMNS` was written as (heading, key)
pairs whose keys were invented rather than taken from `parse_summary_json`, and six of
the thirteen did not exist: every run wrote a table with an empty Image column, empty
Positivity, empty Compartment and empty Pixel size. Nothing raised, no test failed, and
the CSV looked plausible until someone read it. The fix is a function per column; this
test is the thing that keeps the two files in agreement, by driving the real parser.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.reporting.results_table import write_results_table   # noqa: E402
from run_pipeline import parse_summary_json                     # noqa: E402


def _summary(tmp_path, **over):
    d = {
        "image": "slideA.tif",
        "total_cells": 1000,
        "positive_cells": 250,
        "negative_cells": 750,
        "positivity_pct": 25.0,
        "dab_threshold": 0.2,
        "pixel_size_um": 0.5,
        "pixel_size_source": "per_image_override",
    }
    d.update(over)
    p = tmp_path / "slideA_summary.json"
    p.write_text(json.dumps(d))
    return str(p)


def _rows(tmp_path, metrics, cfg):
    out = tmp_path / "out"
    path = write_results_table(metrics, str(out), cfg)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_nuclear_run_fills_every_column_it_should(tmp_path):
    m = parse_summary_json(_summary(tmp_path))
    row = _rows(tmp_path, [m], {"stain_name": "CD8", "preprocess_normalize": True})[0]

    assert row["Image"] == "slideA.tif"
    assert row["Total cells"] == "1000"
    assert row["Positive cells"] == "250"
    assert row["Positivity %"] == "25.0"
    assert row["Marker"] == "CD8"
    assert row["Compartment"] == "nucleus"
    assert row["Positivity rule"] == "nuclear DAB above cutoff"
    assert row["DAB cutoff (OD)"] == "0.2"
    assert row["Cutoff source"] == "cohort"
    assert row["Pixel size (um/px)"] == "0.5"
    assert row["Pixel size source"] == "per_image_override"
    assert row["Normalized"] == "yes"
    assert row["Confidence"]


def test_per_image_override_is_named_not_averaged_in(tmp_path):
    m = parse_summary_json(_summary(
        tmp_path, dab_threshold=0.35, threshold_override=0.35, cohort_threshold=0.2))
    row = _rows(tmp_path, [m], {"stain_name": "CD8"})[0]
    assert row["DAB cutoff (OD)"] == "0.35"
    assert row["Cutoff source"] == "per-image override"


def test_membrane_run_records_both_numbers_of_its_rule(tmp_path):
    m = parse_summary_json(_summary(
        tmp_path, measurement_compartment="cytoplasm",
        membrane_classifier="completeness",
        membrane_pix_thr=0.2, membrane_frac_min=0.15))
    row = _rows(tmp_path, [m], {"stain_name": "TIM-3"})[0]
    assert row["Compartment"] == "cytoplasm"
    assert row["Positivity rule"] == "ring completeness: >=0.15 of ring above 0.2 OD"
    assert row["Ring fraction"] == "0.15"


def test_classifier_run_names_the_classifier(tmp_path):
    m = parse_summary_json(_summary(tmp_path, classifier_name="CD8 cohort v2",
                                    classifier_applied=True))
    row = _rows(tmp_path, [m], {"stain_name": "CD8"})[0]
    assert row["Positivity rule"] == "classifier: CD8 cohort v2"
    assert row["Cutoff source"] == "trained classifier"


def test_refused_image_does_not_credit_the_classifier(tmp_path):
    """A refused image was called by the cutoff on nuclear DAB. The table must say so.

    This is the faint-slide case: the classifier declines, `apply_threshold` re-calls every
    cell on nuclear DAB, and the row used to read 'classifier: X' over a cytoplasm
    compartment for calls neither of them made.
    """
    m = parse_summary_json(_summary(
        tmp_path, classifier_name="TIM-3 cohort", classifier_applied=False,
        classifier_refused_reason="ring_mean below training range",
        measurement_compartment="nucleus"))
    row = _rows(tmp_path, [m], {"stain_name": "TIM-3"})[0]
    assert row["Positivity rule"].startswith("nuclear DAB above cutoff — TIM-3 cohort refused")
    assert "ring_mean below training range" in row["Positivity rule"]
    assert row["Cutoff source"] == "cohort (classifier refused)"
    assert row["Compartment"] == "nucleus"
    assert row["Ring fraction"] == ""


def test_no_column_is_silently_absent(tmp_path):
    """Every heading must appear in the file — a renamed column is a broken pipeline."""
    from oasis.reporting.results_table import COLUMNS
    m = parse_summary_json(_summary(tmp_path))
    row = _rows(tmp_path, [m], {})[0]
    assert [h for h, _ in COLUMNS] == list(row.keys())
