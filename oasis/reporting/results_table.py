"""The results table: one CSV, one row per image.

This replaces the HTML dashboard and the timestamped Excel workbook. Both restated the
same numbers -- the dashboard in a page that is read once and never reopened, the workbook
in a format that has to be converted before anything can be done with it. Whatever a
researcher does next (a statistical test, a figure, a supplementary table) starts from a
plain table, so that is what the run writes.

Columns are chosen so a row is self-describing months later: not just the counts, but the
cutoff that produced them, the pixel size and WHERE that pixel size came from. A positivity
percentage with no record of the cutoff behind it cannot be checked or reproduced.
"""
import csv
import os

# (column heading, key in the per-image metrics dict). Order is the reading order:
# what was measured, what was found, and under which settings.
COLUMNS = [
    ("Image",                "Image"),
    ("Total cells",          "Total_Cells"),
    ("Positive cells",       "Positive_Cells"),
    ("Positivity %",         "Positive_Percentage"),
    ("Marker",               "stain_name"),
    ("Compartment",          "measurement_compartment"),
    ("DAB cutoff (OD)",      "dab_threshold"),
    ("Cutoff source",        "threshold_source"),
    ("Classifier",           "classifier_name"),
    ("Pixel size (µm/px)",   "pixel_size"),
    ("Pixel size source",    "pixel_size_source"),
    ("Normalized",           "preprocess_normalize"),
    ("Confidence",           "Confidence"),
]


def write_results_table(batch_metrics, out_dir, config=None):
    """Write results.csv into out_dir and return its path.

    A missing key becomes an empty cell rather than an error: the columns describe every
    run this pipeline can perform, and no single run fills all of them (a membrane run has
    no nuclear cutoff, a cutoff run has no classifier).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")
    cfg = config or {}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([h for h, _ in COLUMNS])
        for m in batch_metrics or []:
            row = []
            for _, key in COLUMNS:
                v = m.get(key, cfg.get(key, ""))
                if isinstance(v, bool):
                    v = "yes" if v else "no"
                row.append("" if v is None else v)
            w.writerow(row)
    return path
