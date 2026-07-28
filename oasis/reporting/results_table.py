"""The results table: one CSV, one row per image.

This replaces the HTML dashboard and the timestamped Excel workbook. Both restated the
same numbers -- the dashboard in a page that is read once and never reopened, the workbook
in a format that has to be converted before anything can be done with it. Whatever a
researcher does next (a statistical test, a figure, a supplementary table) starts from a
plain table, so that is what the run writes.

Columns are chosen so a row is self-describing months later: not just the counts, but the
rule that produced them, the pixel size and WHERE that pixel size came from. A positivity
percentage with no record of the cutoff behind it cannot be checked or reproduced.

Each column is a FUNCTION of the per-image metrics and the run config, not a key looked up
in a dict. The first version of this file was a list of (heading, key) pairs, and the keys
were written from memory rather than against `parse_summary_json` -- six of the thirteen
did not exist, so every run silently wrote a table whose Image, Positivity, Compartment
and Pixel size columns were empty. Nothing failed; the file was simply wrong. A function
per column cannot drift the same way, because it names the source explicitly.
"""
import csv
import os


def _pct(m):
    v = m.get("Positivity_Index_Pct")
    return "" if v is None else round(float(v), 2)


def _rule(m, cfg):
    """How this image's cells were called, in one cell of a spreadsheet.

    A named classifier is not necessarily the classifier that called this image: one that
    judges an image outside its training range refuses it and hands it back to the fixed
    cutoff. Reading the name alone credited the model for the cutoff's calls.
    """
    if m.get("Classifier_Name") and m.get("Classifier_Applied") is not False:
        return f"classifier: {m['Classifier_Name']}"
    if m.get("Classifier_Name"):
        return (f"nuclear DAB above cutoff — {m['Classifier_Name']} refused this image"
                + (f" ({m['Classifier_Refused_Reason']})"
                   if m.get("Classifier_Refused_Reason") else ""))
    if m.get("Compartment") == "cytoplasm":
        pix, frac = m.get("Membrane_Pix_Thr"), m.get("Membrane_Frac_Min")
        if pix is not None and frac is not None:
            return f"ring completeness: >={frac:g} of ring above {pix:g} OD"
        return "ring mean above cutoff"
    return "nuclear DAB above cutoff"


def _cutoff(m):
    v = m.get("Threshold_Override")
    if v is not None:
        return round(float(v), 4)
    v = m.get("DAB_Threshold")
    return "" if v is None else round(float(v), 4)


def _cutoff_source(m):
    if m.get("Classifier_Name") and m.get("Classifier_Applied") is not False:
        return "trained classifier"
    if m.get("Classifier_Name"):
        return "cohort (classifier refused)"
    if m.get("Threshold_Override") is not None:
        return "per-image override"
    return "cohort"


# (column heading, value function). Order is the reading order: which image, what was
# found, under which rule, on what physical scale, and how much to trust it.
COLUMNS = [
    ("Image",              lambda m, c: m.get("Image_Name", "")),
    ("Total cells",        lambda m, c: m.get("Total_Cells", "")),
    ("Positive cells",     lambda m, c: m.get("Positive_Cells", "")),
    ("Positivity %",       lambda m, c: _pct(m)),
    ("Cells per mm2",      lambda m, c: (round(float(m["Cells_Per_mm2"]))
                                         if m.get("Cells_Per_mm2") is not None else "")),
    ("Marker",             lambda m, c: c.get("stain_name", "")),
    ("Compartment",        lambda m, c: m.get("Compartment", "")),
    ("Positivity rule",    lambda m, c: _rule(m, c)),
    ("DAB cutoff (OD)",    lambda m, c: _cutoff(m)),
    ("Cutoff source",      lambda m, c: _cutoff_source(m)),
    ("Ring fraction",      lambda m, c: (m.get("Membrane_Frac_Min")
                                         if m.get("Membrane_Frac_Min") is not None else "")),
    ("Pixel size (um/px)", lambda m, c: m.get("Pixel_Size_um", "")),
    ("Pixel size source",  lambda m, c: m.get("Pixel_Size_Source", "")),
    ("Normalized",         lambda m, c: "yes" if c.get("preprocess_normalize") else "no"),
    ("Confidence",         lambda m, c: m.get("Confidence", "")),
]


def write_results_table(batch_metrics, out_dir, config=None):
    """Write results.csv into out_dir and return its path.

    A column that does not apply to this run becomes an empty cell rather than an error:
    the columns describe every run this pipeline can perform, and no single run fills all
    of them (a nuclear run has no ring fraction, a cutoff run has no classifier).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")
    cfg = config or {}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([h for h, _ in COLUMNS])
        for m in batch_metrics or []:
            row = []
            for _, fn in COLUMNS:
                try:
                    v = fn(m, cfg)
                except Exception:
                    v = ""
                row.append("" if v is None else v)
            w.writerow(row)
    return path
