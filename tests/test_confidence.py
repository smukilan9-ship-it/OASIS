"""The Confidence column has to reflect everything the pipeline already knows.

It is the one field whose entire job is "how much to trust this row", and it read only the
cell count and the positivity percentage. So an image the trained classifier REFUSED as
outside its trained range — which hands the calls back to the fixed cutoff and flags
`staining_quality: low` — was reported as NORMAL. Observed on a real run:

    Classifier REFUSED this image (area_px=33.36 outside [53, 64])
      — fell back to the fixed cutoff and flagged staining_quality:low
    results.csv -> Confidence: NORMAL

The refusal was recorded in `Positivity rule`, `Cutoff source` and the summary JSON, so the
information existed; the column a reader actually scans for trust did not carry it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import compute_confidence  # noqa: E402

HEALTHY = dict(total=5000, pos_pct=12.0)


def test_a_healthy_row_is_normal():
    assert compute_confidence(**HEALTHY) == "NORMAL"


@pytest.mark.parametrize("total,pos_pct,why", [
    (10, 12.0, "too few cells to characterise"),
    (5000, 0.05, "positivity so low the call is noise"),
    (5000, 99.0, "positivity so high the cutoff is not separating anything"),
])
def test_the_original_count_based_reasons_still_hold(total, pos_pct, why):
    assert compute_confidence(total, pos_pct) == "LOW", why


def test_low_staining_quality_is_not_normal():
    """The pipeline's own verdict on the staining, which it prints and then discarded."""
    assert compute_confidence(**HEALTHY, staining_quality="low") == "LOW"
    assert compute_confidence(**HEALTHY, staining_quality="LOW") == "LOW"


def test_a_classifier_that_refused_the_image_is_not_normal():
    """The numbers did not come from the method the operator chose.

    The fixed-cutoff fallback is a legitimate rule on its own, which is why this is LOW
    rather than an error — but a row scored by a rule the operator did not pick should not
    look identical to one scored by the rule they did.
    """
    assert compute_confidence(**HEALTHY, classifier_name="CD8",
                              classifier_applied=False) == "LOW"


def test_a_classifier_that_ran_is_normal():
    assert compute_confidence(**HEALTHY, classifier_name="CD8",
                              classifier_applied=True) == "NORMAL"


def test_no_classifier_was_requested_is_normal():
    """A plain cutoff run must not be downgraded just because no classifier was named."""
    assert compute_confidence(**HEALTHY, classifier_name=None,
                              classifier_applied=False) == "NORMAL"
    assert compute_confidence(**HEALTHY, staining_quality=None) == "NORMAL"


def test_the_real_refusal_that_prompted_this():
    """The exact combination seen on LL477_CD8_x10_1: classifier named, refused, low quality,
    and counts that are otherwise entirely healthy (12,942 cells, 1.85% positive)."""
    assert compute_confidence(12942, 1.85, staining_quality="low",
                              classifier_name="CD8", classifier_applied=False) == "LOW"
