"""Spatial runs two markers in one job, so the classifier has to be chosen per image.

A single cohort-wide classifier cannot serve a CD8/TIM-3 pair: the CD8 model reads a TIM-3
ring as an out-of-range slide and refuses it, so image B silently drops to the fixed cutoff
while the run still reports "classifier applied". `membrane_overrides` already had to be
per-image for the same reason.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import classifier_for_image  # noqa: E402

CD8 = {"weights": [1, 2], "kind": "membrane"}
TIM3 = {"weights": [3, 4], "kind": "membrane"}


def test_per_image_override_wins_over_the_global_classifier():
    cfg = {"classifier": CD8, "classifier_name": "CD8",
           "classifier_overrides": {"b.tif": {"classifier": TIM3, "classifier_name": "TIM3"}}}
    assert classifier_for_image(cfg, "/data/b.tif") == (TIM3, "TIM3")


def test_an_image_without_an_override_still_gets_the_global_one():
    cfg = {"classifier": CD8, "classifier_name": "CD8",
           "classifier_overrides": {"b.tif": {"classifier": TIM3, "classifier_name": "TIM3"}}}
    assert classifier_for_image(cfg, "/data/a.tif") == (CD8, "CD8")


def test_the_override_is_matched_on_basename_not_full_path():
    cfg = {"classifier_overrides": {"b.tif": {"classifier": TIM3, "classifier_name": "TIM3"}}}
    assert classifier_for_image(cfg, "/somewhere/else/entirely/b.tif") == (TIM3, "TIM3")


def test_no_classifier_anywhere_means_no_classifier():
    assert classifier_for_image({}, "/data/a.tif") == (None, None)


def test_an_override_entry_without_a_model_falls_through_rather_than_crashing():
    """The API writes this key only when a model resolves; an empty entry must not
    shadow the global one or blow up `from_dict(None)` downstream."""
    cfg = {"classifier": CD8, "classifier_name": "CD8",
           "classifier_overrides": {"b.tif": {}}}
    assert classifier_for_image(cfg, "/data/b.tif") == (CD8, "CD8")
