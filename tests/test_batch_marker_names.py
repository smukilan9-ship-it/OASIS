"""Batch asks for folders, not marker names, so it has to take the names from the match.

Two folders is the mode people actually use for a cohort, and it has no marker-name field:
step 1 shows "Marker A folder" / "Marker B folder" and nothing else. The run therefore
inherited whatever was left in the single-pair boxes, which ship with the values CD8 and
TIM-3. Point it at a folder of CD8 and CD45 sections and every surface that names a marker
said TIM-3 — the page subtitle, both panes of the certification canvas, the pixel-size rows,
the activity log, and the marker names recorded with the results.

The numbers were right. All of them were attributed to a marker that was not in the run,
which for a serial-section co-localisation result is the one detail a reader would carry
away. There was also no way to correct it from batch mode: the only inputs that set those
names are hidden unless you switch back to single-pair.

`match_two_folders` has already parsed the real tokens out of the filenames — it cannot pair
the folders without them — so the preview adopts them into the same two inputs the rest of
the tab reads. These tests pin the parsing those tokens come from, and the JS that adopts
them.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oasis.common.file_matcher import normalize_name

INDEX = ROOT / "oasis/webui/index.html"


@pytest.fixture(scope="module")
def js():
    return INDEX.read_text()


def test_the_matcher_reports_the_token_the_ui_will_adopt():
    """The adoption is only as good as the token, so check the two this was found on."""
    assert normalize_name("679_CD8_lm00")[1] == "cd8"
    assert normalize_name("679_CD45_lm00")[1] == "cd45"


def test_the_preview_adopts_matched_markers(js):
    assert "spatialAdoptMatchedMarkers(spatialPairs)" in js, (
        "the batch preview no longer adopts the matched marker names — a CD8/CD45 cohort "
        "will be labelled with whatever the single-pair boxes were left holding")


def test_adoption_writes_into_the_inputs_the_rest_of_the_tab_reads(js):
    """Writing to the shared inputs is the point: six readers are fixed by one assignment."""
    m = re.search(r"function spatialAdoptMatchedMarkers\(pairs\)\s*\{(.*?)\n\}", js, re.S)
    assert m, "spatialAdoptMatchedMarkers is gone"
    body = m.group(1)
    for id_ in ("spatial-label-a", "spatial-label-b"):
        assert id_ in body, f"{id_} is no longer written, so its readers keep the stale name"
    assert "stain_a" in body and "stain_b" in body, "no longer reads the matcher's tokens"


def test_a_disagreeing_folder_is_not_given_a_winner(js):
    """One file out of step must not silently rename the whole cohort."""
    m = re.search(r"function spatialAdoptMatchedMarkers\(pairs\)\s*\{(.*?)\n\}", js, re.S)
    assert "found.length === 1" in m.group(1), (
        "adoption no longer requires the folder to agree on one marker — a mixed folder "
        "would take whichever token happened to sort first")


def test_batch_sends_the_marker_names_with_the_run(js):
    """The config has to carry them too; the inputs alone are a UI detail."""
    batch = js[js.index("folder_mode: spatialFolderMode"):]
    head = batch[:600]
    assert "label_a" in head and "label_b" in head, (
        "the batch branch of spatialBuildConfig no longer sends label_a/label_b, so the "
        "worker falls back to a generic marker name in the written results")
