"""The batch bandwidth verdict has to find the pairs it just measured.

A pair with a drawn region does not reach the worker under its own name. Before the
run, `SpatialAPI` fans each pair out one-per-certified-ROI and rewrites the id
(api.py: `q["sample_id"] = f"{p['sample_id']}__roi{...}"`), so the verdicts come back
keyed `679_lm00__roi0` while the preview list the UI holds still says `679_lm00`.

The original lookup was an equality filter against the preview list. The instant
anyone drew a region it matched nothing, and the failure was silent in the worst
direction: an empty list counts zero fail-closed and zero dense, so the badge fell
through its final `else` and announced ALL PRIMARY (75 µm) over a run it had not
counted a single pair of. On screen that is a green all-clear next to the line
"0 pair(s) -> 75 um reweighted primary null", with no per-pair block beneath it.

These tests pin the three things that fix has to keep true: ROI ids match their
parent pair, preview order is preserved, and an empty result never reads as a pass.
"""
import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parent.parent / "oasis/webui/index.html"


@pytest.fixture(scope="module")
def js():
    return INDEX.read_text(encoding="utf-8")


def parent_of(sid):
    """The rule under test, in Python: the UI splits on the ROI suffix."""
    return str(sid).split("__roi")[0]


def order_batch(pairs, by_pair):
    """A transcription of the batch branch of spatialRunBandwidthPreflight."""
    keys = list(by_pair.keys())
    order = []
    for p in pairs:
        for sid in keys:
            if parent_of(sid) == p:
                order.append(sid)
    for sid in keys:
        if sid not in order:
            order.append(sid)
    return order


def test_roi_expanded_ids_match_their_parent_pair():
    pairs = ["679_lm00", "679_lm01", "679_lm04"]
    by_pair = {"679_lm00__roi0": {}, "679_lm01__roi0": {}, "679_lm04__roi0": {}}
    assert order_batch(pairs, by_pair) == [
        "679_lm00__roi0", "679_lm01__roi0", "679_lm04__roi0"]


def test_several_rois_on_one_pair_are_all_kept():
    """Fan-out is one entry per certified ROI, so a pair can produce more than one."""
    order = order_batch(["a", "b"], {"a__roi0": {}, "a__roi1": {}, "b__roi0": {}})
    assert order == ["a__roi0", "a__roi1", "b__roi0"]


def test_pairs_without_regions_still_match():
    """No region drawn means no rename, and the plain id has to keep working."""
    assert order_batch(["a", "b"], {"a": {}, "b": {}}) == ["a", "b"]


def test_a_verdict_the_preview_list_does_not_know_is_still_shown():
    order = order_batch(["a"], {"a__roi0": {}, "stranger": {}})
    assert order == ["a__roi0", "stranger"], "a measured pair was dropped from the report"


def test_preview_order_is_preserved_not_dict_order():
    order = order_batch(["b", "a"], {"a__roi0": {}, "b__roi0": {}})
    assert order == ["b__roi0", "a__roi0"]


def test_the_badge_cannot_report_all_primary_on_an_empty_list(js):
    """The regression itself: zero counted pairs must not render as a pass.

    Text-level assertion, because the branch is a plain if/else chain in the page
    and the empty case is the one that used to fall through to the green verdict.
    """
    block = re.search(r"if \(!order\.length\)[^\n]*", js)
    assert block, ("the empty-order guard is gone — an unmatched batch will fall "
                   "through to the ALL PRIMARY branch and report a pass for nothing")
    assert "ALL PRIMARY" not in block.group(0)
