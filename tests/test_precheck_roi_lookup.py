"""A pair's pre-flight verdict has to survive the rename that region fan-out applies.

`run_spatial_association` expands a pair with certified regions into one entry per region
and rewrites the id to "<sample_id>__roi0". Everything downstream of the worker is keyed by
those names; everything upstream (the preview list, the wizard's own pair navigator) still
holds the plain id. Any lookup that crosses that boundary with a plain `.get()` silently
misses, and the miss reads as an absent verdict rather than as a failed lookup.

That produced two different wrong screens from one cause:

  * the batch bandwidth badge counted zero of every null and fell through to ALL PRIMARY;
  * the runtime review reported "No valid primary null for this pair" and withheld the
    statistic for three pairs that had all resolved to the dense-tissue null.

The second is the dangerous one, because fail-closed is the safe-looking answer. A user
would read it as the tissue being unsupported rather than as a key mismatch.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oasis.webui.api import API


PLAN_DENSE = {"null_plan": {"primary_null": "dense_morphology"}}
PLAN_PRIMARY = {"null_plan": {"primary_null": "reweighted_75um"}}


def test_a_region_expanded_verdict_is_found_by_the_plain_id():
    by_pair = {"679_lm00__roi0": PLAN_DENSE}
    assert API._precheck_for("679_lm00", by_pair) is PLAN_DENSE


def test_a_pair_with_no_regions_still_matches_exactly():
    by_pair = {"679_lm00": PLAN_PRIMARY}
    assert API._precheck_for("679_lm00", by_pair) is PLAN_PRIMARY


def test_an_exact_key_wins_over_a_region_of_the_same_pair():
    by_pair = {"679_lm00": PLAN_PRIMARY, "679_lm00__roi0": PLAN_DENSE}
    assert API._precheck_for("679_lm00", by_pair) is PLAN_PRIMARY


def test_the_first_region_is_used_when_a_pair_has_several():
    by_pair = {"p__roi1": PLAN_PRIMARY, "p__roi0": PLAN_DENSE}
    assert API._precheck_for("p", by_pair) is PLAN_DENSE, "regions must resolve in order"


def test_a_different_pair_is_never_borrowed():
    """The bug this guards is worse than a miss: reporting another pair's null as this one's."""
    by_pair = {"679_lm01__roi0": PLAN_DENSE}
    assert API._precheck_for("679_lm00", by_pair) == {}


def test_a_pair_whose_name_merely_starts_the_same_is_not_matched():
    by_pair = {"679_lm000__roi0": PLAN_DENSE}
    assert API._precheck_for("679_lm00", by_pair) == {}


@pytest.mark.parametrize("sid", ["", None])
def test_a_missing_sample_id_returns_empty_rather_than_guessing(sid):
    assert API._precheck_for(sid, {"x__roi0": PLAN_DENSE}) == {}
