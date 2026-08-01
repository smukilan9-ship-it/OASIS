"""Every Certify button must reach the same gate with the same FLE.

The Spatial tab has four routes into `certify_local_roi` — the auto sweep, the size ladder,
the drawn regions, and the whole-field attempt. They disagreed about ONE polygon:

    auto   LOCALLY_CERTIFIED   cell error  3.2 µm   FLE shown: (blank)
    drawn  RADIUS_LIMITED      cell error 14.5 µm   FLE shown: 0.17 µm

Identical 810,000 µm² region, identical 82 correspondences, identical fit. The difference was
`fle_fast`, which DECLARED fle_um = 0.7 µm while the other path MEASURED it. Since
σ_fit² = 2·FLE² + deformation², an over-large declared FLE books less residual as deformation,
so 0.7 was lenient in the over-certifying direction.

That was first patched by re-certifying every survivor with a measured FLE, which cost 14.3 s
per region — all of it `loftr_fle`'s cache-bypassing noise trials, and the largest single item
in a certification run. It is now fixed properly: `FLE_WORKING_PX` is measured against a known
warp (validation/validate_loftr_fle_groundtruth.py), so the fast path is the CORRECT path and
there is nothing to re-do. The paths agree by construction.

These tests pin that at the call layer and cost no LoFTR inference.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _source(fn):
    import inspect
    return inspect.getsource(fn)


# ── the constant, and where it came from ─────────────────────────────────────────────
def test_the_declared_fle_is_a_measured_constant_not_a_placeholder():
    from oasis.spatial.loftr_matcher import FLE_WORKING_PX
    # 0.121–0.195 working px across thirteen warps and two working scales; median 0.16.
    assert 0.10 <= FLE_WORKING_PX <= 0.25, FLE_WORKING_PX


def test_the_07_placeholder_is_gone():
    """0.7 µm was never measured and is ~3x the real value, in the lenient direction."""
    from oasis.spatial import loftr_matcher as lm
    src = _source(lm.certify_local_roi)
    assert "fle_um = 0.7" not in src
    assert "FLE_WORKING_PX" in src, "the fast path must use the measured constant"


def test_the_fle_scales_with_the_working_pixel_size():
    """Phase A measured FLE as a matcher-grid property (CV 0.09 in px vs 0.41 in µm), so a
    flat µm constant would be wrong at every working scale but one."""
    from oasis.spatial import loftr_matcher as lm
    src = _source(lm.certify_local_roi)
    assert "FLE_WORKING_PX * px_work" in src


# ── one gate, four doors ─────────────────────────────────────────────────────────────
def test_no_call_site_opts_out_of_the_shared_fle():
    """`fle_fast=False` re-introduces per-region measurement — 14.3 s each, and a different
    FLE from every other path. It stays available for validation; production must not use it."""
    import inspect
    from oasis.webui import api
    src = inspect.getsource(api)
    assert "fle_fast=False" not in src, (
        "a Spatial call site is measuring FLE per region again — that is the cost regression "
        "and the path disagreement, both at once")


def test_the_fast_path_is_the_default():
    """So a new call site that passes nothing gets the calibrated FLE rather than the slow
    per-region measurement."""
    import inspect
    from oasis.spatial.loftr_matcher import certify_local_roi
    assert inspect.signature(certify_local_roi).parameters["fle_fast"].default is True


def test_each_tile_is_certified_once():
    """The double certification existed only because the screen used the wrong FLE.

    Counting every certify call in the function would be wrong — the size ladder legitimately
    certifies a probe circle to CHOOSE a region size, which is a different question from
    whether a given tile passes. What must not recur is the same tile polygon (`roi_t`) being
    put through the gate twice.
    """
    from oasis.webui.api import API
    src = _source(API.auto_certify_regions)
    assert src.count("roi_t, px_t") == 1, (
        "a tile is going through the gate more than once; that was the 14.3 s-per-region "
        "regression, and it is unnecessary now that the screen and the answer agree")


def test_a_returned_region_states_its_fle():
    """A certified region with a blank FLE is not inspectable — FLE is the term that decides
    how much of the residual becomes deformation. The auto path never emitted the key."""
    from oasis.webui.api import API
    assert '"fle_um"' in _source(API.auto_certify_regions)
    assert '"fle_um"' in _source(API.certify_local_roi_multi)


# ── why 0.7 was the unsafe direction, kept as executable documentation ───────────────
def test_an_overlarge_declared_fle_is_the_lenient_direction():
    import numpy as np
    sigma_fit = 4.073          # measured on the disputed ROI, both runs
    from oasis.spatial.loftr_matcher import FLE_WORKING_PX
    px_work = 1.82             # that ROI's working pixel size

    def deformation(fle):
        return float(np.sqrt(max(sigma_fit ** 2 - 2.0 * fle ** 2, 0.0)))

    assert deformation(0.7) < deformation(FLE_WORKING_PX * px_work), (
        "if this flips, the argument for replacing 0.7 has to be re-derived")


def test_the_calibrated_fle_lands_near_the_two_independent_estimates():
    """0.199 µm from loftr_fle on the disputed ROI, 0.409 µm from that pair's variogram
    nugget. The constant must sit in that neighbourhood, not near 0.7."""
    from oasis.spatial.loftr_matcher import FLE_WORKING_PX
    at_disputed_roi = FLE_WORKING_PX * 1.82
    assert 0.15 <= at_disputed_roi <= 0.50, at_disputed_roi


@pytest.mark.parametrize("method", ["auto_certify_regions", "certify_local_roi_multi",
                                    "certify_spatial_auto"])
def test_every_certify_entry_point_still_exists(method):
    """Guards against a rename quietly detaching one of the buttons from these checks."""
    from oasis.webui.api import API
    assert callable(getattr(API, method, None))
