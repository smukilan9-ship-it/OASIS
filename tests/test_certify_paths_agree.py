"""The two Certify buttons have to answer the same question the same way.

The Spatial tab offers two routes to a verdict:

    "Certify (auto: whole field → regions)"   -> API.auto_certify_regions
    "Certify drawn regions"                   -> API.certify_local_roi_multi

Driven on LL477_CD8_x10_1 <-> Tim3_x10_1 they disagreed about ONE polygon:

    auto   LOCALLY_CERTIFIED   cell error  3.2 µm   FLE shown: (blank)
    drawn  RADIUS_LIMITED      cell error 14.5 µm   FLE shown: 0.17 µm

Identical 810,000 µm² polygon, identical 82 correspondences, identical fit. The whole
difference was `fle_fast`: the auto path DECLARED fle_um = 0.7 while the drawn path MEASURED
it. Declaring FLE too high books less of the residual as deformation, so the fast setting is
lenient — it was a screening shortcut being reported as a certification, in the direction that
over-certifies.

research/registration.md § 11.2 measured the real FLE against a known warp at 0.224 µm, so 0.7
is roughly 3x too large, and `loftr_matcher.certify_local_roi`'s own comment already said what
to do: "Re-certify a chosen region without fast mode for the principled measured FLE."

These tests pin the contract at the call layer rather than re-deriving the numbers, so they
cost no LoFTR inference: whatever else the auto path does, the certification it RETURNS must
have been produced with a measured FLE, and the region it returns must carry that FLE.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _source(fn):
    import inspect
    return inspect.getsource(fn)


def test_the_answer_is_never_a_fast_fle_certification():
    """auto_certify_regions may screen with fle_fast, but must re-certify before returning."""
    from oasis.webui.api import API
    src = _source(API.auto_certify_regions)
    assert "fle_fast=True" in src, "the cheap screen is expected to still exist"
    assert "fle_fast=False" in src, (
        "auto_certify_regions returns a fast-FLE verdict as its final answer — that is the "
        "3.2 µm vs 14.5 µm disagreement with the drawn path")
    # the fast call must be the screen (assigned to a probe), the slow one the answer
    assert "probe = lm.certify_local_roi" in src
    assert "cert = lm.certify_local_roi" in src


def test_a_returned_region_states_its_fle():
    """A certified region with a blank FLE is not inspectable.

    FLE is the term that decides how much of the residual becomes deformation, so a verdict
    without it cannot be argued with. The auto path simply never emitted the key.
    """
    from oasis.webui.api import API
    src = _source(API.auto_certify_regions)
    assert '"fle_um"' in src, "auto-certified regions must carry fle_um (the column read blank)"


def test_the_drawn_path_still_measures_fle():
    """The honest path must not be 'fixed' by making it fast — that hides the disagreement
    instead of resolving it (research/registration.md § 6)."""
    from oasis.webui.api import API
    src = _source(API.certify_local_roi_multi)
    assert "fle_fast=True" not in src
    assert '"fle_um"' in src


def test_the_screen_is_lenient_which_is_why_it_cannot_be_the_answer():
    """Documents the direction of the bias, in code rather than prose.

    fle_fast declares 0.7 µm. sigma_fit^2 = 2*FLE^2 + deformation^2, so a larger declared FLE
    leaves a SMALLER deformation term and a smaller cell error. Using the disputed ROI's own
    sigma_fit, the fast setting understates deformation — it can only ever certify more.
    """
    import numpy as np
    sigma_fit = 4.073          # measured on the disputed ROI, both runs
    declared_fast, measured = 0.7, 0.224

    def deformation(fle):
        return float(np.sqrt(max(sigma_fit ** 2 - 2.0 * fle ** 2, 0.0)))

    assert deformation(declared_fast) < deformation(measured), (
        "fast mode must be the lenient one; if this flips, the argument for re-certifying "
        "has to be re-derived")


def test_regions_refused_on_re_certification_are_reported():
    """n=0 with a large drop count means the size ladder picked too big a window, not that
    the pair is unregistrable. Silently returning zero regions loses that distinction."""
    from oasis.webui.api import API
    src = _source(API.auto_certify_regions)
    assert "dropped_on_recertify" in src


@pytest.mark.parametrize("method", ["auto_certify_regions", "certify_local_roi_multi",
                                    "certify_spatial_auto"])
def test_every_certify_entry_point_still_exists(method):
    """Guards against a rename quietly detaching one of the buttons from these checks."""
    from oasis.webui.api import API
    assert callable(getattr(API, method, None))
