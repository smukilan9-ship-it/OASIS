"""The two pixel-size precedence chains, asserted end to end.

Pixel size is the one number every downstream result is scaled by, and OASIS resolves it
through two separate chains that had no test between them:

  Quant   `get_pixel_size_with_source`  per-image override > TIFF/OME metadata > UI default
                                        > filename magnification > interactive > 0.5
  Spatial `run_pipeline.resolve_pixel_size`  manual override > scale image > session default
                                        > bar burned into the image itself > 0.5

Individual pieces were covered (the detector in test_scale_bar.py), but not the ORDER, and
the order is where a silent error lives: every one of these sources can produce a plausible
number, so picking the wrong one is invisible in the output. A slide carrying a resolution
tag quietly overriding a value the operator just typed, or a session default beating the
image's own bar, changes every reported distance and looks completely normal.

The cases below build images whose answer is known by construction — a 133 px bar is
100/133 µm/px — and pin both the value and, for the Quant chain, the reported source, since
the source string is what the UI and results.csv show as provenance.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oasis.common.pixel_size_util import get_pixel_size_with_source  # noqa: E402
from run_pipeline import resolve_pixel_size  # noqa: E402

W, H = 1920, 1440
BAR_PX = 133
BAR_UM = 100.0
FROM_BAR = BAR_UM / BAR_PX          # 0.7519 µm/px — the LL477 cohort's real value


def _slide(directory, name, bar_px=BAR_PX, tiff_resolution=None):
    """A pale field, optionally with a drawn bar and an explicit TIFF resolution tag."""
    arr = np.full((H, W, 3), 235, np.uint8)
    if bar_px:
        y, x = int(H * 0.93), W - 200
        arr[y:y + 6, x:x + bar_px] = 10
    path = os.path.join(directory, name)
    if tiff_resolution:
        import tifffile
        # ResolutionUnit 3 is centimetres, so µm/px = 10000 / resolution.
        tifffile.imwrite(path, arr, resolution=(tiff_resolution, tiff_resolution),
                         resolutionunit=3)
    elif name.endswith(".tif"):
        import tifffile
        tifffile.imwrite(path, arr)
    else:
        Image.fromarray(arr).save(path)
    return path


@pytest.fixture(scope="module")
def slides():
    d = tempfile.mkdtemp()
    return {
        # "x10" in the stem is what the filename parser reads -> 1.0 µm/px
        "plain": _slide(d, "LL_x10_plain.png"),
        "meta": _slide(d, "LL_x10_meta.tif", tiff_resolution=40000),   # tag says 0.25
        "bare": _slide(d, "nobar.png", bar_px=0),
        "dir": d,
    }


# ── Quant chain ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("label,key,cfg,want,want_source", [
    ("a per-image override outranks every other source", "plain",
     {"pixel_overrides": {"LL_x10_plain.png": 0.111}, "_pixel_size_from_ui": True,
      "default_pixel_size": 0.9}, 0.111, "per_image_override"),
    # A slide carrying its own resolution tag must not be overridden by a run-wide default;
    # this is why the UI sends typed values as PER-IMAGE overrides rather than as the default.
    ("embedded TIFF metadata outranks the UI default", "meta",
     {"_pixel_size_from_ui": True, "default_pixel_size": 0.9}, 0.25, "tiff_metadata"),
    ("an explicit UI default outranks filename parsing", "plain",
     {"_pixel_size_from_ui": True, "default_pixel_size": 0.9}, 0.9, "ui_default"),
    ("filename magnification is used only when nothing was set", "plain",
     {"default_pixel_size": 0.5}, 1.0, "filename"),
])
def test_quant_precedence(slides, label, key, cfg, want, want_source):
    value, source = get_pixel_size_with_source(slides[key], dict(cfg))
    assert value == pytest.approx(want), label
    assert source == want_source, f"{label} — provenance is shown to the operator"


def test_quant_falls_back_last_and_says_so(slides):
    """The fallback must be reported as `default_fallback`, not dressed up as a measurement.

    The spatial pipeline keys its "you are running on an untouched default" warning off this
    exact string, so a fallback that reported any other source would silence that warning.
    """
    value, source = get_pixel_size_with_source(slides["bare"], {"default_pixel_size": 0.5})
    assert value == pytest.approx(0.5)
    assert source == "default_fallback"


# ── Spatial chain ────────────────────────────────────────────────────────────────────
def test_spatial_manual_override_wins(slides):
    assert resolve_pixel_size(0.6, slides["plain"], slides["plain"], 0.321) == \
        pytest.approx(0.321)


def test_spatial_scale_image_beats_the_session_default(slides):
    """The whole point of dropping a _scale sibling next to a slide."""
    assert resolve_pixel_size(0.6, slides["plain"], slides["plain"], None) == \
        pytest.approx(FROM_BAR, rel=1e-6)


def test_spatial_session_default_beats_a_bar_inside_the_analysis_image(slides):
    """Deliberate: a bar burned into the ANALYSIS image is the last resort, because an
    operator who typed a session value has stated the scale, and a stray dark rectangle in
    tissue should never silently outrank them."""
    assert resolve_pixel_size(0.6, slides["plain"], None, None) == pytest.approx(0.6)


def test_spatial_reads_a_bar_in_the_image_when_nothing_else_is_set(slides):
    assert resolve_pixel_size(None, slides["plain"], None, None) == \
        pytest.approx(FROM_BAR, rel=1e-6)


def test_spatial_final_fallback(slides):
    assert resolve_pixel_size(None, slides["bare"], None, None) == pytest.approx(0.5)


@pytest.mark.parametrize("bar_um", [50.0, 100.0, 250.0])
def test_the_configured_bar_length_reaches_the_spatial_chain(slides, bar_um):
    """Settings carries the bar length; it has to survive the whole way down, or the app
    measures a 250 µm bar as though it were 100 and mis-scales everything by 2.5x."""
    assert resolve_pixel_size(None, slides["plain"], slides["plain"], None, bar_um) == \
        pytest.approx(bar_um / BAR_PX, rel=1e-6)
