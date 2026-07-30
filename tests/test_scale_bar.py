"""The scale bar sets the physical scale, so a wrong reading rescales everything.

Every distance the spatial statistic reports is derived from µm/px: the 10-20 µm
colocalization band, the 20-50 µm co-infiltration band, the 75 µm architecture bandwidth,
and the certified cell error. A silent 31% error there is not a display bug.

Measured on the LL477 cohort before the fix (true bar 133 px in all six scale images):

    CD8_x10_1   156 px  (+17%)      Tim3_10X_3   165 px  (+24%)
    CD8_x10_2   139 px   (+5%)      Tim3_x10_1   174 px  (+31%)
    CD8_x10_3   133 px    (0%)      Tim3_x10_2   133 px    (0%)

The old detector Otsu-thresholded the bottom strip, accepted anything up to 25% of the crop
HEIGHT as "line-like" (54 px on a 1440 px image), and took the WIDEST survivor -- so a
tissue edge beat the bar. These tests build images where the true bar length is known by
construction, including the tissue-edge case that caused the failure.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.common.pixel_size_util import _detect_scale_bar  # noqa: E402

W, H = 1920, 1440


def _slide(tmp_path, name, bar_len=133, bar_thick=6, extras=()):
    """A pale field with a black bar bottom-right, plus any extra dark shapes."""
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 235, np.uint8)
    img[:int(H * 0.85)] = rng.integers(200, 240, (int(H * 0.85), W, 3), dtype=np.uint8)
    if bar_len:
        y, x = int(H * 0.93), W - 200
        img[y:y + bar_thick, x:x + bar_len] = 10
    for (ex, ey, ew, eh) in extras:
        img[ey:ey + eh, ex:ex + ew] = 20
    p = tmp_path / name
    Image.fromarray(img).save(p)
    return str(p)


def test_measures_a_clean_bar(tmp_path):
    px, bar = _detect_scale_bar(_slide(tmp_path, "clean.png", bar_len=133))
    assert bar == 133
    assert px == pytest.approx(100.0 / 133, rel=1e-6)


def test_a_wider_tissue_edge_does_not_win(tmp_path):
    """The exact failure on the real cohort: a dark structure wider than the bar.

    It is 300 px wide and 40 px tall — inside the old "line-like" allowance of 54 px, and
    wider than the bar, so the old detector returned 300 and a 2.3x scale error.
    """
    path = _slide(tmp_path, "edge.png", bar_len=133,
                  extras=[(100, int(H * 0.88), 300, 40)])
    px, bar = _detect_scale_bar(path)
    assert bar == 133, f"a tissue edge was measured instead of the bar: {bar}px"
    assert px == pytest.approx(100.0 / 133, rel=1e-6)


def test_a_textured_blob_is_rejected(tmp_path):
    """Wide, thin and dark, but not solid — must not be mistaken for a drawn bar."""
    rng = np.random.default_rng(1)
    img = np.full((H, W, 3), 235, np.uint8)
    y, x = int(H * 0.90), 200
    speckle = rng.integers(0, 2, (8, 400)) * 255
    img[y:y + 8, x:x + 400] = np.dstack([speckle] * 3).astype(np.uint8)
    img[int(H * 0.93):int(H * 0.93) + 6, W - 200:W - 200 + 133] = 10
    p = tmp_path / "speckle.png"
    Image.fromarray(img).save(p)
    assert _detect_scale_bar(str(p))[1] == 133


def test_no_bar_fails_closed(tmp_path):
    """No bar must yield None, never a guess — a missing number gets corrected, a wrong
    one silently rescales the analysis."""
    px, bar = _detect_scale_bar(_slide(tmp_path, "none.png", bar_len=0))
    assert px is None and bar is None


def test_bar_length_is_configurable(tmp_path):
    """Not every export preset draws 100 µm."""
    path = _slide(tmp_path, "cfg.png", bar_len=133)
    assert _detect_scale_bar(path, bar_um=250.0)[0] == pytest.approx(250.0 / 133, rel=1e-6)
    assert _detect_scale_bar(path, bar_um=50.0)[0] == pytest.approx(50.0 / 133, rel=1e-6)


def test_the_length_scales_the_answer(tmp_path):
    """A 2x longer bar means 2x smaller pixels."""
    a = _detect_scale_bar(_slide(tmp_path, "a.png", bar_len=100))[0]
    b = _detect_scale_bar(_slide(tmp_path, "b.png", bar_len=200))[0]
    assert a == pytest.approx(2 * b, rel=1e-6)
