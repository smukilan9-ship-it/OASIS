"""The cutoff set on the Spatial review screen must reach the statistic.

Reusing the pre-flight's segmentation is what makes the review screen fast: it appears
seconds after the pre-flight instead of segmenting a second time. But reuse means
run_pipeline skips segmentation and returns the summary it already has, and the DAB cutoff
is only applied INSIDE segmentation — so the statistic went on reading the classifications
baked into the GeoJSON at the pre-flight's cutoff. The slider, the live re-count and the
"cutoffs applied" log line all described a number the run ignored.

This asserts the rewrite happens, on disk, before the run is launched.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _detections(values):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
         "properties": {"measurements": {"DAB: Mean": v},
                        "classification": {"name": "Negative"}}}
        for v in values]}


@pytest.fixture
def staged(tmp_path):
    img = tmp_path / "SAMPLE_CD8.tif"
    img.write_bytes(b"")
    out = tmp_path / "out"
    (out / "SAMPLE_CD8").mkdir(parents=True)
    gj = out / "SAMPLE_CD8" / "SAMPLE_CD8_detections.geojson"
    gj.write_text(json.dumps(_detections([0.05, 0.15, 0.25, 0.35])))
    return img, out, gj


def test_the_reviewed_cutoff_is_written_into_the_cells(staged, monkeypatch):
    img, out, gj = staged
    from oasis.webui.api import API
    api = API()
    seen = {}
    monkeypatch.setattr(API, "run_spatial_association",
                        lambda self, cfg, **kw: seen.setdefault("cfg", cfg) or {"status": "ok"})

    api.spatial_apply_review({"image_a": str(img), "dab_threshold_a": 0.20,
                              "output_dir": str(out)})

    calls = [f["properties"]["classification"]["name"]
             for f in json.loads(gj.read_text())["features"]]
    assert calls == ["Negative", "Negative", "Positive", "Positive"], (
        f"the 0.20 cutoff did not reach the cells: {calls}")
    assert seen["cfg"]["reuse_existing_geojson"] is True, (
        "the run must still reuse the segmentation — rewriting is not re-segmenting")


def test_a_different_cutoff_gives_a_different_call(staged, monkeypatch):
    img, out, gj = staged
    from oasis.webui.api import API
    api = API()
    monkeypatch.setattr(API, "run_spatial_association", lambda self, cfg, **kw: {"status": "ok"})

    api.spatial_apply_review({"image_a": str(img), "dab_threshold_a": 0.30,
                              "output_dir": str(out)})
    calls = [f["properties"]["classification"]["name"]
             for f in json.loads(gj.read_text())["features"]]
    assert calls == ["Negative", "Negative", "Negative", "Positive"]
