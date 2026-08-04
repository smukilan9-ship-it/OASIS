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


# ── The batch review must actually show the pairs it just segmented ─────────────────────
def _batch_staged(tmp_path):
    """A two-folder batch whose detections live where only the pre-check knows to look."""
    fa, fb = tmp_path / "cd8", tmp_path / "cd45"
    fa.mkdir(), fb.mkdir()
    (fa / "679_CD8_lm00.png").write_bytes(b"")
    (fb / "679_CD45_lm00.png").write_bytes(b"")
    out = tmp_path / "out"
    # The fan-out name: a pair with a drawn region is analysed one entry per certified
    # region and writes under "<sample_id>__roi0", which the review never sees.
    roi = out / "679_lm00__roi0"
    roi.mkdir(parents=True)
    ga = roi / "679_CD8_lm00_detections.geojson"
    gb = roi / "679_CD45_lm00_detections.geojson"
    ga.write_text(json.dumps(_detections([0.05, 0.15, 0.25, 0.35])))
    # no value sits ON a cutoff, so the counts below do not pin a boundary convention
    gb.write_text(json.dumps(_detections([0.10, 0.25, 0.30])))
    return fa, fb, out, ga, gb


def _precheck_stub(ga, gb):
    return {"status": "ok", "precheck_by_pair": {"679_lm00__roi0": {
        "sample_id": "679_lm00__roi0", "stain_a": "CD8", "stain_b": "CD45",
        "geojson_a": str(ga), "geojson_b": str(gb),
        "precheck": {"valid": False, "worst_status": "caution"},
        "null_plan": {"primary_null": "dense_morphology", "fail_closed": False},
    }}}


def test_a_batch_review_measures_the_batch_and_not_a_phantom_single_pair(tmp_path, monkeypatch):
    """The screen showed "no measured cells" for every image in a batch.

    `spatial_review_data` branched on `analysis_mode`, which is the QUANT tab's config key.
    Spatial sends `mode`, so a batch fell through to the SINGLE branch, read `image_a` and
    `image_b` (both absent in a batch config), and measured nothing — while the pre-check
    that had just segmented every pair sat unread in the same response.
    """
    fa, fb, out, ga, gb = _batch_staged(tmp_path)
    from oasis.webui.api import API
    api = API()
    monkeypatch.setattr(API, "run_spatial_association",
                        lambda self, cfg, **kw: _precheck_stub(ga, gb))

    res = api.spatial_review_data({"mode": "batch", "folder_a": str(fa), "folder_b": str(fb),
                                   "folder_mode": "two_folder", "output_dir": str(out),
                                   "dab_threshold_a": 0.20, "dab_threshold_b": 0.20})

    assert res["ok"], res.get("msg")
    assert len(res["pairs"]) == 1, "the batch collapsed to a phantom single pair"
    images = res["pairs"][0]["images"]
    assert [im["n_cells"] for im in images] == [4, 3], (
        f"the batch review measured nothing: {[im.get('msg') for im in images]}")
    assert all(im["hist"] for im in images)
    # and the cutoff must be the one the operator set, per image
    assert images[0]["positive_cells"] == 2 and images[1]["positive_cells"] == 2


def test_the_review_reads_the_files_the_precheck_named(tmp_path, monkeypatch):
    """Not the ones a filesystem glob happens to turn up.

    The histograms were re-derived by globbing "<image stem>*_detections.geojson" under the
    output tree. That is a guess about the worker's layout, and the ROI fan-out breaks it:
    here the detections carry a stem that does not match the image at all, which is exactly
    the case a glob cannot recover and a named path can.
    """
    fa, fb, out, ga, gb = _batch_staged(tmp_path)
    renamed_a = ga.with_name("segmented_A_detections.geojson")
    renamed_b = gb.with_name("segmented_B_detections.geojson")
    ga.rename(renamed_a), gb.rename(renamed_b)

    from oasis.webui.api import API
    api = API()
    monkeypatch.setattr(API, "run_spatial_association",
                        lambda self, cfg, **kw: _precheck_stub(renamed_a, renamed_b))

    res = api.spatial_review_data({"mode": "batch", "folder_a": str(fa), "folder_b": str(fb),
                                   "folder_mode": "two_folder", "output_dir": str(out),
                                   "dab_threshold_a": 0.20, "dab_threshold_b": 0.20})
    images = res["pairs"][0]["images"]
    assert [im["n_cells"] for im in images] == [4, 3]
    assert images[0]["geojson"] == str(renamed_a)


def test_a_missing_histogram_says_which_image_and_where_it_looked(tmp_path, monkeypatch):
    """"no measured cells" is not a diagnosis. It cost a full re-run to find out it meant
    "this screen never looked at your batch"."""
    fa, fb, out, ga, gb = _batch_staged(tmp_path)
    ga.unlink(), gb.unlink()

    from oasis.webui.api import API
    api = API()
    monkeypatch.setattr(API, "run_spatial_association",
                        lambda self, cfg, **kw: _precheck_stub(ga, gb))

    res = api.spatial_review_data({"mode": "batch", "folder_a": str(fa), "folder_b": str(fb),
                                   "folder_mode": "two_folder", "output_dir": str(out)})
    msg = res["pairs"][0]["images"][0]["msg"]
    assert "679_CD8_lm00.png" in msg and str(out) in msg


def test_the_precheck_reports_where_it_put_the_detections():
    """The review can only read named files if the worker names them."""
    import inspect

    import run_pipeline

    src = inspect.getsource(run_pipeline)
    i = src.index("BANDWIDTH_PRECHECK_JSON:")
    payload = src[i:i + 1400]
    assert '"geojson_a": geojson_a' in payload and '"geojson_b": geojson_b' in payload


def test_a_batch_cutoff_that_cannot_be_written_stops_the_run(tmp_path, monkeypatch):
    """It used to `continue`, on the reasoning that segmentation would apply the cutoff.

    It would not: `spatial_apply_review` forces `reuse_existing_geojson = True`, so skipping
    the rewrite runs the pair at whatever cutoff the pre-flight happened to use. For a batch
    the per-pair cutoffs live nowhere else — the config carries one A/B pair — so the run
    silently answers at a cutoff nobody chose.
    """
    fa, fb, out, ga, gb = _batch_staged(tmp_path)
    ga.unlink()
    from oasis.webui.api import API
    api = API()
    launched = []
    monkeypatch.setattr(API, "run_spatial_association",
                        lambda self, cfg, **kw: launched.append(cfg) or {"status": "ok"})

    res = api.spatial_apply_review({
        "output_dir": str(out),
        "reviewed_pairs": [{"sample_id": "679_lm00__roi0",
                            "path_a": str(fa / "679_CD8_lm00.png"), "threshold_a": 0.3,
                            "path_b": str(fb / "679_CD45_lm00.png"), "threshold_b": 0.3,
                            "geojson_a": str(ga), "geojson_b": str(gb)}]})

    assert res["status"] == "error"
    assert "679_CD8_lm00.png" in res["error"]
    assert not launched, "the run started anyway, at a cutoff the operator did not choose"


def test_the_reviewed_cutoff_is_written_into_the_reviewed_cells(tmp_path, monkeypatch):
    """Named file wins over a filename search — the two disagree under the ROI fan-out."""
    fa, fb, out, ga, gb = _batch_staged(tmp_path)
    renamed = ga.with_name("segmented_A_detections.geojson")
    ga.rename(renamed)
    from oasis.webui.api import API
    api = API()
    monkeypatch.setattr(API, "run_spatial_association", lambda self, cfg, **kw: {"status": "ok"})

    res = api.spatial_apply_review({
        "output_dir": str(out),
        "reviewed_pairs": [{"sample_id": "679_lm00__roi0",
                            "path_a": str(fa / "679_CD8_lm00.png"), "threshold_a": 0.20,
                            "path_b": str(fb / "679_CD45_lm00.png"), "threshold_b": 0.20,
                            "geojson_a": str(renamed), "geojson_b": str(gb)}]})

    assert res.get("status") != "error", res.get("error")
    calls = [f["properties"]["classification"]["name"]
             for f in json.loads(renamed.read_text())["features"]]
    assert calls == ["Negative", "Negative", "Positive", "Positive"]
