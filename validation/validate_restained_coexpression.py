"""Focused self-test for the isolated restained co-expression workflow."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasis.restained.restained_coexpression import (discover_bundles, preprocess_hematoxylin,
                                    run_bundle, run_config, stain_channels,
                                    summarize_coexpression,
                                    validate_bundle_dimensions)


def _polygon_feature(x0, y0, x1, y1):
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
    ]]}, "properties": {}}


def run_selftest():
    with tempfile.TemporaryDirectory(prefix="restained-selftest-") as temp:
        root = Path(temp)

        colour_probe = np.asarray([[[150, 50, 50], [50, 50, 150], [245, 245, 245]]],
                                  dtype=np.uint8)
        h_od, aec_od = stain_channels(colour_probe)
        assert aec_od[0, 0] > aec_od[0, 1] and aec_od[0, 0] > aec_od[0, 2]
        assert h_od[0, 1] > h_od[0, 0] and h_od[0, 1] > h_od[0, 2]

        h_image = Image.new("RGB", (64, 64), "white")
        h_draw = ImageDraw.Draw(h_image)
        h_draw.ellipse((8, 8, 25, 25), fill=(100, 110, 175))
        h_draw.ellipse((36, 36, 53, 53), fill=(175, 180, 215))
        h_path = root / "sample_Hematoxylin.png"
        h_image.save(h_path)
        processed_path = root / "processed.png"
        provenance = preprocess_hematoxylin(h_path, processed_path)
        processed = np.asarray(Image.open(processed_path))
        assert processed.shape == (64, 64, 3)
        assert processed[16, 16].mean() < processed[0, 0].mean()
        assert provenance["percentiles"] == [1.0, 99.0]

        marker_a = Image.new("RGB", (64, 64), "white")
        marker_b = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(marker_a).rectangle((8, 8, 25, 25), fill=(150, 50, 50))
        ImageDraw.Draw(marker_b).rectangle((36, 36, 53, 53), fill=(150, 50, 50))
        a_path, b_path = root / "sample_CD8.png", root / "sample_FoxP3.png"
        marker_a.save(a_path); marker_b.save(b_path)

        reference_mask = Image.new("RGB", (64, 64), "black")
        mask_draw = ImageDraw.Draw(reference_mask)
        for box in ((8, 8, 25, 25), (36, 36, 53, 53)):
            mask_draw.rectangle(box, fill=(0, 0, 255), outline=(0, 255, 0), width=1)
        mask_path = root / "sample.png"
        reference_mask.save(mask_path)

        detections = {"type": "FeatureCollection", "features": [
            _polygon_feature(8, 8, 25, 25), _polygon_feature(36, 36, 53, 53),
        ]}
        detections_path = root / "detections.geojson"
        detections_path.write_text(json.dumps(detections))

        bundle = {"sample_id": "sample", "hematoxylin": str(h_path),
                  "marker_a": str(a_path), "marker_b": str(b_path),
                  "reference_mask": str(mask_path)}
        validate_bundle_dimensions(bundle)
        config = {
            "pixel_size_um": 0.5, "threshold_a": 0.2, "threshold_b": 0.2,
            "label_a": "CD8", "label_b": "FOXP3", "compartment_a": "nucleus",
            "compartment_b": "nucleus", "cell_expansion_um": 2.0,
            "preprocess_hematoxylin": True, "detections_geojson": str(detections_path),
        }
        result = run_bundle(bundle, config, root / "out")
        assert result["registration"]["performed"] is False
        assert result["coexpression"]["marker_a_only"] == 1
        assert result["coexpression"]["marker_b_only"] == 1
        assert result["coexpression"]["double_positive"] == 0
        assert result["segmentation"]["ground_truth_validation"]["f1"] == 1.0
        assert Path(result["artifacts"]["cells_csv"]).exists()
        assert Path(result["artifacts"]["overlay"]).exists()

        combined = run_config({
            **config, "mode": "single", "output_dir": str(root / "combined"),
            "hematoxylin_image": str(h_path), "marker_a_image": str(a_path),
            "marker_b_image": str(b_path), "reference_mask": str(mask_path),
            "sample_id": "sample_combined",
        }, progress=lambda _pct, _message: None)
        assert combined["n_samples"] == 1
        assert combined["results"][0]["coexpression"]["fisher_q_value_bh"] == 1.0
        assert Path(combined["result_json"]).exists()

        stats = summarize_coexpression([True, True, False, False],
                                       [True, False, True, False])
        assert stats["contingency_table"] == [[1, 1], [1, 1]]
        assert stats["double_positive_enrichment"] == 1.0

        batch = root / "batch"
        batch.mkdir()
        for source, target in ((h_path, "CaseA_Hematoxylin.png"),
                               (a_path, "CaseA_CD8.png"), (b_path, "CaseA_FoxP3.png")):
            Image.open(source).save(batch / target)
        Image.open(h_path).save(batch / "CaseB_Hematoxylin.png")
        complete, incomplete = discover_bundles(batch)
        assert len(complete) == 1 and complete[0]["sample_id"] == "CaseA"
        assert len(incomplete) == 1 and incomplete[0]["sample_id"] == "CaseB"

        mismatch = Image.new("RGB", (63, 64), "white")
        mismatch_path = root / "mismatch.png"
        mismatch.save(mismatch_path)
        bad_bundle = dict(bundle, marker_b=str(mismatch_path), reference_mask=None)
        try:
            validate_bundle_dimensions(bad_bundle)
            raise AssertionError("Dimension mismatch did not fail closed")
        except ValueError as exc:
            assert "dimension mismatch" in str(exc)

    print("RESTAINED CO-EXPRESSION SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_selftest())
