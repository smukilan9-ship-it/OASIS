"""
smoke_test.py — prove a BUILT bundle actually segments, on the machine that built it.

WHY THIS EXISTS. "The build succeeded" and "the application works" are different claims,
and the gap between them is where shipping goes wrong. Every bundle-only failure found so
far — cv2 refusing to import inside a .app, torch missing from a build that completed
happily, index.html resolved relative to a path that does not exist once frozen — produced
a perfectly green build and a broken app. None of them were visible from source.

So the release workflow runs this against the frozen binary on each platform. It drives the
real entry point the same way a background job does (`--oasis-worker run_pipeline`), which
exercises the frozen import path, the vendored model, the image reader and the writers.

The input is the InstanSeg model's own reference image, which is vendored alongside the
weights. That means no synthetic noise and no external download: it is real tissue the model
was validated against, and the expected cell count is therefore meaningful rather than
arbitrary.

Usage (see .github/workflows/release.yml):
    python packaging/smoke_test.py prepare --dir <workdir>    # writes image + config
    <frozen binary> --oasis-worker run_pipeline --config <workdir>/config.yaml
    python packaging/smoke_test.py check --dir <workdir>      # asserts real output
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The reference image is 256x256 of real H-DAB tissue. On this input the model finds a few
# hundred nuclei; anything wildly outside that means the frozen build is not doing the same
# work as the source. Kept deliberately wide — this is a smoke test, not a parity gate.
# Exact reproduction is asserted separately by validate_native_segmenter.py check A.
MIN_CELLS = 50
MAX_CELLS = 2000


def prepare(workdir):
    import tifffile
    from oasis.common.paths import bundled_model_dir

    model = bundled_model_dir()
    if not model:
        sys.exit("vendored InstanSeg model not found — cannot smoke-test")

    arr = np.load(os.path.join(model, "test-input.npy"))   # (1, 3, H, W) float32, 1..255
    rgb = np.ascontiguousarray(arr[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8))

    indir = os.path.join(workdir, "in")
    os.makedirs(indir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "out"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "dash"), exist_ok=True)
    tifffile.imwrite(os.path.join(indir, "smoke.tif"), rgb, photometric="rgb")

    # Written as plain text rather than via yaml so this half runs even where PyYAML is not
    # importable for some reason; the frozen binary is what has to parse it.
    cfg = os.path.join(workdir, "config.yaml")
    with open(cfg, "w") as f:
        f.write(
            "segmenter: native\n"
            "device: cpu\n"            # the only device present on every runner
            "dab_threshold: 0.2\n"
            "default_pixel_size: 0.5\n"
            "magnification: manual\n"
            "mode: automated\n"
            "stain_type: hdab\n"
            "export_geojson: true\n"
            "generate_overlays: false\n"
            "instanseg_threads: 2\n"
            "tile_dims: 512\n"
            "timeout_seconds: 1800\n"
            "image_extensions: ['*.tif']\n"
            f"input_dir: {indir}\n"
            f"output_dir: {os.path.join(workdir, 'out')}\n"
            f"dashboard_dir: {os.path.join(workdir, 'dash')}\n"
        )
    print(f"prepared {rgb.shape} image and config at {cfg}")
    return 0


def check(workdir):
    out = os.path.join(workdir, "out")
    summaries = glob.glob(os.path.join(out, "**", "*_summary.json"), recursive=True)
    geojsons = glob.glob(os.path.join(out, "**", "*_detections.geojson"), recursive=True)

    if not summaries:
        print(f"FAIL: no summary JSON under {out}")
        for root, _, files in os.walk(out):
            for name in files:
                print("   found:", os.path.join(root, name))
        return 1

    with open(summaries[0]) as f:
        summary = json.load(f)
    cells = summary.get("total_cells")

    print(f"summary : {os.path.basename(summaries[0])}")
    print(f"cells   : {cells}")
    print(f"device  : {summary.get('segmenter_device')}")
    print(f"geojson : {len(geojsons)} file(s)")

    if not isinstance(cells, int) or not (MIN_CELLS <= cells <= MAX_CELLS):
        print(f"FAIL: {cells} cells is outside the expected {MIN_CELLS}-{MAX_CELLS} range "
              f"for the model's own reference image")
        return 1
    if not geojsons:
        print("FAIL: no detections GeoJSON was written")
        return 1

    print("PASS: the frozen bundle segmented real tissue end to end")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "check"])
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    return prepare(args.dir) if args.action == "prepare" else check(args.dir)


if __name__ == "__main__":
    sys.exit(main())
