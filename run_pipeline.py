"""
OASIS — Main Pipeline v5
Full automated flow with overlay generation and dashboard.
Usage: python run_pipeline.py [--config path/to/config.yaml]
"""

import os
import sys
import subprocess
import glob
import json
import time
import argparse
import numpy as np

try:
    import yaml
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
    import yaml


# ==========================================================
# CONFIG LOADER
# ==========================================================

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    path_keys = ["input_dir", "output_dir", "dashboard_dir",
                 "qupath_binary", "instanseg_model"]
    for key in path_keys:
        if key in cfg and cfg[key]:
            cfg[key] = os.path.expanduser(str(cfg[key]))

    # input_dir is a quantification-only field. Spatial association supplies
    # explicit image paths via spatial_pairs (+ pixel_overrides), so input_dir is
    # irrelevant there and must not be required. QuPath/InstanSeg are still
    # needed (spatial runs segmentation), so they stay required for every mode.
    required = ["qupath_binary", "instanseg_model"]
    if cfg.get("mode") != "spatial":
        required = ["input_dir"] + required
    for key in required:
        if not cfg.get(key):
            print(f"ERROR: '{key}' is required in config.yaml")
            sys.exit(1)

    if not os.path.exists(cfg["qupath_binary"]):
        print(f"ERROR: QuPath not found at: {cfg['qupath_binary']}")
        sys.exit(1)

    if not os.path.exists(os.path.expanduser(cfg["instanseg_model"])):
        print(f"ERROR: InstanSeg model not found at: {cfg['instanseg_model']}")
        sys.exit(1)

    cfg.setdefault("output_dir", os.path.expanduser("~/Desktop/ihc_results"))
    # Derive dashboard_dir from input_dir only when present (spatial mode has no
    # input_dir); otherwise fall back to output_dir.
    if cfg.get("input_dir"):
        cfg.setdefault("dashboard_dir", os.path.join(cfg["input_dir"], "output_results"))
    else:
        cfg.setdefault("dashboard_dir", cfg["output_dir"])
    cfg.setdefault("dab_threshold", 0.2)
    cfg.setdefault("default_pixel_size", 0.5)
    cfg.setdefault("magnification", "auto")
    cfg.setdefault("instanseg_threads", 4)
    cfg.setdefault("device", "mps")
    cfg.setdefault("tile_dims", 512)
    cfg.setdefault("timeout_seconds", 1800)
    cfg.setdefault("mode", "automated")
    cfg.setdefault("image_extensions", ["*.tif", "*.tiff", "*.svs", "*.ndpi", "*.png"])
    cfg.setdefault("export_geojson", True)
    cfg.setdefault("generate_overlays", True)
    cfg.setdefault("overlay_downsample", 1.0)
    cfg.setdefault("pixel_size_mode", "global")
    cfg.setdefault("objective", "10x")
    # Quant-tab options (also settable from CLI config):
    #   adaptive_threshold  – classify each image at a per-image Otsu cut on the
    #                         cell DAB:Mean distribution instead of a fixed OD.
    #   preprocess_normalize– white-balance each input to a per-image white point
    #                         before segmentation/measurement (corrects tone/
    #                         illumination without rescaling the DAB signal).
    cfg.setdefault("adaptive_threshold", False)
    cfg.setdefault("preprocess_normalize", False)


    return cfg


# ==========================================================
# GROOVY SCRIPT GENERATOR
# ==========================================================

def generate_groovy_script(cfg, script_path="generated_pipeline.groovy", img_path=None,
                           threshold_override=None):
    model_path = os.path.expanduser(cfg["instanseg_model"])

    img_base = os.path.basename(img_path) if img_path else ""

    # An explicit per-image threshold (passed directly, or via
    # cfg["threshold_overrides"] keyed by filename — set by the Spatial
    # Association UI) takes priority over everything and skips the
    # stain_thresholds filename lookup entirely.
    if threshold_override is None and img_base:
        threshold_override = (cfg.get("threshold_overrides") or {}).get(img_base)

    if threshold_override is not None:
        threshold = threshold_override
        print(f"  DAB threshold: {threshold} OD (per-image override from UI)")
    else:
        # Per-stain DAB threshold: if stain_thresholds is configured, match the
        # image filename (case-insensitive substring) against its keys and use the
        # first matching threshold; otherwise fall back to the global dab_threshold.
        threshold = cfg["dab_threshold"]
        stain_thresholds = cfg.get("stain_thresholds")
        if stain_thresholds:
            img_name = img_base.lower()
            matched_token = None
            for token, thr in stain_thresholds.items():
                if img_name and str(token).lower() in img_name:
                    threshold, matched_token = thr, token
                    break
            if matched_token is not None:
                print(f"  DAB threshold: {threshold} OD "
                      f"(stain '{matched_token}' matched in '{img_base}')")
            else:
                why = "no stain token matched filename" if img_name \
                    else "no image filename available"
                print(f"  DAB threshold: {threshold} OD (default dab_threshold; {why})")

    # Adaptive (Otsu) threshold: applies when enabled and no explicit per-image
    # override is in force. The cut is computed in-Groovy from this image's own
    # cell DAB:Mean distribution, so it adapts to per-slide stain intensity.
    adaptive = bool(cfg.get("adaptive_threshold")) and threshold_override is None
    if adaptive:
        threshold_block = f'''def _dabVals = detections.collect {{ it.getMeasurementList().get("DAB: Mean") }}.findAll {{ it != null && !it.isNaN() }}
double threshold
if (_dabVals.size() >= 20) {{
    double _mx = _dabVals.max(); if (_mx <= 0) _mx = 1e-6
    int _NB = 256; int[] _h = new int[_NB]
    _dabVals.each {{ v -> int _bi = (int)Math.min(_NB - 1, Math.max(0, Math.round(v / _mx * (_NB - 1)))); _h[_bi]++ }}
    int _tot = _dabVals.size(); double _sum = 0.0
    for (int i = 0; i < _NB; i++) _sum += (double)i * _h[i]
    double _sumB = 0.0; int _wB = 0; double _maxVar = -1.0; int _thrBin = 0
    for (int i = 0; i < _NB; i++) {{
        _wB += _h[i]; if (_wB == 0) continue; int _wF = _tot - _wB; if (_wF == 0) break
        _sumB += (double)i * _h[i]; double _mB = _sumB / _wB; double _mF = (_sum - _sumB) / _wF
        double _var = (double)_wB * _wF * (_mB - _mF) * (_mB - _mF)
        if (_var > _maxVar) {{ _maxVar = _var; _thrBin = i }}
    }}
    threshold = _thrBin / (double)(_NB - 1) * _mx
    println "DAB threshold: " + threshold + " (ADAPTIVE Otsu over " + _tot + " cells)"
}} else {{
    threshold = {threshold}
    println "DAB threshold: " + threshold + " (fixed fallback; <20 cells for Otsu)"
}}'''
    else:
        threshold_block = (f'double threshold = {threshold}\n'
                           f'println "DAB threshold: " + threshold + " (fixed OD)"')

    device = cfg["device"]
    threads = cfg["instanseg_threads"]
    tile_dims = cfg["tile_dims"]
    output_dir = os.path.expanduser(cfg["output_dir"])
    default_pixel_size = cfg.get("_resolved_pixel_size", cfg.get("default_pixel_size", 0.5))

    script = f"""import qupath.ext.instanseg.core.InstanSeg
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.common.GeneralTools

def imageData = getCurrentImageData()
if (imageData == null) {{ println "ERROR: No image loaded"; return }}
def server = imageData.getServer()

println "======================================"
println "STARTING ANALYSIS"
println "======================================"
println "Image: " + server.getMetadata().getName()
println "Width: " + server.getWidth()
println "Height: " + server.getHeight()

removeAllObjects()
setImageType('BRIGHTFIELD_H_DAB')

double pixelSize = {default_pixel_size}
setPixelSizeMicrons(pixelSize, pixelSize)
println "Pixel size: " + pixelSize + " um/px"

def roi = ROIs.createRectangleROI(0, 0, server.getWidth(), server.getHeight(), ImagePlane.getDefaultPlane())
def annotation = PathObjects.createAnnotationObject(roi)
addObject(annotation)
selectObjects(annotation)
println "Full image annotation created"

println "Running InstanSeg..."
def instanseg = InstanSeg.builder()
    .modelPath("{model_path}")
    .device("{device}")
    .nThreads({threads})
    .tileDims({tile_dims})
    .interTilePadding(32)
    .makeMeasurements(true)
    .randomColors(false)
    .build()

instanseg.detectObjects()
println "InstanSeg completed"

def detections = getDetectionObjects()
println "Total cells detected: " + detections.size()
if (detections.isEmpty()) {{ println "WARNING: No detections found"; return }}

{threshold_block}

def positiveClass = getPathClass("Positive")
def negativeClass = getPathClass("Negative")
int positiveCount = 0
int negativeCount = 0

detections.each {{ cell ->
    def dab = cell.getMeasurementList().get("DAB: Mean")
    if (dab != null && !dab.isNaN() && dab > threshold) {{
        cell.setPathClass(positiveClass); positiveCount++
    }} else {{
        cell.setPathClass(negativeClass); negativeCount++
    }}
}}

fireHierarchyUpdate()
double positivityPct = (positiveCount * 100.0) / detections.size()

println "======================================"
println "FINAL RESULTS"
println "======================================"
println "Total cells:    " + detections.size()
println "Positive cells: " + positiveCount
println "Negative cells: " + negativeCount
println "Positivity %:   " + String.format("%.2f", positivityPct)

def outDir = new File("{output_dir}")
if (!outDir.exists()) outDir.mkdirs()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

def csvPath = new File(outDir, imageName + "_detections.csv").getAbsolutePath()
saveDetectionMeasurements(csvPath)
println "CSV exported to: " + csvPath

def jsonPath = new File(outDir, imageName + "_summary.json").getAbsolutePath()
def summary = \"\"\"{{
    "image": "${{server.getMetadata().getName()}}",
    "pixel_size_um": ${{pixelSize}},
    "image_width": ${{server.getWidth()}},
    "image_height": ${{server.getHeight()}},
    "total_cells": ${{detections.size()}},
    "positive_cells": ${{positiveCount}},
    "negative_cells": ${{negativeCount}},
    "positivity_pct": ${{String.format("%.2f", positivityPct)}},
    "dab_threshold": ${{String.format("%.4f", threshold)}}
}}\"\"\"
new File(jsonPath).text = summary.trim()
println "JSON exported to: " + jsonPath
"""

    if cfg.get("export_geojson", True):
        script += f"""
def geojsonPath = new File("{output_dir}", imageName + "_detections.geojson").getAbsolutePath()
exportObjectsToGeoJson(getDetectionObjects(), geojsonPath, "FEATURE_COLLECTION")
println "GeoJSON exported to: " + geojsonPath
"""

    script += """
println "======================================"
println "PIPELINE FINISHED SUCCESSFULLY"
println "======================================"
"""

    with open(script_path, "w") as f:
        f.write(script)
    return script_path


# ==========================================================
# CONFIDENCE + JSON PARSER
# ==========================================================

def compute_confidence(total, pos_pct):
    if total < 50: return "LOW"
    if pos_pct < 0.1: return "LOW"
    if pos_pct > 95: return "LOW"
    return "NORMAL"


def parse_qupath_output(json_path):
    if not os.path.exists(json_path):
        print(f"  Missing results: {json_path}")
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        total = data["total_cells"]
        pos_pct = float(data["positivity_pct"])
        return {
            "Image_Name": data["image"],
            "Total_Cells": total,
            "Positive_Cells": data["positive_cells"],
            "Negative_Cells": data["negative_cells"],
            "Positivity_Index_Pct": round(pos_pct, 2),
            "DAB_Threshold": data.get("dab_threshold", 0.2),
            "Pixel_Size_um": data.get("pixel_size_um", 0.5),
            "Pixel_Size_Source": data.get("pixel_size_source", "unknown"),
            "Cells_Per_mm2": data.get("cells_per_mm2"),
            "Pixel_Size_Warning": data.get("pixel_size_warning", False),
            "Confidence": compute_confidence(total, pos_pct)
        }
    except Exception as e:
        print(f"  Failed to parse JSON: {e}")
        return None


# ==========================================================
# METADATA
# ==========================================================

def save_metadata(metrics, metadata_dir):
    os.makedirs(metadata_dir, exist_ok=True)
    safe_name = "".join(c for c in metrics["Image_Name"] if c.isalnum() or c in "._- ")
    with open(os.path.join(metadata_dir, f"{safe_name}_metadata.json"), "w") as f:
        json.dump(metrics, f, indent=4)


# ==========================================================
# RUN SINGLE IMAGE
# ==========================================================

def _run_qupath(img_path, cfg, groovy_script):
    """Run QuPath on a single image. Returns json_path or None."""
    img_filename = os.path.basename(img_path)
    command = [cfg["qupath_binary"], "script", "-i", img_path, groovy_script]
    start_time = time.time()
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = "-Djava.awt.headless=true"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True
    )
    stdout_lines = []
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean = line.strip()
                stdout_lines.append(clean)
                if clean and not any(x in clean for x in [
                    "INFO", "WARN", "Measured Detection",
                    "Completed Annotation", "LOADER", "PyInstaller"
                ]):
                    print(clean)
        process.wait(timeout=cfg["timeout_seconds"])
    except subprocess.TimeoutExpired:
        process.kill()
        return None, round(time.time()-start_time,2)

    runtime = round(time.time()-start_time, 2)
    log_dir = os.path.join(cfg["dashboard_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{img_filename}_stdout.log"), "w") as f:
        f.write("\n".join(stdout_lines))
    with open(os.path.join(log_dir, f"{img_filename}_stderr.log"), "w") as f:
        f.write(process.stderr.read())

    if process.returncode != 0:
        return None, runtime

    clean_prefix = os.path.splitext(img_filename)[0]
    matches = glob.glob(os.path.join(cfg["output_dir"], f"{clean_prefix}*_summary.json"))
    return (matches[0] if matches else None), runtime


def _classification_name(props):
    """Read a GeoJSON detection's classification name (dict or string form)."""
    cls = props.get("classification", {})
    if isinstance(cls, dict):
        return cls.get("name", "")
    return str(cls) if cls else ""


def _apply_cytoplasm_measurement(img_path, json_path, cfg):
    """
    Re-measure DAB in the cytoplasmic ring (membrane markers) and reclassify each
    cell, writing the new classification AND all three DAB measurements (nucleus,
    cytoplasm, cell) back into the GeoJSON properties and the summary JSON so that
    overlays / spatial association / dashboard all use the cytoplasm-based classification.

    Cells with degenerate geometry keep their original nuclear classification.
    """
    output_dir = cfg["output_dir"]
    geojson = _find_geojson(img_path, output_dir)
    if not geojson:
        print("  Cytoplasm measurement: GeoJSON not found — skipping")
        return

    pixel_size = cfg.get("_resolved_pixel_size", cfg.get("default_pixel_size", 0.5))
    expansion  = float(cfg.get("cell_expansion_um", 2.0))

    try:
        with open(json_path) as f:
            summ = json.load(f)
    except Exception:
        summ = {}
    # The threshold QuPath actually used is baked into the summary JSON.
    threshold = float(summ.get("dab_threshold", cfg.get("dab_threshold", 0.2)))

    # Membrane-completeness classification (opt-in). When membrane_pix_thr AND
    # membrane_frac_min are configured, a cell is positive iff a sufficient
    # FRACTION of its ring is stained (and, if set, its brightest arc clears
    # membrane_p90_thr) — the statistic that recovers faint membranous TIM-3
    # positives the ring MEAN dilutes away. Cutoffs come from
    # validation/tune_membrane_threshold.py. Absent → legacy mean>threshold.
    # Per-image membrane cutoffs (Spatial Association: image A and image B carry
    # DIFFERENT calibrated markers, e.g. CD8 vs TIM-3, so a single global cutoff
    # would be wrong). A per-image entry wins; fall back to the global cfg value.
    mo = (cfg.get("membrane_overrides") or {}).get(os.path.basename(img_path), {})
    pix_thr  = mo.get("membrane_pix_thr",  cfg.get("membrane_pix_thr"))
    frac_min = mo.get("membrane_frac_min", cfg.get("membrane_frac_min"))
    p90_thr  = mo.get("membrane_p90_thr",  cfg.get("membrane_p90_thr"))
    use_completeness = pix_thr is not None and frac_min is not None

    from cell_expansion import measure_cytoplasm_dab
    results = measure_cytoplasm_dab(
        img_path, geojson, pixel_size,
        expansion_um=expansion, dab_threshold=threshold,
        membrane_pix_thr=float(pix_thr) if use_completeness else None,
    )

    with open(geojson) as f:
        gj = json.load(f)
    features = gj.get("features", [])

    was_pos = 0
    for feat, res in zip(features, results):
        props = feat.setdefault("properties", {})
        if "positive" in _classification_name(props).lower():
            was_pos += 1
        if not res or res.get("cytoplasm_dab_mean") is None:
            continue                       # degenerate cell — keep nuclear class
        if use_completeness:
            frac = res.get("membrane_pos_frac")
            p90  = res.get("cytoplasm_dab_p90")
            is_pos = (frac is not None and frac >= float(frac_min)
                      and (p90_thr is None or (p90 is not None and p90 > float(p90_thr))))
        else:
            is_pos = res["cytoplasm_dab_mean"] > threshold
        props["classification"] = {
            "name":  "Positive" if is_pos else "Negative",
            "color": [255, 0, 0] if is_pos else [0, 200, 0],
        }
        meas = props.get("measurements")
        if not isinstance(meas, dict):
            meas = {}
        meas["Nucleus: DAB OD mean"]   = res.get("nucleus_dab_mean")
        meas["Cytoplasm: DAB OD mean"] = res.get("cytoplasm_dab_mean")
        meas["Cytoplasm: DAB OD p90"]  = res.get("cytoplasm_dab_p90")
        meas["Membrane: positive fraction"] = res.get("membrane_pos_frac")
        meas["Cell: DAB OD mean"]      = res.get("cell_dab_mean")
        props["measurements"] = meas
        # Store the cytoplasm-ring boundary (Voronoi-clipped expanded cell) so the
        # spatial segmentation overlay can draw the measured membrane compartment.
        if res.get("cell_polygon"):
            props["cyto_polygon"] = res["cell_polygon"]

    with open(geojson, "w") as f:
        json.dump(gj, f)

    # Keep the tab-delimited QuPath detection export consistent with the GeoJSON
    # and summary. Previously it silently retained the raw nuclear classes while
    # the UI showed cytoplasm classes.
    try:
        import csv
        stem = os.path.splitext(os.path.basename(img_path))[0]
        csv_matches = glob.glob(os.path.join(cfg["output_dir"], f"{stem}*_detections.csv"))
        if csv_matches:
            csv_path = csv_matches[0]
            with open(csv_path, newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))
            if rows and len(rows) - 1 == len(features):
                header = rows[0]
                class_i = header.index("Classification")
                extra = ["Nucleus: DAB OD mean", "Cytoplasm: DAB OD mean", "Cell: DAB OD mean"]
                for name in extra:
                    if name not in header:
                        header.append(name)
                for row, feat in zip(rows[1:], features):
                    props = feat.get("properties", {})
                    row[class_i] = _classification_name(props)
                    meas = props.get("measurements", {}) or {}
                    for name in extra:
                        idx = header.index(name)
                        while len(row) <= idx:
                            row.append("")
                        row[idx] = meas.get(name, "")
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)
    except Exception as e:
        print(f"  WARNING: could not synchronize cytoplasm detection CSV: {e}")

    total = len(features)
    pos = sum(1 for ft in features
              if "positive" in _classification_name(ft.get("properties", {})).lower())
    neg = total - pos

    summ["positive_cells"] = pos
    summ["negative_cells"] = neg
    if total:
        summ["total_cells"]    = total
        summ["positivity_pct"] = round(pos * 100.0 / total, 2)
    summ["measurement_compartment"] = "cytoplasm"
    summ["cell_expansion_um"]       = expansion
    summ["cytoplasm_dab_calibrated_to_qupath"] = True
    summ["membrane_classifier"] = "completeness" if use_completeness else "ring_mean"
    if use_completeness:
        summ["membrane_pix_thr"]  = float(pix_thr)
        summ["membrane_frac_min"] = float(frac_min)
        summ["membrane_p90_thr"]  = float(p90_thr) if p90_thr is not None else None
        # Membrane-quality gate: on very faint/low-contrast tissue the ring
        # background creeps up to the pixel threshold, so real positives can't be
        # separated and the marker over-calls (validated failure mode: the faint
        # 92290_IM slide gave held-out F1 0.30 with ~50% positive and a
        # threshold-minus-background margin of ~0.016, vs 28–34% positive and
        # margin 0.05–0.07 on callable slides). Flag such runs as low-confidence
        # rather than silently reporting them. Heuristic bounds calibrated on a
        # small labelled set — a soft warning, not a hard block.
        means = [r["cytoplasm_dab_mean"] for r in results
                 if r and r.get("cytoplasm_dab_mean") is not None]
        pos_rate = pos / total if total else 0.0
        bg_margin = None
        if means:
            import statistics as _st
            bg_margin = float(pix_thr) - float(_st.median(means))
        summ["membrane_positive_rate"] = round(pos_rate, 4)
        summ["membrane_bg_margin"] = round(bg_margin, 4) if bg_margin is not None else None
        summ["membrane_quality_warning"] = bool(
            pos_rate > 0.50 or (bg_margin is not None and bg_margin < 0.03))
        if summ["membrane_quality_warning"]:
            print(f"  ⚠ Membrane quality: LOW — positive rate {pos_rate:.1%}, "
                  f"background margin {bg_margin:.3f}. Faint/low-contrast tissue; "
                  f"TIM-3 may not be reliably callable here.")
    with open(json_path, "w") as f:
        json.dump(summ, f, indent=4)

    print(f"  Cytoplasm measurement: {total} cells, {pos} positive "
          f"(was {was_pos} with nuclear), expansion {expansion} µm")


def _normalized_copy(img_path, cfg):
    """Write a per-image white-balanced copy (same basename) and return its path.

    Corrects tone/illumination to a per-image white point so QuPath's H-DAB
    deconvolution and InstanSeg see a consistent input across slides. Scales each
    channel so the slide's own background maps to white — it does NOT rescale the
    DAB signal. Best-effort: on very large WSIs or any failure, returns None so
    the caller falls back to the original image.
    """
    try:
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        # Guard against giant WSIs — normalization loads full-res into memory.
        if os.path.getsize(img_path) > 400 * 1024 * 1024:
            print("  Preprocessing: skipped (image too large to normalize safely)")
            return None
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        flat = rgb.reshape(-1, 3).astype(np.float64)
        bright = flat[flat.mean(1) > np.percentile(flat.mean(1), 80)]
        bg = np.clip(np.percentile(bright, 99, axis=0), 200, 255)
        out = np.clip(rgb.astype(np.float64) * (255.0 / bg.reshape(1, 1, 3)),
                      0, 255).astype(np.uint8)
        pre_dir = os.path.join(cfg["output_dir"], "_preproc")
        os.makedirs(pre_dir, exist_ok=True)
        dest = os.path.join(pre_dir, os.path.basename(img_path))
        Image.fromarray(out).save(dest)
        return dest
    except Exception as e:
        print(f"  Preprocessing: normalization failed ({e}) — using original image")
        return None


def run_single_image(img_path, cfg, groovy_script):
    img_filename = os.path.basename(img_path)

    print(f"\n{'='*50}")
    print(f"PROCESSING: {img_filename}")
    print(f"{'='*50}")

    pixel_size_mode = cfg.get("pixel_size_mode", "global")

    try:
        from pixel_size_util import get_pixel_size_with_source
        pixel_size, pixel_size_source = get_pixel_size_with_source(
            img_path, cfg, interactive=(cfg.get("mode") == "expert"))
    except ImportError:
        pixel_size = cfg.get("default_pixel_size", 0.5)
        pixel_size_source = "default_fallback"
        print(f"  Using default pixel size: {pixel_size} µm/px")

    # Record both the value AND its provenance. The spatial pipeline reads these
    # back so the Ripley's K pixel size is exactly the one this image's
    # segmentation used (and so a silent fall-through can be flagged). The
    # quantification path ignores the source field — behaviour unchanged.
    cfg["_resolved_pixel_size"] = pixel_size
    cfg["_resolved_pixel_size_source"] = pixel_size_source

    # Segmentation reuse: if a prior step (e.g. the 75 µm bandwidth pre-flight) already
    # produced this image's summary JSON + GeoJSON in the output dir and they are newer
    # than the source image, skip the (expensive) QuPath run and reuse them. Gated by an
    # explicit flag so CLI re-runs are never silently short-circuited.
    if cfg.get("reuse_existing_geojson"):
        _prefix = os.path.splitext(img_filename)[0]
        _js = sorted(glob.glob(os.path.join(cfg["output_dir"],
                                            f"{_prefix}*_summary.json")))
        _geo = _find_geojson(img_path, cfg["output_dir"])
        if (_js and _geo and os.path.exists(_geo)
                and os.path.getmtime(_js[0]) >= os.path.getmtime(img_path)):
            print(f"  Reusing existing segmentation for {img_filename} (skip QuPath)")
            return _js[0]

    generate_groovy_script(cfg, groovy_script, img_path)

    # Optional preprocessing: run segmentation/measurement on a per-image
    # white-balanced copy (tone/illumination correction; does NOT rescale DAB).
    # Overlays/naming still key off the original image basename.
    qp_input = img_path
    if cfg.get("preprocess_normalize"):
        norm = _normalized_copy(img_path, cfg)
        if norm:
            qp_input = norm
            print("  Preprocessing: per-image white-balance normalization applied")

    command = [cfg["qupath_binary"], "script", "-i", qp_input, groovy_script]
    start_time = time.time()

    # On macOS, use 'open -gj' equivalent by setting LSUIElement via env
    # This prevents QuPath from stealing focus when launched headlessly
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = "-Djava.awt.headless=true"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True  # prevents focus steal on macOS
    )
    stdout_lines = []
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean = line.strip()
                stdout_lines.append(clean)
                if clean and not any(x in clean for x in [
                    "INFO", "WARN", "Measured Detection",
                    "Completed Annotation", "LOADER", "PyInstaller"
                ]):
                    print(clean)
        process.wait(timeout=cfg["timeout_seconds"])
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"TIMEOUT")
        return None

    runtime = round(time.time() - start_time, 2)

    log_dir = os.path.join(cfg["dashboard_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{img_filename}_stdout.log"), "w") as f:
        f.write("\n".join(stdout_lines))
    with open(os.path.join(log_dir, f"{img_filename}_stderr.log"), "w") as f:
        f.write(process.stderr.read())

    if process.returncode != 0:
        print(f"FAILED (exit code {process.returncode})")
        return None

    print(f"Completed in {runtime}s")

    clean_prefix = os.path.splitext(img_filename)[0]
    matches = glob.glob(os.path.join(cfg["output_dir"],
                                      f"{clean_prefix}*_summary.json"))
    if not matches:
        print(f"ERROR: No JSON results found")
        return None
    json_path = matches[0]

    # Pixel-size provenance + a nucleus-density plausibility check, written into
    # the summary so the UI can flag a run whose pixel size is a silent fallback
    # or implausible (wrong pixel size → nonsensical cell density).
    try:
        with open(json_path) as f:
            _summ = json.load(f)
        _summ["pixel_size_um"] = pixel_size
        _summ["pixel_size_source"] = pixel_size_source
        w = _summ.get("image_width"); h = _summ.get("image_height")
        tot = _summ.get("total_cells", 0)
        dens = None
        if w and h and pixel_size:
            area_mm2 = (w * h * pixel_size * pixel_size) / 1e6
            if area_mm2 > 0:
                dens = tot / area_mm2
                _summ["cells_per_mm2"] = round(dens, 1)
        # Amber flag: default fallback pixel size, or a density outside the broad
        # plausible range for nucleated tissue (~100–20000 cells/mm²).
        _summ["pixel_size_warning"] = bool(
            pixel_size_source == "default_fallback"
            or (dens is not None and (dens < 100 or dens > 20000)))
        with open(json_path, "w") as f:
            json.dump(_summ, f, indent=4)
    except Exception as e:
        print(f"  WARNING: could not annotate pixel-size QC: {e}")

    # Membrane markers: re-measure DAB in the cytoplasmic ring (per-image flag,
    # default off → nuclear CD8/quant behaviour is unchanged unless enabled).
    use_cyto = cfg.get("cytoplasm_overrides", {}).get(
        img_filename, cfg.get("use_cytoplasm_measurement", False))
    if not use_cyto:
        # Per-biomarker membrane routing (Quantification UI): match the filename
        # (case-insensitive substring) against cytoplasm_stain_tokens — symmetric
        # with the stain_thresholds token match used for the DAB threshold.
        lf = (img_filename or "").lower()
        for tok in (cfg.get("cytoplasm_stain_tokens") or []):
            if lf and str(tok).lower() in lf:
                use_cyto = True
                print(f"  Cytoplasm measurement: stain token '{tok}' matched "
                      f"'{img_filename}' → measuring membrane ring")
                break
    if use_cyto:
        try:
            _apply_cytoplasm_measurement(img_path, json_path, cfg)
        except Exception as e:
            print(f"  Cytoplasm measurement failed: {e} — keeping nuclear classification")

    return json_path


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(config_path="config.yaml"):
    cfg = load_config(config_path)

    print(f"\n{'='*55}")
    print(f"  IHC ANALYZER")
    print(f"{'='*55}")
    print(f"  Mode:        {cfg['mode']}")
    print(f"  Stain:       {cfg.get('stain_type', 'hdab').upper()}")
    print(f"  Threshold:   {cfg['dab_threshold']} OD")
    print(f"  Magnification: {cfg['magnification']}")
    print(f"  Input:       {cfg['input_dir']}")
    print(f"  Output:      {cfg['output_dir']}")
    print(f"{'='*55}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["dashboard_dir"], exist_ok=True)
    metadata_dir = os.path.join(cfg["dashboard_dir"], "metadata")

    groovy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "generated_pipeline.groovy"
    )

    # Find images
    images = []
    for ext in cfg["image_extensions"]:
        images.extend(glob.glob(os.path.join(cfg["input_dir"], ext)))

    # Single-image mode (Quant UI): restrict to an explicit basename whitelist.
    whitelist = cfg.get("image_whitelist")
    if whitelist:
        wl = {str(w) for w in whitelist}
        images = [p for p in images if os.path.basename(p) in wl]

    if not images:
        print(f"\nNo images found in: {cfg['input_dir']}")
        return

    print(f"\nFound {len(images)} image(s)\n")

    batch_metrics = []
    for img_path in images:
        json_path = run_single_image(img_path, cfg, groovy_path)
        if json_path is None:
            continue
        metrics = parse_qupath_output(json_path)
        if not metrics:
            continue
        batch_metrics.append(metrics)
        save_metadata(metrics, metadata_dir)
        print(f"  Total cells:  {metrics['Total_Cells']}")
        print(f"  DAB positive: {metrics['Positive_Cells']} ({metrics['Positivity_Index_Pct']}%)")

    if not batch_metrics:
        print("\nNo results to summarize")
        return

    # ── Stage 2+3: Generate overlays ──────────────────────
    if cfg.get("generate_overlays", True) and cfg.get("export_geojson", True):
        print("\nGenerating overlays...")
        try:
            from overlay import generate_overlays_for_batch
            overlay_paths = generate_overlays_for_batch(
                batch_metrics=batch_metrics,
                input_dir=cfg["input_dir"],
                output_dir=cfg["output_dir"],
                downsample=cfg.get("overlay_downsample", 1.0),
            )
            print(f"  Generated {len(overlay_paths)} overlay(s)")
        except Exception as e:
            print(f"  Overlay generation failed: {e}")

    # ── Stage 3: Dashboard + Excel ─────────────────────────
    print(f"\nGenerating dashboard for {len(batch_metrics)} images...")
    try:
        from dashboard import generate_all_outputs
        html_path, excel_path = generate_all_outputs(
            batch_metrics, cfg["dashboard_dir"], cfg
        )
        print(f"\n  Open dashboard: file://{html_path}")
    except Exception as e:
        print(f"  Dashboard generation failed: {e}")

    # ── Final summary ──────────────────────────────────────
    total_cells = sum(m["Total_Cells"] for m in batch_metrics)
    total_pos = sum(m["Positive_Cells"] for m in batch_metrics)
    avg_pct = total_pos / total_cells * 100 if total_cells > 0 else 0

    print(f"\n{'='*55}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*55}")
    print(f"  Images analyzed: {len(batch_metrics)}")
    print(f"  Total cells:     {total_cells:,}")
    print(f"  DAB positive:    {total_pos:,} ({avg_pct:.2f}%)")
    print(f"{'='*55}")
    # ── Cleanup intermediate files ──────────────────────────
    if cfg.get("cleanup_intermediates", False):
        import glob as _glob
        print("\nCleaning up intermediate files...")
        patterns = [
            "*_detections.csv",
            "*_detections.geojson",
            # Keep *_summary.json — needed by UI to display results
        ]
        removed = 0
        for pattern in patterns:
            for f in _glob.glob(os.path.join(cfg["output_dir"], pattern)):
                try:
                    os.remove(f)
                    removed += 1
                except Exception:
                    pass
        # Clean logs dir
        log_dir = os.path.join(cfg["dashboard_dir"], "logs")
        if os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)
        # Clean metadata dir
        meta_dir = os.path.join(cfg["dashboard_dir"], "metadata")
        if os.path.exists(meta_dir):
            import shutil
            shutil.rmtree(meta_dir, ignore_errors=True)
        print(f"  Removed {removed} intermediate files")

    print(f"\nPIPELINE COMPLETE ✓")


# ==========================================================
# SPATIAL ASSOCIATION PIPELINE
# ==========================================================

def _find_geojson(img_path: str, output_dir: str):
    """Find the GeoJSON detection export for an image in output_dir."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    matches = glob.glob(os.path.join(output_dir, f"{stem}*_detections.geojson"))
    return matches[0] if matches else None


def cytoplasm_overrides_for_pair(cfg, path_a, path_b):
    """
    Resolve per-image cytoplasm (membrane) measurement flags for a spatial pair.

    This is the SINGLE SOURCE OF TRUTH for the role-based default, so the CLI
    (`run_pipeline.py --mode spatial`) and the UI (webui/api.py) behave
    identically. Default:

      • reference / image-A role  → OFF  (CD8 nuclear measurement is the
                                          validated path)
      • moving / image-B role     → OFF  (preserve original QuPath classification;
                                          membrane-ring analysis is opt-in)

    Any explicit per-image entry already in cfg["cytoplasm_overrides"] wins, so
    a UI run (which passes explicit flags) and a user-tuned config are both
    respected. Returns a {basename: bool} dict.
    """
    merged = dict(cfg.get("cytoplasm_overrides") or {})
    a, b = os.path.basename(path_a), os.path.basename(path_b)
    merged.setdefault(a, False)   # reference / image A — nuclear (validated)
    merged.setdefault(b, False)   # preserve original QuPath classification by default
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Provenance — makes every spatial result reproducible/traceable for shipping.
# Records package versions, git commit, the analysis params ACTUALLY used (read
# back from the produced result where possible), the registration method that
# ran, and the pixel size + its source. Pure metadata; changes no statistic.
# ──────────────────────────────────────────────────────────────────────────────

def _pkg_version(modname):
    try:
        mod = __import__(modname)
        if modname == "cv2":
            return mod.__version__
        if modname == "SimpleITK":
            return mod.Version_VersionString()
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=here, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_provenance(cfg, assoc_result, reg_method, ref_px, ref_px_source):
    """Assemble the provenance block stamped into every spatial result JSON."""
    import datetime as _dt
    n_perm = kde_bw = dclf_rmin = dclf_rmax = primary_null = None
    arch_scale = None
    if assoc_result:
        for v in (assoc_result.get("association") or {}).values():
            n_perm       = v.get("n_perm", n_perm)
            primary_null = v.get("primary_null", primary_null)
            g = v.get("global") or {}
            dclf_rmin = g.get("dclf_rmin_um", dclf_rmin)
            dclf_rmax = g.get("dclf_rmax_um", dclf_rmax)
            inh = (v.get("nulls") or {}).get("inhomogeneous") or {}
            kde_bw = inh.get("bandwidth_um", kde_bw)
            arch_scale = v.get("architecture_scale", arch_scale)
            break
    reweight_bw = None
    try:  # fall back to library defaults when no association was produced
        from spatial_stats import (_DCLF_RMIN_UM, _DCLF_RMAX_UM, _KDE_BANDWIDTH_UM,
                                    _REWEIGHT_BANDWIDTH_UM)
        dclf_rmin = dclf_rmin if dclf_rmin is not None else _DCLF_RMIN_UM
        dclf_rmax = dclf_rmax if dclf_rmax is not None else _DCLF_RMAX_UM
        kde_bw    = kde_bw    if kde_bw    is not None else _KDE_BANDWIDTH_UM
        reweight_bw = _REWEIGHT_BANDWIDTH_UM
    except Exception:
        pass

    skip = {"spatial_pairs", "coloc_pairs"}
    cfg_snapshot = {k: v for k, v in cfg.items()
                    if not k.startswith("_") and k not in skip}
    cfg_snapshot["spatial_pairs_count"] = len(
        cfg.get("spatial_pairs", cfg.get("coloc_pairs", [])) or [])

    return {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit":    _git_commit(),
        "python":        sys.version.split()[0],
        "package_versions": {
            "numpy":      _pkg_version("numpy"),
            "scipy":      _pkg_version("scipy"),
            "opencv":     _pkg_version("cv2"),
            "shapely":    _pkg_version("shapely"),
            "SimpleITK":  _pkg_version("SimpleITK"),
            "openslide":  _pkg_version("openslide"),
            "matplotlib": _pkg_version("matplotlib"),
            "pillow":     _pkg_version("PIL"),
            "tifffile":   _pkg_version("tifffile"),
        },
        "analysis_params": {
            "n_perm":          n_perm,
            "kde_bandwidth_um": kde_bw,
            # A7: the primary (reweighted) null's actual bandwidth + its fixed seed,
            # which actually drove the result (the retired inhomogeneous block's
            # kde_bandwidth_um above is NOT in the default null set).
            "reweight_bandwidth_um": reweight_bw,
            "null_seed":       0,
            "dclf_band_um":    [dclf_rmin, dclf_rmax],
            "primary_null":    primary_null,
            "max_radius_um":   cfg.get("max_radius_um", 100.0),
            "radius_step_um":  cfg.get("radius_step_um", 2.0),
            # A6: the reweighted-primary test ASSUMES tissue architecture is coarser
            # than this bandwidth (ihc.md §15.5). The architecture scale is now MEASURED
            # per pair (spatial_stats.estimate_architecture_scale) and gated: a "robust"
            # verdict is only trustworthy when the measured scale clears the calibrated
            # validity threshold (validate_architecture_scale.py).
            "architecture_scale_assumption_um": reweight_bw,
            "architecture_scale_measured": bool(arch_scale and arch_scale.get("scale_um") is not None),
            "architecture_scale_um": (arch_scale or {}).get("scale_um"),
            "architecture_scale_status": (arch_scale or {}).get("status"),
            "architecture_scale_ok": (arch_scale or {}).get("ok"),
        },
        "registration_method": reg_method,
        "pixel_size_ref_um":   ref_px,
        "pixel_size_source":   ref_px_source,
        "config_snapshot":     cfg_snapshot,
    }


def resolve_pixel_size(session_pixel_size, image_path, scale_image_path, manual_override):
    """
    Resolve the pixel size (µm/px) for one image, highest priority first:

      1. manual_override     — user-typed value (> 0)
      2. scale_image_path    — extract from a scale-bar calibration image
      3. session_pixel_size  — run-level default (> 0)
      4. scale bar burned into image_path itself
      5. 0.5 µm/px           — final fallback (config default_pixel_size)

    Logs which source won.
    """
    from pixel_size_util import extract_pixel_size_from_scale_bar

    name = os.path.basename(image_path) if image_path else "(session)"

    if manual_override and float(manual_override) > 0:
        val = float(manual_override)
        print(f"  Pixel size [{name}]: {val} µm/px (manual override)")
        return val

    if scale_image_path:
        val = extract_pixel_size_from_scale_bar(scale_image_path)
        if val:
            print(f"  Pixel size [{name}]: {val:.4f} µm/px (scale image)")
            return float(val)

    if session_pixel_size and float(session_pixel_size) > 0:
        val = float(session_pixel_size)
        print(f"  Pixel size [{name}]: {val} µm/px (session default)")
        return val

    if image_path:
        val = extract_pixel_size_from_scale_bar(image_path)
        if val:
            print(f"  Pixel size [{name}]: {val:.4f} µm/px (scale bar in image)")
            return float(val)

    print(f"  Pixel size [{name}]: 0.5 µm/px (default fallback)")
    return 0.5


def _spatial_result_for_json(spatial):
    """
    Prepare a spatial-association result for on-disk JSON.

    Keeps the full cross-type K / g(r) curves, the Monte-Carlo null envelope,
    per-r p-values, the global summary and the figure paths so the Spatial
    Association UI and any downstream figures are fully reproducible. Drops only
    the private "_registered" centroid arrays (used transiently for overlays).
    """
    if not spatial:
        return spatial
    out = {
        "per_marker":         spatial.get("per_marker", {}),
        "tissue_area_um2":    spatial.get("tissue_area_um2"),
        "tissue_mask_method": spatial.get("tissue_mask_method"),
        "association":        {},
    }
    for key, v in spatial.get("association", {}).items():
        out["association"][key] = {kk: vv for kk, vv in v.items()
                                   if kk != "_registered"}
    return out


def evaluate_registration_qc(reg_result, qc_metrics, pixel_size_um,
                             residual_valid_um=5.0, residual_warn_um=10.0,
                             min_overlap=0.5):
    """
    Fail-closed registration QC gate. Decides whether the spatial statistics for a
    pair may be presented as valid, based on objective metrics from
    registration.compute_registration_qc — it never re-runs registration.

    Thresholds (parameters, documented defaults):
      • residual_valid_um (5.0)  — median post-registration residual must be below
        this (half the 10 µm DCLF band lower bound) for a "valid" verdict.
      • residual_warn_um (10.0)  — the DCLF analysis-band lower bound; residual in
        [valid, warn) µm is a caution "warning" zone; residual ≥ warn is "invalid".
      • min_overlap (0.5)        — minimum fraction of moving tissue that must land
        inside the fixed tissue after the transform.

    Returns a registration_qc dict:
      {valid: bool, status: "valid"|"warning"|"invalid", reason: str, method,
       residual_error_um, residual_error_p90_um, tissue_overlap_fraction,
       quality_metric, qc_inlier_ratio, thresholds}
    `valid` is True only when status == "valid"; both "warning" and "invalid"
    flag the statistics as not fully trustworthy (fail closed). Identity fallback,
    an unverifiable alignment (no reliable feature matches), tissue that moved
    off-target, or residual ≥ the analysis scale all force "invalid".
    """
    method  = (reg_result or {}).get("method") or "none"
    qc      = qc_metrics or {}
    res_um  = qc.get("residual_error_um")
    overlap = qc.get("tissue_overlap_fraction")
    base = {
        "method":                  method,
        "residual_error_um":       res_um,
        "residual_error_p90_um":   qc.get("residual_error_p90_um"),
        "tissue_overlap_fraction": overlap,
        "quality_metric":          qc.get("quality_metric"),
        "qc_inlier_ratio":         qc.get("qc_inlier_ratio"),
        "thresholds": {
            "residual_valid_um":   residual_valid_um,
            "residual_warn_um":    residual_warn_um,
            "min_tissue_overlap":  min_overlap,
        },
    }

    # 1. Registration genuinely failed → identity transform.
    if not reg_result or method == "identity":
        return {**base, "valid": False, "status": "invalid",
                "reason": "Registration fell back to identity — spatial "
                          "statistics are not interpretable."}

    # 2. Alignment could not be verified (no reliable feature correspondences).
    if res_um is None:
        return {**base, "valid": False, "status": "invalid",
                "reason": "Registration quality could not be verified (no reliable "
                          "feature correspondences between the hematoxylin "
                          "channels) — statistics treated as unreliable."}

    # 3. Tissue moved off-target.
    if overlap is not None and overlap < min_overlap:
        return {**base, "valid": False, "status": "invalid",
                "reason": f"Only {overlap*100:.0f}% of the moving tissue overlaps "
                          f"the reference after registration (minimum "
                          f"{min_overlap*100:.0f}%) — tissue is misaligned, "
                          f"statistics unreliable."}

    # 4. Residual alignment error vs the analysis scale.
    if res_um >= residual_warn_um:
        return {**base, "valid": False, "status": "invalid",
                "reason": f"Residual alignment error {res_um:.1f} µm meets/exceeds "
                          f"the {residual_warn_um:.0f} µm analysis scale (DCLF band "
                          f"lower bound) — statistics unreliable."}
    if res_um >= residual_valid_um:
        return {**base, "valid": False, "status": "warning",
                "reason": f"Residual alignment error {res_um:.1f} µm is in the "
                          f"caution zone ({residual_valid_um:.0f}–"
                          f"{residual_warn_um:.0f} µm), within the DCLF analysis "
                          f"band — interpret spatial statistics with care."}

    ov_txt = f", tissue overlap {overlap*100:.0f}%" if overlap is not None else ""
    return {**base, "valid": True, "status": "valid",
            "reason": f"Registration passed QC (residual {res_um:.1f} µm < "
                      f"{residual_valid_um:.0f} µm{ov_txt})."}


def run_spatial_association_pipeline(config_path="config.yaml"):
    """
    Cross-type spatial-association pipeline.
    Processes pre-built pairs: runs QuPath on both images, registers them into a
    shared coordinate space, then measures population-level spatial association
    (cross-type Ripley's K / g(r) — see spatial.run_spatial_association). This is
    NOT single-cell co-expression, which serial sections cannot establish.
    """
    cfg   = load_config(config_path)
    # `spatial_pairs` is canonical; `coloc_pairs` kept as a deprecated alias.
    pairs = cfg.get("spatial_pairs", cfg.get("coloc_pairs", []))

    if not pairs:
        print("ERROR: No spatial-association pairs in config")
        return

    print(f"\n{'='*55}")
    print(f"  IHC SPATIAL ASSOCIATION ANALYZER")
    print(f"{'='*55}")
    print(f"  Pairs:        {len(pairs)}")
    print(f"  Threshold:    {cfg['dab_threshold']} OD")
    print(f"  Max distance: {cfg.get('max_distance_um', 10.0)} µm")
    print(f"  Registration: {'enabled' if cfg.get('enable_registration', True) else 'disabled'}")
    print(f"  Output:       {cfg['output_dir']}")
    print(f"{'='*55}")

    groovy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "generated_pipeline.groovy"
    )

    spatial_results = []

    for i, pair in enumerate(pairs):
        sample_id  = pair["sample_id"]
        stain_a    = pair["stain_a"]
        stain_b    = pair["stain_b"]
        path_a     = pair["path_a"]
        path_b     = pair["path_b"]
        landmark_cert = pair.get("certification") or {}

        print(f"\n{'='*50}")
        print(f"PAIR {i+1}/{len(pairs)}: {sample_id}")
        print(f"  {stain_a}: {os.path.basename(path_a)}")
        print(f"  {stain_b}: {os.path.basename(path_b)}")
        print(f"{'='*50}")

        if cfg.get("require_landmark_certification") and not landmark_cert.get("is_certified"):
            reason = landmark_cert.get("reason") or "landmark certification was not completed"
            print(f"  BLOCKED: {sample_id} — {reason}")
            spatial_results.append({
                "sample_id": sample_id, "stain_a": stain_a, "stain_b": stain_b,
                "filename_a": os.path.basename(path_a),
                "filename_b": os.path.basename(path_b),
                "certification": landmark_cert or {
                    "status": "NOT_CERTIFIABLE", "is_certified": False,
                    "method": "manual_landmark_similarity", "reason": reason,
                },
                "statistics_valid": False, "spatial_association": None,
                "error": "blocked_uncertified",
            })
            continue

        # Separate output sub-dir per pair to avoid filename collisions
        pair_out = os.path.join(cfg["output_dir"], sample_id)
        os.makedirs(pair_out, exist_ok=True)
        # Role-based cytoplasm defaults live in cytoplasm_overrides_for_pair so the
        # CLI and UI entry points agree (A=nuclear, B=cytoplasm ring) — see helper.
        cyto_map = cytoplasm_overrides_for_pair(cfg, path_a, path_b)
        pair_cfg = {**cfg, "output_dir": pair_out, "dashboard_dir": pair_out,
                    "cytoplasm_overrides": cyto_map}

        # ── Run QuPath on both images ──────────────────────────────────
        print(f"\nProcessing {stain_a}...")
        json_a = run_single_image(path_a, pair_cfg, groovy_path)
        geojson_a = _find_geojson(path_a, pair_out)
        # Capture the REFERENCE (image-A) pixel size exactly as A's segmentation
        # resolved it (same get_pixel_size chain), BEFORE the image-B run below
        # overwrites _resolved_pixel_size. This is the µm/px the Ripley's K / DCLF
        # band must use — reading cfg["default_pixel_size"] here would silently
        # mis-scale a CLI run whose config still carries the 0.5 default.
        ref_px = pair_cfg.get("_resolved_pixel_size",
                              cfg.get("default_pixel_size", 0.5))
        ref_px_source = pair_cfg.get("_resolved_pixel_size_source",
                                     "default_fallback")

        print(f"\nProcessing {stain_b}...")
        json_b = run_single_image(path_b, pair_cfg, groovy_path)
        geojson_b = _find_geojson(path_b, pair_out)

        if not json_a or not json_b:
            print(f"  SKIPPING pair {sample_id}: QuPath failed for one or both images")
            spatial_results.append({
                "sample_id": sample_id, "stain_a": stain_a, "stain_b": stain_b,
                "filename_a": os.path.basename(path_a),
                "filename_b": os.path.basename(path_b),
                "error": "QuPath failed",
            })
            continue

        metrics_a = parse_qupath_output(json_a)
        metrics_b = parse_qupath_output(json_b)

        if not geojson_a or not geojson_b:
            print(f"  WARNING: GeoJSON missing for pair {sample_id} — skipping matching")
            spatial_results.append({
                "sample_id": sample_id, "stain_a": stain_a, "stain_b": stain_b,
                "filename_a": os.path.basename(path_a),
                "filename_b": os.path.basename(path_b),
                "metrics_a": metrics_a, "metrics_b": metrics_b,
                "spatial_association": None, "error": "GeoJSON not found",
            })
            continue

        # ── Registration ───────────────────────────────────────────────
        reg_result = None
        if landmark_cert.get("is_certified") and landmark_cert.get("matrix"):
            reg_result = {
                "matrix": np.asarray(landmark_cert["matrix"], dtype=np.float32),
                "scale_ref": 1.0, "scale_mov": 1.0,
                "method": "landmark", "success": True,
                "metric": landmark_cert.get("tre_median_um"),
            }
            print(f"\nUsing landmark-certified transform: {landmark_cert.get('status')} "
                  f"(TRE={landmark_cert.get('tre_median_um')} µm)")
        elif cfg.get("enable_registration", True):
            print(f"\nRegistering {stain_b} → {stain_a}...")
            try:
                from registration import compute_registration
                reg_result = compute_registration(path_a, path_b)
            except Exception as e:
                print(f"  Registration error: {e} — proceeding without registration")

        # ── Cross-type spatial association (Ripley's K / g(r)) ──────────
        # Population-level statistic — replaces MNN cell matching. The reference
        # (stain_a / CD8) pixel size (resolved above, identical to A's
        # segmentation) defines the metric; TIM-3 is registered in.
        if ref_px_source == "default_fallback":
            print(f"  ⚠ WARNING: pixel size for {sample_id} fell through to the "
                  f"{ref_px} µm/px hardcoded default (no per-image override, TIFF/"
                  f"OME metadata, UI value, or filename magnification). The spatial "
                  f"statistics (Ripley's K and the 10–50 µm DCLF band) MAY BE "
                  f"MIS-SCALED — provide a scale image, metadata, or a per-image "
                  f"pixel size.")
        print(f"\nComputing cross-type spatial association "
              f"(Ripley's K, ref {ref_px} µm/px, source={ref_px_source})...")
        assoc_result = None
        try:
            from spatial import run_spatial_association
            reg_map = {stain_b: reg_result} if reg_result else {}
            assoc_result = run_spatial_association(
                layer_geojsons={stain_a: geojson_a, stain_b: geojson_b},
                layer_order=[stain_a, stain_b],
                reg_results=reg_map,
                pixel_size_um=ref_px,
                ref_image_path=path_a,
                layer_images={stain_a: path_a, stain_b: path_b},
                max_radius_um=cfg.get("max_radius_um", 100.0),
                radius_step_um=cfg.get("radius_step_um", 2.0),
                certified_roi_polygon=landmark_cert.get("roi_polygon"),
                precheck_only=bool(cfg.get("precheck_bandwidth_only")),
            )
        except Exception as e:
            print(f"  Spatial association error: {e}")

        # Pre-flight-only mode: emit a machine-parseable bandwidth verdict for the UI
        # button and stop before overlays / QC / result JSON. Segmentation has run, so
        # the subsequent full run reuses the GeoJSONs (reuse_existing_geojson).
        if cfg.get("precheck_bandwidth_only"):
            import json as _json
            _pc = (assoc_result or {}).get("bandwidth_precheck") or {
                "valid": None, "worst_status": None,
                "reason": "bandwidth pre-flight could not be computed"}
            print("BANDWIDTH_PRECHECK_JSON:" + _json.dumps({
                "sample_id": sample_id, "stain_a": stain_a, "stain_b": stain_b,
                "precheck": _pc,
                "tissue_area_um2": (assoc_result or {}).get("tissue_area_um2"),
                "tissue_mask_method": (assoc_result or {}).get("tissue_mask_method"),
            }))
            continue

        # ── Visualizations: 3 overlays (A seg, B seg, consolidated) + plot ─────
        reg_method = reg_result["method"] if reg_result else "none"
        if assoc_result and assoc_result.get("association"):
            registered = assoc_result.get("_registered", {})
            # TIM-3 (image B) uses the cytoplasm/membrane compartment when enabled.
            # Read the SAME role-based map the measurement used (cyto_map) so the
            # overlay and the actual classification can never disagree.
            use_cyto_b = cyto_map.get(os.path.basename(path_b), True)
            expansion_um = float(cfg.get("cell_expansion_um", 2.0))
            # Colors (RGB): CD8 positive=red, TIM-3 positive=blue, negatives=green
            RED, GREEN, BLUE = (255, 50, 50), (50, 205, 50), (40, 90, 255)
            try:
                from overlay import (generate_segmentation_overlay,
                                     generate_consolidated_density,
                                     generate_association_plot)
                for key, data in assoc_result["association"].items():
                    p_a = registered.get(stain_a)
                    p_b = registered.get(stain_b)

                    # Image 1 — CD8 segmentation (nuclear; positive=red)
                    seg_a = generate_segmentation_overlay(
                        image_path=path_a, geojson_path=geojson_a,
                        output_path=os.path.join(pair_out, f"{sample_id}_A_segmentation.png"),
                        stain_name=f"{stain_a} (Image A)",
                        pos_color=RED, neg_color=GREEN, line_thickness=2,
                        draw_cyto_ring=False)
                    if seg_a:
                        data["seg_a"] = seg_a

                    # Image 2 — TIM-3 segmentation (positive=blue; cytoplasm ring;
                    # vivid=True → brighter green/blue so it stands out)
                    seg_b = generate_segmentation_overlay(
                        image_path=path_b, geojson_path=geojson_b,
                        output_path=os.path.join(pair_out, f"{sample_id}_B_segmentation.png"),
                        stain_name=f"{stain_b} (Image B)",
                        pos_color=BLUE, neg_color=GREEN,
                        line_thickness=1 if use_cyto_b else 2,
                        draw_cyto_ring=use_cyto_b, ring_thickness=2,
                        expansion_um=expansion_um if use_cyto_b else None,
                        vivid=True)
                    if seg_b:
                        data["seg_b"] = seg_b

                    # Image 3 — consolidated dual-channel density heatmap. When the
                    # certified analysis window is an ROI (LOCALLY_CERTIFIED or an
                    # operator-drawn Certification ROI), burn that region into this
                    # results figure so a reviewer sees the statistics were restricted.
                    cons = generate_consolidated_density(
                        ref_image_path=path_a,
                        points_a=p_a, points_b=p_b,
                        pixel_size_um=ref_px, assoc=data,
                        out_path=os.path.join(pair_out, f"{sample_id}_consolidated.png"),
                        label_a=f"{stain_a}+", label_b=f"{stain_b}+",
                        roi_polygon=landmark_cert.get("roi_polygon"))
                    if cons:
                        data["consolidated"] = cons

                    # Image 4 — the statistical association plot (kept)
                    plot = generate_association_plot(
                        data,
                        os.path.join(pair_out, f"{sample_id}_association_plot.png"),
                        label_a=f"{stain_a}+", label_b=f"{stain_b}+")
                    if plot:
                        data["association_plot"] = plot
            except Exception as e:
                print(f"  Spatial visualization failed: {e}")

        # ── Registration QC gate (FAIL CLOSED) ─────────────────────────────────
        # Measure objective registration quality and decide whether the spatial
        # statistics may be presented as valid. Diagnostic images above are ALWAYS
        # generated; this only governs whether the numbers are trustworthy. Wrapped
        # so a QC failure marks the pair unreliable rather than crashing the run.
        if landmark_cert.get("is_certified"):
            registration_qc = {
                "valid": True, "status": "valid", "method": "landmark",
                "residual_error_um": landmark_cert.get("tre_median_um"),
                "residual_error_p90_um": landmark_cert.get("tre_p90_um"),
                "tissue_overlap_fraction": None,
                "reason": landmark_cert.get("reason") or "Passed landmark certification.",
                "thresholds": {"landmark_tre_um": 5.0},
            }
        else:
            reg_qc_metrics = {}
            try:
                from registration import compute_registration_qc
                if reg_result is not None:
                    reg_qc_metrics = compute_registration_qc(
                        path_a, path_b, reg_result, ref_px,
                        residual_inlier_px=cfg.get("reg_qc_inlier_px", 5.0),
                    )
            except Exception as e:
                print(f"  Registration QC measurement failed: {e}")
            registration_qc = evaluate_registration_qc(
                reg_result, reg_qc_metrics, ref_px,
                residual_valid_um=cfg.get("reg_qc_residual_valid_um", 5.0),
                residual_warn_um=cfg.get("reg_qc_residual_warn_um", 10.0),
                min_overlap=cfg.get("reg_qc_min_overlap", 0.5),
            )

        # Log the verdict clearly.
        print(f"\n  REGISTRATION QC: {registration_qc['status'].upper()} "
              f"(method={registration_qc['method']}, "
              f"residual={registration_qc.get('residual_error_um')} µm, "
              f"overlap={registration_qc.get('tissue_overlap_fraction')})")
        print(f"    {registration_qc['reason']}")
        if not registration_qc["valid"]:
            tag = "UNRELIABLE" if registration_qc["status"] == "invalid" else "CAUTION"
            print(f"  ⚠ Spatial statistics for {sample_id} are flagged {tag} "
                  f"— see registration_qc in the result JSON.")

        # A1/B1: the app's registration QC is NOT the §18–20 landmark certification.
        # §18.4 shows automated registration metrics are unreliable on FOV-crop serial
        # sections, so a legacy-QC pass must never be mistaken for a CERTIFIED pair.
        # Stamp an explicit, fail-closed certification status into the result so no
        # downstream consumer (UI or JSON) can read statistics_valid as certification.
        certification = landmark_cert if landmark_cert.get("is_certified") else {
            "status": "not_performed",
            "method": "legacy_automated_registration_qc",
            "is_certified": False,
            "note": ("Automated registration QC only — NOT the §18–20 landmark "
                     "certification. 'statistics_valid' means the legacy feature-match "
                     "QC gate passed; it does NOT mean the pair is CERTIFIED, and "
                     "§18.4 shows this automated metric is unreliable on FOV-crop "
                     "serial sections. No reported biological finding should rest on "
                     "an uncertified pair (run landmark certification — ihc.md §19)."),
        }

        # ── 75 µm bandwidth spatial-validity gate (SEPARATE from registration) ─────
        # The reweighted primary null is size-controlled only when tissue architecture
        # is coarser than the 75 µm bandwidth. This is a SPATIAL-STATISTIC assumption,
        # NOT registration: a bandwidth failure must never change certification.status
        # (registration passed). It only flags whether the numbers can be trusted. Fail
        # closed: worst_status ∈ {unreliable, unknown} → statistics not valid.
        bw_precheck = (assoc_result or {}).get("bandwidth_precheck") or {}
        bw_valid = bool(bw_precheck.get("valid", True))  # absent (no assoc) → don't block
        spatial_validity = {
            "bandwidth_75um_valid": bw_valid,
            "worst_status":         bw_precheck.get("worst_status"),
            "window_scope":         bw_precheck.get("window_scope"),
            "per_image":            bw_precheck.get("per_image"),
            "reason":               bw_precheck.get("reason"),
        }
        statistics_valid = bool(registration_qc["valid"] and bw_valid)
        if bw_precheck:
            print(f"\n  BANDWIDTH 75 µm VALIDITY (within analysis window): "
                  f"{str(bw_precheck.get('worst_status')).upper()} "
                  f"(valid={bw_valid})")
            print(f"    {bw_precheck.get('reason')}")
            if not bw_valid:
                print(f"  ⚠ Spatial statistics for {sample_id} are flagged NOT "
                      f"TRUSTWORTHY at 75 µm — registration is still CERTIFIED; the "
                      f"reweighted 'robust' claim is withheld (see spatial_validity).")

        # Stamp every association entry with the validity flags + certification status
        # so any consumer of the per-pair association knows the statistics are gated.
        if assoc_result and assoc_result.get("association"):
            for _k, _data in assoc_result["association"].items():
                _data["statistics_valid"] = statistics_valid
                _data["spatial_validity"] = spatial_validity
                _data["certification"] = certification

        result = {
            "sample_id":           sample_id,
            "stain_a":             stain_a,
            "stain_b":             stain_b,
            "filename_a":          os.path.basename(path_a),
            "filename_b":          os.path.basename(path_b),
            "metrics_a":           metrics_a,
            "metrics_b":           metrics_b,
            "spatial_association": assoc_result,
            "registration_method": reg_method,
            "registration_qc":     registration_qc,
            "certification":       certification,
            "spatial_validity":    spatial_validity,
            "statistics_valid":    statistics_valid,
            "pixel_size_ref_um":   ref_px,
            "pixel_size_source":   ref_px_source,
            "provenance":          build_provenance(
                                       cfg, assoc_result, reg_method,
                                       ref_px, ref_px_source),
            "tissue_area_um2":     (assoc_result or {}).get("tissue_area_um2"),
            "tissue_mask_method":  (assoc_result or {}).get("tissue_mask_method"),
            "intersection_overlap_iou": (assoc_result or {}).get("intersection_overlap_iou"),
            "output_dir":          pair_out,
        }
        spatial_results.append(result)

        # Save per-pair result JSON (drop only the large match lists; keep
        # counts, Monte-Carlo null stats, and qc_overlay paths)
        summary = {k: v for k, v in result.items() if k not in ("metrics_a","metrics_b")}
        summary["spatial_association"] = _spatial_result_for_json(result.get("spatial_association"))
        with open(os.path.join(pair_out, f"{sample_id}_spatial_association.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ── Save combined results ──────────────────────────────────────────
    combined = []
    for r in spatial_results:
        sr = {k: v for k, v in r.items() if k not in ("metrics_a", "metrics_b")}
        if "spatial_association" in sr:
            sr["spatial_association"] = _spatial_result_for_json(r.get("spatial_association"))
        combined.append(sr)

    combined_path = os.path.join(cfg["output_dir"], "spatial_association_results.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # A8: cohort-level multiple-comparison correction across per-pair DCLF p-values.
    # Running N pairs and calling the p<0.05 ones "robust" has no family-wise control;
    # apply Benjamini-Hochberg across the cohort (the function already exists). Pairs
    # whose statistics are not valid contribute None and are dropped from the tally.
    try:
        from spatial_stats import cohort_multiple_comparison_correction
        cohort_ps, cohort_labels = [], []
        for r in spatial_results:
            assoc = (r.get("spatial_association") or {}).get("association") or {}
            for key, data in assoc.items():
                cohort_labels.append(f"{r['sample_id']}::{key}")
                p = (data.get("global") or {}).get("global_p_dclf")
                cohort_ps.append(p if data.get("statistics_valid") else None)
        fdr = cohort_multiple_comparison_correction(cohort_ps, method="bh")
        fdr["labels"] = cohort_labels
        fdr["caveat"] = ("Cohort significance MUST use these BH-adjusted q-values, not "
                         "the per-pair p-values. Uncertified pairs "
                         "(certification.status='not_performed') must not contribute to "
                         "a cohort-level biological claim.")
        with open(os.path.join(cfg["output_dir"], "spatial_cohort_fdr.json"), "w") as f:
            json.dump(fdr, f, indent=2)
        print(f"  Cohort FDR (BH): {fdr.get('n_significant_adjusted')}/"
              f"{fdr.get('n_tested')} significant after correction "
              f"(raw {fdr.get('n_significant_raw')}).")
    except Exception as e:
        print(f"  Cohort FDR correction skipped: {e}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SPATIAL ASSOCIATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Pairs analyzed: {len(spatial_results)}")
    n_qc_invalid = sum(1 for r in spatial_results
                       if (r.get("registration_qc") or {}).get("status") == "invalid")
    if n_qc_invalid:
        print(f"  Registration QC: {n_qc_invalid} pair(s) flagged INVALID "
              f"(statistics excluded from interpretation)")
    for r in spatial_results:
        qc = r.get("registration_qc") or {}
        qc_tag = f"  [reg QC: {qc.get('status', 'n/a')}]" if qc else ""
        if r.get("spatial_association"):
            for key, data in r["spatial_association"].get("association", {}).items():
                g = data.get("global", {})
                print(f"  {r['sample_id']}: {key} peak L-r={g.get('peak_L_minus_r')} µm "
                      f"@ r={g.get('peak_r_um')} µm "
                      f"({'significant' if g.get('significant') else 'n.s.'}){qc_tag}")
    print(f"\nSPATIAL ASSOCIATION PIPELINE COMPLETE ✓")

    return spatial_results


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OASIS Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="quant",
                        help="quant = single-stain batch | "
                             "spatial = cross-type spatial association")
    args = parser.parse_args()
    # "coloc" is a hidden, deprecated alias for "spatial" (kept so older
    # callers/scripts don't break); "spatial" is the canonical name.
    mode = "spatial" if args.mode == "coloc" else args.mode
    if mode == "spatial":
        run_spatial_association_pipeline(args.config)
    elif mode == "quant":
        run_pipeline(args.config)
    else:
        parser.error(f"unknown --mode {args.mode!r} (use 'quant' or 'spatial')")
