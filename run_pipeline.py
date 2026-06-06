"""
IHC Analyzer — Main Pipeline v5
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

    required = ["input_dir", "qupath_binary", "instanseg_model"]
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
    cfg.setdefault("dashboard_dir", os.path.join(cfg["input_dir"], "output_results"))
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


    return cfg


# ==========================================================
# GROOVY SCRIPT GENERATOR
# ==========================================================

def generate_groovy_script(cfg, script_path="generated_pipeline.groovy"):
    model_path = os.path.expanduser(cfg["instanseg_model"])
    threshold = cfg["dab_threshold"]
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

double threshold = {threshold}
println "DAB threshold: " + threshold + " (fixed OD)"

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
    "total_cells": ${{detections.size()}},
    "positive_cells": ${{positiveCount}},
    "negative_cells": ${{negativeCount}},
    "positivity_pct": ${{String.format("%.2f", positivityPct)}},
    "dab_threshold": {threshold}
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


def run_single_image(img_path, cfg, groovy_script):
    img_filename = os.path.basename(img_path)

    print(f"\n{'='*50}")
    print(f"PROCESSING: {img_filename}")
    print(f"{'='*50}")

    pixel_size_mode = cfg.get("pixel_size_mode", "global")

    try:
        from pixel_size_util import get_pixel_size
        pixel_size = get_pixel_size(img_path, cfg,
                                    interactive=(cfg.get("mode") == "expert"))
    except ImportError:
        pixel_size = cfg.get("default_pixel_size", 0.5)
        print(f"  Using default pixel size: {pixel_size} µm/px")

    cfg["_resolved_pixel_size"] = pixel_size
    generate_groovy_script(cfg, groovy_script)

    command = [cfg["qupath_binary"], "script", "-i", img_path, groovy_script]
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
    return matches[0]


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
# CO-LOCALIZATION PIPELINE
# ==========================================================

def _find_geojson(img_path: str, output_dir: str):
    """Find the GeoJSON detection export for an image in output_dir."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    matches = glob.glob(os.path.join(output_dir, f"{stem}*_detections.geojson"))
    return matches[0] if matches else None


def run_coloc_pipeline(config_path="config.yaml"):
    """
    Co-localization pipeline.
    Processes pre-built pairs: runs QuPath on both images, registers,
    then does mutual-NN co-expression matching.
    """
    cfg   = load_config(config_path)
    pairs = cfg.get("coloc_pairs", [])

    if not pairs:
        print("ERROR: No co-localization pairs in config")
        return

    print(f"\n{'='*55}")
    print(f"  IHC CO-LOCALIZATION ANALYZER")
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

    coloc_results = []

    for i, pair in enumerate(pairs):
        sample_id  = pair["sample_id"]
        stain_a    = pair["stain_a"]
        stain_b    = pair["stain_b"]
        path_a     = pair["path_a"]
        path_b     = pair["path_b"]

        print(f"\n{'='*50}")
        print(f"PAIR {i+1}/{len(pairs)}: {sample_id}")
        print(f"  {stain_a}: {os.path.basename(path_a)}")
        print(f"  {stain_b}: {os.path.basename(path_b)}")
        print(f"{'='*50}")

        # Separate output sub-dir per pair to avoid filename collisions
        pair_out = os.path.join(cfg["output_dir"], sample_id)
        os.makedirs(pair_out, exist_ok=True)
        pair_cfg = {**cfg, "output_dir": pair_out, "dashboard_dir": pair_out}

        # ── Run QuPath on both images ──────────────────────────────────
        print(f"\nProcessing {stain_a}...")
        json_a = run_single_image(path_a, pair_cfg, groovy_path)
        geojson_a = _find_geojson(path_a, pair_out)

        print(f"\nProcessing {stain_b}...")
        json_b = run_single_image(path_b, pair_cfg, groovy_path)
        geojson_b = _find_geojson(path_b, pair_out)

        if not json_a or not json_b:
            print(f"  SKIPPING pair {sample_id}: QuPath failed for one or both images")
            coloc_results.append({
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
            coloc_results.append({
                "sample_id": sample_id, "stain_a": stain_a, "stain_b": stain_b,
                "filename_a": os.path.basename(path_a),
                "filename_b": os.path.basename(path_b),
                "metrics_a": metrics_a, "metrics_b": metrics_b,
                "coloc": None, "error": "GeoJSON not found",
            })
            continue

        # ── Registration ───────────────────────────────────────────────
        reg_result = None
        if cfg.get("enable_registration", True):
            print(f"\nRegistering {stain_b} → {stain_a}...")
            try:
                from registration import compute_registration
                reg_result = compute_registration(path_a, path_b)
            except Exception as e:
                print(f"  Registration error: {e} — proceeding without registration")

        # ── Co-expression matching ─────────────────────────────────────
        print(f"\nMatching cells (max {cfg.get('max_distance_um', 10.0)} µm)...")
        coloc_result = None
        try:
            from coloc import run_coloc
            reg_map = {stain_b: reg_result} if reg_result else {}
            coloc_result = run_coloc(
                layer_geojsons={stain_a: geojson_a, stain_b: geojson_b},
                layer_order=[stain_a, stain_b],
                reg_results=reg_map,
                max_distance_um=cfg.get("max_distance_um", 10.0),
                pixel_size_um=cfg.get("default_pixel_size", 0.5),
            )
        except Exception as e:
            print(f"  Co-localization matching error: {e}")

        result = {
            "sample_id":           sample_id,
            "stain_a":             stain_a,
            "stain_b":             stain_b,
            "filename_a":          os.path.basename(path_a),
            "filename_b":          os.path.basename(path_b),
            "metrics_a":           metrics_a,
            "metrics_b":           metrics_b,
            "coloc":               coloc_result,
            "registration_method": reg_result["method"] if reg_result else "none",
            "output_dir":          pair_out,
        }
        coloc_results.append(result)

        # Save per-pair result JSON (without large match lists)
        summary = {k: v for k, v in result.items() if k not in ("metrics_a","metrics_b")}
        if summary.get("coloc") and summary["coloc"].get("coexpression"):
            for key in summary["coloc"]["coexpression"]:
                summary["coloc"]["coexpression"][key] = {
                    "count": result["coloc"]["coexpression"][key]["count"]
                }
        with open(os.path.join(pair_out, f"{sample_id}_coloc.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ── Save combined results ──────────────────────────────────────────
    combined = []
    for r in coloc_results:
        sr = {k: v for k, v in r.items() if k not in ("metrics_a", "metrics_b")}
        if sr.get("coloc") and sr["coloc"].get("coexpression"):
            for key in sr["coloc"]["coexpression"]:
                sr["coloc"]["coexpression"][key] = {
                    "count": r["coloc"]["coexpression"][key]["count"]
                }
        combined.append(sr)

    combined_path = os.path.join(cfg["output_dir"], "coloc_results.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CO-LOCALIZATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Pairs analyzed: {len(coloc_results)}")
    for r in coloc_results:
        if r.get("coloc"):
            for key, data in r["coloc"].get("coexpression", {}).items():
                print(f"  {r['sample_id']}: {key} = {data['count']} cells")
    print(f"\nCOLOC PIPELINE COMPLETE ✓")

    return coloc_results


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IHC Analyzer Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="quant", choices=["quant", "coloc"],
                        help="quant = single-stain batch, coloc = co-expression matching")
    args = parser.parse_args()
    if args.mode == "coloc":
        run_coloc_pipeline(args.config)
    else:
        run_pipeline(args.config)