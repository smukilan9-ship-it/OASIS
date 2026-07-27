"""QuPath / Groovy segmentation path — REMOVED from the shipped pipeline 2026-07-27.

WHY IT WAS REMOVED
The native in-process InstanSeg segmenter is the default and is validated as equivalent
(research/ihc.md §7: detection and classification figures within 0.005 over 598 DeepLIIF
images). The QuPath path was kept for one release as cheap insurance while the native one
was new in the field. That release has passed.

Carrying two segmenters means every future change has to be reasoned about twice, and the
QuPath arm additionally required an external binary, a JVM, and a generated Groovy script
that duplicated the thresholding logic in a second language -- so the two could silently
drift apart. It also could not ship inside the standalone bundle.

WHAT THIS FILE IS
The removed code, verbatim, so the equivalence argument stays auditable and so anyone
reproducing the published comparison can run the arm it was measured against. It is not
imported by anything and is not maintained.

Retained alongside it: test_instanseg.groovy, generated_pipeline.groovy.
"""


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



# --- the QuPath arm of run_pipeline.run_single_image ---
                                    pixel_size, pixel_size_source)

    generate_groovy_script(cfg, groovy_script, img_path)
    qp_input = seg_input
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
    return _finish_single_image(img_path, img_filename, matches[0], cfg,
                                pixel_size, pixel_size_source)

