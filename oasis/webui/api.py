"""
api.py — Python backend for pywebview UI
"""
import os
import sys
import json
import glob
import shutil
import subprocess
import threading
import yaml
from statistics import median
from pathlib import Path

CONFIG_DIR       = Path.home() / ".ihc_analyzer"
SETUP_FILE       = CONFIG_DIR / "setup.yaml"
EXPERIMENTS_FILE = CONFIG_DIR / "experiments.yaml"
CONFIG_DIR.mkdir(exist_ok=True)

DEFAULT_SETUP = {
    "microscope": "",
    "camera": "",
    "scanner": "",
    "pixel_size_x": 0.5,
    "pixel_size_y": 0.5,
    "default_objective": "10x",
    "device": "mps",
    "instanseg_threads": 4,
    "qupath_binary": "/Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64",
    "instanseg_model": "~/QuPath/v0.7/instanseg/downloaded/brightfield_nuclei-0.1.1",
    # Root that holds the consolidated validation datasets (Validation tab).
    # Resolved by validation/datasets/resolve.py; kept out of the repo so the
    # project stays lean and the path survives being bundled as a standalone app.
    "validation_data_dir": str(Path.home() / "oasis_validation_datasets"),
}

# Standard pixel sizes per magnification
STANDARD_PIXEL_SIZE = {
    "4x": 2.50, "10x": 1.00, "20x": 0.50, "40x": 0.25, "60x": 0.165, "100x": 0.10
}

# Repo root. This file is <root>/oasis/webui/api.py, so climb THREE levels
# (webui → oasis → root). The restructure moved api.py one level deeper; the old
# .parent.parent pointed at <root>/oasis, so PROJECT_DIR/"run_pipeline.py" and
# PROJECT_DIR/"validation" no longer existed and every spatial/calibrate subprocess
# silently failed (empty stdout → misleading "segmentation may have failed").
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Preloaded calibration presets (data-backed defaults; user calibrations add to these).
BUILTIN_CALIBRATIONS = [
    {"name": "CRC-ICM (TIM-3)", "marker": "tim-3",
     "membrane_pix_thr": 0.30, "membrane_frac_min": 0.14, "auc": 0.93, "builtin": True},
]


def _planar_partition(polygons, min_area=1.0):
    """Split a set of (possibly overlapping, any-shape) polygons into a PLANAR PARTITION:
    non-overlapping pieces where every intersection becomes its own separate piece. This is
    what makes per-region analysis honest — overlapping windows would double-count cells, and
    each piece must carry a single local transform.

    Returns a list of dicts: {polygon: [[x,y]…], origins: [input indices covering it],
    is_intersection: bool}. Pieces outside every input polygon are dropped. Any-shape safe:
    invalid/self-intersecting freehand loops are repaired with buffer(0).
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union, polygonize
    shp = []
    for r in polygons:
        r = [(float(x), float(y)) for x, y in r]
        if len(r) < 3:
            continue
        p = Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            continue
        shp.append(p)
    if not shp:
        return []
    if len(shp) == 1:
        return [{"polygon": [[x, y] for x, y in shp[0].exterior.coords[:-1]],
                 "origins": [0], "is_intersection": False}]
    # Faces of the arrangement formed by ALL polygon boundaries.
    boundaries = unary_union([p.boundary for p in shp])
    out = []
    for face in polygonize(boundaries):
        if face.area < min_area:
            continue
        rp = face.representative_point()
        origins = [i for i, p in enumerate(shp) if p.contains(rp)]
        if not origins:
            continue                       # a hole between polygons — not inside any region
        geom = face
        if geom.geom_type != "Polygon":
            geom = max(getattr(geom, "geoms", [geom]), key=lambda g: g.area)
        out.append({"polygon": [[x, y] for x, y in geom.exterior.coords[:-1]],
                    "origins": origins, "is_intersection": len(origins) > 1})
    return out


class API:
    def __init__(self):
        self._window  = None
        self._process = None
        self._prov_cache = {}     # (ref_path, mov_path, px) -> provisional thumbnail transform

    def set_window(self, window):
        self._window = window

    # ── Setup ──────────────────────────────────────────────────────────────
    def get_setup(self):
        if SETUP_FILE.exists():
            with open(SETUP_FILE) as f:
                data = yaml.safe_load(f) or {}
            result = {**DEFAULT_SETUP, **data}
        else:
            result = dict(DEFAULT_SETUP)
        result["_home"] = str(Path.home())
        return result

    def save_setup(self, data):
        with open(SETUP_FILE, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return {"ok": True}

    def is_first_run(self):
        return not SETUP_FILE.exists()

    # ── Calibration (native cutoff fitting) ────────────────────────────────
    def calibration_prepare(self, image_path, pixel_size):
        """Segment an image and return views + clickable cells for labeling."""
        try:
            from oasis.webui import calibration
            return calibration.prepare(os.path.expanduser(image_path),
                                       float(pixel_size), self.get_setup())
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def calibration_fit(self, image_path, geojson_path, pixel_size, pos_idx, neg_idx):
        """Fit membrane cutoffs from hand-labelled positive/negative cell indices."""
        try:
            from oasis.webui import calibration
            return calibration.fit(os.path.expanduser(image_path), geojson_path,
                                   float(pixel_size), pos_idx, neg_idx)
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def calibration_fit_multi(self, items):
        """Fit membrane cutoffs by POOLING hand-labelled cells across several images.

        `items`: list of {image_path, geojson_path, pixel_size, pos_idx, neg_idx}
        (one per labelled image). Returns pooled cutoffs + leave-one-cell-out F1/AUC.
        """
        try:
            from oasis.webui import calibration
            norm = [{"image_path": os.path.expanduser(it.get("image_path", "")),
                     "geojson_path": it.get("geojson_path"),
                     "pixel_size": float(it.get("pixel_size") or 0.5),
                     "pos_idx": it.get("pos_idx") or [],
                     "neg_idx": it.get("neg_idx") or []}
                    for it in (items or [])]
            if not norm:
                return {"ok": False, "msg": "No labelled images supplied."}
            return calibration.fit_multi(norm)
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def list_calibrations(self):
        """Built-in presets + user-saved calibrations (switchable in Quant)."""
        saved = self.get_setup().get("calibrations") or []
        return {"builtin": BUILTIN_CALIBRATIONS, "saved": saved}

    def save_calibration(self, name, marker, cutoffs):
        setup = self.get_setup(); setup.pop("_home", None)
        cals = [c for c in (setup.get("calibrations") or []) if c.get("name") != name]
        cals.append({"name": str(name), "marker": str(marker).lower(),
                     "membrane_pix_thr": float(cutoffs["membrane_pix_thr"]),
                     "membrane_frac_min": float(cutoffs["membrane_frac_min"]),
                     "auc": float(cutoffs.get("auc", 0) or 0)})
        setup["calibrations"] = cals; self.save_setup(setup)
        return {"ok": True, "calibrations": cals}

    def delete_calibration(self, name):
        setup = self.get_setup(); setup.pop("_home", None)
        setup["calibrations"] = [c for c in (setup.get("calibrations") or [])
                                 if c.get("name") != name]
        self.save_setup(setup)
        return {"ok": True, "calibrations": setup["calibrations"]}

    def _resolve_calibration(self, name, marker):
        """Find a calibration by explicit name, else by marker (newest user save wins)."""
        pool = BUILTIN_CALIBRATIONS + (self.get_setup().get("calibrations") or [])
        if name:
            for c in pool:
                if c["name"] == name:
                    return c
        m = str(marker or "").lower()
        return next((c for c in reversed(pool) if c["marker"] == m), None)

    # ── Experiments ────────────────────────────────────────────────────────
    def get_experiments(self):
        if EXPERIMENTS_FILE.exists():
            with open(EXPERIMENTS_FILE) as f:
                return yaml.safe_load(f) or []
        return []

    def save_experiments(self, experiments):
        with open(EXPERIMENTS_FILE, "w") as f:
            yaml.dump(experiments, f, default_flow_style=False)
        return {"ok": True}

    def pick_folder(self):
        import webview
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if result and len(result) > 0:
            return result[0]
        return None

    def list_images(self, folder):
        if not folder or not os.path.exists(os.path.expanduser(folder)):
            return []
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".svs", ".ndpi"}
        folder = os.path.expanduser(folder)
        try:
            return sorted([
                f for f in os.listdir(folder)
                if os.path.splitext(f.lower())[1] in exts
            ])
        except Exception:
            return []

    def preview_quant_scale_matches(self, folder):
        """
        Preview Quantification batch scale calibration.

        When enabled in the UI, files whose stem contains "scale" are treated as
        calibration-only. They are matched to analysis images by removing that
        token from the scale filename and comparing normalized stems.
        """
        try:
            import re

            folder = os.path.expanduser(folder or "")
            if not os.path.isdir(folder):
                return {"status": "error", "error": "Folder not found"}

            files = self.list_images(folder)

            def is_scale_file(name):
                return "scale" in Path(name).stem.lower()

            def key_for(stem, remove_scale=False):
                s = stem.lower()
                if remove_scale:
                    s = re.sub(r"scale(?:bar)?", " ", s)
                s = re.sub(r"(?<!\d)x(\d+)(?!\d)", r"\1x", s)
                return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

            scale_files = [f for f in files if is_scale_file(f)]
            analysis_files = [f for f in files if not is_scale_file(f)]

            scale_entries = []
            for name in scale_files:
                scale_entries.append({
                    "filename": name,
                    "path": os.path.join(folder, name),
                    "key": key_for(Path(name).stem, remove_scale=True),
                })

            pairs, unmatched_images, used_scales = [], [], set()
            for name in analysis_files:
                image_key = key_for(Path(name).stem)
                candidates = [
                    s for s in scale_entries
                    if s["key"] == image_key
                    or (len(image_key) >= 4 and s["key"].startswith(image_key + "_"))
                    or (len(s["key"]) >= 4 and image_key.startswith(s["key"] + "_"))
                ]
                candidates = [s for s in candidates if s["filename"] not in used_scales]
                if not candidates:
                    unmatched_images.append({
                        "filename": name,
                        "path": os.path.join(folder, name),
                    })
                    continue

                scale = sorted(candidates, key=lambda s: (len(s["key"]), s["filename"]))[0]
                used_scales.add(scale["filename"])
                measured = self.extract_pixel_size(scale["path"])
                row = {
                    "filename": name,
                    "path": os.path.join(folder, name),
                    "scale_filename": scale["filename"],
                    "scale_path": scale["path"],
                    "status": measured.get("status"),
                }
                if measured.get("status") == "ok":
                    row["pixel_size"] = measured["pixel_size"]
                    row["bar_length_px"] = measured.get("bar_length_px")
                else:
                    row["pixel_size"] = None
                    row["error"] = measured.get("error", "scale bar not detected")
                pairs.append(row)

            return {
                "status": "ok",
                "folder": folder,
                "analysis_images": [
                    {"filename": f, "path": os.path.join(folder, f)}
                    for f in analysis_files
                ],
                "scale_images": [
                    {"filename": s["filename"], "path": s["path"]}
                    for s in scale_entries
                ],
                "pairs": pairs,
                "unmatched_images": unmatched_images,
                "unmatched_scale_images": [
                    s["filename"] for s in scale_entries
                    if s["filename"] not in used_scales
                ],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_standard_pixel_size(self, objective):
        return STANDARD_PIXEL_SIZE.get(objective, 0.5)

    def preflight_check(self, workflow="quant"):
        """Fast dependency/path check shown before a costly analysis begins."""
        setup = self.get_setup()
        checks = []
        qpath = os.path.expanduser(setup.get("qupath_binary", ""))
        model = os.path.expanduser(setup.get("instanseg_model", ""))
        checks.append({"name": "QuPath", "ok": bool(qpath and os.path.exists(qpath)), "path": qpath})
        checks.append({"name": "InstanSeg model", "ok": bool(model and os.path.exists(model)), "path": model})
        required = ["numpy", "cv2", "PIL", "yaml"]
        if workflow == "spatial": required += ["shapely", "scipy"]
        for name in required:
            try:
                __import__(name)
                checks.append({"name": name, "ok": True})
            except Exception as e:
                checks.append({"name": name, "ok": False, "error": str(e)})
        return {"ok": all(c["ok"] for c in checks), "checks": checks}

    # ── Pipeline ───────────────────────────────────────────────────────────
    def run_pipeline(self, settings):
        setup = self.get_setup()

        pixel_size_mode = settings.get("pixel_size_mode", "manual")
        pixel_from_ui = pixel_size_mode in ("manual", "global")
        if pixel_size_mode == "scale":
            scale_path = settings.get("scale_image")
            measured = self.extract_pixel_size(scale_path) if scale_path else {}
            if measured.get("status") != "ok":
                return {"ok": False, "msg": measured.get("error", "Scale-bar detection failed")}
            settings["default_pixel_size"] = measured["pixel_size"]
            settings["pixel_size_mode"] = "manual"
            pixel_from_ui = True

        cfg = {
            **setup,
            **settings,
            "qupath_binary":      setup.get("qupath_binary", DEFAULT_SETUP["qupath_binary"]),
            "instanseg_model":    setup.get("instanseg_model", DEFAULT_SETUP["instanseg_model"]),
            "device":             setup.get("device", "mps"),
            "instanseg_threads":  setup.get("instanseg_threads", 4),
            "tile_dims":          512,
            "timeout_seconds":    1800,
            "mode":               "automated",
            "stain_type":         "hdab",
            "image_extensions":   ["*.tif","*.tiff","*.svs","*.ndpi","*.png","*.jpg","*.jpeg"],
            "magnification":      "auto",
            "export_geojson":     True,
            "overlay_downsample": 1.0,
            "_pixel_size_from_ui": pixel_from_ui,
            "objective":          settings.get("objective", "10x"),
        }
        # Single-image Quant has an explicit stain/compartment selection; route it
        # directly to that file rather than depending on hyphen-sensitive filename
        # token matching (TIM-3 vs Tim3). Batch whitelists are used only to exclude
        # calibration images, so batch threshold routing remains filename/stain-based.
        if settings.get("analysis_mode") == "single":
            for filename in settings.get("image_whitelist") or []:
                # Adaptive threshold is computed per-image in-pipeline; a fixed
                # per-image override would suppress it, so only pin the override
                # when the user chose a fixed threshold.
                if not settings.get("adaptive_threshold"):
                    cfg.setdefault("threshold_overrides", {})[filename] = float(
                        settings.get("dab_threshold", 0.2))
                cfg.setdefault("cytoplasm_overrides", {})[filename] = bool(
                    settings.get("use_cytoplasm_measurement", False))

        for k in ["input_dir","output_dir","dashboard_dir","instanseg_model"]:
            if k in cfg and cfg[k]:
                cfg[k] = os.path.expanduser(str(cfg[k]))
        cfg.setdefault("dashboard_dir", str(Path(cfg.get("input_dir","")) / "output_results"))
        os.makedirs(cfg.get("output_dir",""), exist_ok=True)
        os.makedirs(cfg.get("dashboard_dir",""), exist_ok=True)

        # Calibration cutoffs (Calibrate tab / preloaded presets) drive membrane mode:
        # an explicitly chosen profile, else the newest one matching the marker. This
        # overrides any UI default so cutoffs fit to the user's own slides actually run.
        if settings.get("use_cytoplasm_measurement"):
            cal = self._resolve_calibration(settings.get("calibration_name"),
                                            settings.get("stain_name"))
            if cal:
                cfg["membrane_pix_thr"]  = float(cal["membrane_pix_thr"])
                cfg["membrane_frac_min"] = float(cal["membrane_frac_min"])

        config_path = str(CONFIG_DIR / "pipeline_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        def run():
            try:
                skip = ["[INFO ]","[WARN ]","Measured Detection","Completed Annotation",
                        "Processing complete in","Measuring","Loading:","████","WARNING: Unknown"]
                images_total = images_done = 0
                self._process = subprocess.Popen(
                    [str(Path(sys.executable)), str(PROJECT_DIR / "run_pipeline.py"),
                     "--config", config_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, start_new_session=True, cwd=str(PROJECT_DIR)
                )
                while True:
                    line = self._process.stdout.readline()
                    if not line and self._process.poll() is not None:
                        break
                    if not line:
                        continue
                    clean = line.strip()
                    if not clean or any(x in clean for x in skip):
                        continue
                    level = "normal"
                    if any(x in clean for x in ["ERROR","FAILED","TIMEOUT"]):
                        level = "error"
                    elif any(x in clean for x in ["⚠","WARNING"]):
                        level = "warn"
                    elif any(x in clean for x in ["✓","COMPLETE","FINISHED","saved","exported","complete"]):
                        level = "ok"
                    elif any(x in clean for x in ["Running","Processing","Detecting","Rerunning","QC","Generating"]):
                        level = "info"
                    self._emit("log", {"msg": clean, "level": level})
                    if "Found" in clean and "image" in clean:
                        try: images_total = int(clean.split()[1])
                        except: pass
                    if "PIPELINE FINISHED SUCCESSFULLY" in clean:
                        images_done += 1
                        if images_total > 0:
                            self._emit("progress", {"pct": int(images_done/images_total*85)})

                if self._process.returncode == 0:
                    self._emit("progress", {"pct": 100})
                    self._emit("done", {"ok": True, "results": self._load_results(cfg)})
                else:
                    self._emit("done", {"ok": False, "msg": "Pipeline failed"})
            except Exception as e:
                self._emit("done", {"ok": False, "msg": str(e)})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def _emit(self, event, data):
        try:
            js = f"window.onPipelineEvent({json.dumps({'type': event, 'data': data})})"
            self._window.evaluate_js(js)
        except Exception:
            pass

    def _load_results(self, cfg):
        output_dir    = cfg.get("output_dir","")
        dashboard_dir = cfg.get("dashboard_dir","")
        metrics = []
        whitelist = {str(x) for x in (cfg.get("image_whitelist") or [])}
        for jp in sorted(glob.glob(str(Path(output_dir) / "*_summary.json"))):
            try:
                with open(jp) as f:
                    d = json.load(f)
                if whitelist:
                    source_name = str(d.get("image", "")).split(" - ")[0]
                    if source_name not in whitelist:
                        continue
                _tot = d.get("total_cells", 0)
                _pp = float(d.get("positivity_pct", 0))
                _conf = "LOW" if (_tot < 50 or _pp < 0.1 or _pp > 95) else "NORMAL"
                metrics.append({
                    "name":        Path(d.get("image","")).stem.split(" - ")[0],
                    "total_cells": _tot,
                    "positive":    d.get("positive_cells",0),
                    "negative":    d.get("negative_cells",0),
                    "positivity":  _pp,
                    "pixel_size":  d.get("pixel_size_um",0.5),
                    "pixel_size_source": d.get("pixel_size_source", "unknown"),
                    "pixel_size_warning": bool(d.get("pixel_size_warning", False)),
                    "cells_per_mm2": d.get("cells_per_mm2"),
                    "threshold":   d.get("dab_threshold",0.2),
                    "confidence":  _conf,
                    "measurement_compartment": d.get("measurement_compartment", "nucleus"),
                    "membrane_classifier": d.get("membrane_classifier"),
                    "membrane_quality_warning": bool(d.get("membrane_quality_warning", False)),
                    "membrane_positive_rate": d.get("membrane_positive_rate"),
                })
            except Exception:
                continue

        summary_text = ""
        sp = Path(output_dir) / "analysis_summary.txt"
        if sp.exists():
            summary_text = sp.read_text().strip()

        dashboards = sorted(glob.glob(str(Path(dashboard_dir) / "ihc_dashboard_*.html")), key=os.path.getmtime)
        excels     = sorted(glob.glob(str(Path(dashboard_dir) / "ihc_results_*.xlsx")), key=os.path.getmtime)

        overlays_dir = str(Path(output_dir) / "overlays")
        overlays = glob.glob(str(Path(output_dir) / "*_overlay.png"))
        if overlays:
            os.makedirs(overlays_dir, exist_ok=True)
            for ov in overlays:
                dest = Path(overlays_dir) / Path(ov).name
                if not dest.exists():
                    shutil.move(ov, str(dest))

        return {
            "metrics":        metrics,
            "summary":        summary_text,
            "dashboard_path": dashboards[-1] if dashboards else "",
            "excel_path":     excels[-1] if excels else "",
            "overlays_dir":   overlays_dir if overlays else "",
            "output_dir":     output_dir,
        }

    # ── Chat ───────────────────────────────────────────────────────────────
    def send_chat(self, message, context):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            setup    = self.get_setup()
            provider = setup.get("ai_provider","gemini")
            model    = "gemini-2.5-flash" if provider=="gemini" else "claude-sonnet-4-20250514"
            metrics  = context.get("metrics",[])
            total    = sum(m.get("total_cells",0) for m in metrics)
            pos      = sum(m.get("positive",0) for m in metrics)
            avg      = pos/total*100 if total > 0 else 0
            system = f"""You are an expert IHC analysis assistant.
Results: {len(metrics)} images, {total:,} total cells, {pos:,} positive, {avg:.2f}% avg positivity.
Method: InstanSeg brightfield_nuclei, DAB threshold 0.2 OD.
Per image: {json.dumps([{'name':m['name'],'cells':m['total_cells'],'positivity':m['positivity']} for m in metrics])}
Summary: {context.get('summary','')}
Answer concisely and scientifically. Methods sections use past tense passive voice."""
            full = f"{system}\n\nUser: {message}"
            if provider == "gemini":
                from google import genai
                import os as _os
                client = genai.Client(api_key=_os.getenv("GEMINI_API_KEY"))
                r = client.models.generate_content(model=model, contents=full)
                return {"ok": True, "response": r.text.strip()}
            else:
                import anthropic, os as _os
                client = anthropic.Anthropic(api_key=_os.getenv("ANTHROPIC_API_KEY"))
                r = client.messages.create(model=model, max_tokens=800,
                    messages=[{"role":"user","content":full}])
                return {"ok": True, "response": r.content[0].text.strip()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def stop_pipeline(self):
        """Terminate any running pipeline subprocess."""
        if self._process and self._process.poll() is None:
            try:
                import signal
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except Exception:
                self._process.terminate()
        return {"ok": True}

    def _load_spatial_results(self, output_dir: str) -> list:
        """Load combined spatial-association results from disk."""
        path = Path(output_dir) / "spatial_association_results.json"
        if path.exists():
            try:
                with open(path) as f:
                    results = json.load(f)
                # Attach output_dir for UI actions
                for r in results:
                    r.setdefault("output_dir", output_dir)
                return results
            except Exception:
                pass
        return []

    # ── Spatial Association ─────────────────────────────────────────────────
    def pick_file(self):
        """Open a single-file picker (used for image + scale-image selection)."""
        import webview
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False,
            file_types=("Image files (*.tif;*.tiff;*.svs;*.ndpi;*.png;*.jpg;*.jpeg)",
                        "All files (*.*)"),
        )
        if result and len(result) > 0:
            return result[0]
        return None

    def extract_pixel_size(self, image_path: str) -> dict:
        """Extract pixel size (µm/px) from a burned-in 100 µm scale bar."""
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.pixel_size_util import _detect_scale_bar
            px, bar = _detect_scale_bar(os.path.expanduser(image_path))
            if px is None:
                return {"status": "error", "error": "could not detect scale bar"}
            return {"status": "ok", "pixel_size": round(float(px), 4),
                    "bar_length_px": int(bar)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def prepare_landmark_pair(self, ref_path: str, mov_path: str) -> dict:
        """Return compact, blinded browser previews while retaining full-res coordinates."""
        try:
            import base64
            from io import BytesIO
            from PIL import Image

            def preview(path):
                im = Image.open(os.path.expanduser(path)).convert("RGB")
                width, height = im.size
                im.thumbnail((1200, 900), Image.Resampling.LANCZOS)
                buf = BytesIO()
                im.save(buf, "JPEG", quality=86)
                return {
                    "data_url": "data:image/jpeg;base64," +
                                base64.b64encode(buf.getvalue()).decode("ascii"),
                    "width": width, "height": height,
                }

            return {"status": "ok", "ref": preview(ref_path), "mov": preview(mov_path)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def validate_pair_paths(self, ref_path: str, mov_path: str) -> dict:
        """Catch obvious cross-sample selections before landmarking or segmentation."""
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.file_matcher import normalize_name
            a = Path(ref_path).stem
            b = Path(mov_path).stem
            key_a, stain_a = normalize_name(a)
            key_b, stain_b = normalize_name(b)
            ok = key_a == key_b
            return {"status": "ok", "compatible": ok, "key_a": key_a, "key_b": key_b,
                    "stain_a": stain_a, "stain_b": stain_b,
                    "reason": ("Filename sample identifiers match" if ok else
                               f"Sample identifiers differ: {key_a} vs {key_b}")}
        except Exception as e:
            return {"status": "error", "compatible": False, "error": str(e)}

    def certify_landmarks(self, payload: dict) -> dict:
        """Fit and certify a full-resolution moving→reference similarity transform."""
        try:
            import numpy as np
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.spatial.serial_registration import landmark_register_and_verify

            ref = np.asarray(payload.get("ref_points") or [], dtype=float)
            mov = np.asarray(payload.get("mov_points") or [], dtype=float)
            px = float(payload.get("pixel_size_um") or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}
            wh = payload.get("image_wh") or None
            image_wh = tuple(wh) if wh and len(wh) == 2 else None
            # Certification ROI (full-res ref coords, same space as ref_points): if drawn,
            # it constrains the fit to the trusted region AND becomes the certified window.
            user_roi = payload.get("roi_polygon") or None
            result = landmark_register_and_verify(
                ref, mov, px, image_wh=image_wh, user_roi_polygon=user_roi,
            )
            matrix = result.get("matrix")
            result["matrix"] = (matrix.tolist() if hasattr(matrix, "tolist") else matrix)
            # RADIUS_LIMITED is analysable: the transform is distance-preserving and its
            # error only attenuates cross-K, so the pair proceeds with a raised radius
            # floor rather than being withheld. See serial_registration for precedence.
            result["is_certified"] = result.get("verdict") in (
                "CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED")
            result["status"] = result.get("verdict")
            result["method"] = "manual_landmark_similarity"
            result["ref_points"] = ref.tolist()
            result["mov_points"] = mov.tolist()
            result["pixel_size_um"] = px
            return {"status": "ok", "certification": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def suggest_local_certification_roi(self, payload: dict) -> dict:
        """Find an ROI where a deformed/global-failing landmark set certifies locally.

        This is a recovery path, not a relaxed standard: candidate ROIs are accepted
        only if rerunning the ordinary landmark certification inside that ROI returns
        CERTIFIED/LOCALLY_CERTIFIED with the same thresholds used everywhere else.
        """
        try:
            import itertools
            import numpy as np
            from shapely.geometry import MultiPoint, box
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.spatial.serial_registration import (landmark_register_and_verify,
                                             CERTIFICATION_GATES)

            ref = np.asarray(payload.get("ref_points") or [], dtype=float).reshape(-1, 2)
            mov = np.asarray(payload.get("mov_points") or [], dtype=float).reshape(-1, 2)
            px = float(payload.get("pixel_size_um") or 0)
            wh = payload.get("image_wh") or None
            image_wh = tuple(wh) if wh and len(wh) == 2 else None
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}
            if image_wh is None:
                return {"status": "error", "error": "Image dimensions are required"}
            n = min(len(ref), len(mov))
            min_n = int(payload.get("min_n") or CERTIFICATION_GATES["min_n"])
            if n < min_n:
                return {"status": "error",
                        "error": f"Need at least {min_n} landmarks to test local ROIs"}
            ref, mov = ref[:n], mov[:n]

            field = box(0, 0, float(image_wh[0]), float(image_wh[1]))
            min_roi_frac = CERTIFICATION_GATES["min_roi_frac"]
            margin_px = max(60.0 / px, 24.0)
            candidates = set()

            # Spatially coherent neighborhoods around each landmark. Larger k values
            # are included so we can recover a stable local window rather than only a
            # tiny cherry-picked cluster.
            dmat = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
            max_k = min(n, max(12, min_n))
            for i in range(n):
                order = np.argsort(dmat[i])
                for k in range(min_n, max_k + 1):
                    candidates.add(tuple(sorted(int(x) for x in order[:k])))

            # For modest landmark counts, also try all min-size subsets. This catches
            # split fields where nearest-neighbor neighborhoods include one bad point.
            if n <= 12:
                for comb in itertools.combinations(range(n), min_n):
                    candidates.add(tuple(comb))

            best = None
            for idx in candidates:
                pts = ref[list(idx)]
                geom = MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(margin_px)
                geom = geom.intersection(field)
                if geom.is_empty or geom.area <= 0:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: g.area)
                if geom.area / float(image_wh[0] * image_wh[1]) < min_roi_frac:
                    continue
                roi = [[float(x), float(y)] for x, y in geom.exterior.coords[:-1]]
                cert = landmark_register_and_verify(
                    ref, mov, px, image_wh=image_wh, min_n=min_n,
                    min_roi_frac=min_roi_frac, user_roi_polygon=roi)
                if cert.get("verdict") not in ("CERTIFIED", "LOCALLY_CERTIFIED"):
                    continue

                # Because this endpoint is invoked only after the whole field failed,
                # a passing suggested ROI is a local certification even when the
                # within-ROI fit itself is clean enough to say CERTIFIED.
                cert["verdict"] = "LOCALLY_CERTIFIED"
                cert["status"] = "LOCALLY_CERTIFIED"
                cert["is_certified"] = True
                cert["method"] = "auto_suggested_local_roi_landmark_similarity"
                cert["reason"] = (
                    "Global landmark fit was deformed, but this auto-suggested ROI "
                    "passes the ordinary landmark certification gates. " +
                    str(cert.get("reason") or "")
                )
                matrix = cert.get("matrix")
                cert["matrix"] = matrix.tolist() if hasattr(matrix, "tolist") else matrix
                cert["ref_points"] = ref.tolist()
                cert["mov_points"] = mov.tolist()
                cert["pixel_size_um"] = px

                score = (
                    float(cert.get("tre_median_um") if cert.get("tre_median_um") is not None else 1e9),
                    -int(cert.get("n") or 0),
                    -float(cert.get("coverage_frac") or 0),
                )
                if best is None or score < best[0]:
                    best = (score, roi, cert)

            if best is None:
                return {"status": "error",
                        "error": "No local ROI passed the landmark certification gates"}
            _score, roi, cert = best
            return {"status": "ok", "roi_polygon": roi, "certification": cert,
                    "msg": "Auto-suggested local ROI certifies; analysis will be restricted to this window."}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def certify_local_roi_multi(self, payload: dict) -> dict:
        """Certify one or more USER-DRAWN ROIs by a LOCAL rigid fit from LoFTR
        correspondences inside each region (landmark fallback if LoFTR cannot match).

        This is the draw-your-own-ROI path. The user draws a shape on the FIXED image; it is
        mirrored onto the moving image via a provisional transform and the fit is recomputed
        LOCALLY inside the ROI, where serial-section deformation is near-affine. Each ROI is
        certified INDEPENDENTLY through the ordinary Fitzpatrick-West gate — the standard is
        not relaxed: the user chooses WHERE, the gate still decides WHETHER.

        payload:
          ref_path, mov_path      : image paths (fixed = reference, moving)
          pixel_size_um           : full-resolution um/px
          rois                    : list of polygons, each Nx2 in FULL-RES reference pixels
          provisional_matrix      : optional 2x3 or 3x3 moving->reference similarity (full-res).
                                    If absent, one is computed automatically (register_similarity).
        Returns per-ROI: verdict, is_certified, cell-error, source, n_correspondences,
        local_matrix (full-res moving->reference) and mov_roi_polygon (full-res moving px).
        """
        try:
            import numpy as np
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.registration import _load_rgb_thumbnail
            from oasis.spatial import serial_registration as sr
            from oasis.spatial import loftr_matcher as lm

            px = float(payload.get("pixel_size_um") or 0)
            rois = payload.get("rois") or []
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}
            if not rois:
                return {"status": "error", "error": "No ROIs supplied"}

            ref_rgb, ref_scale = _load_rgb_thumbnail(
                os.path.expanduser(payload["ref_path"]), max_side=1920)
            mov_rgb, mov_scale = _load_rgb_thumbnail(
                os.path.expanduser(payload["mov_path"]), max_side=1920)
            if ref_rgb is None or mov_rgb is None:
                return {"status": "error", "error": "Could not load one or both images"}
            px_t = px / max(ref_scale, 1e-9)          # thumbnail pixel size

            # provisional moving->reference transform, in thumbnail coords
            def _full_to_thumb(M):
                M = np.asarray(M, float)
                A = M[:2, :2].copy(); t = M[:2, 2].copy()
                return np.hstack([A, (t * ref_scale).reshape(2, 1)])  # translation scales

            prov = payload.get("provisional_matrix")
            if prov is not None:
                M_t = _full_to_thumb(prov)
            else:
                ck = (payload["ref_path"], payload["mov_path"], round(px, 4))
                if ck not in self._prov_cache:
                    self._prov_cache[ck] = np.asarray(
                        sr.register_similarity(ref_rgb, mov_rgb, px_t)["matrix"], float)
                M_t = self._prov_cache[ck]

            def _thumb_to_full(M):
                A = M[:2, :2].copy(); t = M[:2, 2].copy()
                return np.hstack([A, (t / max(ref_scale, 1e-9)).reshape(2, 1)]).tolist()

            def _map_roi_to_mov(roi_t, M):
                A = M[:2, :2]; t = M[:2, 2]
                return (np.asarray(roi_t, float) - t) @ np.linalg.inv(A).T

            tol_um = float(payload.get("tol_um") or 4.0)
            want_corr = bool(payload.get("return_correspondences"))
            # Overlap handling: split any-shape overlapping regions into a planar partition
            # so each intersection becomes its OWN separate region (no double-counted cells).
            if bool(payload.get("partition")) and len(rois) > 1:
                min_area = (40.0 / max(px, 1e-9)) ** 2      # drop slivers < ~40 µm across
                work_regions = _planar_partition(rois, min_area=min_area)
                if not work_regions:
                    work_regions = [{"polygon": r, "is_intersection": False, "origins": [i]}
                                    for i, r in enumerate(rois)]
            else:
                work_regions = [{"polygon": r, "is_intersection": False, "origins": [i]}
                                for i, r in enumerate(rois)]
            out = []
            for i, wr in enumerate(work_regions):
                roi = np.asarray(wr["polygon"], float).reshape(-1, 2)
                if len(roi) < 3:
                    out.append({"index": i, "verdict": "INVALID_ROI",
                                "error": "need >=3 vertices"})
                    continue
                roi_t = roi * ref_scale
                cert = lm.certify_local_roi(ref_rgb, mov_rgb, roi_t, px_t,
                                            provisional_matrix=M_t, tol_um=tol_um,
                                            return_correspondences=want_corr)
                v = cert.get("verdict")
                local_t = cert.get("local_matrix")
                mov_roi_full = None
                local_full = None
                if local_t is not None:
                    local_full = _thumb_to_full(np.asarray(local_t, float))
                    mov_roi_t = _map_roi_to_mov(roi_t, np.asarray(local_t, float))
                    mov_roi_full = (mov_roi_t / max(mov_scale, 1e-9)).tolist()
                cell = (cert.get("cell_error_p90_um") or cert.get("tre_p90_um")
                        or cert.get("tre_median_um"))
                entry = {
                    "index": i,
                    "verdict": v,
                    "is_certified": v in ("CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED"),
                    "source": cert.get("source"),
                    "n_correspondences": cert.get("n_correspondences"),
                    "cell_error_um": cell,
                    "fle_um": cert.get("fle_um_loftr"),
                    "reason": cert.get("reason"),
                    "roi_polygon": roi.tolist(),
                    "mov_roi_polygon": mov_roi_full,
                    "local_matrix": local_full,
                    "is_intersection": bool(wr.get("is_intersection")),
                    "origins": wr.get("origins", [i]),
                }
                if want_corr and cert.get("corr_ref") is not None:
                    cr = np.asarray(cert["corr_ref"], float) / max(ref_scale, 1e-9)
                    cm = np.asarray(cert["corr_mov"], float) / max(mov_scale, 1e-9)
                    entry["corr_ref"] = cr.tolist()
                    entry["corr_mov"] = cm.tolist()
                out.append(entry)
            n_ok = sum(1 for r in out if r.get("is_certified"))
            return {"status": "ok", "rois": out, "n_certified": n_ok, "n_total": len(out),
                    "provisional_method": (None if prov is not None else "register_similarity")}
        except Exception as e:
            import traceback
            return {"status": "error", "error": str(e), "trace": traceback.format_exc()[-800:]}

    def auto_certify_regions(self, payload: dict) -> dict:
        """Fully automatic: tile the tissue into candidate regions and certify each via
        LoFTR-in-ROI, returning only the regions that PASS. No landmarks, no drawing. The
        provisional transform is computed automatically (register_similarity). Bounded by
        max_regions / attempts so it stays responsive."""
        try:
            import numpy as np
            import cv2
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.registration import _load_rgb_thumbnail
            from oasis.spatial import serial_registration as sr
            from oasis.spatial import loftr_matcher as lm

            px = float(payload.get("pixel_size_um") or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}
            ref_rgb, ref_scale = _load_rgb_thumbnail(
                os.path.expanduser(payload["ref_path"]), max_side=1920)
            mov_rgb, mov_scale = _load_rgb_thumbnail(
                os.path.expanduser(payload["mov_path"]), max_side=1920)
            if ref_rgb is None or mov_rgb is None:
                return {"status": "error", "error": "Could not load one or both images"}
            px_t = px / max(ref_scale, 1e-9)
            ck = (payload["ref_path"], payload["mov_path"], round(px, 4))
            if ck not in self._prov_cache:
                self._prov_cache[ck] = np.asarray(
                    sr.register_similarity(ref_rgb, mov_rgb, px_t)["matrix"], float)
            M_t = self._prov_cache[ck]
            H, W = ref_rgb.shape[:2]
            max_regions = int(payload.get("max_regions") or 9)

            # Tissue mask — lenient (pale H-DAB is bright); close small gaps.
            g = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
            mask = (g < 235).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            ys, xs = np.where(mask > 0)
            if len(xs) < 100:
                x0, y0, x1, y1 = 0, 0, W, H
                mask = np.ones((H, W), np.uint8)
                cxc, cyc = W / 2.0, H / 2.0
            else:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                cxc, cyc = float(xs.mean()), float(ys.mean())     # tissue centroid

            def _circle(cx, cy, r, k=40):
                th = np.linspace(0, 2 * np.pi, k, endpoint=False)
                return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]

            # Region size: use the requested value, else AUTO-SELECT the largest size that
            # certifies at the tissue centroid (largest window = most cells for statistics,
            # while still certifiable). Probe is fast (fast-FLE, small working res).
            requested = payload.get("region_um")
            auto_size = not (requested and float(requested) > 0)
            if not auto_size:
                region_um = float(requested)
            else:
                region_um = 260.0
                for s in (600.0, 450.0, 350.0, 260.0):
                    rr = s / px_t
                    if cxc - rr < 0 or cyc - rr < 0 or cxc + rr > W or cyc + rr > H:
                        continue
                    probe = lm.certify_local_roi(
                        ref_rgb, mov_rgb, _circle(cxc, cyc, rr), px_t,
                        provisional_matrix=M_t, fle_fast=True, work_max_dim=800)
                    if probe.get("ok"):
                        region_um = s
                        break
            R = region_um / px_t
            # Non-overlapping grid of SQUARE tiles CLIPPED to the tissue outline — so the
            # regions follow tissue shape (arbitrary polygons), not circles, and a
            # non-overlapping grid means no intersections to resolve within the auto set.
            from shapely.geometry import Polygon as _Poly, box as _box
            tissue_poly = None
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
                if len(c) >= 3:
                    tp = _Poly([(float(x), float(y)) for x, y in c])
                    if not tp.is_valid:
                        tp = tp.buffer(0)
                    if not tp.is_empty and tp.area > 0:
                        tissue_poly = tp.simplify(2.0)
            import math
            step = max(2.0 * R, 8.0)      # non-overlapping tiles → no auto intersections

            def _centers(c, lo, hi):        # cell centres, aligned so (cxc,cyc) IS a centre
                n0 = int(math.ceil((lo + R - c) / step))
                n1 = int(math.floor((hi - R - c) / step))
                return [c + n * step for n in range(n0, n1 + 1)] or [c]
            tiles = []
            for cy in _centers(cyc, y0, y1):
                for cx in _centers(cxc, x0, x1):
                    sq = _box(cx - R, cy - R, cx + R, cy + R)
                    piece = sq.intersection(tissue_poly) if tissue_poly is not None else sq
                    if (not piece.is_empty) and piece.area >= 0.25 * sq.area:
                        if piece.geom_type != "Polygon":
                            piece = max(piece.geoms, key=lambda g: g.area)
                        # certify the centroid tile first (probe already showed it certifies)
                        pts = [[float(x), float(y)] for x, y in piece.exterior.coords[:-1]]
                        if abs(cx - cxc) < 1 and abs(cy - cyc) < 1:
                            tiles.insert(0, pts)
                        else:
                            tiles.append(pts)

            regions, attempts = [], 0
            attempt_cap = max(max_regions * 3, 18)
            for poly_t in tiles:
                if len(regions) >= max_regions or attempts >= attempt_cap:
                    break
                attempts += 1
                roi_t = np.asarray(poly_t, float)
                cert = lm.certify_local_roi(ref_rgb, mov_rgb, roi_t, px_t,
                                            provisional_matrix=M_t, fle_fast=True,
                                            work_max_dim=800)
                if not cert.get("ok"):
                    continue
                local_t = cert.get("local_matrix")
                local_full = mov_roi_full = None
                if local_t is not None:
                    A = np.asarray(local_t, float)[:2, :2]
                    t = np.asarray(local_t, float)[:2, 2]
                    local_full = np.hstack([A, (t / max(ref_scale, 1e-9)).reshape(2, 1)]).tolist()
                    mov_roi_t = (roi_t - t) @ np.linalg.inv(A).T
                    mov_roi_full = (mov_roi_t / max(mov_scale, 1e-9)).tolist()
                cell = (cert.get("cell_error_p90_um") or cert.get("tre_p90_um")
                        or cert.get("tre_median_um"))
                regions.append({
                    "index": len(regions), "verdict": cert.get("verdict"), "is_certified": True,
                    "source": cert.get("source"), "n_correspondences": cert.get("n_correspondences"),
                    "cell_error_um": cell,
                    "roi_polygon": (roi_t / max(ref_scale, 1e-9)).tolist(),
                    "mov_roi_polygon": mov_roi_full, "local_matrix": local_full,
                })
            return {"status": "ok", "regions": regions, "n": len(regions),
                    "attempted": attempts, "candidates": len(tiles),
                    "region_um": round(region_um, 1), "auto_size": auto_size}
        except Exception as e:
            import traceback
            return {"status": "error", "error": str(e), "trace": traceback.format_exc()[-800:]}

    def certify_spatial_auto(self, payload: dict) -> dict:
        """LoFTR-default certification with the global-FIRST policy the Spatial tab leads with:

          1. Try to certify the WHOLE field (tissue outline as one window). If the FW gate
             returns CERTIFIED, the pair is analysable whole — no regions needed.
          2. Otherwise fall back to all locally-certified sub-regions (disjoint allowed) via
             auto_certify_regions.
          3. If neither certifies, say so plainly (the sections are too deformed) — never
             manufacture a pass.

        Returns {mode: 'global' | 'local' | 'none', regions: [...], ...}. Both modes hand back
        the SAME region shape (roi_polygon, mov_roi_polygon, local_matrix, cell_error_um), so
        the UI and the pipeline fan-out treat 'global' as a single whole-field region."""
        try:
            import numpy as np
            import cv2
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.registration import _load_rgb_thumbnail
            from oasis.spatial import serial_registration as sr
            from oasis.spatial import loftr_matcher as lm

            px = float(payload.get("pixel_size_um") or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}
            ref_rgb, ref_scale = _load_rgb_thumbnail(
                os.path.expanduser(payload["ref_path"]), max_side=1920)
            mov_rgb, mov_scale = _load_rgb_thumbnail(
                os.path.expanduser(payload["mov_path"]), max_side=1920)
            if ref_rgb is None or mov_rgb is None:
                return {"status": "error", "error": "Could not load one or both images"}
            px_t = px / max(ref_scale, 1e-9)
            ck = (payload["ref_path"], payload["mov_path"], round(px, 4))
            if ck not in self._prov_cache:
                self._prov_cache[ck] = np.asarray(
                    sr.register_similarity(ref_rgb, mov_rgb, px_t)["matrix"], float)
            M_t = self._prov_cache[ck]
            H, W = ref_rgb.shape[:2]

            # Whole-tissue outline = the global analysis window.
            g = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
            mask = cv2.morphologyEx((g < 235).astype(np.uint8), cv2.MORPH_CLOSE,
                                    np.ones((15, 15), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            global_roi = None
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
                if len(c) >= 3:
                    global_roi = np.asarray(c, float)
            if global_roi is None:
                global_roi = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)

            def _to_full(cert, roi_t):
                local_t = cert.get("local_matrix")
                local_full = mov_roi_full = None
                if local_t is not None:
                    A = np.asarray(local_t, float)[:2, :2]
                    t = np.asarray(local_t, float)[:2, 2]
                    local_full = np.hstack(
                        [A, (t / max(ref_scale, 1e-9)).reshape(2, 1)]).tolist()
                    mov_roi_t = (roi_t - t) @ np.linalg.inv(A).T
                    mov_roi_full = (mov_roi_t / max(mov_scale, 1e-9)).tolist()
                cell = (cert.get("cell_error_p90_um") or cert.get("tre_p90_um")
                        or cert.get("tre_median_um"))
                return {"index": 0, "verdict": cert.get("verdict"), "is_certified": True,
                        "source": cert.get("source"),
                        "n_correspondences": cert.get("n_correspondences"),
                        "cell_error_um": cell,
                        "roi_polygon": (roi_t / max(ref_scale, 1e-9)).tolist(),
                        "mov_roi_polygon": mov_roi_full, "local_matrix": local_full}

            # 1) GLOBAL — whole field, measured FLE (one real attempt).
            gcert = lm.certify_local_roi(ref_rgb, mov_rgb, global_roi, px_t,
                                         provisional_matrix=M_t, fle_fast=False,
                                         work_max_dim=1000)
            if gcert.get("verdict") == "CERTIFIED":
                return {"status": "ok", "mode": "global", "verdict": "CERTIFIED",
                        "regions": [_to_full(gcert, global_roi)], "n": 1,
                        "cell_error_um": (gcert.get("cell_error_p90_um")
                                          or gcert.get("tre_p90_um"))}

            # 2) LOCAL fallback — disjoint certified regions.
            loc = self.auto_certify_regions(payload)
            if loc.get("status") != "ok":
                return loc
            if loc.get("n", 0) > 0:
                loc.update(mode="local", global_verdict=gcert.get("verdict"))
                return loc

            # 3) Nothing certifies — report it, do not force a pass.
            return {"status": "ok", "mode": "none", "regions": [], "n": 0,
                    "global_verdict": gcert.get("verdict"),
                    "reason": "Neither the whole field nor any sub-region could be certified — "
                              "the sections are too deformed for cell-scale spatial analysis. "
                              "Try the manual-landmark path, or a different pair."}
        except Exception as e:
            import traceback
            return {"status": "error", "error": str(e), "trace": traceback.format_exc()[-800:]}

    def propose_landmarks(self, ref_path: str, mov_path: str,
                          pixel_size_um: float, max_points: int = 8,
                          roi_polygon=None) -> dict:
        """Auto-propose corresponding landmarks for the operator to VERIFY.

        Runs the structural (lumen + corner) proposal on downsampled thumbnails,
        then maps the points back to FULL-RESOLUTION image coordinates so they drop
        straight into the landmark canvas (same coordinate space as hand-placed
        clicks). These are proposals only — the human must confirm them; nothing is
        certified here.

        `roi_polygon` (optional) is the operator's Certification ROI in FULL-RES
        REFERENCE coords (from the canvas). It is converted to the proposal's thumbnail
        space (× ref_scale) on the way in; proposals are then restricted to inside it,
        and the ROI mapped into moving space is returned (mov_roi_polygon, full-res
        moving coords) so the UI can show the cropped moving region.
        """
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.registration import _load_rgb_thumbnail
            from oasis.spatial.serial_registration import propose_landmarks as _propose

            px = float(pixel_size_um or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}

            ref_rgb, ref_scale = _load_rgb_thumbnail(
                os.path.expanduser(ref_path), max_side=1920)
            mov_rgb, mov_scale = _load_rgb_thumbnail(
                os.path.expanduser(mov_path), max_side=1920)
            if ref_rgb is None or mov_rgb is None:
                return {"status": "error", "error": "Could not load one or both images"}

            # Full-res ref ROI → thumbnail coords for the (thumbnail-space) proposal.
            roi_thumb = None
            if roi_polygon and len(roi_polygon) >= 3:
                roi_thumb = [[float(x) * ref_scale, float(y) * ref_scale]
                             for x, y in roi_polygon]

            # Proposal runs in thumbnail space; measure at the thumbnail's pixel size.
            prop = _propose(ref_rgb, mov_rgb, px / max(ref_scale, 1e-9),
                            max_points=int(max_points), roi_polygon=roi_thumb)
            if not prop.get("ok"):
                return {"status": "error", "error": prop.get("msg") or "no proposals"}

            # Thumbnail px → full-res px (canvas coordinates use full-res dims).
            ref_pts = [[x / ref_scale, y / ref_scale] for x, y in prop["ref_points"]]
            mov_pts = [[x / mov_scale, y / mov_scale] for x, y in prop["mov_points"]]
            roi = prop.get("roi_polygon")
            roi_full = ([[x / ref_scale, y / ref_scale] for x, y in roi] if roi else None)
            mov_roi = prop.get("mov_roi_polygon")
            mov_roi_full = ([[x / mov_scale, y / mov_scale] for x, y in mov_roi]
                            if mov_roi else None)
            # No per-point match score is surfaced: the operator verifies every
            # proposed correspondence against the images before it can be certified.
            return {"status": "ok", "ref_points": ref_pts, "mov_points": mov_pts,
                    "n": prop["n"], "msg": prop["msg"],
                    "mode": prop.get("mode"), "coverage_frac": prop.get("coverage_frac"),
                    "fit_residual_um": prop.get("fit_residual_um"),
                    "roi_polygon": roi_full, "mov_roi_polygon": mov_roi_full,
                    "n_lumen_ref": prop["n_lumen_ref"], "n_lumen_mov": prop["n_lumen_mov"]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def suggest_moving_landmark(self, ref_path: str, mov_path: str,
                                pixel_size_um: float, ref_point,
                                ref_points=None, mov_points=None,
                                roi_polygon=None) -> dict:
        """Suggest one moving-tissue landmark after the user clicks fixed tissue.

        All UI coordinates are full-resolution image coordinates. The guidance runs
        on bounded thumbnails for speed, then maps the suggested moving point back
        to full resolution so it can be inserted into the existing landmark list.
        """
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.registration import _load_rgb_thumbnail
            from oasis.spatial.serial_registration import suggest_moving_landmark as _suggest

            px = float(pixel_size_um or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}
            if not ref_point or len(ref_point) < 2:
                return {"status": "error", "error": "A reference landmark is required"}

            ref_rgb, ref_scale = _load_rgb_thumbnail(
                os.path.expanduser(ref_path), max_side=1920)
            mov_rgb, mov_scale = _load_rgb_thumbnail(
                os.path.expanduser(mov_path), max_side=1920)
            if ref_rgb is None or mov_rgb is None:
                return {"status": "error", "error": "Could not load one or both images"}

            ref_thumb = [float(ref_point[0]) * ref_scale,
                         float(ref_point[1]) * ref_scale]
            existing_ref_thumb = [
                [float(x) * ref_scale, float(y) * ref_scale]
                for x, y in (ref_points or [])
            ]
            existing_mov_thumb = [
                [float(x) * mov_scale, float(y) * mov_scale]
                for x, y in (mov_points or [])
            ]
            roi_thumb = None
            if roi_polygon and len(roi_polygon) >= 3:
                roi_thumb = [[float(x) * ref_scale, float(y) * ref_scale]
                             for x, y in roi_polygon]

            prop = _suggest(ref_rgb, mov_rgb, ref_thumb, px / max(ref_scale, 1e-9),
                            existing_ref_pts=existing_ref_thumb,
                            existing_mov_pts=existing_mov_thumb,
                            roi_polygon=roi_thumb)
            if not prop.get("ok"):
                return {"status": "error",
                        "error": prop.get("msg") or "No guided correspondence found",
                        "method": prop.get("method")}

            mx, my = prop["mov_point"]
            return {
                "status": "ok",
                "mov_point": [mx / mov_scale, my / mov_scale],
                "method": prop.get("method"),
                "n_inliers": prop.get("n_inliers"),
                "msg": prop.get("msg"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def guide_landmark_candidates(self, ref_path: str, mov_path: str,
                                  pixel_size_um: float,
                                  ref_points=None, mov_points=None,
                                  roi_polygon=None, max_points: int = 24,
                                  image_wh=None) -> dict:
        """Return next candidate correspondences for the guided landmark workflow.

        The UI presents these for researcher acceptance/rejection, so no automated
        match score is reported: the operator is the arbiter of every correspondence.
        Candidates are full-resolution coordinates and are filtered away from
        already accepted landmarks so the workflow keeps expanding spatial support.

        CERTIFIED is always the target. Candidates are scored ONLY against the
        field-wide verdict, and a candidate that would certify globally always
        outranks one that would not. Local certification is never offered here — it
        is a terminal recovery step the UI takes once the candidate pool is exhausted
        and a global certification has been shown to be unreachable.
        """
        try:
            import numpy as np
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.registration import _load_rgb_thumbnail
            from oasis.spatial.serial_registration import (propose_landmarks as _propose,
                                             _fit_similarity_ls,
                                             landmark_register_and_verify,
                                             CERTIFICATION_GATES)

            px = float(pixel_size_um or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}

            ref_rgb, ref_scale = _load_rgb_thumbnail(
                os.path.expanduser(ref_path), max_side=1920)
            mov_rgb, mov_scale = _load_rgb_thumbnail(
                os.path.expanduser(mov_path), max_side=1920)
            if ref_rgb is None or mov_rgb is None:
                return {"status": "error", "error": "Could not load one or both images"}

            existing_ref = np.asarray(ref_points or [], dtype=float).reshape(-1, 2)
            existing_mov = np.asarray(mov_points or [], dtype=float).reshape(-1, 2)
            eref_thumb = np.asarray(
                [[x * ref_scale, y * ref_scale] for x, y in existing_ref],
                dtype=float).reshape(-1, 2)
            emov_thumb = np.asarray(
                [[x * mov_scale, y * mov_scale] for x, y in existing_mov],
                dtype=float).reshape(-1, 2)

            seed = None
            n = min(len(eref_thumb), len(emov_thumb))
            if n >= 2:
                seed = _fit_similarity_ls(emov_thumb[:n], eref_thumb[:n])
            elif n == 1:
                dx, dy = (eref_thumb[0] - emov_thumb[0]).tolist()
                seed = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=float)

            roi_thumb = None
            if roi_polygon and len(roi_polygon) >= 3:
                roi_thumb = [[float(x) * ref_scale, float(y) * ref_scale]
                             for x, y in roi_polygon]

            prop = _propose(ref_rgb, mov_rgb, px / max(ref_scale, 1e-9),
                            max_points=int(max_points), seed_transform=seed,
                            roi_polygon=roi_thumb)
            if not prop.get("ok"):
                return {"status": "error", "error": prop.get("msg") or "no proposals"}

            ref_full = np.asarray(
                [[x / ref_scale, y / ref_scale] for x, y in prop["ref_points"]],
                dtype=float).reshape(-1, 2)
            mov_full = np.asarray(
                [[x / mov_scale, y / mov_scale] for x, y in prop["mov_points"]],
                dtype=float).reshape(-1, 2)
            image_wh = tuple(image_wh) if image_wh and len(image_wh) == 2 else None
            min_n = CERTIFICATION_GATES["min_n"]

            def score_candidate(rp, mp):
                """Field-wide certification outcome if this candidate were accepted."""
                if image_wh is None:
                    return None
                next_ref = np.vstack([existing_ref, np.asarray(rp, dtype=float).reshape(1, 2)])
                next_mov = np.vstack([existing_mov, np.asarray(mp, dtype=float).reshape(1, 2)])
                if len(next_ref) < min_n:
                    return None
                user_roi = roi_polygon if roi_polygon and len(roi_polygon) >= 3 else None
                return landmark_register_and_verify(
                    next_ref, next_mov, px, image_wh=image_wh,
                    user_roi_polygon=user_roi,
                )

            min_sep_px = max(30.0 / px, 20.0)
            candidates = []
            for rp, mp in zip(ref_full, mov_full):
                if len(existing_ref):
                    d = np.linalg.norm(existing_ref - rp, axis=1)
                    if float(d.min()) < min_sep_px:
                        continue
                    spread = float(d.min())
                else:
                    spread = 1e9
                cert = score_candidate(rp, mp) or {}
                candidates.append({
                    "ref_point": [float(rp[0]), float(rp[1])],
                    "mov_point": [float(mp[0]), float(mp[1])],
                    "spread_px": spread,
                    "certification_status": cert.get("verdict"),
                    "tre_median_um": cert.get("tre_median_um"),
                    "coverage_frac": cert.get("coverage_frac"),
                })

            # Prefer a route to a field-wide CERTIFIED verdict whenever one exists.
            certifying = [c for c in candidates
                          if c.get("certification_status") == "CERTIFIED"]
            if certifying:
                candidates = certifying

            def progress_rank(c):
                """Rank by how close accepting this candidate leaves the GLOBAL fit to
                certification: lower held-out TRE first, then wider spatial support."""
                tre = c.get("tre_median_um")
                tre = float(tre) if tre is not None else 1e9
                coverage = float(c.get("coverage_frac") or 0.0)
                return (tre, -coverage, -float(c.get("spread_px") or 0))

            if certifying:
                candidates.sort(key=progress_rank)
            elif any(c.get("tre_median_um") is not None for c in candidates):
                candidates.sort(key=progress_rank)
            else:
                # Below min_n there is no measurable fit yet — spread out to build one.
                candidates.sort(key=lambda c: c["spread_px"], reverse=True)
            return {"status": "ok", "candidates": candidates,
                    "n": len(candidates),
                    "certification_gates": dict(CERTIFICATION_GATES),
                    "certification_ready": bool(certifying)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def certify_expert_landmarks(self, ref_path: str, mov_path: str,
                                 pixel_size_um: float) -> dict:
        """Certify a pair from the bundled CIMA/ANHIR EXPERT landmark CSVs.

        For validation/demo cohorts whose expert landmarks ship in
        validation/public_landmarks (e.g. lung-lesion_1 Cc10/Ki67/proSPC). The
        two image filenames are matched to their expert landmark CSVs by stain
        token; the certification is otherwise identical to certify_landmarks
        (same thresholds), so a reviewer cannot tell an imported certification
        apart from a hand-placed one except by the `method` field. Returns the
        same shape as certify_landmarks; errors out (never silently passes) if
        expert landmarks for both stains are not found.
        """
        try:
            import re, glob
            import numpy as np
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.file_matcher import normalize_name
            from oasis.spatial.serial_registration import landmark_register_and_verify

            px = float(pixel_size_um or 0)
            if px <= 0:
                return {"status": "error", "error": "A valid pixel size is required"}

            _, stain_a = normalize_name(Path(ref_path).stem)
            _, stain_b = normalize_name(Path(mov_path).stem)
            if not stain_a or not stain_b:
                return {"status": "error", "error": (
                    "Could not identify both stains from the filenames; expert-"
                    "landmark import needs recognised stain tokens in each name.")}

            ann = PROJECT_DIR / "validation" / "public_landmarks" / "annotations"
            key = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
            ka, kb = key(stain_a), key(stain_b)

            def find_csvs(d):
                hit = {}
                for c in glob.glob(os.path.join(d, "*.csv")):
                    ck = key(Path(c).stem)
                    if ka in ck and "a" not in hit: hit["a"] = c
                    if kb in ck and "b" not in hit: hit["b"] = c
                return hit if "a" in hit and "b" in hit else None

            found = None
            for tissue in sorted(glob.glob(str(ann / "*"))):
                for psdir in sorted(glob.glob(os.path.join(tissue, "user-PS_scale-*"))):
                    found = find_csvs(psdir)
                    if found:
                        break
                if found:
                    break
            if not found:
                return {"status": "error", "error": (
                    f"No bundled expert landmarks found for stains "
                    f"'{stain_a}'/'{stain_b}'.")}

            def load_xy(path):
                rows = list(__import__("csv").reader(open(path)))
                pts = [[float(r[1]), float(r[2])] for r in rows[1:] if len(r) >= 3]
                return np.asarray(pts, dtype=float)

            ref, mov = load_xy(found["a"]), load_xy(found["b"])
            n = min(len(ref), len(mov))
            if n < 6:
                return {"status": "error", "error": "Fewer than 6 expert landmarks."}

            # Bind certification to the ACTUAL image being analysed (its real px dims).
            from PIL import Image
            with Image.open(os.path.expanduser(ref_path)) as im:
                image_wh = im.size  # (w, h)

            result = landmark_register_and_verify(
                ref[:n], mov[:n], px, image_wh=image_wh,
                min_n=6, target_n=12, loo_max_um=5.0, fit_max_um=5.0,
                deformed_loo_um=15.0, min_roi_frac=0.10,
            )
            matrix = result.get("matrix")
            result["matrix"] = (matrix.tolist() if hasattr(matrix, "tolist") else matrix)
            # RADIUS_LIMITED is analysable: the transform is distance-preserving and its
            # error only attenuates cross-K, so the pair proceeds with a raised radius
            # floor rather than being withheld. See serial_registration for precedence.
            result["is_certified"] = result.get("verdict") in (
                "CERTIFIED", "LOCALLY_CERTIFIED", "RADIUS_LIMITED")
            result["status"] = result.get("verdict")
            result["method"] = "cima_expert_landmark_import"
            result["landmark_source"] = os.path.relpath(found["a"], str(PROJECT_DIR))
            result["ref_points"] = ref[:n].tolist()
            result["mov_points"] = mov[:n].tolist()
            result["pixel_size_um"] = px
            return {"status": "ok", "certification": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def preview_batch_pairs(self, folder_a: str, folder_b: str = None,
                            mode: str = "two_folder") -> dict:
        """Match images for a batch run (no analysis) — pairs + unmatched + scaled."""
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from oasis.common.file_matcher import match_two_folders, match_single_folder
            if mode == "two_folder":
                if not folder_a or not folder_b:
                    return {"status": "error", "error": "Both folders required"}
                result = match_two_folders(os.path.expanduser(folder_a),
                                           os.path.expanduser(folder_b))
            else:
                if not folder_a:
                    return {"status": "error", "error": "Folder required"}
                result = match_single_folder(os.path.expanduser(folder_a))
            return {"status": "ok", **result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_spatial_association_results(self, output_dir: str) -> dict:
        """Load spatial-association results (incl. null stats + QC overlay paths)."""
        output_dir = os.path.expanduser(output_dir)
        results = self._load_spatial_results(output_dir)   # null stats now persisted
        return {"status": "ok", "results": results, "output_dir": output_dir}

    def _scale_px_map(self, folder) -> dict:
        """Map {analysis_filename: measured_pixel_size} for a folder's scale images.

        Reuses the Quant scale-matcher so Spatial batch mode can calibrate each
        analysis image from its own burned-in scale bar (the "_scale" sibling)
        instead of applying one session pixel size to the whole cohort. Only
        readable pairs are returned; unmatched images fall back to the session
        value in the caller.
        """
        out = {}
        if not folder:
            return out
        prev = self.preview_quant_scale_matches(folder)
        if prev.get("status") != "ok":
            return out
        for p in prev.get("pairs", []):
            if p.get("filename") and p.get("pixel_size"):
                out[p["filename"]] = float(p["pixel_size"])
        return out

    def preview_spatial_scale_matches(self, folder_a: str, folder_b: str = None,
                                      mode: str = "two_folder") -> dict:
        """Return matched scale-bar pixel sizes for Spatial batch mode before landmarking."""
        try:
            scale_px = {}
            if mode == "two_folder":
                scale_px.update(self._scale_px_map(folder_a))
                scale_px.update(self._scale_px_map(folder_b))
            else:
                scale_px.update(self._scale_px_map(folder_a))
            vals = list(scale_px.values())
            return {
                "status": "ok",
                "pixel_sizes": scale_px,
                "count": len(vals),
                "global_pixel_size": float(median(vals)) if vals else None,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def precheck_bandwidth_for_pair(self, config: dict) -> dict:
        """Pre-run '75 µm bandwidth validity' check for one pair OR a whole batch.

        Segments each pair (writing reusable GeoJSONs to the same output dir the full
        run uses), builds the certified analysis window using the LANDMARK-CERTIFIED
        transform, and returns the per-image architecture-scale verdict plus the
        per-pair null plan (which primary null a full run would use) WITHOUT the
        expensive Monte-Carlo statistic. Runs synchronously.

        Certification is REQUIRED: the bandwidth is only meaningful inside a certified
        analysis window, so any pair that is not CERTIFIED/LOCALLY_CERTIFIED is blocked
        by the pipeline and returns no verdict. The mode ('single' or 'batch') is taken
        from the config, so the batch tab validates every matched pair in one pass. A
        subsequent full run with reuse_existing_geojson=True skips re-segmentation.
        """
        return self.run_spatial_association(config, precheck_only=True)

    def run_spatial_association(self, config: dict, precheck_only: bool = False) -> dict:
        """
        Run the spatial-association pipeline (single pair or batch) in a thread.
        Reuses the shared pipeline (run_pipeline.py --mode spatial); per-image
        pixel sizes are resolved up front and injected as pixel_overrides.

        When precheck_only is True, runs synchronously in a special bandwidth-pre-flight
        mode (segment → window → 75 µm verdict, no permutations) and returns the verdict.
        """
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from run_pipeline import resolve_pixel_size
            from oasis.common.pixel_size_util import extract_pixel_size_from_scale_bar
        except Exception as e:
            self._emit("done", {"ok": False, "msg": f"import failed: {e}"})
            return {"ok": False}

        def _exp(p):
            return os.path.expanduser(p) if p else None

        setup      = self.get_setup()
        output_dir = os.path.expanduser(
            config.get("output_dir", str(Path.home() / "Desktop/ihc_spatial_results")))
        os.makedirs(output_dir, exist_ok=True)

        mode          = config.get("mode", "single")
        certifications = config.get("certifications") or {}
        session_px    = config.get("session_pixel_size")
        session_scale = _exp(config.get("session_scale_image"))
        # Per-image DAB thresholds from the UI (Image A vs Image B). These flow to
        # QuPath as per-image overrides (priority over stain_thresholds matching).
        dab_threshold_a = config.get("dab_threshold_a", 0.20)
        dab_threshold_b = config.get("dab_threshold_b", 0.10)
        # Optional per-image membrane remeasurement. Preserve QuPath for both by
        # default; cytoplasm mode must be explicitly requested.
        use_cyto_a = bool(config.get("use_cytoplasm_a", False))
        use_cyto_b = bool(config.get("use_cytoplasm_b", False))
        cell_expansion_um = config.get("cell_expansion_um", 2.0)
        # Batch-mode options mirrored from the Quant tab:
        #   adaptive_threshold   – classify each image at its own per-image Otsu cut
        #   preprocess_normalize – white-balance every input to its own white point
        #   match_scale_images   – calibrate each image from its "_scale" sibling
        # Adaptive is mutually exclusive with a fixed per-image threshold override
        # (run_pipeline suppresses adaptive when threshold_overrides is set), so we
        # skip populating threshold_overrides when adaptive is on.
        adaptive_threshold   = bool(config.get("adaptive_threshold", False))
        preprocess_normalize = bool(config.get("preprocess_normalize", False))
        match_scale_images   = bool(config.get("match_scale_images", False))
        pixel_overrides, threshold_overrides, cytoplasm_overrides = {}, {}, {}
        membrane_overrides = {}          # per-image calibrated membrane cutoffs
        pairs, unmatched = [], []

        def _membrane_cutoffs(filename, profile_name, marker, use_cyto):
            """Resolve the calibrated membrane cutoffs for one image and stash them so
            the pipeline applies the RIGHT per-marker completeness thresholds (CD8 and
            TIM-3 differ). Warns if membrane mode was asked for but no calibration
            exists — the pipeline then falls back to the weaker ring-mean rule."""
            if not use_cyto:
                return
            cal = self._resolve_calibration(profile_name, marker)
            if cal and cal.get("membrane_pix_thr") is not None \
                    and cal.get("membrane_frac_min") is not None:
                membrane_overrides[os.path.basename(filename)] = {
                    "membrane_pix_thr": float(cal["membrane_pix_thr"]),
                    "membrane_frac_min": float(cal["membrane_frac_min"]),
                }
            else:
                self._emit("log", {"msg": f"Membrane mode requested for {marker} but no "
                                   f"calibration found — falling back to ring-mean "
                                   f"(weaker). Calibrate {marker} in the Calibrate tab.",
                                   "level": "warn"})

        # Resolve the session calibration to ONE concrete value up front, so it
        # flows to every image as the default (priority step 3) unless that image
        # has its own per-image override. Without this, scale-image session
        # calibration left session_pixel_size=None and each analysis image fell
        # through to per-image scale-bar extraction (wrong values, e.g. TIM-3).
        session_value = None
        if session_px and float(session_px) > 0:
            session_value = float(session_px)
        elif session_scale:
            session_value = extract_pixel_size_from_scale_bar(session_scale)
        if session_value:
            self._emit("log", {"msg": f"Session pixel size: {session_value:.4f} µm/px "
                                      f"(default for all images unless overridden)",
                               "level": "info"})

        if mode == "single":
            ia = _exp(config.get("image_a")) or ""
            ib = _exp(config.get("image_b")) or ""
            la = (config.get("label_a") or "MARKER_A").upper()
            lb = (config.get("label_b") or "MARKER_B").upper()
            pa = resolve_pixel_size(session_value, ia, _exp(config.get("scale_image_a")),
                                    config.get("pixel_size_a"))
            pb = resolve_pixel_size(session_value, ib, _exp(config.get("scale_image_b")),
                                    config.get("pixel_size_b"))
            pixel_overrides[os.path.basename(ia)] = pa
            pixel_overrides[os.path.basename(ib)] = pb
            if not adaptive_threshold:
                threshold_overrides[os.path.basename(ia)] = dab_threshold_a
                threshold_overrides[os.path.basename(ib)] = dab_threshold_b
            cytoplasm_overrides[os.path.basename(ia)] = use_cyto_a
            cytoplasm_overrides[os.path.basename(ib)] = use_cyto_b
            _membrane_cutoffs(ia, config.get("calib_profile_a"), la, use_cyto_a)
            _membrane_cutoffs(ib, config.get("calib_profile_b"), lb, use_cyto_b)
            ref_px = pa
            sid = os.path.splitext(os.path.basename(ia))[0] or "pair"
            pairs = [{"sample_id": sid, "stain_a": la, "stain_b": lb,
                      "path_a": ia, "path_b": ib,
                      "filename_a": os.path.basename(ia),
                      "filename_b": os.path.basename(ib),
                      "certification": certifications.get(sid)}]
        else:
            prev = self.preview_batch_pairs(config.get("folder_a"),
                                            config.get("folder_b"),
                                            config.get("folder_mode", "two_folder"))
            if prev.get("status") != "ok":
                self._emit("done", {"ok": False, "msg": prev.get("error", "matching failed")})
                return {"ok": False}
            pairs     = prev.get("pairs", [])
            for p in pairs:
                p["certification"] = certifications.get(p.get("sample_id"))
            unmatched = prev.get("unmatched", [])
            # Session pixel size is the default for every image in the batch…
            ref_px = resolve_pixel_size(session_value, "", None, None)
            # …unless "match scale images" is on: then each analysis image is
            # calibrated from its own "_scale" sibling's burned-in bar, and only
            # images without a readable scale fall back to the session value.
            scale_px = {}
            if match_scale_images:
                if config.get("folder_mode", "two_folder") == "two_folder":
                    scale_px.update(self._scale_px_map(config.get("folder_a")))
                    scale_px.update(self._scale_px_map(config.get("folder_b")))
                else:
                    scale_px.update(self._scale_px_map(config.get("folder_a")))
                n = len(scale_px)
                if n:
                    matched_global = float(median(scale_px.values()))
                    ref_px = matched_global
                self._emit("log", {"msg": (
                    f"Matched scale images: {n} image(s) calibrated from their own "
                    f"scale bar; analysis default updated globally to "
                    f"{ref_px:.4f} µm/px for this run; images without a scale match "
                    f"use that value"
                ) if n else (
                    "Match scale images: no readable scale bars found — using the "
                    f"session pixel size ({ref_px:.4f} µm/px) for all images"
                ), "level": "info" if n else "warn"})
            for p in pairs:
                fa, fb = os.path.basename(p["path_a"]), os.path.basename(p["path_b"])
                pixel_overrides[fa] = scale_px.get(fa, ref_px)
                pixel_overrides[fb] = scale_px.get(fb, ref_px)
                if not adaptive_threshold:
                    threshold_overrides[fa] = dab_threshold_a
                    threshold_overrides[fb] = dab_threshold_b
                cytoplasm_overrides[os.path.basename(p["path_a"])] = use_cyto_a
                cytoplasm_overrides[os.path.basename(p["path_b"])] = use_cyto_b
                _membrane_cutoffs(p["path_a"], config.get("calib_profile_a"),
                                  p.get("stain_a"), use_cyto_a)
                _membrane_cutoffs(p["path_b"], config.get("calib_profile_b"),
                                  p.get("stain_b"), use_cyto_b)

        # ── Phase 2: per-ROI fan-out ──────────────────────────────────────────────
        # A pair whose certification carries multiple user-drawn ROIs (from
        # certify_local_roi_multi) is expanded into ONE analyzable pair per CERTIFIED
        # ROI. Each ROI carries its OWN local transform + window and is analysed
        # SEPARATELY — never pooled, because the transforms differ. Deformed ROIs are
        # dropped here (honestly not analysed), not silently downgraded.
        expanded = []
        for p in pairs:
            cert = p.get("certification") or {}
            roi_list = cert.get("roi_certifications")
            if not roi_list:
                expanded.append(p)
                continue
            kept = 0
            for r in roi_list:
                if not r.get("is_certified"):
                    self._emit("log", {"msg": f"{p.get('sample_id')} ROI "
                                       f"{r.get('index')}: {r.get('verdict')} — not "
                                       f"analysed (deformed).", "level": "warn"})
                    continue
                q = dict(p)
                q["sample_id"] = f"{p['sample_id']}__roi{r.get('index', kept)}"
                q["roi_label"] = r.get("label") or f"ROI {r.get('index')}"
                q["certification"] = {
                    "is_certified": True,
                    "status": r.get("verdict"), "verdict": r.get("verdict"),
                    "matrix": r.get("local_matrix"),
                    "roi_polygon": r.get("roi_polygon"),
                    "cell_error_um": r.get("cell_error_um"),
                    "method": "user_roi_loftr_local",
                }
                expanded.append(q)
                kept += 1
            if kept == 0:
                self._emit("log", {"msg": f"{p.get('sample_id')}: no ROI certified — "
                                   f"pair not analysed.", "level": "warn"})
        pairs = expanded

        if not pairs:
            self._emit("done", {"ok": False, "msg": "No pairs to analyze"})
            return {"ok": False}

        cfg = {
            **setup,
            "qupath_binary":       setup.get("qupath_binary", DEFAULT_SETUP["qupath_binary"]),
            "instanseg_model":     setup.get("instanseg_model", DEFAULT_SETUP["instanseg_model"]),
            "device":              setup.get("device", "mps"),
            "instanseg_threads":   setup.get("instanseg_threads", 4),
            "tile_dims":           512,
            "timeout_seconds":     1800,
            "mode":                "spatial",
            "stain_type":          "hdab",
            "image_extensions":    ["*.tif","*.tiff","*.svs","*.ndpi","*.png"],
            "magnification":       "auto",
            "export_geojson":      True,
            "dab_threshold":       config.get("dab_threshold", 0.2),
            "output_dir":          output_dir,
            "dashboard_dir":       output_dir,
            "default_pixel_size":  ref_px,
            "pixel_overrides":     pixel_overrides,
            "threshold_overrides": threshold_overrides,
            "cytoplasm_overrides": cytoplasm_overrides,
            "membrane_overrides":  membrane_overrides,
            "adaptive_threshold":  adaptive_threshold,
            "preprocess_normalize": preprocess_normalize,
            "cell_expansion_um":   cell_expansion_um,
            "_pixel_size_from_ui": True,
            "pixel_size_mode":     "global",
            "max_distance_um":     config.get("max_distance_um", 10.0),
            "enable_registration": config.get("enable_registration", True),
            "cleanup_intermediates": config.get("cleanup_intermediates", False),
            "spatial_pairs":       pairs,
            "require_landmark_certification": True,
            "dense_auto_null":     True,
            "dense_min_positive":  30,
            "dense_min_support":   500,
            # Reuse existing segmentation when a prior pre-flight already produced it
            # (skips a second QuPath pass). Fresh segmentation during the pre-flight.
            "reuse_existing_geojson": (False if precheck_only
                                       else bool(config.get("reuse_existing_geojson", False))),
        }
        if precheck_only:
            cfg["precheck_bandwidth_only"] = True
            # Certification IS required for the bandwidth pre-flight: the 75 µm check
            # is only meaningful inside the certified analysis window (measured with
            # the landmark-certified transform). Any pair that is not CERTIFIED /
            # LOCALLY_CERTIFIED is blocked upstream and returns no verdict, so a
            # DEFORMED / uncertified pair never yields a (misleading) bandwidth call.
            cfg["require_landmark_certification"] = True

        # Per-stain DAB thresholds — mirror the CLI pipeline so the Spatial
        # Association tab also gives TIM-3 images 0.1 and CD8 images 0.2. Use the
        # user's saved setup value if present, otherwise fall back to defaults.
        if setup.get("stain_thresholds"):
            cfg["stain_thresholds"] = setup.get("stain_thresholds", {})
        else:
            cfg["stain_thresholds"] = {
                "cd8": 0.2,
                "tim3": 0.1,
                "tim-3": 0.1,
            }

        config_name = ("spatial_precheck_config.yaml" if precheck_only
                       else "spatial_config.yaml")
        config_path = str(CONFIG_DIR / config_name)
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # ── Synchronous bandwidth pre-flight (UI "Validate 75 µm bandwidth") ──────
        if precheck_only:
            import json as _json
            try:
                proc = subprocess.run(
                    [str(Path(sys.executable)), str(PROJECT_DIR / "run_pipeline.py"),
                     "--config", config_path, "--mode", "spatial"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    cwd=str(PROJECT_DIR), timeout=cfg.get("timeout_seconds", 1800))
            except Exception as e:
                return {"status": "error", "error": f"pre-flight failed: {e}"}
            by_pair = {}
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if line.startswith("BANDWIDTH_PRECHECK_JSON:"):
                    try:
                        d = _json.loads(line[len("BANDWIDTH_PRECHECK_JSON:"):])
                        by_pair[d.get("sample_id")] = d
                    except Exception:
                        pass
            if not by_pair:
                # Surface the REAL cause. The subprocess may have exited non-zero with
                # its traceback on STDERR (bad interpreter/path, import error), or every
                # pair was blocked as uncertified — reading stdout alone hides both and
                # mislabels them "segmentation failed". Prefer stderr in the tail.
                err_tail = "\n".join((proc.stderr or "").splitlines()[-15:])
                out_tail = "\n".join((proc.stdout or "").splitlines()[-15:])
                return {"status": "error",
                        "error": ("Bandwidth pre-flight produced no verdict "
                                  f"(run_pipeline exited {proc.returncode}). Either "
                                  "segmentation failed or no pair was certified."),
                        "returncode": proc.returncode,
                        "log_tail": err_tail or out_tail}
            return {"status": "ok", "precheck_by_pair": by_pair}

        def run():
            try:
                # Surface unmatched files as warnings before starting
                for u in unmatched:
                    self._emit("log", {"msg": f"Unmatched: {u.get('filename')} — "
                                              f"{u.get('reason','no pair')}", "level": "warn"})

                skip = ["[INFO ]","[WARN ]","Measured Detection","Completed Annotation",
                        "Processing complete in","Measuring","Loading:","████","WARNING: Unknown"]
                self._process = subprocess.Popen(
                    [str(Path(sys.executable)), str(PROJECT_DIR / "run_pipeline.py"),
                     "--config", config_path, "--mode", "spatial"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, start_new_session=True, cwd=str(PROJECT_DIR),
                )
                while True:
                    line = self._process.stdout.readline()
                    if not line and self._process.poll() is not None:
                        break
                    if not line:
                        continue
                    clean = line.strip()
                    if not clean or any(x in clean for x in skip):
                        continue
                    level = "normal"
                    if any(x in clean for x in ["ERROR","FAILED","TIMEOUT"]):
                        level = "error"
                    elif any(x in clean for x in ["WARNING","Unmatched"]):
                        level = "warn"
                    elif any(x in clean for x in ["✓","COMPLETE","complete","matched cells","QC overlay"]):
                        level = "ok"
                    elif any(x in clean for x in ["Processing","Registering","Matching",
                                                  "Association","Running","PAIR","Pixel size"]):
                        level = "info"
                    self._emit("log", {"msg": clean, "level": level})

                if self._process.returncode == 0:
                    self._emit("progress", {"pct": 100})
                    res = self.get_spatial_association_results(output_dir)
                    results = res.get("results", [])
                    # Summary across pairs: cross-type association is a curve, so
                    # we aggregate the per-pair verdict. QC-invalid pairs are
                    # EXCLUDED (statistics not interpretable). Among QC-valid pairs we
                    # count SIGNIFICANT findings (verdict 'robust' = significant under
                    # the calibrated reweighted primary) separately from CSR-ONLY
                    # findings (significant only under the weak homogeneous baseline,
                    # i.e. a shared-preference artifact — see ihc.md §15).
                    n_sig, global_ps, n_excluded_qc = 0, [], 0
                    n_robust, n_csr_only = 0, 0
                    for r in results:
                        qc = r.get("registration_qc") or {}
                        if qc.get("status") == "invalid":
                            n_excluded_qc += 1
                            continue
                        for v in (r.get("spatial_association") or {}).get("association", {}).values():
                            g = v.get("global", {}) or {}
                            rob = v.get("robustness", {}) or {}
                            if g.get("significant"):
                                n_sig += 1
                            if rob.get("verdict") == "robust":
                                n_robust += 1
                            elif rob.get("verdict") == "csr_only":
                                n_csr_only += 1
                            if isinstance(g.get("global_p_dclf"), (int, float)):
                                global_ps.append(g["global_p_dclf"])
                    # Cohort-level multiple-comparison correction across the
                    # QC-valid pairs' DCLF p-values. The raw minimum is a
                    # multiplicity trap; any cohort claim must use the FDR result.
                    cohort_fdr = None
                    try:
                        from oasis.spatial.spatial_stats import cohort_multiple_comparison_correction
                        cohort_fdr = cohort_multiple_comparison_correction(
                            global_ps, method="bh")
                    except Exception as e:
                        self._emit("log", {"msg": f"Cohort FDR correction failed: {e}",
                                           "level": "warn"})
                    summary = {
                        "pairs":          len(results),
                        "n_significant":  n_sig,
                        "n_robust":       n_robust,
                        "n_csr_only":     n_csr_only,
                        "n_excluded_qc":  n_excluded_qc,
                        # Raw minimum kept for transparency but explicitly labelled
                        # as NOT a cohort finding (see cohort_fdr / the UI note).
                        "min_global_p_raw": round(min(global_ps), 4) if global_ps else None,
                        "cohort_fdr":     cohort_fdr,
                    }
                    self._emit("spatial_assoc_complete",
                               {"ok": True, "results": results,
                                "summary": summary, "output_dir": output_dir})
                else:
                    stderr = self._process.stderr.read()
                    self._emit("done", {"ok": False,
                                        "msg": "Spatial association pipeline failed",
                                        "stderr": stderr[-500:] if stderr else ""})
            except Exception as e:
                self._emit("done", {"ok": False, "msg": str(e)})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def get_home(self) -> str:
        """Return user home directory for UI path defaults."""
        return str(Path.home())

    def open_file(self, path):
        if path and os.path.exists(path):
            subprocess.Popen(["open", path])
            return {"ok": True}
        return {"ok": False}

    def open_folder(self, path):
        if path and os.path.exists(path):
            subprocess.Popen(["open", path])
            return {"ok": True}
        return {"ok": False}

    # ── Validation framework (Validation tab) ───────────────────────────────
    def _emit_validation(self, event, data):
        try:
            js = f"window.onValidationEvent({json.dumps({'type': event, 'data': data})})"
            self._window.evaluate_js(js)
        except Exception:
            pass

    def get_validation_data_dir(self):
        from validation.datasets import resolve as _R
        return {"path": str(_R.dataset_root()),
                "configured": self.get_setup().get("validation_data_dir")}

    def set_validation_data_dir(self, path):
        setup = self.get_setup()
        setup.pop("_home", None)
        setup["validation_data_dir"] = os.path.expanduser(str(path))
        self.save_setup(setup)
        return {"ok": True, "path": setup["validation_data_dir"]}

    def get_dataset_status(self):
        """Per-dataset presence + checksum status for the Validation tab."""
        from validation.datasets import verify as _V
        try:
            return {"ok": True, "datasets": _V.status()}
        except Exception as e:
            return {"ok": False, "msg": str(e), "datasets": []}

    def dataset_download_info(self, name):
        from validation.datasets import resolve as _R
        rec = _R.datasets().get(name, {})
        return {
            "name": name, "title": rec.get("title"),
            "source_url": rec.get("source_url"), "license": rec.get("license"),
            "redistributable": bool(rec.get("redistributable")),
            "place_under": str(_R.dataset_inputs(name)),
            "citation": rec.get("citation"),
        }

    def list_validations(self):
        """Registry grouped by category, each item enriched with preflight +
        dataset availability + last-run summary — the Validation-tab card data."""
        from validation import registry as _reg, runner as _run
        cats = []
        for cat in _reg.by_category():
            items = []
            for v in cat["validations"]:
                pf = _run.preflight(v["id"])
                last = _run.last_report(v["id"])
                items.append({
                    **{k: v[k] for k in ("id", "title", "claim", "purpose", "why",
                                         "datasets", "assumptions", "limitations",
                                         "interpretation", "expected", "runtime_tier",
                                         "external_deps")},
                    "preflight": pf,
                    "last": ({"status": last["status"],
                              "timestamp_utc": last["timestamp_utc"],
                              "metrics": last.get("metrics", {}),
                              "duration_s": last.get("duration_s")} if last else None),
                })
            cats.append({"key": cat["key"], "title": cat["title"], "validations": items})
        return {"ok": True, "categories": cats}

    def run_validation(self, vid, force=False):
        """Run one validation in the background, streaming log lines to the UI."""
        from validation import runner as _run

        def run():
            try:
                self._emit_validation("start", {"id": vid})
                rep = _run.run_validation(
                    vid, force=bool(force),
                    on_line=lambda ln, lvl: self._emit_validation(
                        "log", {"id": vid, "msg": ln, "level": lvl}))
                self._emit_validation("done", {"id": vid, "report": {
                    "status": rep["status"], "metrics": rep.get("metrics", {}),
                    "duration_s": rep.get("duration_s"),
                    "reason": rep.get("reason"),
                    "dir": str((_run.REPORTS_ROOT / vid / rep["timestamp_utc"]))}})
            except Exception as e:
                self._emit_validation("done", {"id": vid, "report": {
                    "status": "ERROR", "error": str(e)}})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def get_validation_report(self, vid):
        from validation import runner as _run
        rep = _run.last_report(vid)
        if not rep:
            return {"ok": False}
        return {"ok": True, "report": rep,
                "dir": str(_run.REPORTS_ROOT / vid / rep["timestamp_utc"])}


# Isolated additive extension: same-section restained co-expression workflow.
from oasis.webui.restained_api import attach_restained_api
attach_restained_api(API)
