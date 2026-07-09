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

PROJECT_DIR = Path(__file__).parent.parent

# Preloaded calibration presets (data-backed defaults; user calibrations add to these).
BUILTIN_CALIBRATIONS = [
    {"name": "CRC-ICM (TIM-3)", "marker": "tim-3",
     "membrane_pix_thr": 0.30, "membrane_frac_min": 0.14, "auc": 0.93, "builtin": True},
]


class API:
    def __init__(self):
        self._window  = None
        self._process = None

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
            from webui import calibration
            return calibration.prepare(os.path.expanduser(image_path),
                                       float(pixel_size), self.get_setup())
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    def calibration_fit(self, image_path, geojson_path, pixel_size, pos_idx, neg_idx):
        """Fit membrane cutoffs from hand-labelled positive/negative cell indices."""
        try:
            from webui import calibration
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
            from webui import calibration
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
            from pixel_size_util import _detect_scale_bar
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
            from file_matcher import normalize_name
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
            from serial_registration import landmark_register_and_verify

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
                ref, mov, px, image_wh=image_wh,
                min_n=6, target_n=12, loo_max_um=5.0, fit_max_um=5.0,
                deformed_loo_um=15.0, min_roi_frac=0.10,
                user_roi_polygon=user_roi,
            )
            matrix = result.get("matrix")
            result["matrix"] = (matrix.tolist() if hasattr(matrix, "tolist") else matrix)
            result["is_certified"] = result.get("verdict") in (
                "CERTIFIED", "LOCALLY_CERTIFIED")
            result["status"] = result.get("verdict")
            result["method"] = "manual_landmark_similarity"
            result["ref_points"] = ref.tolist()
            result["mov_points"] = mov.tolist()
            result["pixel_size_um"] = px
            return {"status": "ok", "certification": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
            from registration import _load_rgb_thumbnail
            from serial_registration import propose_landmarks as _propose

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
            return {"status": "ok", "ref_points": ref_pts, "mov_points": mov_pts,
                    "confidences": prop.get("confidences") or [],
                    "n": prop["n"], "msg": prop["msg"],
                    "mode": prop.get("mode"), "coverage_frac": prop.get("coverage_frac"),
                    "fit_residual_um": prop.get("fit_residual_um"),
                    "roi_polygon": roi_full, "mov_roi_polygon": mov_roi_full,
                    "n_lumen_ref": prop["n_lumen_ref"], "n_lumen_mov": prop["n_lumen_mov"]}
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
            from file_matcher import normalize_name
            from serial_registration import landmark_register_and_verify

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
            result["is_certified"] = result.get("verdict") in (
                "CERTIFIED", "LOCALLY_CERTIFIED")
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
            from file_matcher import match_two_folders, match_single_folder
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

    def precheck_bandwidth_for_pair(self, config: dict) -> dict:
        """Pre-run '75 µm bandwidth validity' check for one pair (the UI button).

        Segments the pair (writing reusable GeoJSONs to the same output dir the full
        run uses), builds the certified analysis window, and returns the per-image
        architecture-scale verdict WITHOUT the expensive Monte-Carlo statistic. Runs
        synchronously and returns the verdict directly. A subsequent full run with
        reuse_existing_geojson=True skips re-segmentation.
        """
        return self.run_spatial_association({**config, "mode": "single"},
                                            precheck_only=True)

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
            from pixel_size_util import extract_pixel_size_from_scale_bar
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
                self._emit("log", {"msg": (
                    f"Matched scale images: {n} image(s) calibrated from their own "
                    f"scale bar; the rest use the session pixel size ({ref_px:.4f} µm/px)"
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
            # Reuse existing segmentation when a prior pre-flight already produced it
            # (skips a second QuPath pass). Fresh segmentation during the pre-flight.
            "reuse_existing_geojson": (False if precheck_only
                                       else bool(config.get("reuse_existing_geojson", False))),
        }
        if precheck_only:
            cfg["precheck_bandwidth_only"] = True
            # Certification is not required just to validate the bandwidth assumption.
            cfg["require_landmark_certification"] = False

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
                tail = "\n".join((proc.stdout or "").splitlines()[-15:])
                return {"status": "error",
                        "error": "Could not compute the bandwidth pre-flight "
                                 "(segmentation may have failed).",
                        "log_tail": tail}
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
                        from spatial_stats import cohort_multiple_comparison_correction
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
from webui.restained_api import attach_restained_api
attach_restained_api(API)
