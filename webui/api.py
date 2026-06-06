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
}

# Standard pixel sizes per magnification
STANDARD_PIXEL_SIZE = {
    "4x": 2.50, "10x": 1.00, "20x": 0.50, "40x": 0.25, "60x": 0.165, "100x": 0.10
}

PROJECT_DIR = Path(__file__).parent.parent


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

    def get_standard_pixel_size(self, objective):
        return STANDARD_PIXEL_SIZE.get(objective, 0.5)

    # ── Pipeline ───────────────────────────────────────────────────────────
    def run_pipeline(self, settings):
        setup = self.get_setup()

        pixel_size_mode = settings.get("pixel_size_mode", "manual")
        pixel_from_ui = pixel_size_mode in ("manual", "global")

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
            "image_extensions":   ["*.tif","*.tiff","*.svs","*.ndpi","*.png"],
            "magnification":      "auto",
            "export_geojson":     True,
            "overlay_downsample": 1.0,
            "_pixel_size_from_ui": pixel_from_ui,
            "objective":          settings.get("objective", "10x"),
        }

        for k in ["input_dir","output_dir","dashboard_dir","instanseg_model"]:
            if k in cfg and cfg[k]:
                cfg[k] = os.path.expanduser(str(cfg[k]))
        cfg.setdefault("dashboard_dir", str(Path(cfg.get("input_dir","")) / "output_results"))
        os.makedirs(cfg.get("output_dir",""), exist_ok=True)
        os.makedirs(cfg.get("dashboard_dir",""), exist_ok=True)

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
        for jp in sorted(glob.glob(str(Path(output_dir) / "*_summary.json"))):
            try:
                with open(jp) as f:
                    d = json.load(f)
                metrics.append({
                    "name":        Path(d.get("image","")).stem.split(" - ")[0],
                    "total_cells": d.get("total_cells",0),
                    "positive":    d.get("positive_cells",0),
                    "negative":    d.get("negative_cells",0),
                    "positivity":  float(d.get("positivity_pct",0)),
                    "pixel_size":  d.get("pixel_size_um",0.5),
                    "threshold":   d.get("dab_threshold",0.2),
                })
            except Exception:
                continue

        summary_text = ""
        sp = Path(output_dir) / "analysis_summary.txt"
        if sp.exists():
            summary_text = sp.read_text().strip()

        dashboards = sorted(glob.glob(str(Path(dashboard_dir) / "ihc_dashboard_*.html")))
        excels     = sorted(glob.glob(str(Path(dashboard_dir) / "ihc_results_*.xlsx")))

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

    # ── File matching (for Co-localization UI) ─────────────────────────────
    def preview_pairs(self, mode: str, folder_a: str, folder_b: str = None) -> dict:
        """
        Detect image pairs for co-localization without running any analysis.

        mode: "two_folder" | "single_folder"
        Returns pairs list + unmatched lists for UI preview.
        """
        try:
            sys.path.insert(0, str(PROJECT_DIR))
            from file_matcher import match_two_folders, match_single_folder
            if mode == "two_folder":
                if not folder_a or not folder_b:
                    return {"ok": False, "error": "Both folders required for two-folder mode"}
                result = match_two_folders(
                    os.path.expanduser(folder_a),
                    os.path.expanduser(folder_b),
                )
            else:
                if not folder_a:
                    return {"ok": False, "error": "Folder required"}
                result = match_single_folder(os.path.expanduser(folder_a))
            return {"ok": True, **result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Co-localization pipeline ───────────────────────────────────────────
    def run_coloc_pipeline(self, settings: dict) -> dict:
        """Run the co-localization pipeline in a background thread."""
        pairs = settings.get("pairs", [])
        if not pairs:
            return {"ok": False, "error": "No pairs provided"}

        setup = self.get_setup()
        output_dir = os.path.expanduser(
            settings.get("output_dir", str(Path.home() / "Desktop/ihc_coloc_results"))
        )
        os.makedirs(output_dir, exist_ok=True)

        cfg = {
            **setup,
            "qupath_binary":      setup.get("qupath_binary", DEFAULT_SETUP["qupath_binary"]),
            "instanseg_model":    setup.get("instanseg_model", DEFAULT_SETUP["instanseg_model"]),
            "device":             setup.get("device", "mps"),
            "instanseg_threads":  setup.get("instanseg_threads", 4),
            "tile_dims":          512,
            "timeout_seconds":    1800,
            "mode":               "automated",
            "stain_type":         "hdab",
            "image_extensions":   ["*.tif","*.tiff","*.svs","*.ndpi","*.png"],
            "magnification":      "auto",
            "export_geojson":     True,
            "dab_threshold":      settings.get("dab_threshold", 0.2),
            "output_dir":         output_dir,
            "dashboard_dir":      output_dir,
            "default_pixel_size": settings.get("pixel_size", 0.5),
            "_pixel_size_from_ui": True,
            "pixel_size_mode":    "global",
            "max_distance_um":    settings.get("max_distance_um", 10.0),
            "enable_registration": settings.get("enable_registration", True),
            "cleanup_intermediates": settings.get("cleanup_intermediates", False),
            "coloc_pairs":        pairs,
        }

        config_path = str(CONFIG_DIR / "coloc_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        def run():
            try:
                skip = ["[INFO ]","[WARN ]","Measured Detection","Completed Annotation",
                        "Processing complete in","Measuring","Loading:","████","WARNING: Unknown"]
                self._process = subprocess.Popen(
                    [str(Path(sys.executable)), str(PROJECT_DIR / "run_pipeline.py"),
                     "--config", config_path, "--mode", "coloc"],
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
                    elif any(x in clean for x in ["WARNING"]):
                        level = "warn"
                    elif any(x in clean for x in ["✓","COMPLETE","complete","matched cells"]):
                        level = "ok"
                    elif any(x in clean for x in ["Processing","Registering","Matching",
                                                   "Co-expression","Running","PAIR","STARTING"]):
                        level = "info"
                    self._emit("log", {"msg": clean, "level": level})

                if self._process.returncode == 0:
                    self._emit("progress", {"pct": 100})
                    results = self._load_coloc_results(output_dir)
                    self._emit("done", {"ok": True, "results": results})
                else:
                    stderr = self._process.stderr.read()
                    self._emit("done", {"ok": False, "msg": "Co-localization pipeline failed",
                                        "stderr": stderr[-500:] if stderr else ""})
            except Exception as e:
                self._emit("done", {"ok": False, "msg": str(e)})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def _load_coloc_results(self, output_dir: str) -> list:
        """Load combined co-localization results from disk."""
        path = Path(output_dir) / "coloc_results.json"
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