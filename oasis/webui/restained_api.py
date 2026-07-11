"""Additive pywebview API adapter for the isolated restained workflow."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import yaml


def attach_restained_api(api_class):
    """Attach new methods without changing any existing API implementation."""
    project_dir = Path(__file__).parent.parent
    config_dir = Path.home() / ".ihc_analyzer"

    def preview_restained_bundles(self, folder, hematoxylin_token="_Hematoxylin",
                                  marker_a_token="_CD8", marker_b_token="_FoxP3",
                                  reference_mask_folder=None):
        try:
            sys.path.insert(0, str(project_dir))
            from oasis.restained.restained_coexpression import discover_bundles
            complete, incomplete = discover_bundles(
                os.path.expanduser(folder), hematoxylin_token, marker_a_token,
                marker_b_token,
                os.path.expanduser(reference_mask_folder) if reference_mask_folder else None)
            return {"ok": True, "bundles": complete, "incomplete": incomplete}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def run_restained_coexpression(self, config):
        setup = self.get_setup()
        run_config = {
            **config,
            "qupath_binary": setup.get("qupath_binary"),
            "instanseg_model": setup.get("instanseg_model"),
            "device": setup.get("device", "mps"),
            "instanseg_threads": setup.get("instanseg_threads", 4),
            "timeout_seconds": 1800,
        }
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "restained_coexpression_config.yaml"
        with open(config_path, "w") as handle:
            yaml.safe_dump(run_config, handle, default_flow_style=False)

        def emit(event, data):
            self._emit(event, data)

        def run():
            result_path = None
            try:
                self._process = subprocess.Popen(
                    [sys.executable, str(project_dir / "restained_coexpression.py"),
                     "--config", str(config_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    start_new_session=True, cwd=str(project_dir),
                )
                while True:
                    line = self._process.stdout.readline()
                    if not line and self._process.poll() is not None:
                        break
                    clean = line.strip()
                    if not clean:
                        continue
                    if clean.startswith("RESTAINED_PROGRESS "):
                        parts = clean.split(" ", 2)
                        pct = int(parts[1]) if len(parts) > 1 else 0
                        message = parts[2] if len(parts) > 2 else ""
                        emit("restained_progress", {"pct": pct, "msg": message})
                    elif clean.startswith("RESTAINED_RESULT="):
                        result_path = clean.split("=", 1)[1]
                    else:
                        level = "normal"
                        if any(word in clean for word in ("ERROR", "FAILED", "TIMEOUT")):
                            level = "error"
                        elif any(word in clean for word in ("WARNING", "missing", "incomplete")):
                            level = "warn"
                        elif any(word in clean for word in ("complete", "segmented", "verified")):
                            level = "ok"
                        emit("restained_log", {"msg": clean, "level": level})

                stderr = self._process.stderr.read()
                if self._process.returncode != 0:
                    emit("restained_failed", {
                        "msg": "Restained co-expression pipeline failed",
                        "stderr": stderr[-2000:] if stderr else "",
                    })
                    return
                if not result_path or not Path(result_path).exists():
                    emit("restained_failed", {"msg": "Run completed without a result file"})
                    return
                with open(result_path) as handle:
                    result = json.load(handle)
                emit("restained_complete", result)
            except Exception as exc:
                emit("restained_failed", {"msg": str(exc)})

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    api_class.preview_restained_bundles = preview_restained_bundles
    api_class.run_restained_coexpression = run_restained_coexpression

