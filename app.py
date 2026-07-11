"""
app.py — OASIS Desktop App
pywebview + HTML/CSS/JS frontend
"""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _check_dependencies():
    """Fail fast with an ACTIONABLE message if launched on the wrong interpreter /
    an environment missing the pinned deps (rather than a raw ModuleNotFoundError)."""
    required = {"webview": "pywebview", "numpy": "numpy", "cv2": "opencv-python",
                "PIL": "pillow", "scipy": "scipy", "shapely": "shapely",
                "SimpleITK": "SimpleITK"}
    missing = []
    for mod, pkg in required.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        venv_py = Path(__file__).parent / ".venv" / "bin" / "python"
        sys.stderr.write(
            "\nOASIS cannot start — missing dependencies: "
            + ", ".join(missing) + "\n"
            f"You are running: {sys.executable}\n"
            + (f"Use the project venv instead:\n    {venv_py} app.py\n"
               if venv_py.exists() else
               "Create/activate the project venv and `pip install -r requirements.txt`,"
               " then re-run `python app.py`.\n"))
        sys.exit(1)


_check_dependencies()
import webview                       # noqa: E402  (after the actionable dep check)
from oasis.webui.api import API           # noqa: E402

def main():
    api = API()
    html_path = str(Path(__file__).parent / "oasis" / "webui" / "index.html")
    window = webview.create_window(
        title="OASIS",
        url=f"file://{html_path}",
        js_api=api,
        width=1280,
        height=820,
        min_size=(1100, 700),
        background_color="#FFFFFF",
    )
    api.set_window(window)
    webview.start(debug=False)

if __name__ == "__main__":
    main()