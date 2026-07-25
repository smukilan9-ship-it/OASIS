"""
app.py — OASIS Desktop App
pywebview + HTML/CSS/JS frontend
"""
import multiprocessing
import sys
import importlib
from pathlib import Path

# MUST be the first thing that runs in a frozen app on Windows, before any other import
# that might touch multiprocessing. Windows has no fork: a child process is created by
# re-executing this binary and re-importing __main__. In a frozen build that means the
# child starts the whole application again — which spawns another child, and so on. The
# visible symptom is not a crash but a hang (and a growing pile of processes), because
# nothing ever gets far enough to report an error.
#
# freeze_support() makes a re-executed child recognise itself as a child and run its task
# instead of the app. It is a no-op on macOS and Linux, and on any non-frozen run.
multiprocessing.freeze_support()

sys.path.insert(0, str(Path(__file__).parent))

# Background jobs re-invoke this binary with --oasis-worker <module>. That must be handled
# BEFORE anything GUI-related is imported, or a frozen worker would open a second window.
from oasis.common.worker import dispatch_worker          # noqa: E402

if dispatch_worker():
    sys.exit(0)


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
from oasis.common.paths import resource_dir   # noqa: E402

def main():
    api = API()
    # resource_dir() rather than Path(__file__).parent: in a PyInstaller bundle the UI
    # files live under sys._MEIPASS, not beside this script.
    html_path = str(Path(resource_dir()) / "oasis" / "webui" / "index.html")
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