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
from oasis.common.worker import configure_stdio, dispatch_worker   # noqa: E402

# Before the worker runs, so its progress output cannot die on a Windows code page.
configure_stdio()

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
    # as_uri(), NOT "file://" + path. A Windows path is "C:\...", so the f-string produced
    # `file://C:\Users\...\index.html`, where everything after the two slashes is parsed as
    # the URL's HOST and the path is left empty. The window then has nothing to load and the
    # app dies at startup with PyInstaller's "Failed to execute script" dialog. It worked on
    # macOS and Linux only by accident: a POSIX path already starts with "/", so the two
    # slashes plus that one happened to make the three a file URL needs.
    # as_uri() gives file:///C:/Users/... and file:///Users/..., and escapes spaces.
    html_path = Path(resource_dir()) / "oasis" / "webui" / "index.html"
    window = webview.create_window(
        title="OASIS",
        url=html_path.as_uri(),
        js_api=api,
        width=1280,
        height=820,
        min_size=(1100, 700),
        background_color="#FFFFFF",
    )
    api.set_window(window)
    webview.start(debug=False)


def check_ui():
    """Everything startup does except opening the window. Exits non-zero if it cannot.

    THE FROZEN SMOKE TEST NEVER LAUNCHED THE UI. It ran `--oasis-worker run_pipeline`,
    which segments an image headlessly and shares almost no code with startup, so a bundle
    whose window could never open passed on all three platforms and shipped. That is
    precisely the "green build, broken app" failure the smoke test exists to prevent, with
    a hole exactly where the app is most platform-specific.

    Measured, v0.1.0: `url=f"file://{path}"` is a valid URL on POSIX by luck and malformed
    on Windows, where the path becomes the URL's host. Windows users got PyInstaller's
    "Failed to execute script" dialog. Nothing in CI could have caught it.

    So: resolve the UI file, build the URL the window is given, and import the GUI backend
    for this platform. No window is created, so it runs headless on a CI runner.
    """
    html = Path(resource_dir()) / "oasis" / "webui" / "index.html"
    if not html.is_file():
        sys.stderr.write(f"UI CHECK FAILED: no index.html at {html}\n")
        return 1
    url = html.as_uri()
    if not url.startswith("file:///"):
        sys.stderr.write(f"UI CHECK FAILED: malformed file URL {url!r}\n")
        return 1
    try:
        # import_module, not `import webview.guilib as g`. webview/guilib.py declares a
        # module-level `guilib = None` that the package re-exports, so the attribute of the
        # same name shadows the submodule and the plain import binds to None.
        _guilib = importlib.import_module("webview.guilib")
        _guilib.initialize()
    except Exception as e:
        sys.stderr.write(f"UI CHECK FAILED: no GUI backend ({type(e).__name__}: {e})\n")
        return 1
    sys.stdout.write(f"UI CHECK OK: {url}\n")
    return 0


if __name__ == "__main__":
    if "--oasis-check-ui" in sys.argv:
        sys.exit(check_ui())
    main()