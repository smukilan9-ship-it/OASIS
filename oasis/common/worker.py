"""
worker.py — building the command line for OASIS's own background workers.

WHY THIS EXISTS. The UI runs long jobs (the pipeline, the restained co-expression run) in a
subprocess so the window stays responsive and can stream progress. Those call sites all did:

    subprocess.Popen([sys.executable, "run_pipeline.py", "--config", ...])

which is correct from source and WRONG once the app is frozen. In a PyInstaller/py2app bundle
`sys.executable` is not a Python interpreter — it is the OASIS application binary. The command
above therefore launches a second copy of the GUI, passing it arguments it does not understand,
and the job silently never runs. The failure is invisible until you build the bundle, which is
exactly the kind of thing that surfaces the day you try to ship.

The fix is the standard one for frozen apps: re-invoke the bundle with a sentinel argument that
the entry point recognises before it opens a window, and dispatch to the worker instead.

    from source :  [python, "-m", "run_pipeline", ...]
    frozen      :  [OASIS.app/Contents/MacOS/OASIS, "--oasis-worker", "run_pipeline", ...]

`app.py` calls `dispatch_worker()` as its first action, so the sentinel is handled before
pywebview is even imported.

Note the source form uses `-m module` rather than a script path: `oasis.restained.
restained_coexpression` imports `oasis.*`, so it needs the repo root on sys.path, which `-m`
with cwd=PROJECT_DIR provides and a bare script path does not.
"""
import os
import runpy
import sys

WORKER_FLAG = "--oasis-worker"


def is_frozen():
    """True when running from a PyInstaller/py2app bundle rather than from source."""
    return bool(getattr(sys, "frozen", False))


def worker_cmd(module, *args):
    """Argv to run `module` as a background worker, correct from source and when frozen.

    `module` is an importable module path with a __main__ guard — "run_pipeline" or
    "oasis.restained.restained_coexpression".
    """
    argv = [str(a) for a in args]
    if is_frozen():
        return [sys.executable, WORKER_FLAG, module] + argv
    return [sys.executable, "-m", module] + argv


def dispatch_worker(argv=None):
    """If this process was started as a worker, run it and exit; otherwise return False.

    Must be called before any GUI import so a frozen worker never opens a window.
    """
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 3 or argv[1] != WORKER_FLAG:
        return False
    module = argv[2]
    # hand the worker a normal argv: [module, <its args>]
    sys.argv = [module] + argv[3:]
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return True
