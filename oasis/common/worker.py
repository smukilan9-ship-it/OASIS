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


def configure_stdio():
    """Make stdout/stderr UTF-8 so progress output cannot kill a run on Windows.

    Windows defaults console output to the legacy ANSI code page (cp1252), which cannot
    encode characters this pipeline prints routinely — the ✓ in its completion banner, µ in
    "µm/px", arrows in progress lines. Printing one raises UnicodeEncodeError.

    That is not a cosmetic failure. It was found by the bundle smoke test: on Windows the
    pipeline segmented the image correctly, wrote nothing wrong, and then died on the last
    print of the run — after all the real work was done — and the process hung instead of
    exiting. A crash while reporting success is indistinguishable from a crash while
    working, unless you read the traceback.

    errors="replace" rather than "strict" so that a character we did not anticipate degrades
    to "?" instead of destroying a completed analysis.
    """
    for stream in (sys.stdout, sys.stderr):
        # None in a windowed frozen app; older/wrapped streams may lack reconfigure().
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass    # a stream we cannot reconfigure is not worth failing startup over


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
    # -u so the worker's progress is not held in an 8 KB block buffer — see worker_env().
    return [sys.executable, "-u", "-m", module] + argv


def worker_env(base=None):
    """Environment for a streaming worker: unbuffered stdout/stderr.

    Python block-buffers stdout whenever it is a pipe rather than a terminal, and every one
    of these jobs is launched with stdout=PIPE so the UI can stream it. The result is that
    nothing reaches the UI until the child's buffer fills or the process exits: measured on
    a three-pair spatial run, all 182 log lines arrived in one burst at 109 s, after the run
    had already finished. The operator watches a log that stops after the setup lines and an
    app that looks hung for the whole run — on a large cohort, for many minutes.

    `-u` covers the from-source command; a frozen bundle is not launched through a python
    interpreter and has no argv to put it on, so the variable is what covers both.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    return env


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
