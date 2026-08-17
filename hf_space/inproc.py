"""
inproc.py — running OASIS's background jobs inside this process, for ZeroGPU.

WHY THIS EXISTS. Every long job in the desktop app is a subprocess:

    subprocess.Popen(worker_cmd("run_pipeline", "--config", ...), stdout=PIPE, ...)

which is exactly right on a desktop — the window stays responsive and the worker can be
killed by process group. It is fatal on ZeroGPU. A `@spaces.GPU` function is executed by
the ZeroGPU scheduler in a process it owns; an ordinary child this app forks is not that
process and gets no CUDA at all. The pipeline would run, produce correct output, and take
the CPU's time to do it — the same invisible "you paid for a GPU and got a CPU" failure
`oasis/common/device.py` was written to prevent, one level further out.

So on the Space the work has to happen in THIS process, where the ZeroGPU seam
(`hf_space/gpu.py`) can reach it.

Rather than edit the three call sites in `oasis/` — this module is a drop-in stand-in for
the `subprocess` module. `hf_space/app.py` rebinds `oasis.webui.api.subprocess` (and the two
others) to it at startup, and `api.py` goes on calling `subprocess.Popen(...)` and reading
`.stdout.readline()` / `.poll()` / `.returncode` without knowing anything changed. The
desktop app imports the real `subprocess` and is untouched.

WHAT IS FAITHFULLY REPRODUCED, because api.py depends on each of them:
  • line-buffered `.stdout.readline()` that returns "" only at EOF
  • `.poll()` -> None while running, the exit status afterwards
  • non-zero `.returncode` when the job raises, with the traceback on `.stderr`
  • `subprocess.run(...)` returning an object with `.stdout` / `.stderr` / `.returncode`

WHAT DELIBERATELY DIFFERS: `.pid` is -1. `api.stop_pipeline` does
`os.killpg(os.getpgid(self._process.pid), SIGTERM)` inside a try/except, falling back to
`.terminate()`. A plausible pid would be actively dangerous here — `os.getpgid(0)` returns
the SERVER's own process group, so a stop button would kill the Space. -1 cannot name a
process group, `getpgid` raises, and the fallback runs. See `terminate()` for what stopping
can and cannot mean in-process.
"""
import os
import queue
import shutil
import sys
import tempfile
import threading
import traceback

PIPE = -1
STDOUT = -2
DEVNULL = -3

# One job at a time. On a desktop each run is its own process and the OS schedules them;
# on a public Space every visitor shares one interpreter and one ZeroGPU allocation, so
# two runs at once means two jobs interleaving log lines, competing for the GPU and
# doubling the wall-clock of both. Queueing is the honest behaviour, and the waiting run
# says so in its own log rather than appearing to hang.
JOB_LOCK = threading.Lock()

# Called on the job thread before any work starts. hf_space/app.py points this at
# gpu.install_request, which re-establishes the visitor's ZeroGPU quota token across the
# thread boundary. A hook rather than a direct import so this module stays a pure stand-in
# for `subprocess` and can be tested without the GPU seam.
ON_JOB_THREAD = None

# The pipeline is single-threaded (nothing in run_pipeline.py, segment.py, spatial.py or
# serial_registration.py starts a thread — serial_registration.py:449 records that threading
# it measured 1.04x and was not worth it), so routing captured output by thread identity
# below captures all of it. If a stage ever gains a worker thread its prints would fall
# through to the server log instead of the UI; that is a degraded log, not a wrong result.
_ROUTER = None


class _LineStream:
    """The read end of a job's output: writes go in, whole lines come out.

    `readline()` blocks until a line is available and returns "" at EOF, which is the
    contract api.py's reader loop is written against:

        line = self._process.stdout.readline()
        if not line and self._process.poll() is not None:
            break
    """

    def __init__(self):
        self._lines = queue.Queue()
        self._partial = ""
        self._lock = threading.Lock()
        self._eof = False          # write side: sentinel has been queued
        self._drained = False      # read side: sentinel has been seen

    # ── write side (the job thread) ──────────────────────────────────────────
    def write(self, text):
        if not text:
            return 0
        with self._lock:
            self._partial += text
            while True:
                i = self._partial.find("\n")
                if i < 0:
                    break
                self._lines.put(self._partial[:i + 1])
                self._partial = self._partial[i + 1:]
        return len(text)

    def flush(self):
        pass

    def close(self):
        """Flush any unterminated last line and signal EOF."""
        with self._lock:
            if self._partial:
                self._lines.put(self._partial)
                self._partial = ""
            if not self._eof:
                self._eof = True
                self._lines.put(None)          # sentinel

    # ── read side (api.py's reader loop) ─────────────────────────────────────
    def readline(self):
        # EOF is sticky. Without the flag, a write that lands after close() would sit
        # behind the sentinel in the queue and be handed out by a later readline() —
        # api.py's loop would then see output resume after it had already seen EOF.
        if self._drained:
            return ""
        item = self._lines.get()
        if item is None:
            self._drained = True
            return ""
        return item

    def read(self):
        out = []
        while True:
            line = self.readline()
            if not line:
                return "".join(out)
            out.append(line)

    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                return
            yield line


class _ThreadRouter:
    """Stands in for sys.stdout/sys.stderr and routes each write by calling thread.

    A job thread's prints go to that job's `_LineStream`; everything else falls through to
    the real stream, so the Space's own container log keeps working. `contextlib.
    redirect_stdout` cannot be used for this — it swaps the stream process-wide, so a
    second visitor's job would capture the first one's output.
    """

    def __init__(self, real):
        self._real = real
        self._bound = {}
        self._lock = threading.Lock()

    def bind(self, ident, stream):
        with self._lock:
            self._bound[ident] = stream

    def unbind(self, ident):
        with self._lock:
            self._bound.pop(ident, None)

    def _target(self):
        return self._bound.get(threading.get_ident(), self._real)

    def write(self, text):
        return self._target().write(text)

    def flush(self):
        try:
            self._target().flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def fileno(self):
        return self._real.fileno()

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    def writelines(self, lines):
        for line in lines:
            self.write(line)


def install_stdout_router():
    """Install the routers on sys.stdout/sys.stderr. Idempotent, and self-repairing.

    Re-checking on every call rather than only installing once: anything that replaces
    sys.stdout AFTER startup displaces the router, and from then on every worker's output
    goes wherever that replacement points instead of to the UI. The run would stream
    nothing and look hung — the exact symptom `worker_env`'s PYTHONUNBUFFERED exists to
    prevent, from a different cause. When that happens the new stream is adopted as the
    fall-through target so nothing is lost, and the router goes back on top.
    """
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = (_ThreadRouter(sys.stdout), _ThreadRouter(sys.stderr))
    out, err = _ROUTER
    if sys.stdout is not out:
        out._real = sys.stdout
        sys.stdout = out
    if sys.stderr is not err:
        err._real = sys.stderr
        sys.stderr = err
    return _ROUTER


def remove_stdout_router():
    """Put the real streams back. Only needed by tests; the Space never uninstalls."""
    global _ROUTER
    if _ROUTER is None:
        return
    out, err = _ROUTER
    if sys.stdout is out:
        sys.stdout = out._real
    if sys.stderr is err:
        sys.stderr = err._real
    _ROUTER = None


# ── argv -> in-process call ──────────────────────────────────────────────────

def parse_worker_argv(cmd):
    """(module, args) for a command built by oasis.common.worker.worker_cmd.

    From source that is [python, "-u", "-m", <module>, *args]; frozen it is
    [exe, "--oasis-worker", <module>, *args]. Both are recognised so this cannot be
    broken by how the host happens to be running.
    """
    argv = [str(c) for c in cmd]
    if "--oasis-worker" in argv:
        i = argv.index("--oasis-worker")
        return argv[i + 1], argv[i + 2:]
    if "-m" in argv:
        i = argv.index("-m")
        return argv[i + 1], argv[i + 2:]
    raise ValueError(f"not an OASIS worker command: {argv!r}")


def _snapshot_config(args):
    """Copy the job's --config to a private path, and point the job at the copy.

    `api.py` writes every run's YAML to one shared file — `CONFIG_DIR/pipeline_config.yaml`
    (api.py:1114) — which is correct for a single-user desktop app and a data race on a
    Space, where two visitors share one config directory. Queued behind JOB_LOCK, visitor
    B's config would overwrite visitor A's before A's job ever read it, and A would then be
    analysed with B's images, thresholds and output directory. Nothing would look wrong:
    the run succeeds and reports numbers for the wrong slide.

    Snapshotting at construction time — while the caller's write is still the current
    contents — closes that window without needing a lock or touching api.py.

    Returns (args, scratch_dir_or_None).
    """
    if "--config" not in args:
        return args, None
    i = args.index("--config")
    if i + 1 >= len(args):
        return args, None
    src = args[i + 1]
    if not os.path.isfile(src):
        return args, None
    scratch = tempfile.mkdtemp(prefix="oasis-job-")
    dst = os.path.join(scratch, os.path.basename(src))
    shutil.copyfile(src, dst)
    return args[:i + 1] + [dst] + args[i + 2:], scratch


def dispatch(module, args):
    """Run one worker to completion in this thread.

    THE DISPATCH BELOW MIRRORS run_pipeline.py's `__main__` BLOCK. That is the thing it
    replaces, and the two drifting apart would mean the Space quietly running a different
    stage than the CLI for the same arguments — so tests/test_spaces_inproc.py pins every
    --mode / --stage combination against this function.
    """
    if module == "oasis.restained.restained_coexpression":
        from oasis.restained import restained_coexpression
        restained_coexpression.main(args)
        return

    if module != "run_pipeline":
        raise ValueError(f"unknown OASIS worker module: {module!r}")

    import argparse
    import run_pipeline as rp

    parser = argparse.ArgumentParser(description="OASIS Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="quant")
    parser.add_argument("--stage", default="full", choices=["full", "segment", "finish"])
    ns = parser.parse_args(args)

    # "coloc" is run_pipeline.py's hidden deprecated alias for "spatial".
    mode = "spatial" if ns.mode == "coloc" else ns.mode
    if mode == "spatial":
        rp.run_spatial_association_pipeline(ns.config)
    elif mode == "quant":
        if ns.stage == "finish":
            rp.finish_outputs(rp.load_config(ns.config))
        elif ns.stage == "segment":
            cfg = rp.load_config(ns.config)
            cfg["stop_after_segmentation"] = True
            rp.run_pipeline(cfg)
        else:
            rp.run_pipeline(ns.config)
    else:
        raise ValueError(f"unknown --mode {ns.mode!r} (use 'quant' or 'spatial')")


class Popen:
    """The subset of subprocess.Popen that oasis/webui uses, run in a thread instead."""

    def __init__(self, cmd, stdout=None, stderr=None, text=True, cwd=None, env=None,
                 start_new_session=False, **_ignored):
        self.args = cmd
        self.pid = -1                       # see the module docstring: never a real pid
        self.returncode = None
        self.stdout = _LineStream()
        # stderr=STDOUT is how a caller asks for one merged stream.
        self.stderr = self.stdout if stderr == STDOUT else _LineStream()
        self._module, self._args = parse_worker_argv(cmd)
        self._args, self._scratch = _snapshot_config(self._args)
        self._done = threading.Event()
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"oasis-job:{self._module}")
        self._thread.start()

    def _run(self):
        router_out, router_err = install_stdout_router()
        ident = threading.get_ident()
        router_out.bind(ident, self.stdout)
        router_err.bind(ident, self.stderr)
        if ON_JOB_THREAD is not None:
            try:
                ON_JOB_THREAD()
            except Exception:
                pass          # quota attribution is never worth failing a run over
        try:
            if not JOB_LOCK.acquire(blocking=False):
                self.stdout.write(
                    "Waiting for the GPU — another run is in progress on this Space…\n")
                JOB_LOCK.acquire()
            try:
                dispatch(self._module, self._args)
            finally:
                JOB_LOCK.release()
            self.returncode = 0
        except BaseException:
            self.returncode = 1
            tb = traceback.format_exc()
            # Faithful: a real worker's traceback lands on stderr, and api.py's quant reader
            # only watches stdout, so it reports "Pipeline failed" exactly as it does today.
            try:
                self.stderr.write(tb)
            except Exception:
                pass
            # Also to the Space's container log, which is the only place an operator can
            # read it — a real subprocess's traceback would show up there too.
            try:
                sys.__stderr__.write(tb)
            except Exception:
                pass
        finally:
            router_out.unbind(ident)
            router_err.unbind(ident)
            self.stdout.close()
            if self.stderr is not self.stdout:
                self.stderr.close()
            if self._scratch:
                shutil.rmtree(self._scratch, ignore_errors=True)
            self._done.set()

    def poll(self):
        return self.returncode if self._done.is_set() else None

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode

    def communicate(self, input=None, timeout=None):
        out = self.stdout.read()
        err = self.stderr.read() if self.stderr is not self.stdout else ""
        self.wait(timeout)
        return out, err

    def terminate(self):
        """Best effort. In-process work cannot be killed the way a process can.

        A subprocess dies on SIGTERM mid-tile. Here the job runs in this interpreter, and
        killing a thread from outside is not something Python offers — so the flag is set,
        the streams are closed so the UI's reader loop unblocks and the run stops being
        reported, and the remaining work finishes into a directory nobody is watching.
        This is stated plainly rather than papered over: Stop means "stop showing me this"
        on the Space, and "stop computing" on the desktop.
        """
        self._cancelled = True
        self.stdout.close()
        if self.stderr is not self.stdout:
            self.stderr.close()

    kill = terminate


class CompletedProcess:
    def __init__(self, args, returncode, stdout, stderr):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run(cmd, stdout=None, stderr=None, text=True, cwd=None, env=None, timeout=None,
        capture_output=False, **_ignored):
    """Blocking form, used by the bandwidth pre-flight (api.py) and Calibrate.

    Waits for completion BEFORE draining, so `timeout` means something: the reads would
    otherwise block until the job ended on its own and the deadline could never fire. A
    timed-out job is abandoned rather than killed — see Popen.terminate.
    """
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, text=text, cwd=cwd, env=env)
    finished = proc._done.wait(timeout)
    if not finished:
        proc.terminate()
    out = proc.stdout.read()
    err = proc.stderr.read() if proc.stderr is not proc.stdout else ""
    if not finished:
        err += f"\nTimeoutExpired: worker exceeded {timeout}s and was abandoned\n"
    return CompletedProcess(cmd, proc.returncode if finished else 1, out, err)
