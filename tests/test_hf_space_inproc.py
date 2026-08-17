"""
test_hf_space_inproc.py — the Space runs the same stages the CLI does.

WHY THIS TEST EXISTS. On the Hugging Face Space the pipeline cannot run in a subprocess:
a `@spaces.GPU` function is executed by the ZeroGPU scheduler in a process it owns, and an
ordinary child forked by this app is not that process and gets no CUDA. So `hf_space/
inproc.py` stands in for the `subprocess` module and calls `run_pipeline`'s functions
directly instead.

That makes `inproc.dispatch` a SECOND COPY of the argument handling in run_pipeline.py's
`__main__` block. Two copies drift. The failure that drift produces is the quiet kind: a
`--stage segment` run that forgets to set `stop_after_segmentation` does not crash, it
skips the operator's threshold review and writes final numbers from an uninspected cutoff.
So every --mode / --stage combination is pinned here against what the CLI does with it.

The rest of the file pins the parts of `subprocess.Popen` that `oasis/webui/api.py` reads
back — line streaming, `poll()`, `returncode` — since api.py is unmodified and would break
in ways that look like a broken pipeline if any of them changed.
"""
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hf_space import inproc                                             # noqa: E402


@pytest.fixture(autouse=True)
def _restore_streams():
    """Hand sys.stdout/sys.stderr back after each test.

    The router replaces both process-wide, and pytest's own capture also owns them; leaving
    it installed would have this module's plumbing intercepting the rest of the suite's
    output.
    """
    yield
    inproc.remove_stdout_router()


# ── argv parsing ─────────────────────────────────────────────────────────────

def test_parses_the_from_source_worker_command():
    """oasis.common.worker.worker_cmd's non-frozen form."""
    from oasis.common.worker import worker_cmd
    cmd = worker_cmd("run_pipeline", "--config", "/tmp/c.yaml", "--stage", "segment")
    module, args = inproc.parse_worker_argv(cmd)
    assert module == "run_pipeline"
    assert args == ["--config", "/tmp/c.yaml", "--stage", "segment"]


def test_parses_the_frozen_worker_command():
    """The frozen form, [exe, --oasis-worker, module, ...], is recognised too.

    A Space is never frozen, but reading only one of the two shapes would make this shim
    silently host-dependent — and `worker_cmd` picks the shape from `sys.frozen`.
    """
    cmd = ["/Applications/OASIS.app/Contents/MacOS/OASIS", "--oasis-worker",
           "run_pipeline", "--config", "/tmp/c.yaml", "--mode", "spatial"]
    module, args = inproc.parse_worker_argv(cmd)
    assert module == "run_pipeline"
    assert args == ["--config", "/tmp/c.yaml", "--mode", "spatial"]


def test_rejects_a_command_that_is_not_an_oasis_worker():
    with pytest.raises(ValueError):
        inproc.parse_worker_argv(["git", "status"])


# ── dispatch mirrors run_pipeline.py's __main__ ──────────────────────────────

@pytest.fixture
def spy(monkeypatch):
    """Replace run_pipeline's entry points and record which one was called with what."""
    import run_pipeline as rp
    calls = []
    monkeypatch.setattr(rp, "run_pipeline", lambda cfg: calls.append(("quant", cfg)))
    monkeypatch.setattr(rp, "run_spatial_association_pipeline",
                        lambda cfg: calls.append(("spatial", cfg)))
    monkeypatch.setattr(rp, "finish_outputs", lambda cfg: calls.append(("finish", cfg)))
    monkeypatch.setattr(rp, "load_config", lambda path: {"_loaded_from": path})
    return calls


def test_default_stage_runs_the_full_quant_pipeline(spy):
    inproc.dispatch("run_pipeline", ["--config", "/tmp/c.yaml"])
    assert spy == [("quant", "/tmp/c.yaml")]


def test_stage_segment_stops_after_segmentation(spy):
    """The review gate. api.py:run_pipeline asks for --stage segment whenever the operator
    chose to review the DAB cutoff; if the flag were lost the run would sail past the
    review and report numbers from a cutoff nobody looked at."""
    inproc.dispatch("run_pipeline", ["--config", "/tmp/c.yaml", "--stage", "segment"])
    assert len(spy) == 1
    kind, cfg = spy[0]
    assert kind == "quant"
    assert cfg["stop_after_segmentation"] is True
    assert cfg["_loaded_from"] == "/tmp/c.yaml"


def test_stage_finish_generates_outputs_from_summaries_on_disk(spy):
    inproc.dispatch("run_pipeline", ["--config", "/tmp/c.yaml", "--stage", "finish"])
    assert spy == [("finish", {"_loaded_from": "/tmp/c.yaml"})]


def test_mode_spatial_runs_the_association_pipeline(spy):
    inproc.dispatch("run_pipeline", ["--config", "/tmp/c.yaml", "--mode", "spatial"])
    assert spy == [("spatial", "/tmp/c.yaml")]


def test_coloc_is_still_an_alias_for_spatial(spy):
    """run_pipeline.py keeps "coloc" as a hidden deprecated alias; so must this."""
    inproc.dispatch("run_pipeline", ["--config", "/tmp/c.yaml", "--mode", "coloc"])
    assert spy == [("spatial", "/tmp/c.yaml")]


def test_an_unknown_mode_is_refused(spy):
    with pytest.raises(SystemExit):
        inproc.dispatch("run_pipeline", ["--config", "/tmp/c.yaml", "--stage", "bogus"])
    assert spy == []


def test_an_unknown_module_is_refused():
    with pytest.raises(ValueError):
        inproc.dispatch("some.other.module", [])


# ── the Popen surface api.py reads back ──────────────────────────────────────

class _FakeJob:
    """Stands in for the worker so these tests never run a real pipeline."""

    def __init__(self, monkeypatch, body):
        monkeypatch.setattr(inproc, "dispatch", lambda module, args: body())


def _drain(proc):
    """api.py's reader loop, verbatim in shape (api.py:751)."""
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if not line:
            continue
        lines.append(line.rstrip("\n"))
    return lines


def test_worker_output_is_streamed_line_by_line(monkeypatch):
    def body():
        print("Found 2 images")
        print("PIPELINE FINISHED SUCCESSFULLY")

    _FakeJob(monkeypatch, body)
    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE,
                        stderr=inproc.PIPE)
    lines = _drain(proc)
    assert lines == ["Found 2 images", "PIPELINE FINISHED SUCCESSFULLY"]
    assert proc.returncode == 0


def test_a_worker_that_raises_reports_a_non_zero_returncode(monkeypatch):
    """api.py decides a run failed purely from returncode != 0 (api.py:775)."""
    def body():
        print("Segmenting")
        raise RuntimeError("model file is missing")

    _FakeJob(monkeypatch, body)
    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE,
                        stderr=inproc.PIPE)
    assert _drain(proc) == ["Segmenting"]
    assert proc.returncode == 1
    assert "model file is missing" in proc.stderr.read()


def test_output_written_without_a_trailing_newline_still_arrives(monkeypatch):
    """A worker killed or finishing mid-line must not lose that line."""
    def body():
        sys.stdout.write("partial line, no newline")

    _FakeJob(monkeypatch, body)
    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    assert _drain(proc) == ["partial line, no newline"]


def test_eof_is_sticky(monkeypatch):
    """Once the reader has seen EOF it must never see output again — otherwise api.py's
    loop, which breaks on (no line AND poll() is not None), could resume mid-teardown."""
    _FakeJob(monkeypatch, lambda: print("one"))
    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    _drain(proc)
    proc.stdout.write("late arrival\n")
    assert proc.stdout.readline() == ""


def test_the_pid_can_never_name_a_real_process_group(monkeypatch):
    """api.stop_pipeline does os.killpg(os.getpgid(proc.pid), SIGTERM) in a try/except.
    A plausible pid here would kill the Space itself — os.getpgid(0) is the SERVER's own
    process group. -1 makes getpgid raise, so the .terminate() fallback runs instead."""
    import os
    _FakeJob(monkeypatch, lambda: None)
    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    assert proc.pid == -1
    with pytest.raises(OSError):
        os.getpgid(proc.pid)


def test_run_returns_stdout_stderr_and_returncode(monkeypatch):
    """The bandwidth pre-flight (api.py:3128) parses proc.stdout and reads proc.returncode."""
    def body():
        print('BANDWIDTH_PRECHECK_JSON:{"sample_id": "pair1"}')

    _FakeJob(monkeypatch, body)
    done = inproc.run(["python", "-m", "run_pipeline"], capture_output=True, text=True)
    assert done.returncode == 0
    assert "BANDWIDTH_PRECHECK_JSON" in done.stdout


# ── the config snapshot ──────────────────────────────────────────────────────

def test_the_config_is_snapshotted_so_a_second_visitor_cannot_swap_it(monkeypatch, tmp_path):
    """api.py writes every run's YAML to ONE shared path (CONFIG_DIR/pipeline_config.yaml).

    On a desktop that is fine — one user, one run. On a Space two visitors share that file,
    and with runs queued behind JOB_LOCK the second visitor's write lands before the first
    visitor's job has read it. The first run would then analyse the second's images and
    report them under the first's name: a wrong answer that looks entirely healthy.
    """
    shared = tmp_path / "pipeline_config.yaml"
    shared.write_text("input_dir: /visitor-a\n", encoding="utf-8")

    seen = {}

    def body_reader(module, args):
        # Whatever path the job is given, read it at "run" time.
        path = args[args.index("--config") + 1]
        seen["contents"] = Path(path).read_text(encoding="utf-8")

    monkeypatch.setattr(inproc, "dispatch", body_reader)

    proc = inproc.Popen(["python", "-m", "run_pipeline", "--config", str(shared)],
                        stdout=inproc.PIPE)
    # A second visitor overwrites the shared file while the job is queued/running.
    shared.write_text("input_dir: /visitor-b\n", encoding="utf-8")
    proc.wait(10)

    assert seen["contents"] == "input_dir: /visitor-a\n"


def test_the_snapshot_is_cleaned_up(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\n", encoding="utf-8")
    captured = {}

    def body(module, args):
        captured["path"] = args[args.index("--config") + 1]

    monkeypatch.setattr(inproc, "dispatch", body)
    proc = inproc.Popen(["python", "-m", "run_pipeline", "--config", str(cfg)],
                        stdout=inproc.PIPE)
    proc.wait(10)
    assert not Path(captured["path"]).exists()
    assert cfg.exists()              # the caller's own file is untouched


# ── the module really is a drop-in for `subprocess` ──────────────────────────

def test_exposes_everything_the_app_uses_from_the_subprocess_module():
    """app.py rebinds `oasis.webui.api.subprocess` to this module. Anything api.py,
    calibration.py or restained_api.py reaches for must therefore exist here."""
    for name in ("Popen", "run", "PIPE", "STDOUT"):
        assert hasattr(inproc, name), f"inproc.{name} is missing"


def test_one_job_runs_at_a_time(monkeypatch):
    """Two visitors' runs must queue, not interleave: they share one interpreter and one
    ZeroGPU allocation."""
    import threading
    active = []
    peak = []
    gate = threading.Event()

    def body(module, args):
        active.append(1)
        peak.append(len(active))
        gate.wait(2.0)
        active.pop()

    monkeypatch.setattr(inproc, "dispatch", body)
    a = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    b = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    gate.set()
    a.wait(5)
    b.wait(5)
    assert max(peak) == 1
