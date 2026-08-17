"""
test_hf_space_runtime.py — the ZeroGPU seam and where the Space keeps its files.

Both are things that CANNOT be checked by running the Space locally: there is no ZeroGPU
on a laptop (`spaces` is a no-op off-platform, so a broken seam looks identical to a
working one), and the config directory takes the macOS branch here and the Linux branch
there. Getting either wrong fails silently and expensively — the pipeline runs on CPU while
the quota is spent, or every saved classifier disappears on the next Space restart — so
they are pinned here instead of discovered in production.
"""
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── a stand-in for the `spaces` package ──────────────────────────────────────

class _FakeSpaces(types.ModuleType):
    """Records every acquisition the way the real scheduler would charge for one."""

    def __init__(self):
        super().__init__("spaces")
        self.acquisitions = []

    def GPU(self, duration=None, size=None):
        record = self.acquisitions

        def decorate(fn):
            def wrapper(*args, **kwargs):
                record.append({"fn": fn.__name__,
                               "duration": duration(*args, **kwargs) if callable(duration)
                               else duration})
                return fn(*args, **kwargs)
            wrapper.__name__ = fn.__name__
            return wrapper
        return decorate


@pytest.fixture
def seam(monkeypatch):
    """hf_space.gpu with a fake `spaces` installed and the patching undone afterwards."""
    from hf_space import gpu

    fake = _FakeSpaces()
    monkeypatch.setitem(sys.modules, "spaces", fake)
    monkeypatch.setattr(gpu, "_reentrant", False, raising=False)
    monkeypatch.setattr(gpu, "_PATCHED", [], raising=False)
    return gpu, fake


def test_the_seam_is_inert_when_spaces_is_not_installed(monkeypatch):
    """Off-platform the app must run exactly the code the desktop runs."""
    from hf_space import gpu
    monkeypatch.setattr(gpu, "_spaces", lambda: None)
    assert gpu.install() == []


def test_install_patches_every_gpu_entry_point(seam):
    gpu, _ = seam
    patched = gpu.install()
    assert set(patched) == {
        "oasis.quant.segment.segment_image",
        "oasis.spatial.sparse_matcher.correspondences",
        "oasis.spatial.loftr_matcher.loftr_correspondences",
        "oasis.spatial.loftr_matcher.certify_local_roi",
        "oasis.spatial.loftr_matcher.loftr_fle",
    }


def test_install_never_touches_cuda(seam, monkeypatch):
    """Measured on the live Space, 2026-08-04: a CUDA call in THIS process — even one that
    fails and is caught — leaves CUDA partially initialised, and ZeroGPU forks its GPU
    workers from here. Every worker then dies with "No CUDA GPUs are available", for the
    life of the Space, and the only clue at startup is a line saying an optimisation was
    skipped. Preloading the InstanSeg weights onto cuda did exactly this.

    So: setting up the seam must not initialise CUDA, load a model, or move a tensor.
    """
    import torch

    gpu, _ = seam
    touched = []
    monkeypatch.setattr(torch.cuda, "init", lambda *a, **k: touched.append("cuda.init"))
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: touched.append("jit.load"))

    gpu.install()
    assert touched == []


def test_a_wrapped_call_acquires_a_gpu(seam):
    gpu, fake = seam
    wrapped = gpu._wrap(lambda x: x * 2, duration=lambda x: 60)
    assert wrapped(21) == 42
    assert len(fake.acquisitions) == 1


def test_a_nested_wrapped_call_does_not_acquire_a_second_gpu(seam):
    """certify_local_roi -> correspondences -> loftr_correspondences, and certify_local_roi
    -> loftr_fle -> loftr_correspondences. Every one of those is a wrapped entry point.

    Without the re-entrancy guard the inner calls would each try to acquire a GPU from
    inside the process that already holds one — and a single ROI certification, which loops
    the matcher many times, would pay one acquisition per pass. The visitor's daily quota is
    minutes; that is the difference between a demo that works and one that is rate-limited
    on its first click.
    """
    gpu, fake = seam
    calls = []

    def loftr_correspondences():
        calls.append("inner")
        return "inner"

    inner = gpu._wrap(loftr_correspondences, duration=lambda: 60)

    def certify_local_roi():
        calls.append("outer")
        return inner()

    outer = gpu._wrap(certify_local_roi, duration=lambda: 120)

    assert outer() == "inner"
    assert calls == ["outer", "inner"]      # the inner function really did run
    assert len(fake.acquisitions) == 1      # ...on the GPU the outer call already held
    assert fake.acquisitions[0]["duration"] == 120


def test_the_reentrancy_flag_is_cleared_even_when_the_call_raises(seam):
    """A leaked flag would silently disable the seam for the rest of the process."""
    gpu, fake = seam

    def boom():
        raise RuntimeError("segmentation failed")

    wrapped = gpu._wrap(boom, duration=lambda: 60)
    with pytest.raises(RuntimeError):
        wrapped()
    assert gpu._reentrant is False
    # A later call still acquires.
    gpu._wrap(lambda: None, duration=lambda: 60)()
    assert len(fake.acquisitions) == 2


def test_a_progress_callback_is_dropped_rather_than_crossing_the_process_boundary(seam):
    """@spaces.GPU pickles its arguments; a callback cannot be pickled. The pipeline passes
    `progress=` for per-tile log decoration, so dropping it costs log detail and nothing
    else — but passing it would raise PicklingError mid-run."""
    gpu, fake = seam
    seen = {}

    def fn(a, progress=None):
        seen["progress"] = progress
        return a

    wrapped = gpu._wrap(fn, duration=lambda *a, **k: 60)
    wrapped(1, progress=lambda *_: None)
    assert seen["progress"] is None
    assert len(fake.acquisitions) == 1


def test_an_unpicklable_argument_falls_back_to_running_here(seam):
    """Anything else callable in the kwargs is a call we do not understand well enough to
    alter, so it runs in-process rather than being silently changed or blowing up."""
    gpu, fake = seam
    wrapped = gpu._wrap(lambda a, rng=None: "ran", duration=lambda *a, **k: 60)
    assert wrapped(1, rng=lambda: 0) == "ran"
    assert fake.acquisitions == []


# ── falling back to the CPU when the quota runs out ──────────────────────────

def test_quota_exhaustion_falls_back_to_the_cpu(seam):
    """The ZeroGPU allowance is a couple of minutes a day and one real slide can spend it.
    Before this, the run died and the operator got an empty review screen with no reason."""
    gpu, fake = seam
    calls = []

    def segment_image(path, model_dir, px, device="cpu"):
        calls.append(device)
        return "measured"

    # The scheduler refuses: no seconds left today.
    def refuse(*a, **k):
        raise RuntimeError("You have exceeded your ZeroGPU runs limit.")
    fake.GPU = lambda **kw: (lambda fn: refuse)
    wrapped = gpu._wrap(segment_image, duration=lambda *a, **k: 60)

    assert wrapped("a.tif", "/models", 0.5, device="cuda") == "measured"
    assert calls == ["cpu"], "the fallback must force the CPU, not re-ask for cuda"


def test_a_real_failure_is_not_silently_retried_on_the_cpu(seam):
    """Only ZeroGPU's "not now" refusals fall back. A genuine segmentation bug must
    surface, not be quietly re-run and reported as a normal (slow) success."""
    gpu, fake = seam

    def boom(*a, **k):
        raise ValueError("model file is corrupt")

    fake.GPU = lambda **kw: (lambda fn: boom)
    wrapped = gpu._wrap(boom, duration=lambda *a, **k: 60)
    with pytest.raises(ValueError, match="corrupt"):
        wrapped()


def test_the_cpu_fallback_never_initialises_cuda(monkeypatch):
    """THE CONSTRAINT THAT MAKES THIS DELICATE. The fallback runs in the main process, and
    ZeroGPU forks its GPU workers from here — a CUDA context created now is inherited by
    every future worker and breaks them permanently. So the fallback must close every route
    to CUDA, including the one the matchers take by asking torch themselves."""
    import torch
    from hf_space import gpu

    touched = []
    monkeypatch.setattr(torch.cuda, "init", lambda *a, **k: touched.append("cuda.init"))

    seen = {}

    def matcher_style(ref, mov):
        # what loftr/sparse do: ask torch which device to use
        seen["available"] = torch.cuda.is_available()
        return "matched"

    assert gpu._run_on_cpu(matcher_style, ("a", "b"), {}) == "matched"
    assert seen["available"] is False, "cuda must look absent to the callee"
    assert touched == []


def test_the_cpu_fallback_restores_the_device_afterwards(monkeypatch):
    """Pinning the matcher to the CPU permanently would mean one quota blip costs every
    later run its GPU, for the life of the Space."""
    import torch
    from hf_space import gpu
    from oasis.spatial import loftr_matcher

    real = torch.cuda.is_available
    gpu._run_on_cpu(lambda: None, (), {})
    assert torch.cuda.is_available is real
    assert loftr_matcher._DEVICE is None, "must be re-resolved, not left pinned to cpu"


# ── durations ────────────────────────────────────────────────────────────────

def test_duration_grows_with_the_image_and_stays_inside_the_bounds(monkeypatch):
    """Too short and the scheduler kills the run mid-image; too long and the visitor's
    quota drains and the Space queues worse."""
    from hf_space import gpu
    from oasis.quant import segment

    monkeypatch.setattr(segment, "image_dimensions", lambda p: (256, 256))
    small = gpu._segment_duration("small.png")
    monkeypatch.setattr(segment, "image_dimensions", lambda p: (8000, 8000))
    large = gpu._segment_duration("large.svs")

    assert small == gpu.MIN_DURATION
    assert large == gpu.MAX_DURATION
    assert small < large


def test_duration_is_the_ceiling_when_the_image_cannot_be_measured(monkeypatch):
    """An unreadable header must not produce a short slice that kills a long run."""
    from hf_space import gpu
    from oasis.quant import segment

    def explode(_):
        raise OSError("no such file")

    monkeypatch.setattr(segment, "image_dimensions", explode)
    assert gpu._segment_duration("missing.svs") == gpu.MAX_DURATION


# ── whose ZeroGPU quota the run is charged to ────────────────────────────────

def test_the_request_is_re_installed_on_the_job_thread(monkeypatch):
    """ZeroGPU rations GPU time per visitor, identified by a header `spaces` reads from a
    ContextVar at the moment a decorated function is called.

    OASIS runs every job on a thread created inside the request handler, and a ContextVar
    set while handling a request is not visible there. Left alone, every segmentation would
    be billed anonymously to one pool shared by the whole Space — exhausted after a couple
    of runs a day — instead of to the visitor who asked for it.
    """
    pytest.importorskip("gradio")
    from gradio.context import LocalContext
    from hf_space import gpu

    sentinel = object()
    monkeypatch.setattr(gpu, "_request", sentinel, raising=False)
    LocalContext.request.set(None)

    seen = {}
    import threading

    def on_thread():
        gpu.install_request()
        seen["request"] = LocalContext.request.get(None)

    t = threading.Thread(target=on_thread)
    t.start()
    t.join(5)

    assert seen["request"] is sentinel


def test_remembering_a_request_never_overwrites_with_nothing(monkeypatch):
    """A call that could not be wrapped must not erase a good token."""
    from hf_space import gpu
    monkeypatch.setattr(gpu, "_request", "good", raising=False)
    gpu.remember_request(None)
    assert gpu._request == "good"


def test_the_job_thread_hook_runs_before_the_work(monkeypatch):
    """inproc calls the hook; app.py points it at gpu.install_request."""
    from hf_space import inproc

    order = []
    monkeypatch.setattr(inproc, "ON_JOB_THREAD", lambda: order.append("hook"))
    monkeypatch.setattr(inproc, "dispatch", lambda m, a: order.append("work"))

    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    proc.wait(5)
    inproc.remove_stdout_router()
    assert order == ["hook", "work"]


def test_a_failing_hook_does_not_fail_the_run(monkeypatch):
    """Quota attribution is never worth losing an analysis over."""
    from hf_space import inproc

    def explode():
        raise RuntimeError("no gradio context")

    monkeypatch.setattr(inproc, "ON_JOB_THREAD", explode)
    monkeypatch.setattr(inproc, "dispatch", lambda m, a: None)
    proc = inproc.Popen(["python", "-m", "run_pipeline"], stdout=inproc.PIPE)
    proc.wait(5)
    inproc.remove_stdout_router()
    assert proc.returncode == 0


# ── where the Space keeps its files ──────────────────────────────────────────

def test_on_linux_the_config_directory_follows_xdg_config_home(monkeypatch, tmp_path):
    """This is what puts setup.yaml, calibration profiles and saved classifiers on the
    mounted bucket instead of the container's ephemeral disk.

    hf_space/app.py sets XDG_CONFIG_HOME before importing api.py, and relies on
    `_platform_config_dir` honouring it. That branch is Linux-only, so it never executes on
    a macOS or Windows checkout — this test is the only place it is exercised outside the
    Space itself.
    """
    from oasis.common import paths

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(paths.os, "name", "posix")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "bucket" / "config"))

    assert paths._platform_config_dir() == tmp_path / "bucket" / "config" / paths.APP_NAME


def test_an_existing_legacy_directory_still_wins(monkeypatch, tmp_path):
    """`user_config_dir` prefers ~/.ihc_analyzer when it exists, so an upgrade never looks
    like it lost someone's calibration profiles. A Space container has no such directory,
    but pinning it here means the Space's storage story cannot be broken by someone
    changing that precedence for the desktop app."""
    from oasis.common import paths

    legacy = tmp_path / ".ihc_analyzer"
    legacy.mkdir()
    monkeypatch.setattr(paths, "LEGACY_CONFIG_DIR", legacy)
    assert paths.user_config_dir() == legacy


def test_the_data_root_prefers_an_explicit_setting(monkeypatch, tmp_path):
    from hf_space import session
    monkeypatch.setenv("OASIS_SPACE_DATA", str(tmp_path / "mounted"))
    assert session.data_root() == tmp_path / "mounted"


def test_sessions_do_not_share_directories_or_an_event_bus(tmp_path, monkeypatch):
    """Two visitors on a public Space must not see each other's logs or overwrite each
    other's results — `API` keeps per-run state on the instance and the push channel is a
    single buffer, both of which are correct for one desktop user and wrong for a Space."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hf_space import session

    store = session.SessionStore(tmp_path / "data")
    a, sid_a = store.get(None)
    b, sid_b = store.get(None)

    assert sid_a != sid_b
    assert a.output != b.output
    assert a.uploads != b.uploads
    assert a.api is not b.api

    a.bus.push("window.onPipelineEvent({})")
    assert b.bus.since(0, timeout=0.01)[0] == []

    # And a returning visitor gets their own session back.
    again, sid_again = store.get(sid_a)
    assert again is a and sid_again == sid_a


class _FakeRequest:
    def __init__(self, headers=None, cookies=None):
        self.headers = headers or {}
        self.cookies = cookies or {}


def test_the_session_comes_from_a_header_not_a_cookie(tmp_path):
    """MEASURED ON THE LIVE SPACE, 2026-08-04. huggingface.co shows a Space in an iframe
    served from <space>.hf.space, so a cookie this app sets is THIRD-PARTY — Safari discards
    it outright. Every request then minted a new session: the upload landed in one, the run
    happened in a second, and the event poll listened on a third. The pipeline completed and
    wrote its results, while the page sat on "Starting pipeline… 0%" forever.

    It reproduces only inside the iframe — opening the .hf.space URL directly makes the
    cookie first-party and everything works, which is exactly how it got shipped.
    """
    from hf_space import session

    good = "a1b2c3d4e5f60718"
    assert session.session_id_from(_FakeRequest(headers={session.HEADER: good})) == good

    # The header wins: a stale cookie must not split a page's requests across two sessions.
    both = _FakeRequest(headers={session.HEADER: good},
                        cookies={session.COOKIE: "fromcookie"})
    assert session.session_id_from(both) == good

    # Cookie still works for someone opening the .hf.space URL directly.
    assert session.session_id_from(
        _FakeRequest(cookies={session.COOKIE: "cookieonly"})) == "cookieonly"

    assert session.session_id_from(_FakeRequest()) is None


def test_a_client_supplied_id_is_adopted_end_to_end(tmp_path, monkeypatch):
    """The half of the fix the first attempt missed.

    Reading the header is not enough: `SessionStore.get` used to mint a fresh id for any id
    it had not seen before, which is right when the SERVER generates ids and the cookie only
    echoes them back, and wrong now that the page generates its own. With that left in, the
    header was parsed correctly and then thrown away — every request still got its own
    session and the symptom was completely unchanged.

    Checking `session_id_from` alone cannot catch this; the assertion has to run through the
    store, which is how it shipped the first time.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hf_space import session

    store = session.SessionStore(tmp_path / "data")
    sid = "a1b2c3d4e5f60718"

    first, got = store.get(session.session_id_from(_FakeRequest(headers={session.HEADER: sid})))
    assert got == sid, "the client's id must be adopted, not replaced"

    # A later request carrying the same id must reach the SAME session — this is what makes
    # an upload, the run it feeds and the event poll agree on one working directory.
    second, got_again = store.get(
        session.session_id_from(_FakeRequest(headers={session.HEADER: sid})))
    assert second is first and got_again == sid
    assert first.uploads.name == sid


def test_a_too_short_session_id_is_refused(tmp_path, monkeypatch):
    """The id is a bearer token and the client now picks it, so a guessable one must not be
    honoured — a fresh random id is minted instead."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hf_space import session

    assert session.session_id_from(_FakeRequest(headers={session.HEADER: "1"})) is None
    store = session.SessionStore(tmp_path / "data")
    _, got = store.get(session.session_id_from(_FakeRequest(headers={session.HEADER: "1"})))
    assert got != "1" and len(got) >= session.MIN_SESSION_ID


def test_a_hostile_session_header_cannot_escape_the_data_root(tmp_path, monkeypatch):
    """The id names a directory under uploads/ and results/, and it now arrives in a header
    the client controls completely."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hf_space import session

    sid = session.session_id_from(
        _FakeRequest(headers={session.HEADER: "../../../../etc/passwd"}))
    store = session.SessionStore(tmp_path / "data")
    sess, resolved = store.get(sid)
    assert "/" not in resolved and ".." not in resolved
    assert str(sess.uploads.resolve()).startswith(str((tmp_path / "data").resolve()))


def test_uploads_never_escape_the_session_directory(tmp_path, monkeypatch):
    """The browser sends each file's path INSIDE the chosen folder as its filename, which
    is attacker-controlled. `new_upload_dir` names the destination from it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hf_space import session

    store = session.SessionStore(tmp_path / "data")
    sess, _ = store.get(None)
    hostile = session.new_upload_dir(sess, "../../../etc")
    assert str(hostile.resolve()).startswith(str(sess.uploads.resolve()))


def test_a_second_upload_of_the_same_name_gets_its_own_directory(tmp_path, monkeypatch):
    """Otherwise uploading a second folder silently adds its images to the batch the first
    one defined, and the run analyses both under the first one's name."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hf_space import session

    store = session.SessionStore(tmp_path / "data")
    sess, _ = store.get(None)
    first = session.new_upload_dir(sess, "slides")
    second = session.new_upload_dir(sess, "slides")
    assert first != second
