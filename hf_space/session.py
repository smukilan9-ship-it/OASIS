"""
session.py — one OASIS backend per visitor, and where their files live.

`serve.py` builds a single `API()` for the whole process. That is right for the tool it
serves — a desktop app has one user — and wrong for a public Space. `API` keeps per-run
state on the instance (`self._process`, `self._review_cfg`), and the push channel is a
single buffer: with one shared instance, a second visitor's Stop button kills the first
visitor's run, and both watch the same interleaved log. So each browser session gets its
own `API`, its own event bus, and its own directories.

STORAGE. A Space's own disk is ephemeral — it is wiped on restart, on rebuild, and when
the Space sleeps. Anything a visitor should get back later lives under the mounted HF
Storage Bucket instead (`OASIS_SPACE_DATA`, `/data` by default):

    <data>/config/OASIS/       setup.yaml, calibration profiles, saved classifiers
    <data>/uploads/<sid>/      what the visitor uploaded through Browse
    <data>/results/<sid>/      that session's outputs

`config/` is reached by pointing XDG_CONFIG_HOME at it before `oasis.webui.api` is
imported (see app.py) — `oasis/common/paths.py:user_config_dir` already reads that
variable on Linux, so nothing in the app has to know about buckets. The other two are
per-session and handed to the UI through `get_setup`, which is the same channel the
desktop app uses to tell the UI where results go by default.
"""
import os
import threading
import time
import uuid
from pathlib import Path

# Sessions are cheap (an API object and a list of log lines) but not free, and a public
# Space accumulates them. Anything untouched for this long is dropped; its files stay on
# the bucket, so a returning visitor loses their log, not their results.
SESSION_TTL_SECONDS = 6 * 60 * 60

COOKIE = "oasis_sid"

# The shim sends the session id in this header on every request, and it is the PRIMARY
# identifier — see the long note in hf_space/shim.js. In short: on huggingface.co the Space
# runs inside an iframe from another origin, so a cookie this app sets is third-party and
# Safari discards it. Every request then gets a brand-new session, and the page ends up
# polling for events on a session that is not the one running the job — a finished run that
# displays as "Starting pipeline… 0%" forever. The cookie is kept as a fallback for people
# who open the .hf.space URL directly.
HEADER = "x-oasis-session"

# ZeroGPU's per-visitor quota token, attached by the Hub. Named here so both the session
# layer and the health endpoint agree on the spelling.
TOKEN_HEADER = "x-ip-token"


# A session id is a bearer token: whoever presents it gets that session's uploads and
# results. That was true of the cookie too. What changes with a header is that the CLIENT
# picks the value, so a short or guessable one has to be refused — the shim generates 16 hex
# characters from crypto.randomUUID (64 bits, the same as the server's uuid4().hex[:16]),
# and anything materially weaker is treated as absent so a fresh id is minted instead.
MIN_SESSION_ID = 12


def session_id_from(request):
    """The session id this request belongs to: header first, then cookie, else None."""
    sid = request.headers.get(HEADER)
    if sid:
        # Client-controlled, and it names a directory — so it is reduced to an opaque token
        # before it can go anywhere near a path.
        sid = "".join(c for c in sid if c.isalnum() or c in "-_")[:64]
        if len(sid) >= MIN_SESSION_ID:
            return sid
    return request.cookies.get(COOKIE)


def data_root() -> Path:
    """Where persistent state goes: the mounted bucket, or a local dir off-platform."""
    configured = os.environ.get("OASIS_SPACE_DATA")
    if configured:
        return Path(configured)
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        return Path("/data")
    # Running from a checkout (no bucket): keep everything beside the repo rather than
    # scattering it through the developer's home directory.
    return Path(__file__).resolve().parent / ".local-data"


class EventBus:
    """Buffers the JS the backend pushes through `evaluate_js`, for the long-poll bridge.

    Lifted from serve.py:47 — same append-only list and condition variable — but one per
    session instead of one per process.
    """

    def __init__(self):
        self._events = []
        self._cv = threading.Condition()

    def push(self, js):
        with self._cv:
            self._events.append(js)
            self._cv.notify_all()

    def since(self, n, timeout=25.0):
        with self._cv:
            if len(self._events) <= n:
                self._cv.wait(timeout)
            return self._events[n:], len(self._events)


class BrowserWindow:
    """Stand-in for the pywebview window (serve.py:69), per session.

    `create_file_dialog` still returns None: the native dialog does not exist in a browser.
    Browse is not broken by that — the injected shim overrides `pick_folder`/`pick_file`
    on the frontend and uploads instead, so the call never reaches here.
    """

    def __init__(self, bus):
        self._bus = bus

    def evaluate_js(self, js):
        self._bus.push(js)

    def create_file_dialog(self, *a, **k):
        return None


def _session_api_class():
    """Subclass `API` so the UI's defaults point into this session's storage.

    Deliberately a subclass rather than an edit to api.py: `get_setup` is the one call
    that tells the frontend where results go by default (`_default_output_dir`, read at
    index.html:3344) and what to show as home. Overriding it here moves those two answers
    onto the bucket for every visitor without the app knowing a bucket exists.

    Imported lazily so app.py can set XDG_CONFIG_HOME before api.py's module-level
    `CONFIG_DIR = user_config_dir()` runs.
    """
    from oasis.webui.api import API

    class SessionAPI(API):
        def __init__(self, home_dir, output_dir):
            super().__init__()
            self._space_home = str(home_dir)
            self._space_output = str(output_dir)

        def get_setup(self):
            setup = super().get_setup()
            setup["_home"] = self._space_home
            setup["_default_output_dir"] = self._space_output
            return setup

        def get_home(self):
            return self._space_home

        def open_file(self, path):
            # `open_file`/`open_folder` shell out to the desktop's file manager
            # (api.py:3265 `subprocess.Popen(["open", path])`). On a headless container
            # that is meaningless at best; refusing plainly is better than a silent no-op
            # the UI reports as success.
            return {"ok": False, "msg": "Not available in the browser demo — "
                                        "use the download buttons instead"}

        open_folder = open_file

    return SessionAPI


class Session:
    def __init__(self, sid, root):
        self.sid = sid
        self.touched = time.time()
        self.uploads = root / "uploads" / sid
        self.output = root / "results" / sid
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.bus = EventBus()
        self.api = _session_api_class()(root, self.output)
        self.api.set_window(BrowserWindow(self.bus))


class SessionStore:
    def __init__(self, root=None):
        self.root = Path(root or data_root())
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(parents=True, exist_ok=True)
        self._sessions = {}
        self._lock = threading.Lock()

    def get(self, sid):
        """The session for `sid`, creating one if needed. Returns (session, sid).

        AN ID THE SERVER HAS NOT SEEN IS ADOPTED, NOT REPLACED. This used to mint a fresh id
        whenever `sid` was unknown, which was right while the server generated ids and the
        cookie merely echoed them back. It is wrong now that the page generates its own:
        replacing it means the client's id is never honoured, every request gets its own
        session, and the page polls for events belonging to a job it cannot see. Sanitising
        happens in `session_id_from`, so what arrives here is already a bare token.
        """
        with self._lock:
            self._evict_locked()
            if not sid:
                sid = uuid.uuid4().hex[:16]
            if sid not in self._sessions:
                self._sessions[sid] = Session(sid, self.root)
            session = self._sessions[sid]
            session.touched = time.time()
            return session, sid

    def _evict_locked(self):
        cutoff = time.time() - SESSION_TTL_SECONDS
        for sid in [s for s, v in self._sessions.items() if v.touched < cutoff]:
            self._sessions.pop(sid, None)


def new_upload_dir(session, name=None):
    """A fresh directory under this session's uploads, named after what was uploaded.

    Each Browse gets its own directory rather than a shared one, so uploading a second
    folder does not silently add its images to the batch the first one defined.
    """
    stem = "".join(c for c in (name or "upload") if c.isalnum() or c in "-_.") or "upload"
    target = session.uploads / stem
    n = 1
    while target.exists():
        n += 1
        target = session.uploads / f"{stem}-{n}"
    target.mkdir(parents=True, exist_ok=True)
    return target


