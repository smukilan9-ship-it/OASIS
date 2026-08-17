"""
app.py — OASIS as a Hugging Face Space, on ZeroGPU.

This serves THE SAME `oasis/webui/index.html` and THE SAME `oasis/webui/api.py` the desktop
app runs. Nothing under `oasis/` is modified, and `app.py` at the repo root — the desktop
entry point — is not involved. What this file does is supply, in a container, the four
things the desktop provides natively:

  a window          -> a browser, with `hf_space/shim.js` standing in for `window.pywebview`
  a file dialog     -> the shim uploads what the visitor picks and hands back a server path
  a place for files -> the mounted HF Storage Bucket (`hf_space/session.py`)
  a GPU             -> ZeroGPU, reached through `hf_space/gpu.py` and `hf_space/inproc.py`

WHY A GRADIO SERVER AND NOT serve.py. ZeroGPU is only available to Spaces on the Gradio
SDK; on any other SDK `@spaces.GPU` does nothing and the whole pipeline silently runs on
CPU. `gradio.Server` is the way out of the apparent dilemma — it IS a FastAPI application
with Gradio's engine attached, so the Space is a Gradio Space as far as the platform (and
the ZeroGPU scheduler) is concerned, while every route below is an ordinary FastAPI route
serving a hand-written frontend.

WHICH TABS APPEAR is not decided here. `oasis/common/edition.py` defaults to the v1
edition and `applyEdition` (index.html:3296) removes the Validation and Restained nav
buttons unless OASIS_RESEARCH is set. This file deliberately does not set it, so the Space
shows exactly what a lab's install shows: Quant, Spatial, Classifier, Settings.

Run locally:  python hf_space/app.py      then open http://127.0.0.1:7860
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WEBUI = REPO / "oasis" / "webui"
# The repo root only. `HERE` must NOT go on sys.path: this package's siblings would then be
# importable as bare top-level names, and `session`/`gpu` are common enough to shadow
# something. Everything here is reached as `hf_space.<module>` instead.
sys.path.insert(0, str(REPO))

from hf_space import gpu, inproc                                     # noqa: E402
from hf_space import session as session_mod                          # noqa: E402

# ── storage, BEFORE oasis.webui.api is imported ──────────────────────────────
# api.py resolves `CONFIG_DIR = user_config_dir()` at module level, and on Linux
# `oasis/common/paths.py:_platform_config_dir` reads XDG_CONFIG_HOME. Setting it here is
# what puts setup.yaml, calibration profiles and saved classifiers on the persistent
# bucket instead of the container's ephemeral disk — with no change to the app, and no
# override of HOME (which would also drag HF's model caches onto billed storage).
DATA_ROOT = session_mod.data_root()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CONFIG_HOME", str(DATA_ROOT / "config"))

# Progress must reach the UI as it happens, not in one burst at the end — the same reason
# oasis/common/worker.py:worker_env sets it for a real subprocess.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

inproc.install_stdout_router()

# ── run jobs in-process so the ZeroGPU seam can reach them ───────────────────
# api.py and its two siblings do `import subprocess` at module level and then call
# `subprocess.Popen(...)`, so rebinding the module attribute redirects every job without
# editing a line of any of them. See hf_space/inproc.py for why a real child process cannot
# work here.
import oasis.webui.api as _api_mod                                   # noqa: E402
import oasis.webui.calibration as _calib_mod                         # noqa: E402
import oasis.webui.restained_api as _restained_mod                   # noqa: E402

for _mod in (_api_mod, _calib_mod, _restained_mod):
    _mod.subprocess = inproc

# Carry the visitor's ZeroGPU quota token onto the job thread. See hf_space/gpu.py for why
# this is needed and what it costs to get wrong.
inproc.ON_JOB_THREAD = gpu.install_request

_PATCHED = gpu.install()

STORE = session_mod.SessionStore(DATA_ROOT)

# NO SAMPLE DATA IS SHIPPED. A demo image was seeded here and pre-filled into the Quant
# tab's path box so the Space had something to run on first click. In use it was a
# hindrance: the box arrives already populated with a file the visitor did not choose, so
# the first thing they must do is notice it and clear it, and a Browse that is meant to
# replace it instead competes with it. The app opens empty, exactly as the desktop build
# does, and Browse is the way in.
SHIM = "<script>\n" + (HERE / "shim.js").read_text(encoding="utf-8") + "\n</script>"


def _index_html():
    """index.html with the bridge injected, exactly as serve.py:146 does it."""
    html = (WEBUI / "index.html").read_text(encoding="utf-8")
    return html.replace("<head>", "<head>\n" + SHIM, 1)


def build_app():
    from fastapi import Request, UploadFile, File, Form
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

    app = _server()

    # ── the page ─────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        _, sid = STORE.get(session_mod.session_id_from(request))
        response = HTMLResponse(_index_html())
        # NO-STORE, and it matters more here than it looks. This body is assembled per
        # request (index.html + the injected shim), so it carries no ETag and no
        # Last-Modified — nothing a browser can revalidate against. Left alone, browsers
        # apply HEURISTIC caching and keep serving the copy they already have, which means
        # a deploy lands on the server and the visitor goes on using the previous UI. The
        # bug they reported as still broken is then fixed everywhere except on their screen.
        # The assets below are FileResponses and get validators for free; this one cannot.
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        # httponly: the page never needs to read it, only send it back.
        response.set_cookie(session_mod.COOKIE, sid, httponly=True, samesite="lax",
                            max_age=session_mod.SESSION_TTL_SECONDS)
        return response

    # ── static assets ────────────────────────────────────────────────────────
    # index.html references `restained_coexpression.js` and `help/*.png` relatively. These
    # are registered one route per real file rather than behind a `/{path:path}` catch-all,
    # because a catch-all at the root would shadow Gradio's OWN routes — including the ones
    # the ZeroGPU scheduler and queue depend on — and break the GPU rather than a stylesheet.
    for asset in sorted(WEBUI.rglob("*")):
        rel = asset.relative_to(WEBUI)
        if not asset.is_file() or asset.name == "index.html":
            continue
        # Source, caches and dotfiles are not assets. Without the dotfile rule the Space
        # would publish `.DS_Store` at its root, which is both noise and a small leak of
        # the maintainer's directory listing.
        if asset.suffix in (".py", ".pyc") or any(
                p.startswith(".") or p == "__pycache__" for p in rel.parts):
            continue
        route = "/" + str(rel).replace(os.sep, "/")

        def _serve(path=asset):
            return FileResponse(path)

        app.get(route)(_serve)

    # ── the API bridge ───────────────────────────────────────────────────────
    @app.post("/api/{name}")
    async def api(name: str, request: Request):
        sess, _ = STORE.get(session_mod.session_id_from(request))
        # Remembered here, on the request, because by the time the GPU work runs it is two
        # thread hops away and the header that rations the visitor's GPU time is gone.
        gpu.remember_request(gpu.as_gradio_request(request))
        # Same guard rails as serve.py:163 — private helpers and attributes are not
        # callable from the page.
        if name.startswith("_") or not callable(getattr(sess.api, name, None)):
            return JSONResponse({"__error": "no such method: " + name}, status_code=404)
        try:
            raw = await request.body()
            args = json.loads(raw or b"[]")
            if not isinstance(args, list):
                args = [args]
        except Exception as e:
            return JSONResponse({"__error": f"bad request body: {e}"})

        import anyio
        try:
            result = await anyio.to_thread.run_sync(
                lambda: getattr(sess.api, name)(*args))
            return JSONResponse(json.loads(json.dumps({"__result": result}, default=str)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"__error": str(e)})

    # ── the push channel ─────────────────────────────────────────────────────
    @app.get("/__events")
    async def events(request: Request, since: int = 0):
        sess, _ = STORE.get(session_mod.session_id_from(request))
        import anyio
        evts, n = await anyio.to_thread.run_sync(lambda: sess.bus.since(since))
        # A cached long-poll would replay one batch of progress forever and never advance,
        # which looks exactly like a hung run.
        return JSONResponse({"events": evts, "n": n},
                            headers={"Cache-Control": "no-store"})

    # ── Browse -> upload ─────────────────────────────────────────────────────
    @app.post("/__upload")
    async def upload(request: Request,
                     files: list[UploadFile] = File(...),
                     kind: str = Form("file"),
                     name: str = Form("upload")):
        sess, _ = STORE.get(session_mod.session_id_from(request))
        # A folder pick names its directory after the folder. A single file uses the file's
        # stem, so one image lands at `<uploads>/a/a.png` rather than `<uploads>/a.png/a.png`
        # — the path goes straight into a text box the visitor reads.
        target = session_mod.new_upload_dir(
            sess, name if kind == "folder" else Path(name).stem)
        first = None
        for item in files:
            # The client sends the path inside the chosen folder as the filename so a
            # nested layout survives. It is attacker-controlled, so it is rebuilt from
            # its parts with anything that could escape the directory dropped.
            parts = [p for p in (item.filename or "file").replace("\\", "/").split("/")
                     if p not in ("", ".", "..")]
            if not parts:
                continue
            dest = target.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                while chunk := await item.read(1024 * 1024):
                    out.write(chunk)
            first = first or dest
        if first is None:
            return JSONResponse({"__error": "nothing was uploaded"})
        # A folder pick answers with the directory; a single file answers with the file —
        # which is what the UI's two callers each put in their text box.
        return JSONResponse({"path": str(target if kind == "folder" else first)})

    @app.get("/__health")
    def health(request: Request):
        # `ip_token` answers the question that decides whether this Space is usable: ZeroGPU
        # rations GPU time per visitor, identified by an X-IP-Token header, and without it
        # every run — signed in or not — is charged to one small anonymous pool shared by
        # everyone on the same address. `seen` counts requests that carried one since boot,
        # so the answer does not depend on who is asking right now.
        return {
            "ok": True,
            "zerogpu": gpu.on_zerogpu(),
            "gpu_patched": _PATCHED,
            "data_root": str(DATA_ROOT),
            "ip_token_on_this_request": session_mod.TOKEN_HEADER in request.headers,
            "ip_token_requests_seen": gpu.tokened_requests(),
        }

    return app


def _server():
    """A `gradio.Server` when one is available, else a plain FastAPI app.

    The fallback is what makes `python hf_space/app.py` work in a checkout that has not
    installed gradio — the UI and the whole pipeline can then be exercised locally. It is
    NOT a substitute on the platform: without the Gradio SDK there is no ZeroGPU, so the
    difference is announced rather than left to be discovered from a slow run.
    """
    try:
        from gradio import Server
        return Server()
    except Exception as e:
        print(f"[hf_space] gradio.Server unavailable ({type(e).__name__}: {e}) — "
              "falling back to plain FastAPI. THERE IS NO ZEROGPU ON THIS PATH.",
              file=sys.stderr)
        from fastapi import FastAPI
        return FastAPI()


def main():
    app = build_app()
    port = int(os.environ.get("PORT", 7860))
    print(f"[hf_space] data root   : {DATA_ROOT}")
    print(f"[hf_space] ZeroGPU     : {gpu.on_zerogpu()}")
    print(f"[hf_space] GPU patched : {', '.join(_PATCHED) or 'nothing'}")
    if hasattr(app, "launch"):
        app.launch(server_name="0.0.0.0", server_port=port)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
