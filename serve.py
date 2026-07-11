"""
serve.py — HTTP shell for OASIS.

The desktop app (app.py) runs the same webui inside a pywebview window and binds
webui.api.API as `js_api`, so JS calls `window.pywebview.api.<method>(...)`. That window
is NOT an HTTP server, so it cannot be driven by a browser (or by an automated agent).

This module serves the SAME frontend + the SAME validated API over plain HTTP, so the
identical UI runs in a real browser:

  • POST /api/<method>   body = JSON array of positional args  ->  {"__result": ...}
                                                        (or {"__error": "..."} on failure)
  • GET  /__events?since=N   long-poll bridge for the backend's push channel. The API pushes
                             UI updates via self._window.evaluate_js(js) (onPipelineEvent /
                             onValidationEvent). A stand-in window buffers those JS strings;
                             the injected shim polls this endpoint and eval()s them in the
                             page, reproducing the desktop push semantics.
  • GET  /  (and static)     serves webui/index.html with a small shim injected that defines
                             window.pywebview.api over fetch. Any other path serves a file
                             under webui/.

Native file dialogs (pick_folder / pick_file) don't exist in a browser; the stand-in window
returns None and the UI falls back to typed paths (which an agent CAN drive).

pywebview desktop (app.py) is untouched — this is an additive second entry point.

Run:  .venv/bin/python serve.py    then open  http://127.0.0.1:8765
"""
import json
import mimetypes
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).parent
WEBUI = ROOT / "oasis" / "webui"
sys.path.insert(0, str(ROOT))
from oasis.webui.api import API                                            # noqa: E402

HOST = "127.0.0.1"
PORT = 8765


class _EventBus:
    """Buffers the JS strings the backend pushes through evaluate_js, and hands them to the
    browser via long-poll. Single-user local tool: one append-only list + a condition var."""
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


BUS = _EventBus()


class _BrowserWindow:
    """Stand-in for the pywebview window. evaluate_js pushes go to the event bus; native
    dialogs return None so path-based UI still works in a browser."""
    def evaluate_js(self, js):
        BUS.push(js)

    def create_file_dialog(self, *a, **k):
        return None


api = API()
api.set_window(_BrowserWindow())


SHIM = """<script>
(function () {
  window.pywebview = window.pywebview || {};
  window.pywebview.api = new Proxy({}, {
    get: function (_t, name) {
      return function () {
        var args = Array.prototype.slice.call(arguments);
        return fetch('/api/' + name, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(args)
        }).then(function (r) { return r.json(); }).then(function (res) {
          if (res && res.__error) { throw new Error(res.__error); }
          return res ? res.__result : undefined;
        });
      };
    }
  });
  // Bridge the backend push channel (onPipelineEvent / onValidationEvent).
  var _n = 0;
  function poll() {
    fetch('/__events?since=' + _n).then(function (r) { return r.json(); }).then(function (d) {
      _n = d.n;
      (d.events || []).forEach(function (js) {
        try { (0, eval)(js); } catch (e) { console.error('event bridge', e); }
      });
      setTimeout(poll, 10);
    }).catch(function () { setTimeout(poll, 1000); });
  }
  poll();
  document.addEventListener('DOMContentLoaded', function () {
    try { window.dispatchEvent(new Event('pywebviewready')); } catch (e) {}
  });
})();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/__events":
            q = parse_qs(urlparse(self.path).query)
            since = int((q.get("since") or ["0"])[0])
            events, n = BUS.since(since)
            self._send(200, json.dumps({"events": events, "n": n}))
            return
        if path in ("/", "/index.html"):
            html = (WEBUI / "index.html").read_text(encoding="utf-8")
            html = html.replace("<head>", "<head>\n" + SHIM, 1)
            self._send(200, html, "text/html; charset=utf-8")
            return
        rel = path.lstrip("/")
        f = (WEBUI / rel).resolve()
        if str(f).startswith(str(WEBUI.resolve())) and f.is_file():
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype)
            return
        self._send(404, json.dumps({"__error": "not found: " + path}))

    def do_POST(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            self._send(404, json.dumps({"__error": "not found"}))
            return
        name = path[len("/api/"):]
        if name.startswith("_") or not hasattr(api, name) or not callable(getattr(api, name)):
            self._send(404, json.dumps({"__error": "no such method: " + name}))
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"[]"
        try:
            args = json.loads(raw or b"[]")
            if not isinstance(args, list):
                args = [args]
            result = getattr(api, name)(*args)
            self._send(200, json.dumps({"__result": result}, default=str))
        except Exception as e:
            traceback.print_exc()
            self._send(200, json.dumps({"__error": str(e)}))


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"OASIS serving at http://{HOST}:{PORT}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
