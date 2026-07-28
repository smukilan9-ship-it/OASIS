"""Every wire in the web UI has to land somewhere real.

Three failure modes, all of which have shipped in this UI before and none of which any
other test can see, because the frontend is one HTML file with no build step:

  1. an onclick/onchange naming a function that does not exist -- the control does nothing
     when clicked, silently
  2. JS reading an element id that is not in the markup -- `getElementById` returns null and
     the feature is dead; if the code does not guard, it throws at click time
  3. JS calling `window.pywebview.api.X` where the API class has no `X` -- throws when used

Found on introduction: `spatialCertPropose()` wrote to a button id that had been removed
from the markup, so a working backend path (`propose_landmarks`) both had no way in AND
would have thrown if reached; and the guided-landmark accept/reject buttons were missing,
leaving `spatialCertRejectGuide()` unreachable while a toast told the operator to use it.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "oasis/webui/index.html"
API = ROOT / "oasis/webui/api.py"

# Globals and language constructs that appear inside inline handlers but are not app code.
NOT_APP_CODE = {
    "if", "for", "while", "return", "typeof", "new", "await", "switch",
    "alert", "confirm", "prompt", "console", "window", "document", "location", "event",
    "parseInt", "parseFloat", "Number", "String", "Boolean", "Array", "Object", "JSON",
    "Math", "Date", "setTimeout", "setInterval", "encodeURIComponent", "toFixed",
}
# Created by JS at runtime rather than present in the markup.
BUILT_AT_RUNTIME = {"cls-use-btn"}


@pytest.fixture(scope="module")
def ui():
    html = INDEX.read_text()
    script = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S))
    markup = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
    defined = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", script))
    for kw in ("const", "let", "var"):
        defined |= set(re.findall(rf"\b{kw}\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", script))
        defined |= set(re.findall(rf"\b{kw}\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?function", script))
    return {"markup": markup, "script": script, "defined": defined,
            "ids": set(re.findall(r'\bid="([^"]+)"', markup))}


def test_every_inline_handler_calls_a_function_that_exists(ui):
    missing = {}
    handlers = re.findall(r'\bon(?:click|change|input|submit|keydown)="([^"]*)"', ui["markup"])
    assert handlers, "no inline handlers found — the regex has drifted from the markup"
    for body in handlers:
        for call in re.findall(r"([A-Za-z_$][\w$.]*)\s*\(", body):
            base = call.split(".")[0]
            if "." in call or base in NOT_APP_CODE or base in ui["defined"]:
                continue
            missing.setdefault(base, body.strip()[:80])
    assert not missing, f"handlers call undefined functions: {missing}"


def test_every_element_id_read_by_js_exists_in_the_markup(ui):
    read = set(re.findall(r"""getElementById\(\s*['"]([^'"]+)['"]\s*\)""", ui["script"]))
    read |= set(re.findall(r"""querySelector\(\s*['"]#([A-Za-z0-9_-]+)['"]""", ui["script"]))
    missing = sorted(read - ui["ids"] - BUILT_AT_RUNTIME)
    assert not missing, f"JS reads element ids that are not in the markup: {missing}"


def test_every_backend_call_names_a_real_api_method(ui):
    tree = ast.parse(API.read_text())
    methods = {f.name
               for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    called = set(re.findall(r"pywebview\.api\.([A-Za-z_]\w*)", ui["script"]))
    missing = sorted(called - methods)
    assert not missing, f"UI calls API methods that do not exist: {missing}"


def test_the_page_script_parses(ui):
    """The frontend is one HTML file with no build step, so nothing else catches a syntax
    error until the app is launched and the whole UI is blank. Cheap gate; skipped where
    node is unavailable rather than silently passing."""
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(ui["script"])
        path = fh.name
    r = subprocess.run([node, "--check", path], capture_output=True, text=True)
    assert r.returncode == 0, f"page script does not parse:\n{r.stderr[:2000]}"


def test_functions_reachable_from_the_ui_are_actually_reachable(ui):
    """A handler-less function that the UI's own copy tells the user to use is a dead end.

    Narrow on purpose: only the certification helpers whose absence produced a hard
    dead-end state (a toast instructing the operator to press a control that did not
    exist). Not a general dead-code check -- plenty of functions are called from JS only.
    """
    for fn in ("spatialCertAcceptGuide", "spatialCertRejectGuide", "spatialCertPropose"):
        assert fn in ui["defined"], f"{fn} is gone; remove its button too"
        assert re.search(rf'onclick="{fn}\(', ui["markup"]), \
            f"{fn} has no control that invokes it"
