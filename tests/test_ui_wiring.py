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


def test_no_element_id_is_used_twice(ui):
    """Two elements sharing an id is a silent half-broken control.

    `getElementById` returns the first one, so the second renders, accepts clicks and is
    read by nothing. Introduced while restructuring the Spatial wizard: a lifted card
    already contained the cleanup toggle and a second copy was written into the new final
    step, leaving a switch on screen that no run ever consulted.
    """
    import collections

    counts = collections.Counter(re.findall(r'\bid="([^"]+)"', ui["markup"]))
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"duplicate element ids: {dupes}"


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


def test_the_manual_landmark_path_offers_no_suggestions(ui):
    """The manual path's entire claim is that OASIS does not guess the correspondence.

    "Auto-propose landmarks" and "Guide next" contradicted that from inside it, and once
    LoFTR-in-ROI became the primary automatic engine they were a second, weaker automatic
    path living in the manual one. They are retired to legacy/webui_guided_landmarks.js;
    this keeps them from creeping back into the shipped UI.

    (An earlier version of this test asserted the opposite — that these functions must have
    buttons — because a wiring audit reported "function with no control". The audit was
    right that something was inconsistent; the correct repair was to remove the functions.)
    """
    retired = ("spatialCertPropose", "spatialCertGuideRequestNext", "spatialCertAcceptGuide",
               "spatialCertRejectGuide", "spatialCertAcceptCandidate", "spatialCertGuideContinue")
    present = [fn for fn in retired if fn in ui["defined"]]
    assert not present, f"retired suggestion helpers are back in the UI: {present}"
    wired = [fn for fn in retired if re.search(rf'onclick="{fn}\(', ui["markup"])]
    assert not wired, f"retired suggestion helpers have controls again: {wired}"


def test_no_micro_sign_is_left_inside_an_uppercased_heading(ui):
    """`text-transform: uppercase` turns µ into Greek capital Mu, which looks like M.

    "Bandwidth (75 µm) validity" therefore rendered as "BANDWIDTH (75 MM)" — millimetres,
    a 1000x error in a science label, produced purely by CSS. Units inside an uppercased
    heading must be wrapped in `.unit`, which opts back out of the transform.
    """
    uppercased = re.findall(r"^\.([a-z-]+)\s*\{[^}]*text-transform:\s*uppercase",
                            ui["script"] + ui["markup"], re.M)
    caps = set(uppercased) | {"card-title", "section-label", "metric-label",
                              "parameter-name", "terminal-title"}
    offenders = []
    for cls in caps:
        for m in re.finditer(rf'<div class="[^"]*\b{re.escape(cls)}\b[^"]*"[^>]*>', ui["markup"]):
            tail = ui["markup"][m.end():m.end() + 600]
            block = tail.split("</div>")[0]
            stripped = re.sub(r'<span class="unit">.*?</span>', "", block, flags=re.S)
            if "µ" in re.sub(r"<[^>]+>", "", stripped):
                offenders.append((cls, " ".join(re.sub(r"<[^>]+>", "", block).split())[:70]))
    assert not offenders, f"micro sign inside an uppercased heading: {offenders}"


# What each wizard step is actually called, so copy that cites a step number can be checked
# against the markup rather than against memory.
WIZARDS = {
    "q":  ["Inputs & outputs", "Pixel size", "Segmentation", "Classification", "Save & run"],
    "sp": ["Inputs & outputs", "Pixel size", "Certify registration", "Segmentation",
           "Classification", "Save & run"],
}
# The instruction shapes that send an operator somewhere: "<topic> ... in/on step N".
# Deliberately narrow. An earlier, looser version keyed on the topic word appearing anywhere
# within 90 characters and flagged three correct sentences — "the null-model check on step 6"
# was caught because "certified analysis window" sat nearby. A step-reference test that cries
# wolf is worse than none, because the next person learns to skip it.
# "certify"/"certification" only, never "certified": the latter is an adjective on a noun
# phrase ("inside a certified analysis window"), not an instruction to go anywhere, and it
# put the correct sentence "the null-model check on step 6" on the offenders list.
STEP_TOPIC = {
    r"pixel size":                "Pixel size",
    r"certif(?:y|ication)\w*":    "Certify registration",
    r"(?:input |image )folder":   "Inputs & outputs",
}


def _messages(ui):
    """Each self-contained piece of user-facing copy, one per element.

    Comments are dropped: they legitimately discuss step numbers from the code's point of
    view ("Step 2: name the pixel-size rows after what step 1 holds") and instruct nobody.
    Only literals long enough to be prose are kept, so identifiers and CSS selectors that
    happen to contain a digit cannot look like a step reference.
    """
    script = re.sub(r"/\*.*?\*/", " ", ui["script"], flags=re.S)
    script = re.sub(r"(?m)^\s*//.*$", " ", script)
    script = re.sub(r"(?m)\s//\s.*$", " ", script)
    out = [re.sub(r"<[^>]+>", " ", block)
           for block in re.split(r"</(?:div|p|span|li|td|h\d)>", ui["markup"])]
    for quote in ("'", '"', "`"):
        out += [lit for lit in re.findall(rf"{quote}([^{quote}\\\n]{{25,400}}){quote}", script)]
    return [u for u in out if re.search(r"\bstep \d", u, re.I)]


def test_copy_that_sends_you_to_a_step_names_the_right_one(ui):
    """Telling the operator to go to the wrong step is worse than not telling them.

    Both wizards were renumbered when Spatial grew a Pixel size step and a Certify step, and
    the prose did not move with them. Two survived: a run-time warning about the default
    pixel size said "Verify it in step 1 (Inputs)" when pixel size is step 2, and the
    bandwidth pre-flight said "Complete landmark certification in step 2" when certification
    is step 3. Both are dead ends — the operator arrives at a step with no such control.
    """
    offenders = []
    # Scoped to one message at a time, not to a sliding character window. The pixel-size
    # warning put its topic two sentences before its pointer — "Pixel size is 0.5 µm/px …
    # Verify it in step 1 (Inputs)" — which no same-clause window can span, and its
    # parenthesised title was self-consistent (step 1 IS "Inputs & outputs"), so only the
    # topic gives it away. A message is the right unit: it is what the operator reads at once.
    for unit in _messages(ui):
        for m in re.finditer(r"\bstep (\d)\b", unit, re.I):
            n = int(m.group(1))
            for topic, want in STEP_TOPIC.items():
                if not re.search(topic, unit, re.I):
                    continue
                if not any(n <= len(s) and s[n - 1] == want for s in WIZARDS.values()):
                    offenders.append(f"'step {n}' should be the '{want}' step: "
                                     f"{' '.join(unit.split())[:120]}")
                break
    assert not offenders, "copy points at the wrong wizard step:\n  " + "\n  ".join(offenders)
