#!/usr/bin/env python
"""Regenerate the "Need help" pictures by driving the app, not the machine.

The first set was captured by moving the real mouse and cropping the screen to the window's
rectangle. Two problems with that, and the second one is why these exist:

  * it takes the machine over, so nobody can use it for the half hour it runs; and
  * a screen crop shows whatever is in that rectangle. Any window that happens to sit on
    top of OASIS lands in the picture, and the help card ships it.

This drives the UI through `evaluate_js` — the same calls the buttons make — and captures
with `screencapture -l <windowid>`, which asks the window server for that one window's
image. Both problems go away: the pointer is never touched, and nothing that overlaps the
window can appear in the shot.

    OASIS_DEMO_DIR=~/oasis_help_demo ./.venv/bin/python tools/capture_help.py
    ./.venv/bin/python tools/capture_help.py quant settings   # one group at a time

Runs against demo data listed in DEMO below (HyReCo CD8/CD45 brightfield). It is real IHC
and a real run every time: nothing here fakes a number onto a screen.

Expect it to look hung and not be. Segmentation, certification and the statistic all run in
`run_pipeline` SUBPROCESSES, so this process sits at 0% CPU polling while the real work
happens elsewhere. Judging progress by this process's CPU says "nothing is running" for the
entire length of every slow step. Read the log, or look for a python process under this one.

Run the groups one at a time. Two instances drive two windows fine, but they share
~/.ihc_analyzer/*.yaml and compete for the same cores, which turns a slow step into a
much slower one for no gain.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import webview
import Quartz

from oasis.webui.api import API
from oasis.common.paths import resource_dir

HELP = ROOT / "oasis" / "webui" / "help"

# Where the demo tissue lives. Not committed: these are rendered HyReCo fields, and the
# dataset's terms are the dataset's to set. Point OASIS_DEMO_DIR at a directory holding
# quant_cd8/, spatial_cd8/ and spatial_cd45/ and this runs anywhere; the default keeps it
# out of the repo and off any one machine's temp directory.
DEMO = Path(os.environ.get("OASIS_DEMO_DIR", "~/oasis_help_demo")).expanduser()
OUT = Path(os.environ.get("OASIS_DEMO_OUT", "~/oasis_help_runs")).expanduser()

QUANT_IN = DEMO / "quant_cd8"
SPATIAL_A = DEMO / "spatial_cd8"
SPATIAL_B = DEMO / "spatial_cd45"
PX = "0.7519"                     # HyReCo rendered fields, measured from their own bars
TARGET_W = 1400                   # the card renders at 880 CSS px; 1400 stays crisp

_window = None
_failures = []


# ── plumbing ──────────────────────────────────────────────────────────────────────────
def js(code):
    """Evaluate in the page. Returns the JS value (pywebview marshals via JSON)."""
    return _window.evaluate_js(code)


def install_recorder():
    """Record what the UI would have told a person, so a timeout can say why.

    Every failure in here is reported to the operator as a toast and to nobody else. Driven
    headless that message goes nowhere, and the run simply sits in a wait loop until it
    times out — which reports "timed out waiting for the classifier to fit" when the app
    knew, and said, exactly what was wrong seconds in.
    """
    js("""(function(){
      window.__cap = [];
      const t = window.showToast;
      window.showToast = function(m){ window.__cap.push('toast: ' + m); return t.apply(this, arguments); };
      window.addEventListener('error', e => window.__cap.push('error: ' + e.message));
      window.addEventListener('unhandledrejection',
        e => window.__cap.push('rejected: ' + ((e.reason && e.reason.message) || e.reason)));
    })()""")


def recorded():
    try:
        return [str(x) for x in (js("window.__cap || []") or [])]
    except Exception:
        return []


def wait_for(expr, timeout=900, poll=1.0, what=""):
    """Poll a JS boolean expression until it is true. Returns False on timeout."""
    deadline = time.time() + timeout
    seen = len(recorded())
    while time.time() < deadline:
        try:
            if js(f"(function(){{ try {{ return !!({expr}); }} catch(e) {{ return false; }} }})()"):
                return True
        except Exception:
            pass
        # Echo as it happens rather than only at the deadline. A wait that is going to fail
        # usually knows within seconds, and thirty minutes of silence afterwards helps no one.
        now = recorded()
        for line in now[seen:]:
            print(f"      · {line}", flush=True)
        seen = len(now)
        time.sleep(poll)
    said = recorded()[seen:]
    _failures.append(f"timed out after {timeout}s waiting for {what or expr}"
                     + (("; the app said — " + " | ".join(said[-4:])) if said else
                        "; the app said nothing"))
    return False


def window_id():
    """The window server's id for our own window, so the capture is of that window only."""
    pid = os.getpid()
    best = None
    for w in Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") != pid:
            continue
        b = w.get("kCGWindowBounds") or {}
        if b.get("Width", 0) < 600 or b.get("Height", 0) < 400:
            continue
        area = b["Width"] * b["Height"]
        if best is None or area > best[1]:
            best = (w["kCGWindowNumber"], area)
    return best[0] if best else None


def shot(name, settle=1.2):
    """Capture the OASIS window alone into help/<name>."""
    time.sleep(settle)                       # let a re-render and any image decode finish
    wid = window_id()
    if wid is None:
        _failures.append(f"{name}: could not find the OASIS window")
        return
    HELP.mkdir(parents=True, exist_ok=True)
    dst = HELP / name
    # -l <id>: that window's own image, so an overlapping window cannot appear in it.
    # -o: no drop shadow.  -x: no shutter sound.
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(dst)], check=True)
    if not dst.exists() or dst.stat().st_size < 4000:
        _failures.append(f"{name}: capture produced nothing usable")
        return
    # A retina grab is 2560 px wide and the help card renders it at 880. Twenty of those at
    # full size is most of the bundle's UI weight for detail no one can see, so halve it.
    from PIL import Image
    with Image.open(dst) as im:
        im = im.convert("RGB")
        if im.width > TARGET_W:
            im = im.resize((TARGET_W, round(im.height * TARGET_W / im.width)), Image.LANCZOS)
        im.save(dst, optimize=True)
    print(f"    {name:26s} {dst.stat().st_size / 1024:6.0f} KB")


def scroll_to(selector=None, top=None):
    """Position the active page's scroller, by element or by absolute offset."""
    if selector:
        js(f"""(function(){{
            const el = document.querySelector({json.dumps(selector)});
            if (el) el.scrollIntoView({{block:'center'}});
        }})()""")
    else:
        js(f"""(function(){{
            const s = document.querySelector('.page.active .scroll');
            if (s) s.scrollTop = {int(top or 0)};
        }})()""")
    time.sleep(0.5)


def scroll_to_text(text, selector=".card-title, .section-label"):
    """Position by a heading's words. Two fields on one row share a scroll position, so
    anchoring on an input silently produced two identical pictures."""
    ok = js(f"""(function(){{
        const t = {json.dumps(text)}.toLowerCase();
        const el = [...document.querySelectorAll('.page.active ' +
                    {json.dumps(selector)}.split(',').join(', .page.active '))]
                   .find(e => (e.textContent || '').toLowerCase().includes(t));
        if (!el) return false;
        el.scrollIntoView({{block:'start'}});
        const s = document.querySelector('.page.active .scroll');
        if (s) s.scrollTop = Math.max(0, s.scrollTop - 24);
        return true;
    }})()""")
    if not ok:
        _failures.append(f"could not find a heading containing {text!r} to scroll to")
    time.sleep(0.5)


def set_field(el_id, value):
    js(f"""(function(){{
        const e = document.getElementById({json.dumps(el_id)});
        if (!e) return false;
        e.value = {json.dumps(str(value))};
        e.dispatchEvent(new Event('input',  {{bubbles:true}}));
        e.dispatchEvent(new Event('change', {{bubbles:true}}));
        return true;
    }})()""")


def click(el_id):
    js(f"(document.getElementById({json.dumps(el_id)})||{{click(){{}}}}).click()")
    time.sleep(0.4)


# ── groups ────────────────────────────────────────────────────────────────────────────
def do_settings():
    print("  settings")
    js("showPage('settings')")
    # The tab loads its saved values asynchronously, so a value written too early is simply
    # overwritten by the load and the picture shows the stale one.
    wait_for("document.getElementById('s-px-x').value !== ''", timeout=30,
             what="settings to load")
    time.sleep(1.2)
    set_field("s-px-x", PX)
    if str(js("document.getElementById('s-px-x').value")) != PX:
        _failures.append("settings: pixel size did not take")
    # By fraction of the scroll height, not by heading. Settings is barely two viewports
    # tall, so scrolling the last three headings to the top all clamp at the bottom and
    # produce three byte-identical pictures — which is how the first set shipped a duplicate.
    for name, frac in (("set-3-scalebar.png", 0.00),
                       ("set-1-pixelsize.png", 0.34),
                       ("set-2-priority.png", 0.62),
                       ("set-4-engine.png", 1.00)):
        js(f"""(function(){{
            const s = document.querySelector('.page.active .scroll');
            if (s) s.scrollTop = (s.scrollHeight - s.clientHeight) * {frac};
        }})()""")
        time.sleep(0.6)
        shot(name)
    seen = {}
    for name in ("set-3-scalebar.png", "set-1-pixelsize.png",
                 "set-2-priority.png", "set-4-engine.png"):
        p = HELP / name
        if not p.exists():
            continue
        digest = p.read_bytes()
        for other, blob in seen.items():
            if blob == digest:
                _failures.append(f"settings: {name} is identical to {other} — "
                                 "two steps would show the same picture")
        seen[name] = digest


def do_quant():
    print("  quant")
    js("showPage('quant')")
    # The tab finishes setting itself up asynchronously (saved settings, folder state) and
    # that setup calls quantSetMode('single'). Switching to batch before it lands is undone
    # a moment later with nothing on screen to say so — the fields still read batch while
    # the run takes the single-image path and fails on an empty filename.
    time.sleep(3.0)
    # Marker BEFORE mode. Choosing the marker runs quantStainChanged(), which re-derives the
    # step's state and puts the mode back to single — so setting it after the mode toggle
    # silently undoes the toggle, and the run then fails on the single-image path with an
    # empty filename while every field on screen still says batch.
    set_field("q-stain", "CD8")        # required; the run refuses without it
    mode = ""
    for _ in range(10):                 # set, then confirm it survived the next tick
        js("quantSetMode('batch')")
        time.sleep(1.0)
        mode = js("typeof quantMode !== 'undefined' ? String(quantMode) : 'undefined'")
        if mode == "batch":
            break
    print(f"      mode after switch: {mode!r}", flush=True)
    if mode != "batch":
        _failures.append(f"quant: batch mode did not stick (quantMode={mode!r})")
        return
    set_field("q-input", str(QUANT_IN))
    set_field("q-output", str(OUT / "quant"))
    # The folder scan is async. A fixed sleep photographs an empty count on a slow disk.
    wait_for("document.getElementById('q-image-count').textContent.trim().length > 0",
             timeout=90, what="the input folder to be scanned")
    js("quantWizardGoTo(1)")
    shot("quant-1-inputs.png")

    js("quantWizardGoTo(2)")
    set_field("q-px-x", PX)
    time.sleep(1.0)
    shot("quant-2-pixelsize.png")

    js("quantWizardGoTo(4)")
    time.sleep(0.8)
    shot("quant-3-classification.png")

    js("quantWizardGoTo(5)")
    time.sleep(0.5)
    click("quant-run-btn")
    # Fail here, not thirty minutes later. runQuantAnalysis() returns early on several
    # guards, and if none of them toasts, the only symptom is a review screen that never
    # arrives — which the next wait reports as a timeout on the wrong thing entirely.
    time.sleep(6)
    state = js("""({
        running: typeof pipelineRunning !== 'undefined' ? pipelineRunning : 'undefined',
        mode: typeof quantMode !== 'undefined' ? quantMode : 'undefined',
        input: (document.getElementById('q-input')||{}).value || '',
        step: typeof quantWizardStep !== 'undefined' ? quantWizardStep : -1,
        label: (document.getElementById('quant-progress-label')||{}).textContent || '',
    })""")
    print(f"      after Run: {state}", flush=True)
    if not state or not state.get("running"):
        _failures.append(f"quant: the run did not start — page state {state}; "
                         f"said {recorded()[-3:]}")
        return
    # The progress bar plus the activity log filling up is the picture for "running".
    wait_for("document.getElementById('quant-progress-label').textContent.trim() "
             "&& !/^Initializing/.test(document.getElementById('quant-progress-label').textContent)",
             timeout=180, what="the run to start")
    time.sleep(12)
    shot("quant-4-running.png")

    if wait_for("document.getElementById('quant-review').classList.contains('active')",
                timeout=1800, what="the Quant review screen"):
        time.sleep(2.0)
        shot("quant-5-review.png")
        js("qrApply()")

    if wait_for("document.getElementById('quant-results').classList.contains('active')",
                timeout=1800, what="the Quant results screen"):
        time.sleep(2.5)
        scroll_to(top=0)
        shot("quant-6-results.png")


def do_spatial(stop_after_certify=False):
    print("  spatial")
    js("showPage('spatial')")
    time.sleep(0.8)

    # Batch mode has to SURVIVE A TICK before anything else happens, for the same reason
    # quantSetMode does. Init is `pywebviewready → await loadSetup() → spatialAssocSetMode
    # ('single')`, so the reset lands whenever that await resolves — after this script has
    # clicked Batch and, at 0.8 s in, usually after it has previewed the folders too. The
    # call is a no-op for a person (the mode is already 'single' when init runs and the
    # function returns early), but here it flips the mode back AND clears
    # `spatialCertifications`, so the certify loop then finds no pairs and every wait after
    # it sits out its timeout. Observed exactly that: 3 pairs previewed, mode 'single',
    # zero pairs at the gate.
    mode = ""
    for _ in range(10):
        click("spatial-mode-batch-btn")
        time.sleep(1.0)
        mode = js("typeof spatialMode !== 'undefined' ? String(spatialMode) : 'undefined'")
        if mode == "batch":
            break
    if mode != "batch":
        _failures.append(f"spatial: batch mode did not stick (spatialMode={mode!r})")
        return
    click("spatial-fmode-two-btn")
    set_field("spatial-folder-a", str(SPATIAL_A))
    set_field("spatial-folder-b", str(SPATIAL_B))
    set_field("spatial-output", str(OUT / "spatial"))
    click("spatial-preview-btn")
    wait_for("spatialPairs && spatialPairs.length === 3", timeout=120, what="3 matched pairs")
    time.sleep(1.0)
    js("spatialWizardGoTo(1)")
    scroll_to(selector="#spatial-preview-card")
    shot("spatial-1-inputs.png")

    js("spatialWizardGoTo(2)")
    set_field("spatial-session-px-val", PX)
    time.sleep(1.0)
    shot("spatial-2-pixelsize.png")

    # Certify all three. Each is a LoFTR pass; the wizard exposes both actions by name.
    js("spatialWizardGoTo(3)")
    time.sleep(1.0)
    for i in range(3):
        js(f"spatialCertGoToPair({i})")
        time.sleep(1.5)
        # Switching pair already loads it; call Load only if that did not happen.
        js("""(function(){
            const b = document.getElementById('spatial-cert-load');
            if (b && b.offsetParent !== null) spatialCertLoad();
        })()""")
        # Wait on the LOADED PAIR, not on the Certify button. That button is static markup
        # inside the panel, so getElementById finds it before anything is loaded — the wait
        # returned at once, Certify fired on an empty canvas, said "Load the pair first" to
        # nobody, and the next wait then sat out its full timeout three times over.
        if not wait_for("spatialCertWork && spatialCertWork.pair && spatialCertWork.ref",
                        timeout=420, what=f"pair {i + 1} to load"):
            break
        js("loftrAutoFind()")
        if not wait_for(f"spatialCertifications && "
                        f"Object.values(spatialCertifications).filter(c=>c&&c.is_certified).length >= {i + 1}",
                        timeout=600, what=f"pair {i + 1} to certify"):
            break
        print(f"      certified {i + 1}/3")
    time.sleep(1.5)
    shot("spatial-3-certify.png")
    # Step 4 is about moving BETWEEN pairs, so it has to show a different pair — the strip
    # lives in the same non-scrolling panel as step 3, so anchoring on it produced a second
    # copy of the step-3 picture, byte for byte.
    js("spatialCertGoToPair(0)")
    wait_for("spatialCertWork && spatialCertWork.pair && spatialCertWork.ref",
             timeout=420, what="pair 1 to come back up")
    time.sleep(2.0)
    shot("spatial-4-pairnav.png")
    if (HELP / "spatial-3-certify.png").read_bytes() == (HELP / "spatial-4-pairnav.png").read_bytes():
        _failures.append("spatial: steps 3 and 4 captured the same picture")

    if stop_after_certify:
        return

    js("spatialWizardGoTo(6)")
    time.sleep(0.8)

    # SAY WHAT THE GATE SEES BEFORE OPENING IT. spatialBandwidthCheck() returns early on
    # "no pairs" with a toast and NOTHING written to #spatial-bw-result, so the wait below
    # has nothing to fail on and sits out its full forty minutes in silence. Observed: the
    # certify phase finished, the pair list was empty by the time step 6 ran, and the run
    # stalled with no worker, no output and no message.
    gate = js("""(function(){
        const u = (typeof spatialUncertifiedPairs === 'function') ? spatialUncertifiedPairs() : null;
        return { mode: String(typeof spatialMode !== 'undefined' ? spatialMode : '?'),
                 previewed: (typeof spatialPairs !== 'undefined' && spatialPairs) ? spatialPairs.length : -1,
                 gate_pairs: u ? u.pairs.length : -1,
                 uncertified: u ? u.uncertified.length : -1 };
    })()""") or {}
    print(f"      null-check gate: {gate}", flush=True)
    if not gate.get("gate_pairs"):
        _failures.append(f"spatial: the null check has no pairs to run on ({gate}) — "
                         f"the pair list was lost between certification and step 6")
        return
    if gate.get("uncertified"):
        _failures.append(f"spatial: {gate['uncertified']} pair(s) uncertified at step 6")
        return

    js("spatialBandwidthCheck()")
    # Fail on the badge too. It reaches ERROR / CERT REQUIRED in seconds, and waiting for
    # prose that will never be written turns a clear failure into a silent forty minutes.
    if not wait_for("(document.getElementById('spatial-bw-result') && "
                    " document.getElementById('spatial-bw-result').innerHTML.length > 200) || "
                    "['ERROR','CERT REQUIRED'].includes("
                    " (document.getElementById('spatial-bw-badge')||{}).textContent)",
                    timeout=2400, what="the null-model check"):
        _failures.append("spatial: the null check never reported")
        return
    badge = js("(document.getElementById('spatial-bw-badge')||{}).textContent")
    if badge in ("ERROR", "CERT REQUIRED"):
        _failures.append(f"spatial: the null check came back {badge}")
        return
    print(f"      null check: {badge}", flush=True)
    time.sleep(1.5)

    js("spatialAssocRun()")
    if wait_for("document.getElementById('spatial-review').classList.contains('active')",
                timeout=2400, what="the Spatial review screen"):
        time.sleep(2.5)
        scroll_to(top=0)
        shot("spatial-5-review.png")
        js("spReviewApply()")

    # Wait for the CARDS, not for the view. The results view is switched on first and filled
    # afterwards from the results JSON, so anchoring on `.active` shot an empty window: the
    # picture that shipped was 12 KB of background with a title bar.
    if wait_for("document.getElementById('spatial-results').classList.contains('active') && "
                "document.getElementById('spatial-results-cards').innerHTML.length > 500",
                timeout=2400, what="the Spatial results to render"):
        time.sleep(2.0)
        scroll_to(top=0)
        shot("spatial-6-results.png")
    else:
        _failures.append("spatial: the results screen never rendered any cards")


def do_classifier():
    print("  classifier")
    js("showPage('classifier')")
    time.sleep(0.8)
    scroll_to(top=0)
    shot("cls-1-when.png")

    set_field("cls-marker", "CD8")
    set_field("cls-px", PX)
    for name in ("679_CD8_lm00.png", "679_CD8_lm01.png", "679_CD8_lm04.png"):
        js(f"clsAddImagePath({json.dumps(str(QUANT_IN / name))})")
        time.sleep(0.4)
    time.sleep(1.0)
    scroll_to(selector="#cls-req")
    shot("cls-2-cohort.png")

    js("clsSegment()")
    if not wait_for("calib && calib.images && calib.images.length === 3 && "
                    "calib.images.every(i => i.data)", timeout=1800,
                    what="the three slides to segment"):
        return

    # Label programmatically. The labeller stores one state per detected cell, so the same
    # array the mouse writes into can be written here — no synthetic cells, no synthetic
    # image, just the choice of WHICH real cells get a label.
    #
    # Seeded from each slide's own measured DAB with a deliberate gap: clearly stained cells
    # positive, clearly unstained negative, the ambiguous middle left unlabelled — which is
    # what a careful labeller does, and what the abstain gate expects. NOTE this makes the
    # held-out score a property of that rule, not of a human's judgement; it is here to
    # populate a screenshot reproducibly, not to stand as a validation result.
    #
    # PER_SLIDE is a count, not a fraction, and that matters: the scoring is leave-one-CELL-
    # out, so its cost grows with the number of labels rather than the size of the image.
    # Taking a percentage of a 6,000-cell field produced ~6,500 labels, and the fit ran for
    # half an hour without finishing. A real labelling session is tens of cells per slide.
    PER_SLIDE = 80
    from oasis.quant import reclassify as RC
    import numpy as np

    # The canvas cell carries only its outline and its index into the GeoJSON — the DAB
    # value is never sent to the page, because the labeller is meant to judge from the
    # picture. So read the values here, from the same file the fit will measure, and map
    # them back through each cell's own feature index. Cells are not 1:1 with features:
    # `_cells` skips any feature without a usable ring, so the position in `im.state` and
    # the index in the GeoJSON are different numbers, and using one for the other would
    # label whichever cell happened to sit at that offset.
    metas = js("""calib.images.map(im => ({
        geo: im.data.geojson,
        idx: im.data.cells.map(c => c.i),
    }))""")
    states, tp, tn = [], 0, 0
    for m in metas or []:
        idx = list(m["idx"])
        state = [0] * len(idx)
        try:
            values = RC.read_dab_values(m["geo"])
        except (OSError, ValueError) as e:
            _failures.append(f"classifier: could not read {m['geo']}: {e}")
            states.append(state)
            continue
        ranked = sorted(
            ((float(values[f]), pos) for pos, f in enumerate(idx)
             if f < len(values) and np.isfinite(values[f])),
            key=lambda t: -t[0])
        if len(ranked) >= 4 * PER_SLIDE:       # enough to leave a gap between the classes
            for _, pos in ranked[:PER_SLIDE]:
                state[pos] = 1; tp += 1
            for _, pos in ranked[-PER_SLIDE:]:
                state[pos] = 2; tn += 1
        states.append(state)
    js(f"""(function(){{
      const s = {json.dumps(states)};
      calib.images.forEach((im, i) => {{ if (s[i]) im.state = s[i]; }});
      calibBuild();          // repaint the canvas from the states just written
    }})()""")
    n = [tp, tn]
    print(f"      labelled {n[0]} positive / {n[1]} negative")
    time.sleep(2.0)
    shot("cls-3-label.png")

    # The held-out report is deliberately NOT captured. Labels seeded from the DAB ranking
    # are separable by the very features the rule is fitted on, so the screen comes out at
    # F1 1.00 / AUC 1.00 — true of this labelling and of nothing else. Shipping that inside
    # the app's own help reads as a claim about the software, so that step is text only.
    # See tests/test_help_images.py. The fit still runs here, because a fit that has stopped
    # working should fail this tool rather than be discovered by a user.
    js("clsFit()")
    if wait_for("cls && cls.fit && cls.fit.ok", timeout=420, what="the classifier to fit"):
        f1 = js("(cls.fit.holdout && cls.fit.holdout.f1) ?? cls.fit.f1 ?? null")
        print(f"      fit OK (held-out F1 {f1}); report screen intentionally not captured")


def do_spatial_certify():
    """Steps 1-4 only: everything up to the certified pairs, without spending the statistic.

    Re-shooting the certification pictures should not cost another null check and another
    Monte-Carlo run, which is most of the group's half hour and produces the two pictures
    this one does not touch.
    """
    do_spatial(stop_after_certify=True)


def do_spatial_results():
    """Step 6 alone, from a run that has already finished.

    The results screen is filled from the results JSON on disk, so re-shooting it does not
    need another certification and another Monte-Carlo run — the previous half hour already
    produced the numbers. Same code path the run itself uses: the completion event hands
    `spatialAssocShowResults` exactly what `get_spatial_association_results` returns.
    """
    print("  spatial results")
    js("showPage('spatial')")
    time.sleep(0.8)
    ok = js(f"""(async function(){{
        const r = await window.pywebview.api.get_spatial_association_results(
            {json.dumps(str(OUT / 'spatial'))});
        if (!r || !(r.results || []).length) return false;
        // Adopt the markers off the results themselves. In a real run the folder preview
        // does this; loading from disk skips the preview, so the page subtitle keeps its
        // MARKER_A/TIM-3 default and the picture then names markers this cohort never had.
        const p0 = r.results[0] || {{}};
        const set = (id, v) => {{ const e = document.getElementById(id); if (e && v) e.value = v; }};
        set('spatial-label-a', p0.stain_a); set('spatial-label-b', p0.stain_b);
        if (typeof spatialSyncPageSubtitle === 'function') spatialSyncPageSubtitle();
        spatialAssocShowResults(r);
        return (r.results || []).length;
    }})()""")
    if not wait_for("document.getElementById('spatial-results').classList.contains('active') && "
                    "document.getElementById('spatial-results-cards').innerHTML.length > 500",
                    timeout=120, what="the stored results to render"):
        _failures.append(f"spatial results: nothing rendered from {OUT / 'spatial'} "
                         f"(loader returned {ok!r}) — run the `spatial` group first")
        return
    time.sleep(2.0)
    scroll_to(top=0)
    shot("spatial-6-results.png")


GROUPS = {"settings": do_settings, "quant": do_quant,
          "spatial": do_spatial, "certify": do_spatial_certify,
          "results": do_spatial_results, "classifier": do_classifier}


def drive():
    wanted = [a for a in sys.argv[1:] if a in GROUPS] or list(GROUPS)
    if not wait_for("typeof showPage === 'function'", timeout=60, what="the page to load"):
        _window.destroy(); return
    install_recorder()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\ncapturing into {HELP}\n")
    for name in wanted:
        try:
            GROUPS[name]()
        except Exception as e:
            _failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"  !! {name}: {e}")
    print("\n" + ("all shots captured" if not _failures else
                  "INCOMPLETE:\n  " + "\n  ".join(_failures)))
    _window.destroy()


def main():
    global _window
    api = API()
    html = str(Path(resource_dir()) / "oasis" / "webui" / "index.html")
    _window = webview.create_window(title="OASIS", url=f"file://{html}", js_api=api,
                                    width=1280, height=820, min_size=(1100, 700),
                                    background_color="#FFFFFF")
    api.set_window(_window)
    webview.start(drive, debug=False)
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
