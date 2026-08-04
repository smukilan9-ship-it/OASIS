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

    ./.venv/bin/python tools/capture_help.py            # everything
    ./.venv/bin/python tools/capture_help.py quant set  # only those groups

Runs against demo data listed in DEMO below (HyReCo CD8/CD45 brightfield). It is real IHC
and a real run every time: nothing here fakes a number onto a screen.
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
SCRATCH = Path("/private/tmp/claude-501/-Users-mukilan-PycharmProjects-ihc-original-copy"
               "/8d3ccfb2-fce9-4fea-8daf-9f7309a1100d/scratchpad")
DEMO = SCRATCH / "oasis_demo"
OUT = SCRATCH / "help_runs"

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
    scroll_to_text("what the burned-in bar represents")
    shot("set-3-scalebar.png")
    scroll_to_text("cohort pixel size")
    shot("set-1-pixelsize.png")
    scroll_to_text("which pixel size wins")
    shot("set-2-priority.png")
    scroll_to_text("analysis engine")
    shot("set-4-engine.png")


def do_quant():
    print("  quant")
    js("showPage('quant')")
    time.sleep(0.8)
    click("q-mode-batch-btn")
    set_field("q-input", str(QUANT_IN))
    set_field("q-output", str(OUT / "quant"))
    time.sleep(2.0)                                   # folder scan populates the count
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


def do_spatial():
    print("  spatial")
    js("showPage('spatial')")
    time.sleep(0.8)
    click("spatial-mode-batch-btn")
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
        if not wait_for("document.getElementById('spatial-loftr-auto')", timeout=420,
                        what=f"pair {i + 1} to load"):
            break
        js("loftrAutoFind()")
        if not wait_for(f"spatialCertifications && "
                        f"Object.values(spatialCertifications).filter(c=>c&&c.is_certified).length >= {i + 1}",
                        timeout=1200, what=f"pair {i + 1} to certify"):
            break
        print(f"      certified {i + 1}/3")
    time.sleep(1.5)
    shot("spatial-3-certify.png")
    scroll_to(selector=".cert-pairnav")
    shot("spatial-4-pairnav.png")

    js("spatialWizardGoTo(6)")
    time.sleep(0.8)
    js("spatialBandwidthCheck()")
    wait_for("document.getElementById('spatial-bw-result') && "
             "document.getElementById('spatial-bw-result').innerHTML.length > 200",
             timeout=2400, what="the null-model check")
    time.sleep(1.5)

    js("spatialAssocRun()")
    if wait_for("document.getElementById('spatial-review').classList.contains('active')",
                timeout=2400, what="the Spatial review screen"):
        time.sleep(2.5)
        scroll_to(top=0)
        shot("spatial-5-review.png")
        js("spReviewApply()")

    if wait_for("document.getElementById('spatial-results').classList.contains('active')",
                timeout=2400, what="the Spatial results screen"):
        time.sleep(3.0)
        scroll_to(top=0)
        shot("spatial-6-results.png")


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
    n = js("""(function(){
      const K = %d;
      let tp = 0, tn = 0;
      calib.images.forEach(im => {
        const cells = (im.data && im.data.cells) || [];
        const order = cells.map((c,i) => [c.v, i])
                           .filter(p => p[0] != null && isFinite(p[0]))
                           .sort((a,b) => b[0] - a[0]);
        if (order.length < 4 * K) return;      // too few to leave a gap between the classes
        im.state = new Array(cells.length).fill(0);
        order.slice(0, K).forEach(p => { im.state[p[1]] = 1; tp++; });
        order.slice(order.length - K).forEach(p => { im.state[p[1]] = 2; tn++; });
      });
      calibBuild();          // repaint the canvas from the states just written
      return [tp, tn];
    })()""" % PER_SLIDE)
    print(f"      labelled {n[0]} positive / {n[1]} negative")
    time.sleep(2.0)
    shot("cls-3-label.png")

    js("clsFit()")
    if wait_for("cls && cls.fit && cls.fit.ok", timeout=1800, what="the classifier to fit"):
        time.sleep(2.0)
        js("showPage('classifier')")
        scroll_to(selector="#cls-results")
        shot("cls-4-report.png")


GROUPS = {"settings": do_settings, "quant": do_quant,
          "spatial": do_spatial, "classifier": do_classifier}


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
