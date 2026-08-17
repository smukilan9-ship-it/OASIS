#!/usr/bin/env python3
"""
make_review_html.py — one self-contained review page per analysed image.

Split screen: the ORIGINAL on the left, the SEGMENTED overlay on the right, with a single
shared zoom/pan so both panes always show the same field at the same magnification. Panning
one pans the other; there is no way to compare the wrong two regions.

Every setting that produced the result is printed on the page, read from the run's own
summary JSON rather than re-derived here — pixel size AND where it came from, the downsample
the segmenter actually applied, whether the image was enlarged to the model's resolution, the
seed (detection) threshold, the DAB cutoff and whether it was the cohort value or an override,
the device, and the resulting counts and density. A page that cannot state its own provenance
is not a review page.

Overlays come from oasis.reporting.overlay, the same renderer the pipeline itself uses, so
what you review is what the pipeline drew — not a lookalike written for this script.

Usage:
  python validation/make_review_html.py --images DIR --results DIR --out DIR [--title NAME]
"""
import argparse
import base64
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JPEG_Q = 86


def _b64_jpeg(path_or_arr, quality=JPEG_Q):
    from PIL import Image
    import numpy as np
    im = (Image.open(path_or_arr).convert("RGB")
          if isinstance(path_or_arr, str) else Image.fromarray(np.asarray(path_or_arr)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), im.size


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def _settings_rows(s, ctx=None):
    """(label, value, note) triples. `note` flags anything a reviewer should not skim past.

    `ctx` carries cohort-level context the per-image summary cannot know on its own — the
    median positivity, and the pipeline's own per-image Confidence from results.csv. Both
    exist upstream; neither reached the reviewer before, so a slide the pipeline had already
    marked LOW rendered exactly like a clean one.
    """
    ctx = ctx or {}
    px_src = s.get("pixel_size_source")
    src_note = ("measured / per-image override" if px_src == "per_image_override" else
                "FROM FILENAME MAP — not measured" if px_src == "filename" else
                "FELL BACK TO DEFAULT" if px_src == "default_fallback" else
                "set in the UI" if px_src == "ui_default" else "")
    ds = s.get("segmenter_downsample")
    ds_note = ("enlarged to the model's 0.5 µm/px" if (ds and ds < 1) else
               "shrunk to the model's 0.5 µm/px" if (ds and ds > 1) else
               "fed at native resolution")
    thr_note = ""
    if s.get("threshold_override") is not None:
        thr_note = f"OVERRIDE — cohort value was {_fmt(s.get('cohort_threshold'))}"
    elif s.get("dab_threshold_method") == "adaptive_otsu":
        thr_note = "per-image Otsu, not a fixed cohort cutoff"
    # Absent means the run predates the field, NOT that it used the default. Say which,
    # because a page that prints a number it never read is asserting, not reporting.
    seed = s.get("segmenter_seed_threshold")
    if seed is None:
        seed_txt, seed_note = "0.7 (assumed)", "NOT RECORDED by this run — value is the bundle default, not read back"
    elif abs(float(seed) - 0.7) < 1e-9:
        seed_txt, seed_note = "0.7", "the bundle/QuPath default"
    else:
        seed_txt, seed_note = _fmt(seed), "NOT the default 0.7 — counts not comparable"
    dens = s.get("cells_per_mm2")
    dens_note = ("outside the plausible 100–20000 /mm² band"
                 if dens is not None and (dens < 100 or dens > 20000) else "")
    # Positivity has no absolute plausible band — it depends on the marker — so judge it
    # against this cohort. An image several times the cohort median is where background
    # gets called, and that is exactly what the reviewer should look at first.
    pct = float(s.get("positivity_pct") or 0.0)
    med = float(ctx.get("median_positivity") or 0.0)
    pos_note = ""
    if med > 0 and pct > 5.0 and pct > 3.0 * med:
        pos_note = (f"OUTLIER — {pct/med:.1f}x this cohort's median ({med:.2f}%); "
                    f"check for background being called")
    conf = ctx.get("confidence")
    conf_note = "" if conf in (None, "", "NORMAL") else f"pipeline flagged this slide {conf}"
    rows = [
        ("Pixel size", f"{_fmt(s.get('pixel_size_um'))} µm/px", src_note),
        ("Downsample applied", _fmt(ds), ds_note),
        ("Upsampled to model", _fmt(s.get("segmenter_upsampled")), ""),
        ("Detection (seed) threshold", seed_txt, seed_note),
        ("DAB cutoff", f"{_fmt(s.get('dab_threshold'))} OD", thr_note),
        ("Cutoff method", _fmt(s.get("dab_threshold_method")), ""),
        ("Segmenter", _fmt(s.get("segmenter")), ""),
        ("Device", _fmt(s.get("segmenter_device")), ""),
        ("Image size", f"{s.get('image_width')} × {s.get('image_height')} px", ""),
        ("Total cells", f"{s.get('total_cells'):,}" if s.get("total_cells") is not None else "—", ""),
        ("Positive", f"{s.get('positive_cells'):,}  ({float(s.get('positivity_pct') or 0):.2f}%)"
         if s.get("positive_cells") is not None else "—", pos_note),
        ("Density", f"{_fmt(dens, 1)} cells/mm²", dens_note),
    ]
    if conf is not None:
        rows.append(("Pipeline confidence", str(conf), conf_note))
    return rows


def build(image_path, geojson_path, summary_path, out_path, px, ctx=None):
    from oasis.reporting.overlay import generate_overlay
    import tempfile
    with open(summary_path, encoding="utf-8") as f:
        summ = json.load(f)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    ov_tmp = os.path.join(tempfile.gettempdir(), f"_ov_{stem}.png")
    generate_overlay(image_path, geojson_path, ov_tmp, pixel_size_um=px,
                     downsample=1.0, line_thickness=2, show_negative=True)
    raw_b64, (W, H) = _b64_jpeg(image_path)
    seg_b64, _ = _b64_jpeg(ov_tmp)
    try:
        os.remove(ov_tmp)
    except OSError:
        pass

    rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td>"
        f"<td class='note{' bad' if note and (note.isupper() or 'NOT' in note or 'OVERRIDE' in note or 'outside' in note or 'FELL' in note or 'OUTLIER' in note or 'flagged' in note) else ''}'>{note}</td></tr>"
        for k, v, note in _settings_rows(summ, ctx))

    html = _TEMPLATE
    for k, v in [("__STEM__", stem), ("__W__", str(W)), ("__H__", str(H)),
                 ("__RAW__", raw_b64), ("__SEG__", seg_b64), ("__ROWS__", rows),
                 ("__NCELL__", f"{summ.get('total_cells', 0):,}"),
                 ("__NPOS__", f"{summ.get('positive_cells', 0):,}"),
                 ("__PCT__", f"{float(summ.get('positivity_pct') or 0):.2f}")]:
        html = html.replace(k, v)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return {"stem": stem, "file": os.path.basename(out_path),
            "cells": summ.get("total_cells"), "pos": summ.get("positive_cells"),
            "pct": summ.get("positivity_pct"), "px": summ.get("pixel_size_um"),
            "ds": summ.get("segmenter_downsample"),
            "thr": summ.get("dab_threshold"),
            "bytes": os.path.getsize(out_path)}


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__STEM__ — segmentation review</title>
<style>
 :root{--bg:#0e0e11;--panel:#17171c;--line:#2a2a32;--txt:#e8e8ec;--dim:#9a9aa6;--bad:#ff8a5c;--acc:#7fb4ff}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
   font:13px/1.5 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;overflow:hidden}
 #top{height:46px;display:flex;align-items:center;gap:10px;padding:0 14px;background:var(--panel);
   border-bottom:1px solid var(--line);flex-wrap:nowrap;overflow:hidden}
 #top b{font-size:14px;white-space:nowrap}
 button{background:#26262e;color:var(--txt);border:1px solid #3a3a44;border-radius:6px;
   padding:5px 11px;cursor:pointer;font-size:12px;white-space:nowrap} button:hover{background:#32323c}
 /* Zoom is a real control, not just wheel-and-hope: the slider is authoritative, the
    readout is exact, and the buttons step it by a fixed factor. */
 #zoomwrap{display:flex;align-items:center;gap:8px;background:#1e1e25;border:1px solid #3a3a44;
   border-radius:6px;padding:3px 10px}
 #zr{width:150px;accent-color:var(--acc)}
 #zv{min-width:52px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
 #wrap{position:absolute;top:46px;left:0;right:0;bottom:0;display:flex;min-height:0}
 #main{flex:1 1 auto;display:flex;gap:2px;background:#000;min-width:0}
 .pane{position:relative;flex:1 1 0;min-width:0;overflow:hidden;cursor:grab;touch-action:none}
 .pane.drag{cursor:grabbing}
 .pane .lab{position:absolute;top:8px;left:8px;z-index:3;background:rgba(0,0,0,.72);
   padding:3px 9px;border-radius:5px;font-size:11px;letter-spacing:.04em;pointer-events:none}
 .w{position:absolute;transform-origin:0 0;will-change:transform}
 .w img{display:block}
 #side{flex:0 0 320px;background:var(--panel);border-left:1px solid var(--line);
   overflow:auto;padding:14px}
 /* Never let the sidebar squeeze the images to nothing — below this width it goes under. */
 @media (max-width:1000px){
   #wrap{flex-direction:column} #side{flex:0 0 auto;border-left:none;border-top:1px solid var(--line);max-height:42%}
   #main{flex:1 1 auto;min-height:0}
 }
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
   margin:0 0 10px;font-weight:600}
 table{width:100%;border-collapse:collapse;font-size:12px}
 th{text-align:left;font-weight:500;color:var(--dim);padding:5px 8px 5px 0;
   vertical-align:top;white-space:nowrap;width:44%}
 td{padding:5px 0;vertical-align:top}
 td.note{font-size:11px;color:var(--dim);font-style:italic;padding-left:8px}
 td.note.bad{color:var(--bad);font-style:normal;font-weight:600}
 tr{border-bottom:1px solid #202028}
 .big{display:flex;gap:10px;margin-bottom:16px}
 .big div{flex:1;background:#1e1e25;border-radius:8px;padding:10px}
 .big .n{font-size:18px;font-weight:600} .big .l{font-size:11px;color:var(--dim)}
 .hint{color:var(--dim);font-size:11px;margin-left:auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 kbd{background:#26262e;border:1px solid #3a3a44;border-radius:4px;padding:0 5px;font-size:11px}
</style></head><body>
<div id="top">
  <b>__STEM__</b>
  <div id="zoomwrap">
    <button onclick="stepZoom(1/1.4)" title="Zoom out">&minus;</button>
    <input type="range" id="zr" min="0" max="1000" value="0" oninput="sliderZoom(this.value)">
    <button onclick="stepZoom(1.4)" title="Zoom in">+</button>
    <b id="zv">100%</b>
  </div>
  <button onclick="fit()">Fit</button>
  <button onclick="setZoom(1)">1:1</button>
  <button id="ob" onclick="toggleOnly()">Show: both</button>
  <span class="hint">drag or arrow keys to move &middot; wheel or slider to zoom &middot; panes locked together &middot; <kbd>f</kbd> fit &middot; <kbd>space</kbd> swap</span>
</div>
<div id="wrap">
  <div id="main">
    <div class="pane" id="pL"><div class="lab">ORIGINAL — no segmentation</div>
      <div class="w" id="wL"><img id="iL" src="__RAW__"></div></div>
    <div class="pane" id="pR"><div class="lab">SEGMENTED — red = positive, green = negative</div>
      <div class="w" id="wR"><img id="iR" src="__SEG__"></div></div>
  </div>
  <div id="side">
    <div class="big">
      <div><div class="n">__NCELL__</div><div class="l">cells</div></div>
      <div><div class="n">__NPOS__</div><div class="l">positive</div></div>
      <div><div class="n">__PCT__%</div><div class="l">positivity</div></div>
    </div>
    <h2>Settings used for this image</h2>
    <table>__ROWS__</table>
  </div>
</div>
<script>
const W=__W__, H=__H__;
const wL=document.getElementById('wL'), wR=document.getElementById('wR');
const pL=document.getElementById('pL'), pR=document.getElementById('pR');
const zr=document.getElementById('zr'), zv=document.getElementById('zv');
// Zoom is logarithmic between these bounds so the slider feels even at both ends.
const ZMIN=0.05, ZMAX=40;
let sc=1, tx=0, ty=0, only=0;

function apply(){
  const t='translate('+tx+'px,'+ty+'px) scale('+sc+')';
  wL.style.transform=t; wR.style.transform=t;
  zv.textContent = (sc*100 < 10 ? (sc*100).toFixed(1) : Math.round(sc*100)) + '%';
  const f=(Math.log(sc)-Math.log(ZMIN))/(Math.log(ZMAX)-Math.log(ZMIN));
  zr.value = Math.max(0, Math.min(1000, Math.round(f*1000)));
}
function clampSc(v){ return Math.max(ZMIN, Math.min(ZMAX, v)); }
function zoomAt(f, cx, cy){
  const ns=clampSc(sc*f); f=ns/sc; if(f===1) return;
  tx=cx-(cx-tx)*f; ty=cy-(cy-ty)*f; sc=ns; apply();
}
// Measure whichever pane is actually on screen. toggleOnly() can hide pL, and a hidden
// element reports 0x0 — which silently killed fit() and anchored every zoom to the corner.
function vis(){ return only===2 ? pR : pL; }
function centre(){ const r=vis().getBoundingClientRect(); return [r.width/2, r.height/2]; }
function stepZoom(f){ const c=centre(); zoomAt(f, c[0], c[1]); }
function setZoom(v){ const c=centre(); zoomAt(clampSc(v)/sc, c[0], c[1]); }
function sliderZoom(val){
  const t=val/1000, ns=Math.exp(Math.log(ZMIN)+t*(Math.log(ZMAX)-Math.log(ZMIN)));
  const c=centre(); zoomAt(ns/sc, c[0], c[1]);
}
function fit(){ const r=vis().getBoundingClientRect(); if(!r.width||!r.height) return;
  sc=clampSc(Math.min(r.width/W, r.height/H)*0.98);
  tx=(r.width-W*sc)/2; ty=(r.height-H*sc)/2; apply(); }
// Swapping panes must NOT throw away where you are. The whole point of the tool is
// inspecting one cell at high zoom and flipping between raw and segmented; re-fitting on
// every swap made that impossible. Keep the transform, just re-apply it to the pane that
// is now visible.
function toggleOnly(){ only=(only+1)%3;
  document.getElementById('ob').textContent='Show: '+['both','original','segmented'][only];
  pL.style.display = only===2?'none':''; pR.style.display = only===1?'none':'';
  apply(); }

[pL,pR].forEach(p=>{
  p.addEventListener('wheel',e=>{e.preventDefault();
    const r=p.getBoundingClientRect();
    zoomAt(e.deltaY<0?1.15:1/1.15, e.clientX-r.left, e.clientY-r.top);},{passive:false});
  p.addEventListener('pointerdown',e=>{p.setPointerCapture(e.pointerId);
    p.dataset.d='1'; p.dataset.x=e.clientX; p.dataset.y=e.clientY; p.classList.add('drag');});
  p.addEventListener('pointermove',e=>{ if(p.dataset.d!=='1') return;
    tx+=e.clientX-parseFloat(p.dataset.x); ty+=e.clientY-parseFloat(p.dataset.y);
    p.dataset.x=e.clientX; p.dataset.y=e.clientY; apply();});
  ['pointerup','pointercancel','pointerleave'].forEach(k=>p.addEventListener(k,()=>{
    p.dataset.d='0'; p.classList.remove('drag');}));
});
addEventListener('keydown',e=>{
  // Leave focused controls alone — the zoom slider needs its own arrow stepping, and a
  // focused button needs Space. Only drive the viewer when nothing is focused.
  const t=e.target, tag=(t&&t.tagName||'').toLowerCase();
  if(tag==='input'||tag==='select'||tag==='textarea'||tag==='button') return;
  const STEP = e.shiftKey ? 200 : 60;
  if(e.key==='ArrowLeft'){ tx+=STEP; apply(); e.preventDefault(); }
  else if(e.key==='ArrowRight'){ tx-=STEP; apply(); e.preventDefault(); }
  else if(e.key==='ArrowUp'){ ty+=STEP; apply(); e.preventDefault(); }
  else if(e.key==='ArrowDown'){ ty-=STEP; apply(); e.preventDefault(); }
  else if(e.key==='f'){ fit(); }
  else if(e.key==='+'||e.key==='='){ stepZoom(1.4); }
  else if(e.key==='-'||e.key==='_'){ stepZoom(1/1.4); }
  else if(e.key===' '){ e.preventDefault(); toggleOnly(); }
});
addEventListener('resize',fit);
document.getElementById('iL').onload=fit; fit(); setTimeout(fit,60);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder holding the source images")
    ap.add_argument("--results", required=True, help="folder holding *_summary.json etc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    summaries = sorted(glob.glob(os.path.join(a.results, "*_summary.json")))
    if not summaries:
        raise SystemExit(f"no *_summary.json under {a.results}")
    # Pass 1: cohort context. Positivity can only be judged relative to its cohort, and the
    # per-image Confidence the pipeline computed lives in results.csv, not in the summaries.
    import csv as _csv
    import statistics as _st
    pcts = []
    for sp in summaries:
        with open(sp, encoding="utf-8") as f:
            pcts.append(float(json.load(f).get("positivity_pct") or 0.0))
    median_pos = _st.median(pcts) if pcts else 0.0
    conf_by_image = {}
    rcsv = os.path.join(a.results, "results.csv")
    if os.path.exists(rcsv):
        with open(rcsv, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = os.path.splitext(str(row.get("Image", "")).strip())[0]
                if name and row.get("Confidence"):
                    conf_by_image[name] = row["Confidence"].strip()
    print(f"cohort median positivity {median_pos:.2f}% | "
          f"confidence flags read for {len(conf_by_image)} images", flush=True)

    index = []
    for i, sp in enumerate(summaries):
        stem = os.path.basename(sp)[:-len("_summary.json")]
        geo = os.path.join(a.results, f"{stem}_detections.geojson")
        img = None
        for pat in (f"{stem}.tif", f"{stem}.tiff", f"{stem}.png", f"{stem}.jpg"):
            cand = glob.glob(os.path.join(a.images, "**", pat), recursive=True)
            if cand:
                img = cand[0]
                break
        if not img or not os.path.exists(geo):
            print(f"  SKIP {stem}: image={bool(img)} geojson={os.path.exists(geo)}", flush=True)
            continue
        with open(sp, encoding="utf-8") as f:
            px = json.load(f).get("pixel_size_um", 0.5)
        rec = build(img, geo, sp, os.path.join(a.out, f"{stem}.html"), px,
                    ctx={"median_positivity": median_pos,
                         "confidence": conf_by_image.get(stem)})
        index.append(rec)
        print(f"  [{i+1}/{len(summaries)}] {stem}  {rec['cells']:,} cells  "
              f"({rec['bytes']/1e6:.1f} MB)", flush=True)

    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    _write_index(a.out, index, a.title or os.path.basename(a.out))
    print(f"\n{len(index)} pages -> {a.out}")


def _cohort(index):
    """The DAB cutoff most images share, so per-image overrides stand out in the index."""
    from collections import Counter
    return Counter(float(r["thr"]) for r in index).most_common(1)[0][0]


def _write_index(out, index, title):
    rows = "\n".join(
        f'<tr><td><a href="{r["file"]}">{r["stem"]}</a></td>'
        f'<td class=n>{r["cells"]:,}</td><td class=n>{r["pos"]:,}</td>'
        f'<td class=n>{r["pct"]:.2f}%</td><td class=n>{r["px"]:.4f}</td>'
        f'<td class=n>{r["ds"]:.3f}</td>'
        f'<td class="n{" odd" if abs(float(r["thr"]) - _cohort(index)) > 1e-9 else ""}">{r["thr"]}</td></tr>'
        for r in index)
    tot = sum(r["cells"] for r in index)
    pos = sum(r["pos"] for r in index)
    html = f"""<!doctype html><meta charset=utf-8><title>{title} — review index</title>
<style>
 body{{background:#0e0e11;color:#e8e8ec;font:13px/1.6 -apple-system,system-ui,sans-serif;
   margin:0;padding:28px}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#9a9aa6;margin:0 0 20px}}
 table{{border-collapse:collapse;width:100%;max-width:1000px}}
 th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.07em;
   color:#9a9aa6;border-bottom:1px solid #2a2a32;padding:8px 10px}}
 td{{padding:7px 10px;border-bottom:1px solid #1c1c22}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 a{{color:#7fb4ff;text-decoration:none}} a:hover{{text-decoration:underline}}
 tfoot td{{font-weight:600;border-top:1px solid #2a2a32}}
 td.odd{{color:#ff8a5c;font-weight:700}}
</style>
<h1>{title}</h1>
<p class=sub>{len(index)} images · {tot:,} cells · {pos:,} positive
({pos/tot*100 if tot else 0:.2f}%) · click an image to open its split-screen review</p>
<table><thead><tr><th>Image</th><th class=n>Cells</th><th class=n>Positive</th>
<th class=n>Positivity</th><th class=n>µm/px</th><th class=n>Downsample</th>
<th class=n>DAB OD</th></tr></thead>
<tbody>{rows}</tbody>
<tfoot><tr><td>{len(index)} images</td><td class=n>{tot:,}</td><td class=n>{pos:,}</td>
<td class=n>{pos/tot*100 if tot else 0:.2f}%</td><td colspan=3></td></tr></tfoot></table>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
