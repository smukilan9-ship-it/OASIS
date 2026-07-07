#!/usr/bin/env python3
"""
Build a self-contained HTML labeling tool for a TIM-3 (or any membranous) image.

Given the image + its InstanSeg detection GeoJSON (with QuPath `DAB: Mean`), emit
a single portable HTML file: the image with clickable per-cell nucleus outlines,
where you LEFT-click a cell to mark it TIM-3 POSITIVE and RIGHT-click to mark it
NEGATIVE (click again to clear). Zoom with the mouse wheel, pan by dragging.
"Export labelled GeoJSON" downloads the full detection set with your labels written
as `membrane_pos` / `membrane_neg` classifications (all measurements preserved),
ready for:

  python validation/tune_membrane_threshold.py --image IMG --labelled LABELLED.geojson \\
      --pixel-size 0.5 --pos-label membrane_pos --neg-label membrane_neg \\
      --out validation/membrane_cutoffs.yaml

Existing classifications are stripped in the tool so every cell starts UNLABELLED
(don't let the nuclear pass bias your hand labels). Label the membrane by the
brown ring around each nucleus — aim for ~50+ positive AND ~50+ negative across
images, including faint/borderline cells.

Usage:
  python validation/make_tim3_label_tool.py --image IMG.jpg --geojson DET.geojson --out TOOL.html
"""
import os, sys, json, base64, argparse, mimetypes


def build(image_path, geojson_path, out_path):
    with open(geojson_path) as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    from PIL import Image
    import numpy as np, io
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cell_expansion import (_estimate_background, _estimate_stain_vectors,
                                _od_channels, _QUPATH_STAINS, _DEFAULT_BACKGROUND)

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = rgb.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]

    def _b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # (1) Original. (2) Normalized: per-image white point removes the CRC-ICM
    # green/blue cast so brown reads true. (3) DAB signal: per-image deconvolution
    # showing ONLY marker-dominant DAB (DAB_OD > H_OD) as brown-on-white, so real
    # TIM-3 membrane staining is unmistakable against background.
    bg = _estimate_background(rgb)
    norm = np.clip(rgb.astype(float) * (255.0 / bg.reshape(1, 1, 3)), 0, 255).astype(np.uint8)
    est = _estimate_stain_vectors(rgb, bg)
    vecs = est if est else _QUPATH_STAINS
    bg_use = bg.tolist() if est else _DEFAULT_BACKGROUND
    hem_od, dab_od = _od_channels(rgb, vecs, bg_use)
    d = np.clip(dab_od, 0, None).astype(float)
    dom = (dab_od > hem_od) & (dab_od > 0)
    p99 = float(np.percentile(d[dom], 99)) if dom.any() else 1.0
    inten = (np.clip(d / (p99 or 1.0), 0, 1) * dom)[..., None]
    white = np.array([245, 245, 242]); brown = np.array([110, 64, 32])
    dab_img = (white * (1 - inten) + brown * inten).astype(np.uint8)

    # Strip existing classifications so cells start unlabelled in the export copy.
    for ft in feats:
        ft.setdefault("properties", {}).pop("classification", None)

    data = {"stem": stem, "w": int(W), "h": int(H), "geo": gj,
            "views": {"Original": _b64(rgb), "Normalized": _b64(norm),
                      "DAB signal": _b64(dab_img)}}

    html = _TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  {stem}: {len(feats)} labellable cells → {out_path}")


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>TIM-3 cell labeler</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#eee;font:13px/1.4 system-ui,sans-serif;overflow:hidden}
  #bar{position:fixed;top:0;left:0;right:0;height:46px;background:#1b1b1f;border-bottom:1px solid #333;
       display:flex;align-items:center;gap:16px;padding:0 14px;z-index:10}
  #bar b{color:#fff} .pos{color:#ff5a5a} .neg{color:#4aa3ff}
  #stage{position:absolute;top:46px;left:0;right:0;bottom:0;overflow:hidden;cursor:grab}
  #stage.drag{cursor:grabbing}
  #world{position:absolute;transform-origin:0 0}
  #world img{display:block;position:absolute;top:0;left:0}
  svg{position:absolute;top:0;left:0}
  polygon{fill:transparent;stroke:#888;stroke-opacity:.35;stroke-width:1;vector-effect:non-scaling-stroke;cursor:pointer}
  polygon:hover{stroke:#fff;stroke-opacity:1;stroke-width:2}
  polygon.pos{stroke:#ff3b3b;stroke-opacity:1;stroke-width:2;fill:rgba(255,60,60,.28)}
  polygon.neg{stroke:#2f8dff;stroke-opacity:1;stroke-width:2;fill:rgba(50,140,255,.25)}
  button{background:#2a2a30;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
  button:hover{background:#35353c} .hint{color:#888;margin-left:auto;font-size:12px}
</style></head><body>
<div id="bar">
  <b id="ttl"></b>
  <span>Positive <b class="pos" id="np">0</b></span>
  <span>Negative <b class="neg" id="nn">0</b></span>
  <button onclick="zoom(1.25)">+</button><button onclick="zoom(0.8)">−</button>
  <button onclick="resetView()">Fit</button>
  <button id="viewbtn" onclick="cycleView()">View: Normalized</button>
  <button onclick="exportGeo()">Export labelled GeoJSON</button>
  <span class="hint">Left-click = <span class="pos">positive</span> · Right-click = <span class="neg">negative</span> · click again to clear · wheel = zoom · drag = pan</span>
</div>
<div id="stage"><div id="world"><img id="im"><svg id="ov"></svg></div></div>
<script>
const D = /*__DATA__*/;
const world=document.getElementById('world'), stage=document.getElementById('stage'),
      im=document.getElementById('im'), ov=document.getElementById('ov');
document.getElementById('ttl').textContent = D.stem;
im.width=D.w; im.height=D.h;
const views=Object.keys(D.views); let vi=1;   // default = Normalized
function setView(){ im.src=D.views[views[vi]]; document.getElementById('viewbtn').textContent='View: '+views[vi]; }
function cycleView(){ vi=(vi+1)%views.length; setView(); }
setView();
ov.setAttribute('width',D.w); ov.setAttribute('height',D.h);
ov.setAttribute('viewBox','0 0 '+D.w+' '+D.h);
const feats=D.geo.features;
const state=new Array(feats.length).fill(0);   // 0 none,1 pos,2 neg
function ringOf(ft){ const g=ft.geometry||{}, c=g.coordinates||[];
  if(g.type==='Polygon'&&c.length) return c[0];
  if(g.type==='MultiPolygon'&&c.length&&c[0]) return c[0][0]; return null; }
// build polygons
const frag=document.createDocumentFragment();
feats.forEach((ft,i)=>{ const r=ringOf(ft); if(!r) return;
  const p=document.createElementNS('http://www.w3.org/2000/svg','polygon');
  p.setAttribute('points', r.map(pt=>pt[0]+','+pt[1]).join(' '));
  p.dataset.i=i;
  p.addEventListener('click',e=>{e.preventDefault(); set(i,1);});
  p.addEventListener('contextmenu',e=>{e.preventDefault(); set(i,2);});
  frag.appendChild(p);
});
ov.appendChild(frag);
const polys={}; ov.querySelectorAll('polygon').forEach(p=>polys[p.dataset.i]=p);
function set(i,v){ state[i]=(state[i]===v)?0:v; const p=polys[i];
  p.classList.toggle('pos',state[i]===1); p.classList.toggle('neg',state[i]===2); counts(); }
function counts(){ let a=0,b=0; state.forEach(s=>{if(s===1)a++;else if(s===2)b++;});
  document.getElementById('np').textContent=a; document.getElementById('nn').textContent=b; }
// pan/zoom
let sc=1,tx=0,ty=0;
function apply(){ world.style.transform='translate('+tx+'px,'+ty+'px) scale('+sc+')'; }
function resetView(){ const r=stage.getBoundingClientRect(); sc=Math.min(r.width/D.w,r.height/D.h)*0.98;
  tx=(r.width-D.w*sc)/2; ty=(r.height-D.h*sc)/2; apply(); }
function clampF(f){ const M=10, m=0.02; if(sc*f>M) return M/sc; if(sc*f<m) return m/sc; return f; }
function zoom(f){ f=clampF(f); const r=stage.getBoundingClientRect(); const cx=r.width/2,cy=r.height/2;
  tx=cx-(cx-tx)*f; ty=cy-(cy-ty)*f; sc*=f; apply(); }
stage.addEventListener('wheel',e=>{e.preventDefault(); let f=e.deltaY<0?1.12:0.89; f=clampF(f);
  const r=stage.getBoundingClientRect(); const mx=e.clientX-r.left,my=e.clientY-r.top;
  tx=mx-(mx-tx)*f; ty=my-(my-ty)*f; sc*=f; apply();},{passive:false});
let dragging=false,px,py,moved=0;
stage.addEventListener('pointerdown',e=>{dragging=true;moved=0;px=e.clientX;py=e.clientY;stage.classList.add('drag');});
stage.addEventListener('pointermove',e=>{if(!dragging)return; const dx=e.clientX-px,dy=e.clientY-py;
  moved+=Math.abs(dx)+Math.abs(dy); tx+=dx;ty+=dy;px=e.clientX;py=e.clientY;apply();});
addEventListener('pointerup',()=>{dragging=false;stage.classList.remove('drag');});
// suppress cell click if it was a drag
ov.addEventListener('click',e=>{if(moved>4){e.stopImmediatePropagation();}},true);
function exportGeo(){ const g=JSON.parse(JSON.stringify(D.geo));
  g.features.forEach((ft,i)=>{ ft.properties=ft.properties||{};
    if(state[i]===1) ft.properties.classification={name:'membrane_pos',color:[255,0,0]};
    else if(state[i]===2) ft.properties.classification={name:'membrane_neg',color:[0,120,255]};
    else delete ft.properties.classification; });
  const blob=new Blob([JSON.stringify(g)],{type:'application/geo+json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=D.stem+'_labelled.geojson'; a.click(); }
im.onload=resetView; setTimeout(resetView,50);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.image, a.geojson, a.out)


if __name__ == "__main__":
    main()
