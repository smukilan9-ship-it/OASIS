"""
make_landmark_tool.py — generate browser-viewable image copies + a self-contained
local HTML landmark-clicking tool for gold-standard TRE (Phase A, A4).

v2 (guided re-click): pre-loads any prior clicks from phase_a_qc/landmarks.json,
COLOUR-CODED green=consistent (keep) / red=outlier (delete & redo), with per-point
delete and a small registration-overlay reference per pair. References the JPEG
copies by relative path (kept on disk, NOT embedded) so the HTML stays tiny.
"""
import os, sys, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "phase_a_qc")
IMG = os.path.join(OUT, "img")
DISP_W = 1280
DATA = "/Users/mukilan/Desktop/052526"

PAIRS = [
    ("Tumor_1", "Tumor/LL477_CD8_x10_1.tif", "Tumor/LL477_Tim3_x10_1.tif"),
    ("Tumor_2", "Tumor/LL477_CD8_x10_2.tif", "Tumor/LL477_Tim3_x10_2.tif"),
    ("Tumor_3", "Tumor/LL477_CD8_x10_3.tif", "Tumor/LL477_Tim3_10X_3.tif"),
    ("Liver_1", "Liver/LL477_Liver_CD8_10X_1.tif", "Liver/LL477_Liver_Tim3_10X_1.tif"),
    ("Liver_2", "Liver/LL477_Liver_CD8_10X_2.tif", "Liver/LL477_Liver_Tim3_10X_2.tif"),
    ("Liver_3", "Liver/LL477_Liver_CD8_10X_3.tif", "Liver/LL477_Liver_Tim3_10X_3.tif"),
    ("Liver_4", "Liver/LL477_Liver_CD8_10X_4.tif", "Liver/LL477_Liver_Tim3_10X_4.tif"),
]


def _flags_for(points):
    """green(1)=consistent / red(0)=outlier, via RANSAC similarity on the clicks."""
    import cv2
    pts = np.array(points, float)
    if len(pts) < 4:
        return [0] * len(pts)            # too few to trust → mark all redo
    ref, mov = pts[:, :2], pts[:, 2:]
    _M, inl = cv2.estimateAffinePartial2D(
        mov.astype(np.float32), ref.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if inl is None:
        return [0] * len(pts)
    return [int(x) for x in inl.ravel()]


def main():
    os.makedirs(IMG, exist_ok=True)
    prior = {}
    lm_path = os.path.join(OUT, "landmarks.json")
    if os.path.exists(lm_path):
        txt = open(lm_path).read().strip()
        if txt.startswith("LANDMARKS"):
            txt = txt[len("LANDMARKS"):].strip()
        prior = json.loads(txt)

    meta = []
    for sid, a, b in PAIRS:
        rec = {"id": sid, "orig_w": None, "disp_w": DISP_W}
        for tag, rel in (("cd8", a), ("tim3", b)):
            im = Image.open(os.path.join(DATA, rel)).convert("RGB")
            rec["orig_w"] = im.width
            im2 = im.resize((DISP_W, int(im.height * DISP_W / im.width)))
            fn = f"{sid}_{tag}.jpg"
            im2.save(os.path.join(IMG, fn), "JPEG", quality=72)
            rec[tag] = f"img/{fn}"
        # small overlay reference (MI registration) if present
        ov = os.path.join(OUT, f"{sid}_overlay.png")
        if os.path.exists(ov):
            ref_small = Image.open(ov).convert("RGB")
            ref_small.thumbnail((760, 760))
            ref_small.save(os.path.join(IMG, f"{sid}_ref.jpg"), "JPEG", quality=70)
            rec["ref"] = f"img/{sid}_ref.jpg"
        pts = prior.get(sid, {}).get("points", [])
        rec["points"] = pts
        rec["flags"] = _flags_for(pts) if pts else []
        meta.append(rec)

    html = _HTML.replace("__META__", json.dumps(meta))
    path = os.path.join(OUT, "landmark_tool.html")
    with open(path, "w") as f:
        f.write(html)
    print("Wrote", path)
    print("Open:  file://" + path)
    for m in meta:
        g = sum(m["flags"]); n = len(m["flags"])
        print(f"  {m['id']}: {n} prior pts, {g} green / {n-g} red")
    return 0


_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Phase A landmark tool</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:16px;color:#1a1a1a;max-width:1000px}
 h1{font-size:20px;font-weight:500} .muted{color:#666;font-size:13px;line-height:1.6}
 .bar{display:flex;align-items:center;gap:10px;margin:10px 0;flex-wrap:wrap}
 button{font-size:14px;padding:6px 12px;border:1px solid #bbb;background:#fff;border-radius:6px;cursor:pointer}
 button:hover{background:#f2f2f2}
 .imgwrap{position:relative;display:inline-block;border:1px solid #ccc;line-height:0}
 .imgwrap img{width:100%;display:block;cursor:crosshair}
 body.blind .imgwrap img{filter:grayscale(1) contrast(1.08)}
 .lbl{font-size:13px;font-weight:500;margin:8px 0 2px}
 .ov{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
 .dot{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
      border:2px solid #fff;font-size:10px;color:#fff;text-align:center;line-height:11px;font-weight:700}
 #loupe{position:absolute;width:160px;height:160px;border:2px solid #333;border-radius:50%;
        overflow:hidden;pointer-events:none;display:none;z-index:50;box-shadow:0 0 6px rgba(0,0,0,.4)}
 .next{font-weight:500}
 table{border-collapse:collapse;font-size:12px;margin-top:6px} td,th{border:1px solid #ddd;padding:2px 6px}
 #out{width:100%;height:80px;font-family:monospace;font-size:11px;margin-top:8px}
 .legend{font-size:12px} .gd{color:#1f8a4c;font-weight:700} .bd{color:#c62828;font-weight:700} .nw{color:#1565c0;font-weight:700}
 #ref{max-width:380px;border:1px solid #ccc;margin-top:6px}
</style></head><body>
<h1>Phase A — guided landmark re-click</h1>
<p class="muted">Your prior points are pre-loaded: <span class="gd">green = consistent (keep)</span>,
<span class="bd">red = outlier (delete &amp; redo)</span>, <span class="nw">blue = new</span>.
Goal per pair: <b>≥6 (ideally 8)</b> well-spread <span class="gd">green/blue</span> points on
<b>unambiguous structures</b> (vessel/sinusoid bifurcations, lumen centres, sharp boundary corners — never a single nucleus).
Click a row's <b>✕</b> to delete a point. Click CD8 (top) then the matching TIM-3 (bottom) to add one.
Images are shown greyscale (<b>Blind: ON</b>) so you mark structure, not stain colour. Do all 7 pairs, then <b>Copy results</b> and paste back.</p>
<div class="bar">
 <button id="prev">&larr; Prev</button><span class="next" id="pairlbl"></span><button id="next">Next &rarr;</button>
 <span id="status" class="muted"></span>
 <button id="undo">Undo last add</button><button id="clearbad">Delete all red</button>
 <button id="clear">Clear pair</button><button id="blind">Blind: ON</button><button id="copy">Copy results</button>
</div>
<div class="lbl" id="cd8lbl">CD8 (reference) — click first</div>
<div class="imgwrap" id="w_cd8"><img id="img_cd8"><div class="ov" id="ov_cd8"></div></div>
<div class="lbl" id="tim3lbl">TIM-3 (moving) — click the matching point</div>
<div class="imgwrap" id="w_tim3"><img id="img_tim3"><div class="ov" id="ov_tim3"></div></div>
<div id="loupe"><canvas id="lc" width="160" height="160"></canvas></div>
<div class="lbl">Registration overlay (green=CD8, magenta=TIM-3; grey=agree) — guidance only</div>
<img id="ref">
<h3 style="font-size:14px;font-weight:500">Points (this pair)</h3>
<div id="list"></div>
<textarea id="out" readonly placeholder="Copy results appears here"></textarea>
<script>
const META=__META__;
let pi=0, pending=null;
const data={}, flags={};
META.forEach(m=>{data[m.id]=(m.points||[]).map(p=>p.slice(0,4)); flags[m.id]=(m.flags||[]).slice();});
const E=id=>document.getElementById(id);
const imgC=E("img_cd8"), imgT=E("img_tim3");
function cur(){return META[pi];}
function load(){const m=cur();imgC.src=m.cd8;imgT.src=m.tim3;E("ref").src=m.ref||"";
 E("pairlbl").textContent=`Pair ${pi+1}/${META.length}: ${m.id}`;pending=null;render();}
function factor(){return cur().orig_w/cur().disp_w;}
function color(f){return f===1?"#1f8a4c":(f===0?"#c62828":"#1565c0");}
function addDot(ovId,imgEl,xn,yn,n,col){const m=cur();
 const orig_h=m.orig_w*(imgEl.naturalHeight/imgEl.naturalWidth);
 const d=document.createElement("div");d.className="dot";
 d.style.left=(xn/m.orig_w*100)+"%";d.style.top=(yn/orig_h*100)+"%";
 d.style.background=col;d.textContent=n;E(ovId).appendChild(d);}
function render(){E("ov_cd8").innerHTML="";E("ov_tim3").innerHTML="";
 const pts=data[cur().id], fl=flags[cur().id];
 pts.forEach((p,i)=>{const c=color(fl[i]);addDot("ov_cd8",imgC,p[0],p[1],i+1,c);addDot("ov_tim3",imgT,p[2],p[3],i+1,c);});
 if(pending) addDot("ov_cd8",imgC,pending[0],pending[1],pts.length+1,"#1565c0");
 const ngood=fl.filter(f=>f!==0).length;
 E("status").textContent=`${pts.length} pts (${ngood} keep/new, ${fl.filter(f=>f===0).length} red)`+(pending?" — click TIM-3":"");
 E("cd8lbl").style.fontWeight=pending?"400":"700";E("tim3lbl").style.fontWeight=pending?"700":"400";
 let h="<table><tr><th>#</th><th>state</th><th>CD8</th><th>TIM3</th><th></th></tr>";
 pts.forEach((p,i)=>{const s=fl[i]===1?'<span class="gd">keep</span>':(fl[i]===0?'<span class="bd">redo</span>':'<span class="nw">new</span>');
  h+=`<tr><td>${i+1}</td><td>${s}</td><td>${p[0]|0},${p[1]|0}</td><td>${p[2]|0},${p[3]|0}</td><td><button onclick="del(${i})">✕</button></td></tr>`;});
 h+="</table>";E("list").innerHTML=h;
 const obj={};META.forEach(m=>{if(data[m.id].length)obj[m.id]={orig_w:m.orig_w,points:data[m.id]};});
 E("out").value="LANDMARKS "+JSON.stringify(obj);}
window.del=i=>{data[cur().id].splice(i,1);flags[cur().id].splice(i,1);render();};
function clickPos(ev,img){const r=img.getBoundingClientRect();
 const xd=(ev.clientX-r.left)/r.width*img.naturalWidth, yd=(ev.clientY-r.top)/r.height*img.naturalHeight;
 const f=factor();return [xd*f,yd*f];}
imgC.addEventListener("click",ev=>{if(pending)return;pending=clickPos(ev,imgC);render();});
imgT.addEventListener("click",ev=>{if(!pending)return;const p=clickPos(ev,imgT);
 data[cur().id].push([pending[0],pending[1],p[0],p[1]]);flags[cur().id].push(-1);pending=null;render();});
E("prev").onclick=()=>{pi=(pi-1+META.length)%META.length;load();};
E("next").onclick=()=>{pi=(pi+1)%META.length;load();};
E("undo").onclick=()=>{if(pending){pending=null;}else{data[cur().id].pop();flags[cur().id].pop();}render();};
E("clearbad").onclick=()=>{const d=data[cur().id],f=flags[cur().id];for(let i=d.length-1;i>=0;i--)if(f[i]===0){d.splice(i,1);f.splice(i,1);}render();};
E("clear").onclick=()=>{data[cur().id]=[];flags[cur().id]=[];pending=null;render();};
E("copy").onclick=()=>{E("out").select();document.execCommand("copy");E("status").textContent="copied — paste into chat";};
document.body.classList.add("blind");
E("blind").onclick=()=>{const b=document.body.classList.toggle("blind");E("blind").textContent="Blind: "+(b?"ON":"OFF");};
const loupe=E("loupe"),lc=E("lc"),lx=lc.getContext("2d"),Z=3;
function moveLoupe(ev,img){const r=img.getBoundingClientRect();
 if(ev.clientX<r.left||ev.clientX>r.right||ev.clientY<r.top||ev.clientY>r.bottom){loupe.style.display="none";return;}
 loupe.style.display="block";loupe.style.left=(ev.clientX+window.scrollX+18)+"px";loupe.style.top=(ev.clientY+window.scrollY-80)+"px";
 const sx=(ev.clientX-r.left)/r.width*img.naturalWidth, sy=(ev.clientY-r.top)/r.height*img.naturalHeight;
 lx.clearRect(0,0,160,160);lx.filter=document.body.classList.contains("blind")?"grayscale(1) contrast(1.08)":"none";try{lx.drawImage(img,sx-160/(2*Z),sy-160/(2*Z),160/Z,160/Z,0,0,160,160);}catch(e){}lx.filter="none";
 lx.strokeStyle="#e11";lx.beginPath();lx.moveTo(80,70);lx.lineTo(80,90);lx.moveTo(70,80);lx.lineTo(90,80);lx.stroke();}
imgC.addEventListener("mousemove",e=>moveLoupe(e,imgC));imgT.addEventListener("mousemove",e=>moveLoupe(e,imgT));
[imgC,imgT].forEach(im=>im.addEventListener("mouseleave",()=>loupe.style.display="none"));
imgC.onload=render;load();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
