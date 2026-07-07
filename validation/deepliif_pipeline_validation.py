#!/usr/bin/env python3
"""
Batch-validate the REAL IHC pipeline (run_pipeline.py: InstanSeg + DAB
classification + overlay) against DeepLIIF's IF-derived expert ground truth.

Folder layout, per condition (raw_instanseg | preprocessed):
  <cond>/changed_inputs/  cropped IHC panel that is fed to the pipeline
  <cond>/expert_overlay/  ground-truth cells rendered from DeepLIIF SegMask
  <cond>/our_overlay/     the pipeline's classification overlay
  <cond>/f1/              matched overlay + metrics (precision/recall/acc/F1)
  <cond>/_pipeline_out/   raw pipeline outputs (geojson/csv/summary)

Ground truth = DeepLIIF SegMask panel (red=positive cell, blue=negative),
derived from co-registered IF, not hand-labelled.

Usage:
  python validation/deepliif_pipeline_validation.py prep   --n 8
  python validation/deepliif_pipeline_validation.py overlay-gt
  python validation/deepliif_pipeline_validation.py score  --cond raw_instanseg
"""
import os, sys, json, glob, argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

BASE = os.path.expanduser("~/Desktop/DeepLIIF_data/pipeline_validation")
SRC  = os.path.expanduser("~/Desktop/DeepLIIF_data/DeepLIIF_Testing_Set")
PANEL = 512

# ---------- shared helpers ----------
def _white_balance(rgb):
    flat = rgb.reshape(-1, 3).astype(np.float64)
    bright = flat[flat.mean(1) > np.percentile(flat.mean(1), 80)]
    bg = np.clip(np.percentile(bright, 99, axis=0), 200, 255)
    return np.clip(rgb.astype(np.float64) * (255.0 / bg.reshape(1, 1, 3)), 0, 255).astype(np.uint8)

def _gt_cells(mask_rgb):
    """Return list of (centroid_xy, label, comp_mask). pos=red, neg=blue."""
    r, g, b = (mask_rgb[..., i].astype(int) for i in range(3))
    pos = (r > 90) & (r > b + 25) & (r >= g)
    neg = (b > 90) & (b > r + 25) & (b >= g)
    out = []
    for m, lab in [(pos, 1), (neg, 0)]:
        L, n = ndimage.label(m)
        for i in range(1, n + 1):
            comp = L == i
            if comp.sum() >= 20:
                ys, xs = np.where(comp)
                out.append(((float(xs.mean()), float(ys.mean())), lab, comp))
    return out

def _feat_centroid(feat):
    g = feat.get("geometry", {})
    if g.get("type") != "Polygon" or not g.get("coordinates"):
        return None
    ring = np.asarray(g["coordinates"][0], float)
    return float(ring[:, 0].mean()), float(ring[:, 1].mean())

def _is_pos(feat):
    c = feat.get("properties", {}).get("classification") or {}
    return str(c.get("name", "")).lower().startswith("pos")

# ---------- stages ----------
def prep(n, preprocess_too=True):
    files = sorted(glob.glob(os.path.join(SRC, "*.png")))[:n]
    for cond in ("raw_instanseg", "preprocessed"):
        for sub in ("changed_inputs", "expert_overlay", "our_overlay", "f1", "_pipeline_out"):
            os.makedirs(os.path.join(BASE, cond, sub), exist_ok=True)
    gt_all = {}
    for f in files:
        im = np.asarray(Image.open(f).convert("RGB"))
        npan = im.shape[1] // PANEL
        ihc = im[:, 0:PANEL]
        mask = im[:, (npan - 1) * PANEL:npan * PANEL]
        stem = os.path.splitext(os.path.basename(f))[0]
        # raw crop
        Image.fromarray(ihc).save(os.path.join(BASE, "raw_instanseg", "changed_inputs", stem + ".png"))
        # preprocessed input (white balance)
        Image.fromarray(_white_balance(ihc)).save(
            os.path.join(BASE, "preprocessed", "changed_inputs", stem + ".png"))
        # ground truth cells
        cells = _gt_cells(mask)
        gt_all[stem] = [{"xy": c[0], "label": c[1]} for c in cells]
        # expert overlay (draw on the raw IHC)
        ov = Image.fromarray(ihc.copy()); d = ImageDraw.Draw(ov)
        for (x, y), lab, comp in cells:
            col = (255, 0, 0) if lab == 1 else (0, 90, 255)
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col, width=2)
        for cond in ("raw_instanseg", "preprocessed"):
            ov.save(os.path.join(BASE, cond, "expert_overlay", stem + ".png"))
    json.dump(gt_all, open(os.path.join(BASE, "ground_truth.json"), "w"))
    print(f"prep: {len(files)} images -> {BASE}")
    print(f"  GT cells total: {sum(len(v) for v in gt_all.values())}")

def score(cond, match_tol=15.0, adaptive=False):
    gt_all = json.load(open(os.path.join(BASE, "ground_truth.json")))
    out_dir = os.path.join(BASE, cond, "_pipeline_out")
    agg = dict(tp=0, fp=0, fn=0, tn=0, gt_matched=0, gt_total=0, pred_total=0)
    per_image = {}
    for stem, gts in gt_all.items():
        geo = glob.glob(os.path.join(out_dir, f"{stem}*.geojson"))
        if not geo:
            print(f"  MISSING geojson for {stem}"); continue
        feats = json.load(open(geo[0])).get("features", [])
        preds = []
        for ft in feats:
            c = _feat_centroid(ft)
            if c: preds.append((c, _is_pos(ft)))
        # optional adaptive threshold on DAB:Mean (Otsu across this image's cells)
        if adaptive:
            dabs = []
            for ft in feats:
                m = (ft.get("properties", {}).get("measurements", {}) or {})
                dabs.append(m.get("DAB: Mean", m.get("Nucleus: DAB OD mean")))
            dv = np.array([x for x in dabs if isinstance(x, (int, float))], float)
            if len(dv) > 20:
                thr = _otsu(dv)
                preds = []
                for ft in feats:
                    c = _feat_centroid(ft)
                    if not c: continue
                    m = (ft.get("properties", {}).get("measurements", {}) or {})
                    val = m.get("DAB: Mean", m.get("Nucleus: DAB OD mean"))
                    preds.append((c, isinstance(val, (int, float)) and val > thr))
        # greedy ONE-TO-ONE matching of GT<->pred by ascending centroid distance
        pcen = np.array([p[0] for p in preds]) if preds else np.zeros((0, 2))
        pairs = []
        for gi, g in enumerate(gts):
            gx, gy = g["xy"]
            for pj in range(len(pcen)):
                d2 = (pcen[pj, 0] - gx) ** 2 + (pcen[pj, 1] - gy) ** 2
                if d2 <= match_tol ** 2:
                    pairs.append((d2, gi, pj))
        pairs.sort()
        gt_used, pr_used, match = set(), set(), {}
        for d2, gi, pj in pairs:
            if gi in gt_used or pj in pr_used: continue
            gt_used.add(gi); pr_used.add(pj); match[gi] = pj
        matched = len(match)
        # classification-only (matched pairs) + end-to-end confusion
        tp = fp = fn = tn = 0          # end-to-end (positive class)
        ctp = cfp = cfn = ctn = 0      # classification-only (matched cells)
        for gi, g in enumerate(gts):
            glab = g["label"]
            if gi in match:
                plab = preds[match[gi]][1]
                if glab == 1 and plab: tp += 1; ctp += 1
                elif glab == 1 and not plab: fn += 1; cfn += 1
                elif glab == 0 and plab: fp += 1; cfp += 1
                else: tn += 1; ctn += 1
            else:                       # GT cell the pipeline never detected
                if glab == 1: fn += 1
                else: tn += 1
        for pj, (c, plab) in enumerate(preds):   # predicted+ with no GT match
            if plab and pj not in pr_used: fp += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_image[stem] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(prec, 3),
                               recall=round(rec, 3), f1=round(f1, 3),
                               gt=len(gts), matched=matched, pred=len(preds))
        for k in ("tp", "fp", "fn", "tn"): agg[k] += locals()[k]
        for k in ("ctp", "cfp", "cfn", "ctn"): agg[k] = agg.get(k, 0) + locals()[k]
        agg["gt_matched"] += matched; agg["gt_total"] += len(gts); agg["pred_total"] += len(preds)
        # render our_overlay + f1 matched overlay
        _render_overlays(cond, stem, preds, gts)
    def _prf(tp, fp, fn, tn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        a = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
        s = tn / (tn + fp) if tn + fp else 0.0
        return dict(precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                    accuracy=round(a, 3), specificity=round(s, 3),
                    TP=tp, FP=fp, FN=fn, TN=tn)
    e2e = _prf(agg["tp"], agg["fp"], agg["fn"], agg["tn"])
    clsf = _prf(agg.get("ctp", 0), agg.get("cfp", 0), agg.get("cfn", 0), agg.get("ctn", 0))
    det_recall = agg["gt_matched"] / agg["gt_total"] if agg["gt_total"] else 0.0
    det_prec = agg["gt_matched"] / agg["pred_total"] if agg["pred_total"] else 0.0
    summary = dict(condition=cond, adaptive=adaptive, images=len(per_image),
                   detection_recall=round(det_recall, 3), detection_precision=round(det_prec, 3),
                   classification_only=clsf, end_to_end=e2e, per_image=per_image)
    json.dump(summary, open(os.path.join(BASE, cond, "f1", "metrics.json"), "w"), indent=2)
    print(f"\n=== {cond}{' (adaptive)' if adaptive else ''} ===")
    print(f"  images {len(per_image)}  segmentation: detection-recall {det_recall:.3f} "
          f"detection-precision {det_prec:.3f}")
    print(f"  CLASSIFICATION-ONLY (matched cells): precision {clsf['precision']:.3f} "
          f"recall {clsf['recall']:.3f} F1 {clsf['f1']:.3f} acc {clsf['accuracy']:.3f} "
          f"spec {clsf['specificity']:.3f}")
    print(f"  END-TO-END (seg+class): precision {e2e['precision']:.3f} recall {e2e['recall']:.3f} "
          f"F1 {e2e['f1']:.3f} acc {e2e['accuracy']:.3f} spec {e2e['specificity']:.3f}")
    print(f"  end-to-end confusion: TP {e2e['TP']} FP {e2e['FP']} FN {e2e['FN']} TN {e2e['TN']}")

def _otsu(x):
    h, e = np.histogram(x, 128); cx = (e[:-1] + e[1:]) / 2
    wB = np.cumsum(h); wF = h.sum() - wB; mB = np.cumsum(h * cx); mT = mB[-1]
    with np.errstate(all="ignore"):
        var = wB * wF * ((mB / wB) - ((mT - mB) / wF)) ** 2
    var[~np.isfinite(var)] = -1
    return float(cx[np.argmax(var)])

def _render_overlays(cond, stem, preds, gts):
    inp = os.path.join(BASE, cond, "changed_inputs", stem + ".png")
    base = Image.open(inp).convert("RGB")
    ov = base.copy(); d = ImageDraw.Draw(ov)
    for (x, y), plab in preds:
        col = (255, 0, 0) if plab else (0, 90, 255)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col, width=2)
    ov.save(os.path.join(BASE, cond, "our_overlay", stem + ".png"))
    # f1 side-by-side (our | expert)
    exp = Image.open(os.path.join(BASE, cond, "expert_overlay", stem + ".png")).convert("RGB")
    sheet = Image.new("RGB", (ov.width * 2 + 30, ov.height + 26), "white")
    dd = ImageDraw.Draw(sheet)
    sheet.paste(ov, (10, 22)); sheet.paste(exp, (ov.width + 20, 22))
    dd.text((10, 6), "OUR pipeline (red=+, blue=-)", fill="black")
    dd.text((ov.width + 20, 6), "EXPERT / IF ground truth", fill="black")
    sheet.save(os.path.join(BASE, cond, "f1", stem + "_match.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "score"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cond", default="raw_instanseg")
    ap.add_argument("--adaptive", action="store_true")
    a = ap.parse_args()
    if a.stage == "prep": prep(a.n)
    elif a.stage == "score": score(a.cond, adaptive=a.adaptive)
