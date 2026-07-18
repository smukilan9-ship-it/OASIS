"""
Score InstanSeg and DeepLIIF nuclear segmentation vs HNSCC expert masks.

Mask convention (verified visually, same for GT and DeepLIIF):
  blue = nucleus interior, red = marker+ nucleus interior, green = separating
  ring/boundary, black = background. Green rings cleanly split touching nuclei,
  so connected components of (blue|red) = individual nucleus instances.

GT (expert) uses blue-filled nuclei separated by green. InstanSeg gives polygons.
The hematoxylin image and every mask are the SAME registered 512x512 tile, so no
alignment transform is needed. "Allow for some shifts" = centroid-match tolerance
(detection F1) + a dilation tolerance (pixel F1).

Both detectors scored identically against the same GT.
"""
import os, json, glob, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

GT_DIR = os.path.expanduser("~/Desktop/HNSCC_raw_dataset/inputs/Segmentation")
IS_DIR = os.path.expanduser("~/Desktop/HNSCC_seg_comparison/instanseg_segmentation")
DL_DIR = os.path.expanduser("~/Desktop/HNSCC_seg_comparison/deeplif_segmentation")
MIN_COMP = 20            # px, drop specks
TOLS = [6, 10, 15]      # centroid-match tolerances (px)
PIX_TOL = 3             # px dilation for tolerant pixel F1

def _rgb(p):
    return np.array(Image.open(p).convert("RGB")).astype(int)

def nucleus_mask(im, blue_only=False):
    """Interior nuclei = blue-dominant (|red-dominant). Excludes green rings + black bg."""
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    blue = (b > r + 20) & (b > g + 20) & (b > 60)
    if blue_only:
        return blue
    red = (r > b + 20) & (r > g + 20) & (r > 60)
    return blue | red

def instances(mask, min_size=MIN_COMP):
    L, n = ndimage.label(mask)
    cens = []
    for i in range(1, n + 1):
        ys, xs = np.where(L == i)
        if len(xs) >= min_size:
            cens.append((float(xs.mean()), float(ys.mean())))
    return cens

def gt_load(stem):
    p = os.path.join(GT_DIR, stem + ".png")
    return _rgb(p) if os.path.exists(p) else None

def instanseg_polys(stem):
    g = glob.glob(os.path.join(IS_DIR, stem + "*_detections.geojson"))
    if not g: return None
    return json.load(open(g[0])).get("features", [])

def instanseg_centroids(feats):
    out = []
    for ft in feats:
        gm = ft.get("geometry", {})
        if gm.get("type") == "Polygon" and gm.get("coordinates"):
            rr = np.asarray(gm["coordinates"][0], float)
            out.append((float(rr[:, 0].mean()), float(rr[:, 1].mean())))
    return out

def instanseg_fgmask(feats, shape=(512, 512)):
    img = Image.new("L", (shape[1], shape[0]), 0); d = ImageDraw.Draw(img)
    for ft in feats:
        gm = ft.get("geometry", {})
        if gm.get("type") == "Polygon" and gm.get("coordinates"):
            pts = [(float(x), float(y)) for x, y in gm["coordinates"][0]]
            if len(pts) >= 3: d.polygon(pts, fill=255)
    return np.array(img) > 127

def deepliif_seg(stem):
    g = glob.glob(os.path.join(DL_DIR, stem + "*_Seg.png"))
    return _rgb(g[0]) if g else None

def match(gt_cen, pred_cen, tol):
    pc = np.array(pred_cen) if pred_cen else np.zeros((0, 2))
    pairs = []
    for gi, (gx, gy) in enumerate(gt_cen):
        for pj in range(len(pc)):
            d2 = (pc[pj, 0] - gx) ** 2 + (pc[pj, 1] - gy) ** 2
            if d2 <= tol * tol: pairs.append((d2, gi, pj))
    pairs.sort(); gu, pu = set(), set()
    for d2, gi, pj in pairs:
        if gi in gu or pj in pu: continue
        gu.add(gi); pu.add(pj)
    return len(gu)

def tol_pixel_f1(gt, pr, tol=PIX_TOL):
    gtd = ndimage.binary_dilation(gt, iterations=tol)
    prd = ndimage.binary_dilation(pr, iterations=tol)
    tp_p = np.logical_and(pr, gtd).sum(); prec = tp_p / pr.sum() if pr.sum() else 0
    tp_r = np.logical_and(gt, prd).sum(); rec = tp_r / gt.sum() if gt.sum() else 0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def run():
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(GT_DIR + "/*.png"))
    n_is = len(glob.glob(IS_DIR + "/*_detections.geojson"))
    n_dl = len(glob.glob(DL_DIR + "/*_Seg.png"))
    print(f"GT tiles {len(stems)} | InstanSeg geojson {n_is} | DeepLIIF Seg {n_dl}\n")

    rows = {}
    for name in ("InstanSeg", "DeepLIIF"):
        agg = {t: [0, 0, 0] for t in TOLS}; pixf = []; nimg = 0; miss = 0
        gt_tot = pred_tot = 0
        for stem in stems:
            gtim = gt_load(stem)
            if gtim is None: continue
            gt_cen = instances(nucleus_mask(gtim, blue_only=True))
            gt_fg = nucleus_mask(gtim, blue_only=True)
            if name == "InstanSeg":
                feats = instanseg_polys(stem)
                if feats is None: miss += 1; continue
                pred_cen = instanseg_centroids(feats); pred_fg = instanseg_fgmask(feats, gtim.shape[:2])
            else:
                seg = deepliif_seg(stem)
                if seg is None: miss += 1; continue
                pred_cen = instances(nucleus_mask(seg)); pred_fg = nucleus_mask(seg)
            nimg += 1; gt_tot += len(gt_cen); pred_tot += len(pred_cen)
            for t in TOLS:
                m = match(gt_cen, pred_cen, t)
                agg[t][0] += m; agg[t][1] += len(gt_cen); agg[t][2] += len(pred_cen)
            pixf.append(tol_pixel_f1(gt_fg, pred_fg))
        rows[name] = (agg, np.mean(pixf) if pixf else 0, nimg, miss, gt_tot, pred_tot)

    for name, (agg, pf, nimg, miss, gt_tot, pred_tot) in rows.items():
        print(f"=== {name}  (scored {nimg} tiles, {miss} missing; GT nuclei {gt_tot}, pred {pred_tot}, "
              f"pred/GT ratio {pred_tot/gt_tot:.2f}) ===")
        for t in TOLS:
            mtc, g, p = agg[t]
            rec = mtc / g if g else 0; prec = mtc / p if p else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            print(f"  tol {t:2d}px: det-recall {rec:.3f}  det-precision {prec:.3f}  det-F1 {f1:.3f}")
        print(f"  tolerant pixel-F1 (±{PIX_TOL}px): {pf:.3f}\n")

if __name__ == "__main__":
    run()
