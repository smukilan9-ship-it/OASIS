#!/usr/bin/env python3
"""
Membranous-marker validation of the hardened ring/completeness method on REAL
CD8 tissue with IF-DERIVED per-cell ground truth (HNSCC-mIF-mIHC-comparison v2).

Same principle as the DeepLIIF validation (Phase 2), but for a MEMBRANOUS marker:
  - input        : chromogenic CD8 (mIHC_Data/*_CD8.png)  ← our method runs here
  - ground truth : CD8 immunofluorescence (mIF_Data/*_CD8.png), co-registered
  - cells        : expert nuclear masks (Segmentation/*.png; blue=nucleus,
                   green=boundary) → connected components on blue
Per-cell CD8+ label is derived from the membranous CD8-IF in each cell's ring —
DeepLIIF's ground truth is likewise IF-derived, so this is the same kind of label,
not a weaker proxy.

Caveat (honest): the chromogen is AEC, not DAB. This validates the membranous
METHOD — per-image stain-vector estimation (Macenko, marker vs hematoxylin), the
marker-dominance gate, and ring completeness — on a real membranous immune marker.
The DAB-specific constants in cell_expansion stay validated via CRC-ICM + nuclear
DeepLIIF (Ki67).

Usage: python validation/validate_membrane_cd8_hnscc.py [n_tiles]
"""
import os, sys, glob, math
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.quant.cell_expansion import _estimate_background, _norm_vec, _REF_H

ROOT = os.path.expanduser("~/PKG - HNSCC-mIF-mIHC-comparison_v2")
RING_PX = 4            # ~2 µm ring at ~0.5 µm/px
GT_IF_FRAC = 0.10      # cell is CD8+ if >=10% of its ring shows CD8-IF above Otsu


def _estimate_marker_vectors(rgb, bg):
    """Macenko stain estimation, assigning marker = the NON-hematoxylin vector
    (stain-agnostic: works for AEC and DAB). Returns (hem_vec, marker_vec) or None."""
    rgb = np.asarray(rgb, float)[..., :3]
    od = -np.log10((rgb.reshape(-1, 3) + 1.0) / np.asarray(bg, float).reshape(1, 3))
    tis = od[od.sum(1) > 0.15]
    if len(tis) < 1000:
        return None
    cov = np.cov(tis.T); w, V = np.linalg.eigh(cov)
    plane = V[:, [2, 1]].astype(float)
    if plane[:, 0] @ tis.mean(0) < 0:
        plane[:, 0] = -plane[:, 0]
    proj = tis @ plane; ang = np.arctan2(proj[:, 1], proj[:, 0])
    a1, a2 = np.percentile(ang, 1.0), np.percentile(ang, 99.0)

    def pos(v):
        v = np.asarray(v, float); return _norm_vec(-v if v.sum() < 0 else v)
    v1 = pos(plane @ np.array([np.cos(a1), np.sin(a1)]))
    v2 = pos(plane @ np.array([np.cos(a2), np.sin(a2)]))
    # hematoxylin = vector closer to the H reference; the OTHER is the marker.
    hem, mark = (v1, v2) if (v1 @ _REF_H) >= (v2 @ _REF_H) else (v2, v1)
    if (hem @ _REF_H) < 0.6 or (hem @ mark) > 0.95:
        return None
    return hem, mark


def _od_two(rgb, hem, mark, bg):
    """Deconvolve to (hematoxylin_od, marker_od) with a 2-stain pseudo-inverse."""
    od = -np.log10((np.asarray(rgb, float)[..., :3] + 1.0) /
                   np.asarray(bg, float).reshape(1, 1, 3))
    conc = od @ np.linalg.pinv(np.array([hem, mark]))
    return conc[..., 0].astype(np.float32), conc[..., 1].astype(np.float32)


def _otsu(x):
    x = x[np.isfinite(x)]
    if len(x) < 10: return float("inf")
    h, e = np.histogram(x, 128); cx = (e[:-1] + e[1:]) / 2
    wB = np.cumsum(h); wF = h.sum() - wB; mB = np.cumsum(h * cx); mT = mB[-1]
    with np.errstate(all="ignore"):
        var = wB * wF * ((mB / wB) - ((mT - mB) / wF)) ** 2
    var[~np.isfinite(var)] = -1
    return float(cx[np.argmax(var)])


def _roc_auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    o = np.argsort(-s); y = y[o]; P = y.sum(); N = len(y) - P
    if P == 0 or N == 0: return float("nan")
    tpr = np.cumsum(y) / P; fpr = np.cumsum(1 - y) / N
    return float(np.trapezoid(np.concatenate([[0], tpr]), np.concatenate([[0], fpr])))


def _prf(s, y, thr):
    pred = np.asarray(s) > thr; y = np.asarray(y).astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    a = (tp + tn) / len(y); sp = tn / (tn + fp) if tn + fp else 0.0
    return dict(precision=p, recall=r, f1=f, accuracy=a, specificity=sp,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    n_tiles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    samples = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(ROOT, "Segmentation", "*.png")))
    samples = samples[:n_tiles]
    print(f"CD8 membranous validation on {len(samples)} HNSCC tiles "
          f"(chromogenic CD8 vs CD8-IF ground truth)...")

    recs = []   # (tile_idx, pred_frac, gt_label)
    skipped = 0
    for ti, s in enumerate(samples):
        try:
            mihc = np.asarray(Image.open(os.path.join(ROOT, "mIHC_Data", f"{s}_CD8.png")).convert("RGB"))
            mif  = np.asarray(Image.open(os.path.join(ROOT, "mIF_Data",  f"{s}_CD8.png")).convert("L")).astype(float)
            seg  = np.asarray(Image.open(os.path.join(ROOT, "Segmentation", f"{s}.png")).convert("RGB"))
        except Exception:
            skipped += 1; continue
        nuc_mask = (seg[..., 2] > 150) & (seg[..., 0] < 100) & (seg[..., 1] < 100)  # blue interiors
        lab, n = ndimage.label(nuc_mask)
        if n < 5:
            skipped += 1; continue
        bg = _estimate_background(mihc)
        vecs = _estimate_marker_vectors(mihc, bg)
        if vecs is None:
            skipped += 1; continue
        hem_od, mark_od = _od_two(mihc, vecs[0], vecs[1], bg)
        # per-image marker-positive pixel threshold (Otsu over marker-dominant tissue)
        dom = (mark_od > hem_od) & (mark_od > 0)
        mthr = _otsu(mark_od[dom]) if dom.sum() > 200 else np.inf
        mif_otsu = _otsu(mif[mif > 0])
        all_nuc = nuc_mask
        for i in range(1, n + 1):
            nucd = lab == i
            ring = ndimage.binary_dilation(nucd, iterations=RING_PX) & ~nucd & ~(all_nuc & ~nucd)
            if ring.sum() < 10:
                continue
            # OUR chromogenic prediction feature: ring completeness with marker-dominance gate
            mk = mark_od[ring]; hh = hem_od[ring]
            pred_frac = float(((mk > mthr) & (mk > hh)).mean())
            # IF-derived ground truth: membranous CD8-IF completeness in the ring
            gt_frac = float((mif[ring] > mif_otsu).mean())
            recs.append((ti, pred_frac, gt_frac >= GT_IF_FRAC))

    if not recs:
        sys.exit("No cells measured — check dataset path / formats.")
    idx = np.array([r[0] for r in recs]); pred = np.array([r[1] for r in recs])
    y = np.array([r[2] for r in recs], int)
    fit = idx % 2 == 0; hold = ~fit
    # fit the completeness cutoff by max-F1 on the fit split
    cand = np.unique(pred[fit])
    best_t, best = cand[0] if len(cand) else 0.0, -1
    for t in cand:
        f = _prf(pred[fit], y[fit], t)["f1"]
        if f > best: best, best_t = f, t
    m = _prf(pred[hold], y[hold], best_t); auc = _roc_auc(pred[hold], y[hold])
    print(f"  cells: {len(recs)}  CD8+ (IF) {int(y.sum())} ({100*y.mean():.1f}%)  "
          f"skipped tiles {skipped}")
    print(f"  fitted completeness cutoff (max-F1 on fit split) = {best_t:.3f}")
    print(f"\n  === HELD-OUT per-cell metrics (chromogenic CD8 vs CD8-IF truth) ===")
    print(f"    precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
    print(f"    accuracy  {m['accuracy']:.3f}  specificity {m['specificity']:.3f}  ROC-AUC {auc:.3f}")
    print(f"    confusion TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}")
    print(f"    positivity: IF-truth {100*y[hold].mean():.1f}%  vs  predicted "
          f"{100*(pred[hold]>best_t).mean():.1f}%")


if __name__ == "__main__":
    main()
