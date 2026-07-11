"""
calibration.py — native in-app cutoff calibration backend.

Flow: segment an image (InstanSeg) → hand-label cells positive/negative in the UI
→ fit membrane-completeness cutoffs (the same tune_membrane_threshold logic) →
save per-marker so Quant uses them. Keeps the heavy logic out of api.py.
"""
import os, io, json, base64, glob, subprocess, sys, tempfile
import numpy as np
from pathlib import Path

# Repo root: <root>/oasis/webui/calibration.py → climb three levels (webui → oasis →
# root). run_pipeline.py lives at the root; the old .parent.parent pointed at <root>/oasis
# after the restructure and broke the Calibrate tab's segmentation subprocess.
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ── views (Original / Normalized / DAB-signal), same as the label tool ──────────
def _b64(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _views(image_path):
    from PIL import Image
    from oasis.quant.cell_expansion import (_estimate_background, _estimate_stain_vectors,
                                _od_channels, _QUPATH_STAINS, _DEFAULT_BACKGROUND)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = rgb.shape[:2]
    bg = _estimate_background(rgb)
    norm = np.clip(rgb.astype(float) * (255.0 / bg.reshape(1, 1, 3)), 0, 255).astype(np.uint8)
    est = _estimate_stain_vectors(rgb, bg)
    vecs = est if est else _QUPATH_STAINS
    bg_use = bg.tolist() if est else _DEFAULT_BACKGROUND
    h, dab = _od_channels(rgb, vecs, bg_use)
    d = np.clip(dab, 0, None).astype(float)
    dom = (dab > h) & (dab > 0)
    p99 = float(np.percentile(d[dom], 99)) if dom.any() else 1.0
    inten = (np.clip(d / (p99 or 1.0), 0, 1) * dom)[..., None]
    white = np.array([245, 245, 242]); brown = np.array([110, 64, 32])
    dab_img = (white * (1 - inten) + brown * inten).astype(np.uint8)
    return int(W), int(H), {"Original": _b64(rgb), "Normalized": _b64(norm),
                            "DAB signal": _b64(dab_img)}


def _cells(geojson_path):
    gj = json.load(open(geojson_path))
    out = []
    for i, ft in enumerate(gj.get("features", [])):
        g = ft.get("geometry", {}) or {}
        c = g.get("coordinates") or []
        ring = c[0] if g.get("type") == "Polygon" and c else \
               (c[0][0] if g.get("type") == "MultiPolygon" and c and c[0] else None)
        if ring:
            out.append({"i": i, "points": [[round(float(x), 1), round(float(y), 1)]
                                           for x, y in ring]})
    return out


# ── segmentation (reuse the real pipeline) ──────────────────────────────────────
def segment(image_path, pixel_size, setup):
    """Run InstanSeg on one image via run_pipeline; return the detection GeoJSON path."""
    work = Path(tempfile.mkdtemp(prefix="ihc_calib_"))
    in_dir = work / "in"; out_dir = work / "out"
    in_dir.mkdir(); out_dir.mkdir()
    dst = in_dir / Path(image_path).name
    import shutil as _sh; _sh.copy(image_path, dst)
    ext = Path(image_path).suffix.lower()
    cfg = {
        "mode": "automated", "stain_type": "hdab",
        "input_dir": str(in_dir), "output_dir": str(out_dir),
        "dashboard_dir": str(out_dir / "_dash"),
        "qupath_binary": os.path.expanduser(setup.get("qupath_binary", "")),
        "instanseg_model": os.path.expanduser(setup.get("instanseg_model", "")),
        "device": setup.get("device", "mps"), "instanseg_threads": setup.get("instanseg_threads", 4),
        "default_pixel_size": float(pixel_size), "dab_threshold": 0.1,
        "export_geojson": True, "generate_overlays": False,
        "image_extensions": [f"*{ext}"],
    }
    import yaml
    cfg_path = work / "cfg.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    subprocess.run([sys.executable, str(PROJECT_DIR / "run_pipeline.py"),
                    "--config", str(cfg_path), "--mode", "quant"],
                   cwd=str(PROJECT_DIR), capture_output=True, timeout=1800)
    geo = glob.glob(str(out_dir / "*_detections.geojson"))
    return geo[0] if geo else None


def prepare(image_path, pixel_size, setup):
    """Segment + build views + cells for the labeling canvas."""
    geojson = segment(image_path, pixel_size, setup)
    if not geojson:
        return {"ok": False, "msg": "Segmentation produced no cells"}
    W, H, views = _views(image_path)
    return {"ok": True, "geojson": geojson, "w": W, "h": H,
            "views": views, "cells": _cells(geojson)}


# ── fit cutoffs (same statistic as tune_membrane_threshold + the DAB>H gate) ─────
def _roc_auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); o = np.argsort(-s); y = y[o]
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0: return float("nan")
    return float(np.trapezoid(np.r_[0, np.cumsum(y) / P], np.r_[0, np.cumsum(1 - y) / N]))


def _best_f1_cut(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    best_t, best = 0.0, -1.0
    for t in np.unique(s):
        p = s > t
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0
        if f > best: best, best_t = f, t
    return float(best_t), float(best)


def _measure_labeled(image_path, geojson_path, pixel_size, pos_idx, neg_idx):
    """Ring measurement for the labelled cells of ONE image → list of (label, rv, rh),
    where rv/rh are the calibrated ring DAB / hematoxylin OD pixel arrays."""
    from oasis.quant.cell_expansion import measure_cytoplasm_dab
    pos_idx = set(int(i) for i in pos_idx); neg_idx = set(int(i) for i in neg_idx)
    res = measure_cytoplasm_dab(image_path, geojson_path, float(pixel_size),
                                keep_ring_values=True)
    cells = []
    for i, r in enumerate(res):
        if not r or (i not in pos_idx and i not in neg_idx):
            continue
        vals = r.get("ring_values")
        if not vals:
            continue
        hv = r.get("ring_h_values")
        cells.append((1 if i in pos_idx else 0, np.asarray(vals, float),
                      np.asarray(hv, float) if hv else None))
    return cells


def _neg_t_pix(cells, neg_pct):
    """Pixel-OD threshold = the neg_pct percentile of all NEGATIVE ring pixels pooled."""
    neg_px = [rv for lab, rv, _rh in cells if lab == 0]
    return float(np.percentile(np.concatenate(neg_px), neg_pct)) if neg_px else 0.0


def _ring_frac(cells, t_pix):
    """Per-cell membrane-completeness fraction at pixel threshold t_pix (DAB>H gated)."""
    fr = []
    for _lab, rv, rh in cells:
        pos = rv > t_pix
        if rh is not None and len(rh) == len(rv):
            pos = pos & (rv > rh)
        fr.append(float(pos.mean()))
    return np.asarray(fr)


def _loo_f1_auc(cells, neg_pct):
    """Leave-one-CELL-out held-out F1/AUC. Each cell is scored by a threshold fit on
    all the OTHER cells (t_pix from their negatives + best-F1 ring-fraction cut), so
    the number is an UNBIASED estimate of callability — not the optimistic in-sample
    fit that uses the same cells to choose and to score the cutoff. Returns
    (loo_f1, loo_auc, n) or (None, None, n) when there are too few cells to hold out."""
    n = len(cells)
    y = np.array([c[0] for c in cells])
    if n < 4 or y.sum() < 2 or (n - y.sum()) < 2:
        return None, None, n
    held_frac = np.zeros(n)
    held_pred = np.zeros(n, int)
    for i in range(n):
        train = [cells[j] for j in range(n) if j != i]
        y_tr = np.array([c[0] for c in train])
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            return None, None, n           # a fold with one class → LOO undefined
        t = _neg_t_pix(train, neg_pct)
        cut, _ = _best_f1_cut(_ring_frac(train, t), y_tr)
        fi = _ring_frac([cells[i]], t)[0]
        held_frac[i] = fi
        held_pred[i] = 1 if fi > cut else 0
    tp = int(((held_pred == 1) & (y == 1)).sum())
    fp = int(((held_pred == 1) & (y == 0)).sum())
    fn = int(((held_pred == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return float(f1), _roc_auc(held_frac, y), n


def fit_multi(items, neg_pct=99.0):
    """Fit membrane cutoffs by POOLING hand-labelled cells across one or more images.

    Pooling several representative slides captures the antibody/scanner staining
    variability that a single field cannot, so the cutoff transfers better. The
    headline callability metric is the honest LEAVE-ONE-CELL-OUT F1/AUC; the
    in-sample AUC/F1 are also returned but are optimistic (same cells choose and
    score the cutoff). `items`: list of
    {image_path, geojson_path, pixel_size, pos_idx, neg_idx}.
    """
    cells, n_images, per_image = [], 0, []
    for it in items:
        try:
            c = _measure_labeled(it["image_path"], it["geojson_path"],
                                 it["pixel_size"], it.get("pos_idx") or [],
                                 it.get("neg_idx") or [])
        except Exception as e:
            return {"ok": False, "msg": "measurement failed on "
                    f"{os.path.basename(str(it.get('image_path', '?')))}: {e}"}
        per_image.append({"image": os.path.basename(str(it.get("image_path", "?"))),
                          "n_pos": sum(1 for x in c if x[0] == 1),
                          "n_neg": sum(1 for x in c if x[0] == 0)})
        if c:
            n_images += 1
            cells.extend(c)

    n_pos = sum(1 for c in cells if c[0] == 1)
    n_neg = sum(1 for c in cells if c[0] == 0)
    if n_pos < 5 or n_neg < 5:
        return {"ok": False, "per_image": per_image,
                "msg": f"Too few labelled cells with a valid ring (pooled {n_pos} pos "
                       f"/ {n_neg} neg across {n_images} image(s); need ≥5 each)."}

    # Global cutoffs (what Quant will use), fit on ALL pooled cells.
    t_pix = _neg_t_pix(cells, neg_pct)
    frac = _ring_frac(cells, t_pix)
    y = np.array([c[0] for c in cells])
    auc = _roc_auc(frac, y)
    cut, f1 = _best_f1_cut(frac, y)
    loo_f1, loo_auc, _n = _loo_f1_auc(cells, neg_pct)

    judge_auc = loo_auc if loo_auc is not None else auc     # judge on held-out if we can
    return {"ok": True,
            "membrane_pix_thr": round(t_pix, 4), "membrane_frac_min": round(cut, 4),
            "auc": round(auc, 3), "f1": round(f1, 3),
            "loo_auc": (round(loo_auc, 3) if loo_auc is not None else None),
            "loo_f1": (round(loo_f1, 3) if loo_f1 is not None else None),
            "n_pos": n_pos, "n_neg": n_neg, "n_images": n_images,
            "per_image": per_image,
            "callable": bool(judge_auc is not None and judge_auc >= 0.75)}


def fit(image_path, geojson_path, pixel_size, pos_idx, neg_idx, neg_pct=99.0):
    """Single-image cutoff fit — backward-compatible wrapper over fit_multi."""
    return fit_multi([{"image_path": image_path, "geojson_path": geojson_path,
                       "pixel_size": pixel_size, "pos_idx": pos_idx,
                       "neg_idx": neg_idx}], neg_pct=neg_pct)
