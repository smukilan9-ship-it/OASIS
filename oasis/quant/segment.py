"""
segment.py — in-process nuclear segmentation, replacing the QuPath/Groovy subprocess.

WHY THIS EXISTS. Segmentation was the last hard dependency on an external application: the
pipeline generated a Groovy script, launched the QuPath binary, and parsed the files it wrote.
That makes OASIS unshippable as a standalone executable (QuPath is a ~400 MB JVM application the
user must install and configure separately) and makes the numbers depend on a version of QuPath
we do not pin.

Nothing in that arrangement was load-bearing. The InstanSeg `brightfield_nuclei` model is a
TorchScript bundle; loaded directly it reproduces the reference output shipped with the model
EXACTLY (336/336 labels, foreground IoU 1.0 — see validation/validate_native_segmenter.py).
QuPath's contribution around it is image reading, resampling, tiling, colour deconvolution and
export, all of which this module does in Python against the same conventions.

WHAT QuPATH DID THAT THIS MUST ALSO DO. These are the non-obvious ones; each is implemented
below and pinned by the parity harness:

  1. RESAMPLE TO THE MODEL'S PIXEL SIZE. The model is trained at a fixed resolution declared in
     rdf.yaml (`scale: 0.5` µm/px). QuPath's InstanSeg extension resamples the image to it via
     `InstanSegModel.getPreferredDownsample(cal)` — the image is NOT fed at its native pixel
     size. Skipping this silently changes nucleus size in model space and therefore the counts.
  2. TILE WITH OVERLAP, THEN RECONCILE. QuPath requests 512x512 tiles with 32 px of inter-tile
     padding. The padding is context, not output: an object must be attributed to exactly one
     tile or it is double-counted at every seam. Here each object is assigned to the tile whose
     CORE (unpadded) region contains its centroid, which is duplicate-free by construction.
  3. COLOUR DECONVOLUTION IN QuPath's CONVENTION. `setImageType('BRIGHTFIELD_H_DAB')` installs
     fixed H-DAB stain vectors and a 255 white point; "DAB: Mean" is the per-object mean of the
     deconvolved DAB optical density. Reproduced in `_od_channels` (parity: r=0.9985,
     slope 0.994, MAE 0.0037 OD over 800 nuclei against a real QuPath export).
  4. MEASUREMENTS AND EXPORT SHAPE. Downstream code reads a specific set of names out of the
     GeoJSON and the tab-delimited CSV. Those names are reproduced verbatim; see `_measure`.

The pixel size is supplied by the caller and is authoritative — as it already was, since the
Groovy called `setPixelSizeMicrons` with OASIS's resolved value rather than trusting metadata.
"""
import json
import os

import numpy as np

# Model geometry. QuPath's defaults, matched so counts are comparable to the QuPath era.
TILE = 512
PADDING = 32
MIN_TILE = 32          # rdf.yaml: spatial axes have min size 32

# QuPath's fixed BRIGHTFIELD_H_DAB stain vectors and white point. Not estimated: the Groovy
# never called estimateStainVectors, so these are exactly what produced the reference numbers.
QUPATH_HEMATOXYLIN = (0.651, 0.701, 0.29)
QUPATH_DAB = (0.269, 0.568, 0.778)
QUPATH_WHITE = (255.0, 255.0, 255.0)

_MODEL_CACHE = {}


def _norm_vec(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def model_pixel_size(model_dir):
    """The model's trained pixel size in µm, read from rdf.yaml (mean of the x and y spatial
    scales). Returns None if it cannot be determined, in which case no resampling is done."""
    rdf = os.path.join(os.path.expanduser(model_dir), "rdf.yaml")
    if not os.path.exists(rdf):
        return None
    try:
        import yaml
        with open(rdf) as f:
            spec = yaml.safe_load(f)
        scales = []
        for tensor in spec.get("inputs", []):
            for axis in tensor.get("axes", []):
                if axis.get("type") == "space" and axis.get("scale"):
                    scales.append(float(axis["scale"]))
        return float(np.mean(scales)) if scales else None
    except Exception:
        return None


def preferred_downsample(image_px_um, model_px_um, snap=0.1):
    """QuPath's `getPreferredDownsample`: the factor to reduce the image by so it reaches the
    model's trained resolution, snapped to a whole number when it is already close.

    QuPath snaps (its javadoc: a computed 2.04 may be returned as 2.0) to avoid interpolation
    artefacts and needless floating-point vertices in the output. `snap` is the tolerance for
    that rounding.

    CLAMPED AT 1.0 — never upsample. When the image is COARSER than the model's trained
    resolution the arithmetic asks for a downsample below 1, i.e. interpolating the image up to
    manufacture detail that was never scanned. Measured on LL477 (0.752 µm/px, so the raw factor
    is 0.665): upsampling yields 6719 nuclei against QuPath's 4796 (ratio 1.40) because the
    model, seeing enlarged nuclei, splits them; at downsample 1.0 it yields 5380 (ratio 1.12).
    QuPath clamps the same way, and it is the conservative choice regardless — inventing
    resolution to feed a segmenter inflates counts.
    """
    if not image_px_um or not model_px_um or image_px_um <= 0 or model_px_um <= 0:
        return 1.0
    ds = float(model_px_um) / float(image_px_um)
    if ds <= 1.0:
        return 1.0
    nearest = round(ds)
    if abs(ds - nearest) <= snap:
        return float(nearest)
    return ds


def load_model(model_dir, device="cpu"):
    """Load (and cache) the InstanSeg TorchScript bundle. `model_dir` is the same directory the
    config's `instanseg_model` points at — the one holding instanseg.pt and rdf.yaml."""
    import torch
    model_dir = os.path.expanduser(model_dir)
    pt = model_dir if model_dir.endswith(".pt") else os.path.join(model_dir, "instanseg.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError(f"InstanSeg model not found: {pt}")
    key = (pt, device)
    if key not in _MODEL_CACHE:
        m = torch.jit.load(pt, map_location=device)
        m.eval()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _normalize(tile_rgb):
    """rdf.yaml `scale_range` preprocessing: per-channel percentile normalisation over the
    spatial axes, 0.1 / 99.9, eps 1e-6. Verified to reproduce the model's shipped reference
    output exactly."""
    x = np.asarray(tile_rgb, dtype=np.float32).transpose(2, 0, 1)[None]
    for c in range(x.shape[1]):
        ch = x[0, c]
        lo = np.percentile(ch, 0.1)
        hi = np.percentile(ch, 99.9)
        x[0, c] = (ch - lo) / (hi - lo + 1e-6)
    return x


def _infer(model, tile_rgb, device="cpu"):
    """Run the model on one RGB tile, returning an integer label mask."""
    import torch
    x = torch.from_numpy(_normalize(tile_rgb))
    if device != "cpu":
        x = x.to(device)
    with torch.no_grad():
        y = model(x)
    y = y[0] if isinstance(y, (tuple, list)) else y
    return np.asarray(y.cpu().numpy().squeeze(), dtype=np.int32)


def _tile_grid(h, w, tile=TILE, pad=PADDING):
    """Tiles covering (h, w). Each entry is (core, padded) as (y0, y1, x0, x1). The core tiles
    partition the image exactly; the padded ones overlap and supply context."""
    out = []
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            out.append(((y0, y1, x0, x1), (py0, py1, px0, px1)))
    return out


def segment_labels(rgb, model, device="cpu", tile=TILE, pad=PADDING, progress=None):
    """Tile, infer, and reconcile into one whole-image label mask.

    Reconciliation rule: an object belongs to the tile whose CORE region contains its centroid.
    An object straddling a seam is therefore emitted once, by exactly one tile, and always by a
    tile that saw it with padding context on that side. Objects whose centroid falls outside the
    core are dropped here because the neighbouring tile owns them.
    """
    import cv2
    h, w = rgb.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    tiles = _tile_grid(h, w, tile, pad)
    for i, ((y0, y1, x0, x1), (py0, py1, px0, px1)) in enumerate(tiles):
        if progress:
            progress(i + 1, len(tiles))
        sub = rgb[py0:py1, px0:px1]
        if sub.shape[0] < MIN_TILE or sub.shape[1] < MIN_TILE:
            ph = max(MIN_TILE - sub.shape[0], 0)
            pw = max(MIN_TILE - sub.shape[1], 0)
            sub = cv2.copyMakeBorder(sub, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        lab = _infer(model, sub, device)
        lab = lab[:py1 - py0, :px1 - px0]
        ids = np.unique(lab)
        for oid in ids:
            if oid == 0:
                continue
            m = (lab == oid)
            ys, xs = np.nonzero(m)
            cy = ys.mean() + py0
            cx = xs.mean() + px0
            # core ownership — the half-open convention makes the cores a true partition
            if not (y0 <= cy < y1 and x0 <= cx < x1):
                continue
            labels[py0:py1, px0:px1][m] = next_id
            next_id += 1
    return labels


def _od_channels(rgb, background=QUPATH_WHITE,
                 hematoxylin=QUPATH_HEMATOXYLIN, dab=QUPATH_DAB):
    """Per-pixel hematoxylin and DAB optical density by colour deconvolution, QuPath's
    convention:  OD = -log10((I + 1) / white);  [H, DAB, residual] = OD . inv(stain_matrix).

    The residual (third) vector is the cross product of the two stains, which is what QuPath
    uses when only two are specified. Returns (hematoxylin_od, dab_od) as float32 HxW.
    """
    rgb = np.asarray(rgb, dtype=np.float64)[..., :3]
    bg = np.asarray(background, dtype=np.float64).reshape(1, 1, 3)
    od = -np.log10((rgb + 1.0) / bg)
    h = _norm_vec(hematoxylin)
    d = _norm_vec(dab)
    M = np.array([h, d, _norm_vec(np.cross(h, d))])
    Minv = np.linalg.inv(M)
    return (od @ Minv[:, 0]).astype(np.float32), (od @ Minv[:, 1]).astype(np.float32)


def _polygon(mask_sub, x0, y0):
    """Outer contour of one object as a closed polygon in full-image pixel coords."""
    import cv2
    cnts, _ = cv2.findContours(mask_sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    c[:, 0] += x0
    c[:, 1] += y0
    if len(c) < 3:
        return None
    return np.vstack([c, c[:1]])       # GeoJSON rings are closed


def _shape_measurements(mask_sub, contour, px_um):
    """Area / perimeter / circularity / solidity / calliper diameters, in µm, matching the
    QuPath measurement names the rest of the pipeline reads."""
    import cv2
    area_px = float(mask_sub.sum())
    c = np.asarray(contour[:-1], dtype=np.float32)
    perim_px = float(cv2.arcLength(c.reshape(-1, 1, 2), True))
    hull = cv2.convexHull(c.reshape(-1, 1, 2))
    hull_area = float(cv2.contourArea(hull))
    solidity = (area_px / hull_area) if hull_area > 0 else 0.0
    circ = (4.0 * np.pi * area_px / (perim_px ** 2)) if perim_px > 0 else 0.0
    if len(c) >= 5:
        (_cx, _cy), (d1, d2), _ang = cv2.minAreaRect(c.reshape(-1, 1, 2))
        dmax, dmin = max(d1, d2), min(d1, d2)
    else:
        dmax = dmin = float(np.sqrt(area_px))
    return {
        "Area µm^2": area_px * px_um * px_um,
        "Length µm": perim_px * px_um,
        "Circularity": min(circ, 1.0),
        "Solidity": solidity,
        "Max diameter µm": dmax * px_um,
        "Min diameter µm": dmin * px_um,
    }


def _stain_measurements(values, name):
    """QuPath emits Mean/Median/Min/Max/Std.Dev. per stain channel per object."""
    return {
        f"{name}: Mean": float(np.mean(values)),
        f"{name}: Median": float(np.median(values)),
        f"{name}: Min": float(np.min(values)),
        f"{name}: Max": float(np.max(values)),
        f"{name}: Std.Dev.": float(np.std(values)),
    }


def _measure(labels, hem_od, dab_od, px_um):
    """One record per object: polygon, centroid, shape and both stain channels."""
    from scipy import ndimage
    out = []
    n = int(labels.max())
    if n == 0:
        return out
    slices = ndimage.find_objects(labels)
    for oid in range(1, n + 1):
        sl = slices[oid - 1]
        if sl is None:
            continue
        ys, xs = sl
        sub = (labels[ys, xs] == oid)
        if sub.sum() < 3:
            continue
        poly = _polygon(sub, xs.start, ys.start)
        if poly is None:
            continue
        yy, xx = np.nonzero(sub)
        cx = (xx.mean() + xs.start)
        cy = (yy.mean() + ys.start)
        meas = {}
        meas.update(_shape_measurements(sub, poly, px_um))
        meas.update(_stain_measurements(hem_od[ys, xs][sub], "Hematoxylin"))
        meas.update(_stain_measurements(dab_od[ys, xs][sub], "DAB"))
        out.append({
            "id": oid,
            "polygon": poly,
            "centroid_px": (cx, cy),
            "centroid_um": (cx * px_um, cy * px_um),
            "measurements": meas,
        })
    return out


def otsu_threshold(values, nbins=256):
    """Otsu's cut over the object DAB means. Ported from the Groovy the pipeline generated, bin
    for bin, so the adaptive-threshold mode keeps producing the same cut it used to."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size < 20:
        return None
    mx = v.max()
    if mx <= 0:
        mx = 1e-6
    bins = np.clip(np.round(v / mx * (nbins - 1)).astype(int), 0, nbins - 1)
    hist = np.bincount(bins, minlength=nbins).astype(np.float64)
    idx = np.arange(nbins, dtype=np.float64)
    total = v.size
    sum_all = float((idx * hist).sum())
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    thr_bin = 0
    for i in range(nbins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += idx[i] * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            thr_bin = i
    return thr_bin / (nbins - 1) * mx


def segment_image(image_path, model_dir, pixel_size_um, device="cpu",
                  dab_threshold=0.2, adaptive_threshold=False, rgb=None,
                  progress=None):
    """Segment one image and measure every nucleus.

    Returns {records, labels, threshold, threshold_method, pixel_size_um, width, height,
    downsample}. Coordinates in `records` are FULL-RESOLUTION image pixels regardless of the
    resampling used for inference.
    """
    import cv2
    if rgb is None:
        from oasis.quant.cell_expansion import _load_rgb_full
        rgb = _load_rgb_full(os.path.expanduser(image_path))
    rgb = np.asarray(rgb)[..., :3]
    h, w = rgb.shape[:2]

    # 1. resample to the model's trained resolution (see module docstring, item 1)
    mpx = model_pixel_size(model_dir)
    ds = preferred_downsample(pixel_size_um, mpx)
    if abs(ds - 1.0) > 1e-6:
        # ds > 1 means the image is finer than the model wants -> shrink by ds.
        interp = cv2.INTER_AREA if ds > 1 else cv2.INTER_LINEAR
        work = cv2.resize(rgb, (max(int(round(w / ds)), 1), max(int(round(h / ds)), 1)),
                          interpolation=interp)
    else:
        work = rgb

    # 2. tile, infer, reconcile
    model = load_model(model_dir, device)
    labels_work = segment_labels(work, model, device=device, progress=progress)

    # back to full resolution — nearest keeps label identity
    if labels_work.shape[:2] != (h, w):
        labels = cv2.resize(labels_work, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        labels = labels_work

    # 3. measure on the ORIGINAL pixels, so stain values are not resampling artefacts
    hem_od, dab_od = _od_channels(rgb)
    records = _measure(labels, hem_od, dab_od, float(pixel_size_um))

    # 4. classify
    method = "fixed"
    thr = float(dab_threshold)
    if adaptive_threshold:
        auto = otsu_threshold([r["measurements"]["DAB: Mean"] for r in records])
        if auto is not None:
            thr, method = float(auto), "adaptive_otsu"
        else:
            method = "fixed_fallback_few_cells"
    for r in records:
        r["classification"] = "Positive" if r["measurements"]["DAB: Mean"] > thr else "Negative"

    return {"records": records, "labels": labels, "threshold": thr,
            "threshold_method": method, "pixel_size_um": float(pixel_size_um),
            "width": w, "height": h, "downsample": ds}


# ── exports, in the shapes the rest of the pipeline already reads ────────────────────────

_CSV_COLUMNS = [
    "Image", "Object ID", "Object type", "Name", "Classification", "Parent", "ROI",
    "Centroid X µm", "Centroid Y µm",
    "Area µm^2", "Length µm", "Circularity", "Solidity",
    "Max diameter µm", "Min diameter µm",
    "Hematoxylin: Mean", "Hematoxylin: Median", "Hematoxylin: Min",
    "Hematoxylin: Max", "Hematoxylin: Std.Dev.",
    "DAB: Mean", "DAB: Median", "DAB: Min", "DAB: Max", "DAB: Std.Dev.",
]

_CLASS_COLORS = {"Positive": [200, 100, 40], "Negative": [112, 112, 225]}


def write_geojson(result, path, image_name=""):
    """FEATURE_COLLECTION of detections. `cell_expansion` reads `properties.measurements`
    ("DAB: Mean") and the polygon geometry out of this file."""
    feats = []
    for r in result["records"]:
        feats.append({
            "type": "Feature",
            "id": str(r["id"]),
            "geometry": {"type": "Polygon",
                         "coordinates": [[[float(x), float(y)] for x, y in r["polygon"]]]},
            "properties": {
                "objectType": "detection",
                "classification": {"name": r["classification"],
                                   "color": _CLASS_COLORS.get(r["classification"],
                                                              [128, 128, 128])},
                "measurements": r["measurements"],
            },
        })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return path


def write_detections_csv(result, path, image_name=""):
    """Tab-delimited detection table. `spatial.py` reads 'Centroid X µm' / 'Centroid Y µm' and
    'Classification' from it."""
    import csv
    with open(path, "w", newline="") as f:
        wr = csv.writer(f, delimiter="\t", lineterminator="\n")
        wr.writerow(_CSV_COLUMNS)
        for r in result["records"]:
            m = r["measurements"]
            row = [image_name, str(r["id"]), "Detection", "", r["classification"], "", "",
                   r["centroid_um"][0], r["centroid_um"][1]]
            row += [m.get(c, "") for c in _CSV_COLUMNS[9:]]
            wr.writerow(row)
    return path


def write_summary(result, path, image_name="", extra=None):
    """The per-image summary JSON `parse_qupath_output` consumes."""
    recs = result["records"]
    total = len(recs)
    pos = sum(1 for r in recs if r["classification"] == "Positive")
    summ = {
        "image": image_name,
        "pixel_size_um": result["pixel_size_um"],
        "image_width": result["width"],
        "image_height": result["height"],
        "total_cells": total,
        "positive_cells": pos,
        "negative_cells": total - pos,
        "positivity_pct": round(pos * 100.0 / total, 2) if total else 0.0,
        "dab_threshold": round(result["threshold"], 4),
        "dab_threshold_method": result["threshold_method"],
        "segmenter": "instanseg_native",
        "segmenter_downsample": result["downsample"],
    }
    if extra:
        summ.update(extra)
    with open(path, "w") as f:
        json.dump(summ, f, indent=4)
    return path
