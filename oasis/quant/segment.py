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


def _normalize(rgb):
    """rdf.yaml `scale_range` preprocessing: per-channel percentile normalisation over the
    spatial axes, 0.1 / 99.9, eps 1e-6. Verified to reproduce the model's shipped reference
    output exactly.

    APPLIED ONCE OVER THE WHOLE IMAGE, NOT PER TILE. Normalising each tile against its own
    percentiles makes the model's input — and therefore the cell count — depend on the tile
    size, which is an arbitrary performance knob. Measured on LL477: per-tile normalisation
    gives 5396 objects at 256 px, 5379 at 512 px and 4657 as a single tile (a 16% spread),
    while normalising globally gives 4611 / 4632 / 4651 / 4657 — a 1% spread, and closer to
    QuPath's 4796. A tissue-density statistic must not move when someone changes a tiling
    parameter for speed.

    Returns NCHW float32, batch size 1.
    """
    return _apply_norm(rgb, norm_range(rgb))


# Below this 0.1-99.9 intensity span (any channel) an image carries no usable contrast and
# normalisation would stretch sensor noise across the model's full input range. Measured
# consequence on a 99.95%-background ACROBAT crop: 3926 "nuclei" out of pure noise.
#
# The floor is set from the validated corpus, not by taste: the minimum channel span over all
# 598 DeepLIIF panels is 98 and LL477's is 144, while the blank crop is 10 and a whole ACROBAT
# slide is ~99. A floor of 40 therefore cannot alter any validated result (2.5x margin) and
# still catches the degenerate case by 4x.
MIN_DYNAMIC_RANGE = 40.0


def low_contrast(ranges):
    """True when no channel has enough dynamic range for normalisation to be meaningful."""
    return any((hi - lo) < MIN_DYNAMIC_RANGE for lo, hi in ranges)


def norm_range(rgb):
    """The per-channel (lo, hi) percentile pair `_normalize` uses. Split out so a whole-slide
    run can compute the pair ONCE — from a downsampled level it can afford to hold — and then
    apply the same constants to every streamed tile, preserving global normalisation without
    ever materialising the full-resolution image."""
    a = np.asarray(rgb)
    return [(float(np.percentile(a[..., c], 0.1)), float(np.percentile(a[..., c], 99.9)))
            for c in range(a.shape[2])]


def _apply_norm(rgb, ranges):
    x = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None]
    for c, (lo, hi) in enumerate(ranges):
        x[0, c] = (x[0, c] - lo) / (hi - lo + 1e-6)
    return x


def _forward(model, chw, device="cpu"):
    """Run the model on one already-normalised NCHW tile, returning an integer label mask."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(chw))
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


def segment_labels(rgb, model, device="cpu", tile=TILE, pad=PADDING, progress=None,
                   ranges=None):
    """Tile, infer, and reconcile into one whole-image label mask.

    Reconciliation rule: an object belongs to the tile whose CORE region contains its centroid.
    An object straddling a seam is therefore emitted once, by exactly one tile, and always by a
    tile that saw it with padding context on that side. Objects whose centroid falls outside the
    core are dropped here because the neighbouring tile owns them.

    CONTESTED PIXELS. Ownership decides who EMITS an object, but two neighbouring tiles still
    infer independently over their shared padded band and can disagree about where one nucleus
    ends and the next begins. A later tile writing its object over the same pixels would erase
    part of an object an earlier tile already emitted. Measured on LL477 with 256 px tiles before
    this guard: one object was overwritten to zero pixels and one reduced to a fragment, out of
    5397. Contested pixels therefore go to whoever claimed them first, and an object left with
    fewer than 3 pixels is not emitted at all — so a zero-pixel label can never reach the
    measurement stage (where it would produce a NaN centroid). Single-tile images have no
    padded band and so are unaffected by this rule.
    """
    h, w = rgb.shape[:2]
    # `ranges` lets a streaming caller supply globally-computed percentiles instead of
    # recomputing them on this block (which would be per-block normalisation by another name).
    if ranges is None:
        ranges = norm_range(rgb)
    labels = np.zeros((h, w), dtype=np.int32)
    if low_contrast(ranges):        # nothing to segment — see MIN_DYNAMIC_RANGE
        return labels
    norm = _apply_norm(rgb, ranges)
    next_id = 1
    contested = 0
    tiles = _tile_grid(h, w, tile, pad)
    for i, ((y0, y1, x0, x1), (py0, py1, px0, px1)) in enumerate(tiles):
        if progress:
            progress(i + 1, len(tiles))
        sub = norm[:, :, py0:py1, px0:px1]
        th, tw = sub.shape[2], sub.shape[3]
        if th < MIN_TILE or tw < MIN_TILE:      # model needs >= 32 px per spatial axis
            sub = np.pad(sub, ((0, 0), (0, 0), (0, max(MIN_TILE - th, 0)),
                               (0, max(MIN_TILE - tw, 0))), mode="edge")
        lab = _forward(model, sub, device)
        lab = lab[:py1 - py0, :px1 - px0]
        window = labels[py0:py1, px0:px1]
        for oid in np.unique(lab):
            if oid == 0:
                continue
            m = (lab == oid)
            ys, xs = np.nonzero(m)
            cy = ys.mean() + py0
            cx = xs.mean() + px0
            # core ownership — the half-open convention makes the cores a true partition
            if not (y0 <= cy < y1 and x0 <= cx < x1):
                continue
            free = m & (window == 0)
            n_free = int(free.sum())
            if n_free < 3:
                continue
            if n_free != int(m.sum()):
                contested += 1
            window[free] = next_id
            next_id += 1
    if contested and progress:
        progress(len(tiles), len(tiles))
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


# ── whole-slide streaming ────────────────────────────────────────────────────────────────

# Above this many pixels the image is streamed rather than held in RAM. The binding constraint
# is not the RGB array but the optical-density maths: `_od_channels` builds float64 HxWx3
# intermediates, i.e. 24 bytes/px, so a 776 Mpx ACROBAT slide (48128x16128) needs ~18.6 GB on a
# 17 GB machine — it does not merely swap, it fails. 64 Mpx keeps the OD peak near 1.5 GB.
WSI_STREAM_THRESHOLD_PX = 64_000_000

# Pixel budget for estimating the global normalisation percentiles on a slide too large to hold.
_NORM_ESTIMATE_PX = 12_000_000
_NORM_PATCH = 512


def open_slide(image_path):
    """openslide handle, or None if this is not a slide openslide can read."""
    try:
        import openslide
        return openslide.OpenSlide(os.path.expanduser(image_path))
    except Exception:
        return None


def image_dimensions(image_path):
    """(width, height) without decoding the pixels."""
    slide = open_slide(image_path)
    if slide is not None:
        try:
            return slide.dimensions
        finally:
            slide.close()
    from PIL import Image
    with Image.open(os.path.expanduser(image_path)) as im:
        return im.size


def _percentiles_from_hist(hist, lo_q=0.001, hi_q=0.999):
    """Nearest-rank percentiles from a 256-bin uint8 histogram. Exact for 8-bit data."""
    total = hist.sum()
    if total == 0:
        return 0.0, 0.0
    c = np.cumsum(hist)
    lo = int(np.searchsorted(c, lo_q * total, side="left"))
    hi = int(np.searchsorted(c, hi_q * total, side="left"))
    return float(min(lo, 255)), float(min(hi, 255))


def slide_scan(slide, tile=TILE, progress=None):
    """One counting pass over a slide: EXACT global normalisation percentiles, plus which
    stripes contain tissue.

    Sampling was tried first and rejected on measurement. Estimating the percentiles from a grid
    of level-0 patches (11.5 Mpx of 197) put the normalisation 0.28 normalised units off the
    exact value, and segmenting a tissue region with the estimate rather than the exact range
    agreed with it on only 63% of nuclei — the counts looked close (120 vs 126) while a third of
    the objects were different. Reading a downsampled pyramid level was worse still (0.34),
    because downsampling averages away the very tails the 0.1/99.9 percentiles are made of.

    Exactness is affordable here because the data is 8-bit: a 256-bin histogram per channel is
    exact by construction, and the only cost is reading the pixels once more. That cost is
    repaid immediately — the same pass records which stripes are pure background, and on a
    typical slide (this ACROBAT slide is ~0.5% tissue) skipping them saves far more inference
    time than the extra read costs.

    Returns (ranges, tissue_stripes, stats).
    """
    W, H = slide.dimensions
    hist = np.zeros((3, 256), dtype=np.int64)
    tissue = []
    stripes = list(range(0, H, tile))
    for i, y0 in enumerate(stripes):
        if progress:
            progress(i + 1, len(stripes))
        y1 = min(y0 + tile, H)
        rgb = np.asarray(slide.read_region((0, y0), 0, (W, y1 - y0)).convert("RGB"))
        for c in range(3):
            hist[c] += np.bincount(rgb[..., c].ravel(), minlength=256)
        # "tissue" here only has to be permissive enough not to skip a stripe that could
        # contain nuclei; the real decision is still the model's.
        if float((rgb.mean(axis=2) < 225).mean()) > 0.0005:
            tissue.append(y0)
    ranges = [_percentiles_from_hist(hist[c]) for c in range(3)]
    stats = {"stripes": len(stripes), "tissue_stripes": len(tissue),
             "skipped": len(stripes) - len(tissue)}
    return ranges, tissue, stats


def _slide_norm_ranges(slide):
    """Exact global normalisation percentiles for a slide (counting pass, no sampling)."""
    W, H = slide.dimensions
    if W * H <= _NORM_ESTIMATE_PX:
        region = np.asarray(slide.read_region((0, 0), 0, (W, H)).convert("RGB"))
        return norm_range(region), "level0_full"
    ranges, _tissue, stats = slide_scan(slide)
    return ranges, f"level0_histogram_{stats['stripes']}stripes"


def segment_slide_streaming(image_path, model, pixel_size_um, device="cpu",
                            tile=TILE, pad=PADDING, progress=None):
    """Segment and measure a whole slide without ever holding it in memory.

    Processed in STRIPES one tile-row tall and the full image wide. Each stripe is read from
    openslide with `pad` rows of vertical context, normalised with the GLOBAL constants (so the
    tile-size-independence property still holds — see `_normalize`), segmented tile by tile, and
    measured immediately; only the per-object records survive the stripe. Peak memory is a
    stripe, not a slide.

    Cross-stripe conflicts are handled the same way as cross-tile ones: a `carry` band holds the
    pixels the previous stripe already claimed, so an object emitted there cannot be overwritten
    by this stripe. Nuclei are ~10-20 px across and the band is `2*pad` rows, so no object can
    span it undetected.

    Returns the same record list as `_measure`, in full-resolution image coordinates.
    """
    import cv2
    slide = open_slide(image_path)
    if slide is None:
        raise ValueError(f"not an openslide-readable slide: {image_path}")
    try:
        W, H = slide.dimensions
        if W * H <= _NORM_ESTIMATE_PX:
            ranges, _how = _slide_norm_ranges(slide)
            stripes = list(range(0, H, tile))
        else:
            # counting pass: exact percentiles + which stripes are worth inferring on
            ranges, stripes, _stats = slide_scan(slide, tile=tile, progress=progress)
        if low_contrast(ranges):     # blank / no-tissue slide — see MIN_DYNAMIC_RANGE
            return []
        records = []
        next_id = 1
        carry = None            # (y_start, bool array) claimed pixels from the previous stripe
        for si, sy0 in enumerate(stripes):
            if progress:
                progress(si + 1, len(stripes))
            sy1 = min(sy0 + tile, H)
            ry0, ry1 = max(sy0 - pad, 0), min(sy1 + pad, H)
            rgb = np.asarray(
                slide.read_region((0, ry0), 0, (W, ry1 - ry0)).convert("RGB"))
            norm = _apply_norm(rgb, ranges)

            claimed = np.zeros((ry1 - ry0, W), dtype=bool)
            if carry is not None:
                cy0, cband = carry
                # overlap between the previous stripe's read window and this one
                a0, a1 = max(cy0, ry0), min(cy0 + cband.shape[0], ry1)
                if a1 > a0:
                    claimed[a0 - ry0:a1 - ry0] |= cband[a0 - cy0:a1 - cy0]

            labels = np.zeros((ry1 - ry0, W), dtype=np.int32)
            for x0 in range(0, W, tile):
                x1 = min(x0 + tile, W)
                px0, px1 = max(x0 - pad, 0), min(x1 + pad, W)
                sub = norm[:, :, :, px0:px1]
                th, tw = sub.shape[2], sub.shape[3]
                if th < MIN_TILE or tw < MIN_TILE:
                    sub = np.pad(sub, ((0, 0), (0, 0), (0, max(MIN_TILE - th, 0)),
                                       (0, max(MIN_TILE - tw, 0))), mode="edge")
                lab = _forward(model, sub, device)[:ry1 - ry0, :px1 - px0]
                for oid in np.unique(lab):
                    if oid == 0:
                        continue
                    m = (lab == oid)
                    ys, xs = np.nonzero(m)
                    cy = ys.mean() + ry0
                    cx = xs.mean() + px0
                    # own it only if the centroid is in this tile's core, in BOTH axes
                    if not (sy0 <= cy < sy1 and x0 <= cx < x1):
                        continue
                    win_l = labels[:, px0:px1]
                    win_c = claimed[:, px0:px1]
                    free = m & (win_l == 0) & (~win_c)
                    if int(free.sum()) < 3:
                        continue
                    win_l[free] = next_id
                    next_id += 1

            # measure this stripe's objects on ORIGINAL pixels, then discard the stripe
            hem_od, dab_od = _od_channels(rgb)
            stripe_records = _measure(labels, hem_od, dab_od, float(pixel_size_um))
            for r in stripe_records:                     # stripe -> full-image coordinates
                r["polygon"][:, 1] += ry0
                cx, cy = r["centroid_px"]
                r["centroid_px"] = (cx, cy + ry0)
                r["centroid_um"] = (cx * pixel_size_um, (cy + ry0) * pixel_size_um)
                r["id"] = len(records) + 1
                records.append(r)
            carry = (ry0, labels > 0)
        return records
    finally:
        slide.close()


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
    resampling used for inference. `labels` is None on the streamed whole-slide path, where a
    full-resolution label array cannot be held (3.1 GB for a 776 Mpx slide).
    """
    import cv2

    # Whole slides are streamed. Deciding by DIMENSIONS rather than by file extension means a
    # large flat TIFF is handled too, and a small .svs is not needlessly streamed.
    if rgb is None:
        try:
            w0, h0 = image_dimensions(image_path)
        except Exception:
            w0 = h0 = 0
        if w0 * h0 > WSI_STREAM_THRESHOLD_PX and open_slide(image_path) is not None:
            return _segment_image_streamed(image_path, model_dir, pixel_size_um, device,
                                           dab_threshold, adaptive_threshold, progress)

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
    thr, method = _classify(records, dab_threshold, adaptive_threshold)

    return {"records": records, "labels": labels, "threshold": thr,
            "threshold_method": method, "pixel_size_um": float(pixel_size_um),
            "width": w, "height": h, "downsample": ds, "streamed": False,
            "device": device, "low_contrast": bool(low_contrast(norm_range(work)))}


def _classify(records, dab_threshold, adaptive_threshold):
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
    return thr, method


def _segment_image_streamed(image_path, model_dir, pixel_size_um, device,
                            dab_threshold, adaptive_threshold, progress):
    """Whole-slide branch of `segment_image`.

    NOTE ON RESAMPLING: the streaming path reads level 0 and does not resample. In practice
    whole slides are scanned at 0.25-1.0 µm/px, where `preferred_downsample` is either clamped
    to 1.0 (coarser than the model's 0.5 µm) or a small integer. If a slide is fine enough to
    want a downsample > 1, that is reported in the result so the caller can see it was not
    applied, rather than it being silently ignored.
    """
    model = load_model(model_dir, device)
    w, h = image_dimensions(image_path)
    ds = preferred_downsample(pixel_size_um, model_pixel_size(model_dir))
    records = segment_slide_streaming(image_path, model, pixel_size_um,
                                      device=device, progress=progress)
    thr, method = _classify(records, dab_threshold, adaptive_threshold)
    return {"records": records, "labels": None, "threshold": thr,
            "threshold_method": method, "pixel_size_um": float(pixel_size_um),
            "width": w, "height": h, "downsample": 1.0, "streamed": True,
            "device": device, "downsample_requested": ds}


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
    """The per-image summary JSON `parse_summary_json` consumes."""
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
        # Provenance: CPU and MPS agree to ~1.5e-5 OD with identical cell counts and
        # classifications (597/598 DeepLIIF panels byte-identical), but they are not
        # bit-identical in every case. Recording the device makes a run reproducible.
        "segmenter_device": result.get("device", "cpu"),
        "segmenter_streamed": bool(result.get("streamed")),
    }
    if extra:
        summ.update(extra)
    with open(path, "w") as f:
        json.dump(summ, f, indent=4)
    return path
