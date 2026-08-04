"""
cell_expansion.py
Cytoplasmic-ring DAB measurement for MEMBRANE markers (CD8, TIM-3).

InstanSeg segments nuclei and QuPath measures DAB inside the nucleus polygon —
the wrong compartment for a membranous stain, which sits in a ring *outside* the
nucleus. This module re-measures DAB in the cytoplasmic ring (expanded cell minus
nucleus) so membranous positivity is detected correctly.

Biological safety: the expansion of each nucleus is clipped to that nucleus's
Voronoi cell, so an expanded cell can NEVER cross the midline into a neighbouring
nucleus and steal its membrane DAB. In dense lymphocyte infiltrate, naive
expansion produces false positives; the Voronoi clip is what makes this valid
(this mirrors QuPath's detectionsToCells behaviour). The Voronoi constraint is
implemented exactly via perpendicular-bisector half-plane clipping against the
relevant nearby nuclei.

Dependencies: shapely, scipy, numpy, PIL (all already in the project).
"""

import os
import json
import numpy as np


# Initial H-DAB vectors. The resulting channel is calibrated per image against
# QuPath's exported nuclear "DAB: Mean" values before it may reclassify cells;
# the parity gate below fails closed if the two channels do not agree.
_QUPATH_STAINS = {
    "hematoxylin": [0.721,  0.646, 0.249],
    "dab":         [0.532,  0.656, 0.535],
    "residual":    [0.539, -0.750, 0.384],
}
_DEFAULT_BACKGROUND = [255, 255, 254]


# ──────────────────────────────────────────────────────────────────────────────
# Image + deconvolution
# ──────────────────────────────────────────────────────────────────────────────

def _load_rgb_full(image_path: str) -> np.ndarray:
    """Full-resolution RGB array. openslide for WSI (SVS/NDPI), PIL fallback."""
    try:
        import openslide
        slide = openslide.OpenSlide(image_path)
        w, h  = slide.dimensions
        region = slide.read_region((0, 0), 0, (w, h)).convert("RGB")
        slide.close()
        return np.asarray(region)
    except Exception:
        pass
    from PIL import Image
    return np.asarray(Image.open(image_path).convert("RGB"))


def _norm_vec(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _od_channels(rgb: np.ndarray, stain_vectors: dict, background):
    """
    Per-pixel hematoxylin AND DAB optical density via colour deconvolution,
    QuPath convention:
        OD = -log10((I + 1) / background)   (per channel)
        [H, DAB, residual] = OD · inv(normalized_stain_matrix)
    Returns (H_od, DAB_od), each HxW float32; DAB_od is directly comparable to
    QuPath's "DAB: Mean". The hematoxylin channel enables the per-pixel
    DAB-dominance gate (a stained pixel must be more brown than blue).
    """
    rgb = np.asarray(rgb, dtype=np.float64)[..., :3]
    bg  = np.asarray(background, dtype=np.float64).reshape(1, 1, 3)
    od  = -np.log10((rgb + 1.0) / bg)                      # HxWx3 optical density
    M = np.array([_norm_vec(stain_vectors["hematoxylin"]),
                  _norm_vec(stain_vectors["dab"]),
                  _norm_vec(stain_vectors["residual"])])   # rows = stain vectors
    Minv = np.linalg.inv(M)
    hem = (od @ Minv[:, 0]).astype(np.float32)             # HxW
    dab = (od @ Minv[:, 1]).astype(np.float32)             # HxW
    return hem, dab


# Ruifrok reference stain vectors, used only to LABEL the two estimated stains
# (which is hematoxylin, which is DAB) — not for the deconvolution itself.
_REF_H   = _norm_vec([0.650, 0.704, 0.286])
_REF_DAB = _norm_vec([0.268, 0.570, 0.776])


def _estimate_background(rgb: np.ndarray):
    """Per-image white point = 99th percentile per channel over bright pixels.
    Corrects illumination/white balance so OD is comparable across slides — the
    fixed [255,255,254] white is what let the counterstain leak into DAB on
    tone-cast slides (CRC-ICM)."""
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    means = flat.mean(1)
    # `>=` (not `>`) so a large saturated-white border can't empty this set: when
    # ≥20% of the frame is clipped white the 80th-percentile brightness equals the
    # ceiling, and strict `>` would drop every pixel → np.percentile of an empty
    # array crashes ("index -1 is out of bounds for axis 0 with size 0").
    bright = flat[means >= np.percentile(means, 80)]
    if bright.size == 0:                      # degenerate (fully uniform) image
        bright = flat
    return np.clip(np.percentile(bright, 99, axis=0), 200, 255)


def _estimate_stain_vectors(rgb: np.ndarray, background) -> dict:
    """
    Per-image H-DAB stain-vector estimation (Macenko) on optical density.

    The fixed QuPath vectors mis-deconvolve slides whose staining/white balance
    differs from QuPath's convention — the counterstain bleeds into the DAB
    channel and manufactures false positives (this was the ~99%-positive failure
    on CRC-ICM TIM-3). Estimating the two dominant stain vectors from THIS image
    fixes that. Returns a stain_vectors dict (hematoxylin/dab/residual) or None if
    the estimate is degenerate (too little tissue, or the two stains don't
    separate), in which case the caller keeps the fixed vectors.
    """
    try:
        rgb = np.asarray(rgb, dtype=np.float64)[..., :3]
        bg  = np.asarray(background, dtype=np.float64).reshape(1, 3)
        od  = -np.log10((rgb.reshape(-1, 3) + 1.0) / bg)
        tissue = od[od.sum(1) > 0.15]                      # drop glass/background
        if len(tissue) < 1000:
            return None
        cov = np.cov(tissue.T)
        w, V = np.linalg.eigh(cov)
        plane = V[:, [2, 1]].astype(np.float64)            # top-2 eigenvectors
        # Orient the primary axis toward the data mean so the projected angles
        # cluster near 0 rather than straddling the ±π wrap — otherwise the 1st
        # and 99th angle percentiles collapse onto the SAME vector (degenerate).
        mean_od = tissue.mean(0)
        if plane[:, 0] @ mean_od < 0:
            plane[:, 0] = -plane[:, 0]
        proj = tissue @ plane
        ang = np.arctan2(proj[:, 1], proj[:, 0])
        a1, a2 = np.percentile(ang, 1.0), np.percentile(ang, 99.0)

        def _orient_pos(v):
            # Flip the WHOLE vector (not per-component abs, which folds opposite
            # directions together) so its RGB-OD components are non-negative.
            v = np.asarray(v, dtype=np.float64)
            return _norm_vec(-v if v.sum() < 0 else v)

        v1 = _orient_pos(plane @ np.array([np.cos(a1), np.sin(a1)]))
        v2 = _orient_pos(plane @ np.array([np.cos(a2), np.sin(a2)]))
        # Assign by cosine to the DAB reference; the other becomes hematoxylin.
        dab, hem = (v1, v2) if (v1 @ _REF_DAB) >= (v2 @ _REF_DAB) else (v2, v1)
        # Sanity: the two stains must point in the DAB / H directions and be
        # separated, else the estimate is degenerate → fall back to fixed.
        if (dab @ _REF_DAB) < 0.6 or (hem @ _REF_H) < 0.6 or (dab @ hem) > 0.93:
            return None
        residual = _norm_vec(np.cross(hem, dab))
        return {"hematoxylin": hem.tolist(), "dab": dab.tolist(),
                "residual": residual.tolist()}
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _feature_polygon(feat):
    """shapely (Multi)Polygon from a GeoJSON detection feature, or None."""
    from shapely.geometry import Polygon, MultiPolygon
    geom   = feat.get("geometry", {})
    gtype  = geom.get("type")
    coords = geom.get("coordinates", [])
    if not coords:
        return None
    try:
        if gtype == "Polygon":
            return Polygon(coords[0], coords[1:] if len(coords) > 1 else None)
        if gtype == "MultiPolygon":
            polys = [Polygon(p[0], p[1:] if len(p) > 1 else None) for p in coords]
            return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    except Exception:
        return None
    return None


def _halfplane_clip(poly, p, q):
    """
    Clip `poly` to the side of the perpendicular bisector of segment pq that
    contains p (i.e. the locus of points closer to nucleus p than nucleus q).
    Intersecting with every relevant neighbour's bisector reproduces p's Voronoi
    cell exactly.
    """
    from shapely.geometry import Polygon
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = q - p
    L = float(np.linalg.norm(d))
    if L == 0:
        return poly
    n_p = (p - q) / L                       # unit normal pointing toward p
    t   = np.array([-n_p[1], n_p[0]])       # tangent along the bisector
    mid = (p + q) / 2.0

    # Build a half-plane quad on p's side, large enough to cover `poly`
    minx, miny, maxx, maxy = poly.bounds
    R = max(np.hypot(cx - mid[0], cy - mid[1])
            for cx in (minx, maxx) for cy in (miny, maxy)) + L + 10.0
    quad = Polygon([mid + t * R, mid - t * R,
                    mid - t * R + n_p * 2 * R, mid + t * R + n_p * 2 * R]).buffer(0)
    try:
        return poly.intersection(quad)
    except Exception:
        return poly


def ring_connectivity(xy, vals, centroid, thr, hvals=None, *, n_sectors=72,
                      sector_min_frac=0.5, dab_dominance_gate=True):
    """Read a ring as an ordered arc: how much of the circumference is continuously stained.

    Membrane completeness (`membrane_pos_frac`) counts *how many* ring pixels are stained
    and cannot distinguish one clean arc from the same number of pixels scattered around
    the ring as noise. Real membrane staining is contiguous; speckle from debris, an
    adjacent cell's membrane, or counterstain bleed is not. This is the measurement the
    HER2 literature settled on — Visiopharm's HER2-CONNECT skeletonises membrane fragments
    and scores connectivity on 0-1, cutting at 0.12 / 0.56 for kappa 0.86 against
    pathologists, and Aperio's Membrane algorithm scores completeness alongside intensity.

    The ring is binned into `n_sectors` angular sectors about the nucleus centroid. A
    sector counts as stained when at least `sector_min_frac` of its pixels pass the same
    threshold and DAB-dominance gate that completeness uses, so the two features are
    measured off identical pixels and differ only in whether order is taken into account.

    Returns (connectivity, arc_count, covered_frac):
      connectivity  longest run of consecutive stained sectors / n_sectors, wrapping
                    around 0/2pi. One clean half-membrane arc scores ~0.5.
      arc_count     number of separate stained runs. A real membrane is one or two arcs;
                    scattered speckle is many.
      covered_frac  stained sectors / n_sectors, ignoring order. Included so a caller can
                    see directly how much of connectivity is contiguity and how much is
                    simply coverage.
    """
    if xy is None or vals is None or len(vals) == 0:
        return 0.0, 0, 0.0
    xy = np.asarray(xy, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    if xy.shape[0] != vals.shape[0]:
        return 0.0, 0, 0.0

    pos = vals > float(thr)
    if dab_dominance_gate and hvals is not None:
        hvals = np.asarray(hvals, dtype=np.float64)
        if hvals.shape == vals.shape:
            pos = pos & (vals > hvals)

    ang = np.arctan2(xy[:, 1] - float(centroid[1]), xy[:, 0] - float(centroid[0]))
    sector = np.floor((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int)
    np.clip(sector, 0, n_sectors - 1, out=sector)

    total = np.bincount(sector, minlength=n_sectors)
    stained = np.bincount(sector, weights=pos.astype(np.float64), minlength=n_sectors)
    # An empty sector is not evidence of absence — the ring simply has no pixels at that
    # angle (thin rings, clipped Voronoi cells). Treating it as unstained would sever an
    # otherwise continuous arc, so it is excluded from both numerator and denominator.
    occupied = total > 0
    if not occupied.any():
        return 0.0, 0, 0.0
    frac = np.zeros(n_sectors, dtype=np.float64)
    frac[occupied] = stained[occupied] / total[occupied]
    on = (frac >= float(sector_min_frac)) & occupied

    covered = float(on.sum()) / float(occupied.sum())
    if not on.any():
        return 0.0, 0, covered
    if on.all():
        return 1.0, 1, covered

    # Longest circular run, and how many runs there are. Rotating the array so it starts
    # at an unstained sector makes the wrap-around case fall out of a linear scan.
    start = int(np.argmin(on))
    rolled = np.roll(on, -start)
    runs, cur = [], 0
    for flag in rolled:
        if flag:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    longest = max(runs) if runs else 0
    return float(longest) / float(n_sectors), len(runs), covered


def auto_pixel_threshold(ring_value_arrays, k=3.0):
    """The OD above which a ring pixel is unusual FOR THIS IMAGE, from its own ring pixels.

    Returned in the raw channel's units, i.e. directly comparable to the values in
    `ring_value_arrays` (the caller maps it to calibrated OD for the record).

    Why an estimator rather than a number. A membranous cell is positive when enough of
    its ring is stained, which needs a definition of "stained" -- and that definition has
    to exist at APPLY time, on an unlabelled image, not just at fit time. A fitted
    constant cannot: cutoffs fitted on labelled cells have ranged from 0.08 to 0.30 across
    cohorts, so any single number is wrong for most slides, and carrying one from a
    different cohort silently mis-scales every completeness measurement made with it.

    median + k·MAD, not a high percentile. The pooled ring distribution is mostly
    background, so its median and MAD describe the background and are unmoved by the
    stained tail -- which is exactly the property required, because that tail's size is
    the positivity rate and must not feed back into the threshold. A 99th percentile of
    pooled pixels does the opposite: on a marker with many positive cells it lands inside
    the positive tail and pushes the threshold up, so a slide with more positive cells
    gets a stricter definition of positive. 1.4826 makes MAD a standard-deviation estimate
    for a Gaussian background, so k is in familiar units; k=3 is the usual outlier bar.

    This is the same principle as `local_background` in the classifier: judge a
    measurement against its own surroundings rather than a constant chosen elsewhere.
    """
    vals = [np.asarray(v, dtype=np.float64).ravel()
            for v in ring_value_arrays if v is not None and len(v)]
    if not vals:
        return float("inf")            # nothing measurable: no pixel counts as stained
    pooled = np.concatenate(vals)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("inf")
    med = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - med)))
    if mad <= 0:                       # a flat ring channel has no scale to speak of
        return med
    return med + k * 1.4826 * mad


def _mask_stats(geoms, dab_od, x0, y0, w, h, want_values=None, aux_od=None):
    """
    Rasterize each geometry over the [x0:x0+w, y0:y0+h] window using shapely's
    vectorized point-in-polygon test, and return a per-geom dict:
        {"mean": float|None, "p90": float|None, "n": int,
         "values": np.ndarray|None, "aux_values": np.ndarray|None,
         "xy": np.ndarray|None}

    `mean` is the ring-average DAB OD (the legacy statistic). `p90` is the 90th
    percentile of the ring's pixel OD — it tracks the *brightest arc* and is not
    diluted by the empty part of the ring, which is what makes a faint membranous
    stain (TIM-3) detectable where the mean collapses below threshold.
    `values` (the raw per-pixel DAB OD array) is returned only for geometry
    indices in `want_values`. When `aux_od` (e.g. the hematoxylin OD channel) is
    supplied, `aux_values` holds the same pixels' aux OD — used for the per-pixel
    DAB-dominance gate (DAB > hematoxylin). `xy` holds those pixels' image
    coordinates, which is what lets a ring be read as an ordered arc rather than an
    unordered bag of pixels (see `ring_connectivity`).
    """
    import shapely
    xs = np.arange(x0, x0 + w) + 0.5
    ys = np.arange(y0, y0 + h) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    fx, fy = gx.ravel(), gy.ravel()
    sub = dab_od[y0:y0 + h, x0:x0 + w]
    sub_aux = aux_od[y0:y0 + h, x0:x0 + w] if aux_od is not None else None
    out = []
    for idx, g in enumerate(geoms):
        if g is None or g.is_empty:
            out.append({"mean": None, "p90": None, "n": 0,
                        "values": None, "aux_values": None, "xy": None})
            continue
        mask = shapely.contains_xy(g, fx, fy).reshape(h, w)
        n = int(mask.sum())
        if not n:
            out.append({"mean": None, "p90": None, "n": 0,
                        "values": None, "aux_values": None, "xy": None})
            continue
        vals = sub[mask]
        want = (want_values is not None and idx in want_values)
        keep = vals.astype(np.float32) if want else None
        keep_aux = (sub_aux[mask].astype(np.float32)
                    if (want and sub_aux is not None) else None)
        keep_xy = (np.stack([gx[mask], gy[mask]], axis=1).astype(np.float32)
                   if want else None)
        out.append({"mean": float(vals.mean()),
                    "p90": float(np.percentile(vals, 90)),
                    "n": n, "values": keep, "aux_values": keep_aux,
                    "xy": keep_xy})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def measure_cytoplasm_dab(
    image_path: str,
    geojson_path: str,
    pixel_size_um: float,
    expansion_um: float = 2.0,
    stain_vectors: dict = None,
    background: list = None,
    dab_threshold: float = None,
    membrane_pix_thr: float = None,
    keep_ring_values: bool = False,
    estimate_stains: bool = True,
    dab_dominance_gate: bool = True,
) -> list:
    """
    Measure DAB in the cytoplasmic ring of every detected nucleus.

    Returns a list aligned 1:1 with the GeoJSON `features` (so the caller can zip
    it straight back), each entry:
        {
          "nucleus_dab_mean":   float | None,   # inside nucleus (recomputed)
          "cytoplasm_dab_mean": float | None,   # the ring (expanded ∩ Voronoi − nucleus)
          "cytoplasm_dab_p90":  float | None,   # 90th pct of ring OD (brightest arc)
          "cell_dab_mean":      float | None,   # whole Voronoi-clipped expanded cell
          "membrane_pos_frac":  float | None,   # frac of ring pixels > membrane_pix_thr
          "membrane_connectivity": float | None, # longest contiguous stained arc / ring
          "membrane_arc_count":    int | None,   # number of separate stained arcs
          "centroid":           [x, y] | None,
        }
    `membrane_pos_frac` counts stained ring pixels; `membrane_connectivity` reads the same
    pixels in angular order, so a clean arc and the same pixel count scattered as speckle
    are separable (`ring_connectivity`). Both need `membrane_pix_thr`.
    Non-polygon features yield an all-None entry.

    `dab_threshold`, if given, is used only to print a validation summary
    (how many cells the nuclear vs cytoplasm classification disagree on).

    `membrane_pix_thr` (calibrated OD): when given, `membrane_pos_frac` is computed
    as the fraction of ring pixels whose calibrated OD exceeds it — the membrane
    *completeness* feature that separates a faint concentrated arc (real positive)
    from diffuse low background (false positive), which the ring mean cannot.
    `keep_ring_values=True` additionally attaches the calibrated per-pixel ring OD
    array as `"ring_values"` (and `"ring_h_values"`, the matching hematoxylin OD)
    so the threshold-tuning harness can sweep cutoffs AND replicate the gate.

    `estimate_stains=True` (default) estimates this image's H-DAB stain vectors
    (Macenko) instead of using the fixed QuPath vectors — fixed vectors
    mis-deconvolve variable slides and let the counterstain masquerade as DAB
    (the CRC-ICM ~99%-positive failure). Falls back to the fixed vectors if the
    estimate is degenerate. `dab_dominance_gate=True` requires each positive ring
    pixel to be more DAB than hematoxylin (DAB_OD > H_OD), which removes
    dark-counterstain false positives at low OD.
    """
    from scipy.spatial import cKDTree

    stain_vectors = stain_vectors or _QUPATH_STAINS
    background    = background or _DEFAULT_BACKGROUND
    px = float(pixel_size_um) if pixel_size_um and pixel_size_um > 0 else 0.5
    exp_px = expansion_um / px

    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)
    features = gj.get("features", [])

    rgb    = _load_rgb_full(image_path)
    H, W   = rgb.shape[:2]

    # Fix geometry and collect valid nuclei (aligned to feature index)
    fixed = []
    for feat in features:
        poly = _feature_polygon(feat)
        if poly is not None:
            try:
                poly = poly.buffer(0)
                if poly.is_empty:
                    poly = None
            except Exception:
                poly = None
        fixed.append(poly)

    # Choose stain vectors by PARITY: prefer the per-image Macenko estimate, but
    # fall back to the fixed QuPath vectors when they deconvolve BETTER. Low-DAB
    # slides estimate poorly — the estimate would otherwise fail the parity gate
    # and silently revert the whole membrane measurement to nuclear (the LL477
    # case). Selection uses the correlation of a centroid-window DAB sample with
    # QuPath's exported nuclear "DAB: Mean" — cheap, no ring rasterization.
    candidates = [("fixed", _QUPATH_STAINS, _DEFAULT_BACKGROUND)]
    if estimate_stains:
        bg_est = _estimate_background(rgb)
        est = _estimate_stain_vectors(rgb, bg_est)
        if est is not None:
            candidates.insert(0, ("estimated", est, bg_est.tolist()))

    def _centroid_parity(dab_ch):
        s, q = [], []
        for i, feat in enumerate(features):
            if fixed[i] is None:
                continue
            qm = (feat.get("properties", {}).get("measurements", {}) or {}).get("DAB: Mean")
            if not isinstance(qm, (int, float)):
                continue
            cx = int(round(fixed[i].centroid.x)); cy = int(round(fixed[i].centroid.y))
            if 0 <= cy < H and 0 <= cx < W:
                y0, y1 = max(cy - 2, 0), min(cy + 3, H)
                x0, x1 = max(cx - 2, 0), min(cx + 3, W)
                s.append(float(dab_ch[y0:y1, x0:x1].mean())); q.append(float(qm))
        if len(s) < 50:
            return -2.0
        c = float(np.corrcoef(s, q)[0, 1])
        return c if np.isfinite(c) else -2.0

    def _h_validity(h, d):
        """Median hematoxylin OD over tissue. Physically it MUST be positive (hematoxylin
        absorbs). A negative value means the stain vectors / white point are wrong for this
        (usually colour-tinted) slide; the DAB>H dominance gate then saturates and over-calls
        — this is the faint-TIM-3 92290 failure. Used to reject physically-invalid candidates
        BEFORE parity ranking (parity is circular here: it rewards matching QuPath's own
        fixed-vector DAB, so it keeps the fixed channel even when it is broken)."""
        tissue = (np.abs(h) + np.abs(d)) > 0.1
        return float(np.median(h[tissue])) if bool(tissue.any()) else float(np.median(h))

    scored = []
    for name, vecs, bg in candidates:
        h, d = _od_channels(rgb, vecs, bg)
        scored.append((_centroid_parity(d), _h_validity(h, d), name, vecs, bg, h, d))
    # Prefer physically-valid H (median > 0.02), ranked by parity within that set. Fall back
    # to best-parity-overall only if NO candidate has a valid H — so a degenerate estimate on
    # a genuinely low-DAB slide (the LL477 case) still reverts to fixed exactly as before.
    valid = [s for s in scored if s[1] > 0.02]
    pool = valid if valid else scored
    best = max(pool, key=lambda s: s[0])
    _, h_med, chosen, stain_vectors, background, hem_od, dab_od = best
    print(f"  Cytoplasm measurement: {chosen} stain vectors "
          f"(centroid-parity r={best[0]:.3f}, H-median {h_med:+.3f})")
    del rgb

    valid_idx  = [i for i, p in enumerate(fixed) if p is not None]
    centroids  = np.array([[fixed[i].centroid.x, fixed[i].centroid.y] for i in valid_idx]) \
        if valid_idx else np.empty((0, 2))
    pos_of     = {i: k for k, i in enumerate(valid_idx)}
    tree       = cKDTree(centroids) if len(centroids) else None

    results      = [None] * len(features)
    raw_ring_vals = {}          # feature idx -> raw (uncalibrated) ring pixel DAB OD
    raw_ring_h    = {}          # feature idx -> matching ring pixel hematoxylin OD
    raw_ring_xy   = {}          # feature idx -> matching ring pixel image coordinates
    ring_fallbacks = 0

    for i, poly in enumerate(fixed):
        if poly is None:
            results[i] = {"nucleus_dab_mean": None, "cytoplasm_dab_mean": None,
                          "cell_dab_mean": None, "centroid": None}
            continue

        k = pos_of[i]
        p = centroids[k]
        expanded = poly.buffer(exp_px)

        # ── Voronoi clip: intersect with bisector half-planes of nearby nuclei ──
        bx0, by0, bx1, by1 = expanded.bounds
        r_self = max(np.hypot(cx - p[0], cy - p[1])
                     for cx in (bx0, bx1) for cy in (by0, by1))
        clipped = expanded
        if tree is not None:
            for nid in tree.query_ball_point(p, 2.0 * r_self):
                if nid == k:
                    continue
                clipped = _halfplane_clip(clipped, p, centroids[nid])
                if clipped.is_empty:
                    break

        ring = clipped.difference(poly) if not clipped.is_empty else None

        # Raster window = expanded bounds, clamped to the image
        x0 = max(int(np.floor(bx0)), 0)
        y0 = max(int(np.floor(by0)), 0)
        x1 = min(int(np.ceil(bx1)), W)
        y1 = min(int(np.ceil(by1)), H)
        if x1 <= x0 or y1 <= y0:
            results[i] = {"nucleus_dab_mean": None, "cytoplasm_dab_mean": None,
                          "cell_dab_mean": None,
                          "centroid": [float(p[0]), float(p[1])]}
            continue

        nuc_s, cell_s, ring_s = _mask_stats(
            [poly, clipped, ring], dab_od, x0, y0, x1 - x0, y1 - y0,
            want_values={2}, aux_od=hem_od)
        nuc_m  = nuc_s["mean"]
        cell_m = cell_s["mean"]
        ring_m, ring_n, ring_p90 = ring_s["mean"], ring_s["n"], ring_s["p90"]

        if ring_m is None or ring_n == 0:        # degenerate ring → fall back
            ring_m = nuc_m
            ring_p90 = nuc_m
            ring_fallbacks += 1
        else:
            raw_ring_vals[i] = ring_s["values"]
            raw_ring_h[i]    = ring_s["aux_values"]
            raw_ring_xy[i]   = ring_s["xy"]
        if cell_m is None:
            cell_m = nuc_m

        # Exterior boundary of the Voronoi-clipped expanded cell = the cytoplasm
        # measurement region. Stored so the segmentation overlay can draw the
        # actual membrane compartment that was measured (largest part if multi).
        cell_polygon = None
        try:
            geom = clipped
            if geom is not None and not geom.is_empty:
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda gg: gg.area)
                if geom.geom_type == "Polygon":
                    cell_polygon = [[round(float(x), 2), round(float(y), 2)]
                                    for x, y in geom.exterior.coords]
        except Exception:
            cell_polygon = None

        results[i] = {
            "nucleus_dab_mean":   round(nuc_m, 5)   if nuc_m   is not None else None,
            "cytoplasm_dab_mean": round(ring_m, 5)  if ring_m  is not None else None,
            "cytoplasm_dab_p90":  round(ring_p90, 5) if ring_p90 is not None else None,
            "cell_dab_mean":      round(cell_m, 5)  if cell_m  is not None else None,
            "membrane_pos_frac":  None,   # filled after calibration (needs cal. thr)
            "membrane_connectivity": None,   # longest contiguous stained arc / circumference
            "membrane_arc_count":    None,   # number of separate stained arcs
            "centroid":           [float(p[0]), float(p[1])],
            "cell_polygon":       cell_polygon,
        }

    measured = [r for r in results if r and r["cytoplasm_dab_mean"] is not None]

    # Anchor the recomputed channel to QuPath's actual DAB scale for THIS image.
    # Different QuPath stain/background conventions can otherwise preserve rank
    # while shifting the channel enough to turn a 0.1 threshold into thousands of
    # false positives. Nuclear polygons give us a direct parity reference because
    # QuPath exported their "DAB: Mean" values in the same GeoJSON.
    pairs = []
    for feat, r in zip(features, results):
        qm = (feat.get("properties", {}).get("measurements", {}) or {}).get("DAB: Mean")
        if r and r.get("nucleus_dab_mean") is not None and isinstance(qm, (int, float)):
            pairs.append((float(r["nucleus_dab_mean"]), float(qm)))
    if len(pairs) < 50:
        raise RuntimeError("cytoplasm DAB calibration failed: too few QuPath nuclear references")
    raw_n, qp_n = np.asarray(pairs, float).T
    corr = float(np.corrcoef(raw_n, qp_n)[0, 1])
    slope, intercept = np.polyfit(raw_n, qp_n, 1)
    calibrated = raw_n * slope + intercept
    mae = float(np.mean(np.abs(calibrated - qp_n)))
    if not np.isfinite(corr) or corr < 0.90 or slope <= 0 or mae > 0.015:
        raise RuntimeError(
            f"cytoplasm DAB calibration failed parity gate "
            f"(corr={corr:.3f}, slope={slope:.3f}, MAE={mae:.4f} OD)")
    for r in results:
        if not r:
            continue
        for key in ("nucleus_dab_mean", "cytoplasm_dab_mean",
                    "cytoplasm_dab_p90", "cell_dab_mean"):
            if r.get(key) is not None:
                r[key] = round(max(0.0, float(r[key]) * slope + intercept), 5)
    print(f"  Cytoplasm DAB parity calibration: corr={corr:.3f}, "
          f"scale={slope:.3f}, offset={intercept:.4f}, MAE={mae:.4f} OD")

    # Membrane completeness. A pixel threshold expressed in CALIBRATED OD maps to
    # the raw channel by the inverse affine (slope>0 enforced above), so we count
    # in raw space without re-applying the calibration per pixel.
    if membrane_pix_thr is not None:
        if isinstance(membrane_pix_thr, str) and membrane_pix_thr == "auto":
            raw_thr = auto_pixel_threshold(raw_ring_vals.values())
        else:
            raw_thr = (float(membrane_pix_thr) - intercept) / slope
        for i, r in enumerate(results):
            vals = raw_ring_vals.get(i)
            if r and vals is not None and len(vals):
                pos = vals > raw_thr
                # DAB-dominance gate: a positive membrane pixel must be more DAB
                # than hematoxylin, so dark counterstain can't count as stain.
                hvals = raw_ring_h.get(i)
                if dab_dominance_gate and hvals is not None and len(hvals) == len(vals):
                    pos = pos & (vals > hvals)
                r["membrane_pos_frac"] = round(float(pos.mean()), 5)
                # Same pixels, same threshold, same dominance gate — the only difference
                # is that connectivity reads them in angular order. Measured off the raw
                # channel against the back-mapped threshold, exactly as completeness is.
                conn, arcs, _covered = ring_connectivity(
                    raw_ring_xy.get(i), vals, r["centroid"], raw_thr,
                    hvals=raw_ring_h.get(i),
                    dab_dominance_gate=dab_dominance_gate)
                r["membrane_connectivity"] = round(conn, 5)
                r["membrane_arc_count"] = int(arcs)
        # Record the threshold these three features were computed against, in calibrated
        # OD. With "auto" it is a property of the image, not of any setting, so nothing
        # downstream can reconstruct it and it has to travel with the measurement.
        _cal_thr = round(float(raw_thr) * slope + intercept, 5)
        for r in results:
            if r:
                r["membrane_pix_thr_used"] = _cal_thr
        print(f"  Membrane pixel threshold: {_cal_thr:.4f} OD"
              + (" (auto, from this image's own ring pixels)"
                 if isinstance(membrane_pix_thr, str) else ""))
    if keep_ring_values:
        for i, r in enumerate(results):
            vals = raw_ring_vals.get(i)
            if r and vals is not None:
                r["ring_values"] = (vals.astype(np.float64) * slope + intercept).tolist()
                hvals = raw_ring_h.get(i)
                if hvals is not None:
                    # hematoxylin OD in the SAME calibrated space (affine applies
                    # to the channel, so the gate comparison stays valid).
                    r["ring_h_values"] = (hvals.astype(np.float64) * slope + intercept).tolist()

    print(f"  Cytoplasm measurement: {len(measured)} cells measured, "
          f"{ring_fallbacks} degenerate rings (fell back to nucleus), "
          f"expansion {expansion_um} µm")

    if dab_threshold is not None and measured:
        thr     = float(dab_threshold)
        nuc_pos  = sum(1 for r in measured if (r["nucleus_dab_mean"] or 0)   > thr)
        cyto_pos = sum(1 for r in measured if (r["cytoplasm_dab_mean"] or 0) > thr)
        disagree = sum(1 for r in measured
                       if ((r["nucleus_dab_mean"] or 0) > thr)
                       != ((r["cytoplasm_dab_mean"] or 0) > thr))
        print(f"  Cytoplasm vs nuclear @ threshold {thr}: nuclear+={nuc_pos}, "
              f"cytoplasm+={cyto_pos}, DISAGREE={disagree} "
              f"(cells the fix actually changes)")

    return results
