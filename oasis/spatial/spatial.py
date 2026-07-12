"""
spatial.py
Cross-section, population-level SPATIAL ASSOCIATION between markers on
registered serial sections.

The active entry point is `run_spatial_association()`, which loads DAB-positive
cell centroids from two (or more) IHC serial sections, registers them into a
shared coordinate space, and measures cross-type spatial association with
Ripley's K / pair-correlation g(r) (implemented in spatial_stats.py). This is a
POPULATION statistic: it does NOT claim that individual cells co-express both
markers. Serial sections physically cannot establish single-cell co-expression
(different Z-planes, TIM-3 is not CD8-restricted, membrane-vs-nuclear
compartments), so no per-cell pairing is asserted anywhere in the active path.

Marker-agnostic: marker names are passed as strings, results are keyed by those
names, and the analysis scales to N markers.

DEPRECATED / UNUSED LEGACY (kept for reference only, not called by the
pipeline): the mutual-nearest-neighbour matcher `match_layers`, its translational
null `spatial_permutation_null`, the `run_coloc` driver and `generate_qc_overlay`.
MNN matching implies single-cell pairing => co-expression, which is exactly the
claim serial sections cannot support; it was replaced by the Ripley's K path and
remains only so the history is auditable. Do not use these for new work.
"""

import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON loaders
# ──────────────────────────────────────────────────────────────────────────────

def _polygon_centroid(ring) -> list:
    """
    Area-weighted (shoelace) centroid of a polygon outer ring.

    Uses the standard shoelace centroid formula, which gives the true geometric
    centroid of the enclosed area rather than the mean of the vertices (the
    vertex mean is biased toward regions where vertices are densely sampled).

    Works for both closed rings (first vertex repeated at the end — the duplicate
    edge contributes zero) and open rings. Falls back to the vertex mean for
    degenerate (near-zero-area / collinear) polygons.

    Args:
        ring: sequence of [x, y, ...] vertices (extra dims ignored)

    Returns:
        [cx, cy] as a Python list of floats
    """
    pts = np.asarray(ring, dtype=np.float64)[:, :2]
    if len(pts) < 3:
        return pts.mean(axis=0).tolist() if len(pts) else [0.0, 0.0]

    x, y   = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross  = x * y1 - x1 * y
    area   = cross.sum() / 2.0

    if abs(area) < 1e-9:                       # degenerate polygon → vertex mean
        return pts.mean(axis=0).tolist()

    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    return [float(cx), float(cy)]


def load_positive_centroids(geojson_path: str):
    """
    Extract XY centroids of DAB-positive cells from a QuPath GeoJSON export.

    Returns:
        centroids: Nx2 float32 array of (x, y) in image pixel coordinates
        features:  list of original GeoJSON feature dicts (positive cells only)
    """
    try:
        with open(geojson_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Could not load GeoJSON {Path(geojson_path).name}: {e}")
        return np.empty((0, 2), dtype=np.float32), []

    centroids, features = [], []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        cls   = props.get("classification", {}).get("name", "")
        if cls != "Positive":
            continue
        geom   = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        gtype = geom["type"]
        if gtype == "Point":
            xy = coords[:2]
        elif gtype == "Polygon" and coords:
            xy = _polygon_centroid(coords[0])
        elif gtype == "MultiPolygon" and coords:
            xy = _polygon_centroid(coords[0][0])
        else:
            continue
        centroids.append(xy)
        features.append(feat)

    arr = np.array(centroids, dtype=np.float32) if centroids \
        else np.empty((0, 2), dtype=np.float32)
    return arr, features


# ──────────────────────────────────────────────────────────────────────────────
# Core matching
# ──────────────────────────────────────────────────────────────────────────────

def match_layers(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    max_distance_um: float,
    pixel_size_um: float,
) -> list:
    """
    Mutual nearest-neighbour matching between two positive-cell centroid sets.

    A pair (i, j) is accepted only when:
      1. Cell i in A's nearest neighbour in B is cell j
      2. Cell j in B's nearest neighbour in A is cell i   ← mutual condition
      3. Their Euclidean distance ≤ max_distance_um

    Mutual NN prevents many-to-one matches in dense regions.

    Returns:
        List of dicts: {idx_a, idx_b, dist_um, centroid_a, centroid_b}
    """
    if len(centroids_a) == 0 or len(centroids_b) == 0:
        return []

    try:
        from scipy.spatial import KDTree
    except ImportError:
        print("  scipy not installed — install with: pip install scipy")
        return []

    max_dist_px = max_distance_um / pixel_size_um

    tree_b = KDTree(centroids_b)
    tree_a = KDTree(centroids_a)

    dist_a_to_b, idx_a_to_b = tree_b.query(centroids_a, k=1)
    dist_b_to_a, idx_b_to_a = tree_a.query(centroids_b, k=1)

    matches = []
    for i, (dist, j) in enumerate(zip(dist_a_to_b, idx_a_to_b)):
        j = int(j)
        if dist <= max_dist_px and idx_b_to_a[j] == i:
            matches.append({
                "idx_a":      i,
                "idx_b":      j,
                "dist_um":    round(float(dist) * pixel_size_um, 3),
                "centroid_a": centroids_a[i].tolist(),
                "centroid_b": centroids_b[j].tolist(),
            })
    return matches


# ──────────────────────────────────────────────────────────────────────────────
# Monte-Carlo spatial null model
# ──────────────────────────────────────────────────────────────────────────────

N_PERMUTATIONS = 1000
_NULL_SEED     = 0       # fixed seed → reproducible significance numbers

# Smallest positive-cell count that can support ANY spatial claim for a marker.
# Below this a marker is treated as effectively absent for the spatial test (its
# arrangement is not estimable); a segregation/association finding needs at least
# this many cells to be meaningful. Between this and dense_min_positive (30) the
# marker is SPARSE: the test still runs on the marker-independent all-cell support
# null (which does not need the sparse marker's own intensity), but the result
# carries an explicit underpowered flag. See ihc.md §15.7 and Q3.
_SPARSE_MIN_POSITIVE = 5


def spatial_permutation_null(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    observed_count: int,
    max_distance_um: float,
    pixel_size_um: float,
    n_perm: int = N_PERMUTATIONS,
) -> dict:
    """
    Monte-Carlo translational null model for co-localization significance.

    In dense tissue, two cell populations co-locate by chance simply because of
    their density — so a raw match count is uninterpretable on its own. This test
    asks: how many matches would we expect if marker B's positive cells kept their
    own internal spatial structure but were placed at a random offset within the
    tissue bounding box?

    Each of `n_perm` iterations rigidly translates centroids_b by a random offset
    (constrained so the cloud stays inside the combined bounding box), then counts
    mutual-NN matches against centroids_a. The observed count is compared against
    the resulting null distribution.

    Returns:
        {null_mean, null_std, z_score, p_value, n_perm}
        z_score / p_value are None when the null is degenerate (empty inputs).
    """
    result = {
        "null_mean": 0.0, "null_std": 0.0,
        "z_score": None, "p_value": None, "n_perm": n_perm,
    }
    if len(centroids_a) == 0 or len(centroids_b) == 0:
        return result

    # Tissue extent in reference space = bounding box of both registered clouds
    allpts = np.vstack([centroids_a, centroids_b]).astype(np.float64)
    r_min  = allpts.min(axis=0)
    r_max  = allpts.max(axis=0)

    b       = centroids_b.astype(np.float64)
    b_min   = b.min(axis=0)
    b_extent = b.max(axis=0) - b_min
    # Room for B's lower-left corner so the whole cloud stays inside the bbox
    span    = np.maximum((r_max - r_min) - b_extent, 0.0)

    rng    = np.random.default_rng(_NULL_SEED)
    counts = np.empty(n_perm, dtype=np.int64)
    for k in range(n_perm):
        new_origin = r_min + rng.random(2) * span
        shifted_b  = (b - b_min + new_origin).astype(np.float32)
        counts[k]  = len(match_layers(
            centroids_a, shifted_b, max_distance_um, pixel_size_um))

    null_mean = float(counts.mean())
    null_std  = float(counts.std())
    z_score   = ((observed_count - null_mean) / null_std) if null_std > 0 else None
    # One-sided permutation p-value with +1 correction (never reports p = 0)
    p_value   = float((np.count_nonzero(counts >= observed_count) + 1) / (n_perm + 1))

    return {
        "null_mean": round(null_mean, 3),
        "null_std":  round(null_std, 3),
        "z_score":   round(z_score, 3) if z_score is not None else None,
        "p_value":   round(p_value, 5),
        "n_perm":    n_perm,
    }


# ──────────────────────────────────────────────────────────────────────────────
# QC overlay
# ──────────────────────────────────────────────────────────────────────────────

def generate_qc_overlay(
    image_a_path: str,
    image_b_path: str,
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    matches: list,
    max_distance_um: float,
    stats: dict,
    registration_method: str,
    out_path: str,
    max_side: int = 1024,
) -> str:
    """
    Registered-point QC overlay (OpenCV only — no matplotlib).

    Side-by-side thumbnails of both serial sections:
      • left  = marker-A image, all A-positive centroids as hollow white circles
      • right = marker-B image, all B-positive centroids as hollow white circles
      • each matched pair joined by a line spanning both panels, coloured by
        distance — green ≤ T/3, yellow ≤ 2T/3, red otherwise (T = max_distance_um)
      • a stats box (top-left) with matched count, z-score, p-value, reg method

    centroids_a / centroids_b are NATIVE full-resolution coordinates; matches
    carry idx_a / idx_b into those arrays plus dist_um. Saved to out_path.

    Returns out_path on success, "" on failure.
    """
    try:
        import cv2
    except ImportError:
        print("  QC overlay: opencv-python not installed")
        return ""

    from oasis.common.registration import _load_rgb_thumbnail

    def _panel(path, cents):
        rgb, scale = _load_rgb_thumbnail(path, max_side)
        if rgb is not None:
            img = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
            return img, float(scale)
        # Fallback: blank canvas sized to the point cloud when the image can't load
        if len(cents):
            mx, my = cents.max(axis=0)
            scale = min(max_side / max(float(mx), float(my), 1.0), 1.0)
            h, w = int(my * scale) + 20, int(mx * scale) + 20
        else:
            scale, h, w = 1.0, max_side, max_side
        return np.full((max(h, 50), max(w, 50), 3), 245, np.uint8), scale

    img_a, scale_a = _panel(image_a_path, centroids_a)
    img_b, scale_b = _panel(image_b_path, centroids_b)

    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    gap = 24
    H, W = max(ha, hb), wa + gap + wb
    canvas = np.full((H, W, 3), 28, np.uint8)
    canvas[:ha, :wa] = img_a
    canvas[:hb, wa + gap:wa + gap + wb] = img_b
    xoff = wa + gap

    WHITE  = (255, 255, 255)
    GREEN  = (0, 200, 0)
    YELLOW = (0, 215, 235)
    RED    = (40, 40, 230)

    # All positive centroids as hollow white circles
    for (x, y) in centroids_a:
        cv2.circle(canvas, (int(x * scale_a), int(y * scale_a)), 3, WHITE, 1, cv2.LINE_AA)
    for (x, y) in centroids_b:
        cv2.circle(canvas, (int(x * scale_b) + xoff, int(y * scale_b)), 3, WHITE, 1, cv2.LINE_AA)

    # Matched pairs joined across panels, coloured by distance
    t1, t2 = max_distance_um / 3.0, 2.0 * max_distance_um / 3.0
    for m in matches:
        ia, ib = m["idx_a"], m["idx_b"]
        if ia >= len(centroids_a) or ib >= len(centroids_b):
            continue
        ax, ay = centroids_a[ia]
        bx, by = centroids_b[ib]
        d = m.get("dist_um", 0.0)
        color = GREEN if d <= t1 else YELLOW if d <= t2 else RED
        cv2.line(canvas,
                 (int(ax * scale_a), int(ay * scale_a)),
                 (int(bx * scale_b) + xoff, int(by * scale_b)),
                 color, 1, cv2.LINE_AA)

    # Stats box (top-left)
    z = stats.get("z_score")
    p = stats.get("p_value")
    lines = [
        f"Matched: {stats.get('count', 0)}",
        f"z = {z:.2f}" if isinstance(z, (int, float)) else "z = n/a",
        f"p = {p:.4f}" if isinstance(p, (int, float)) else "p = n/a",
        f"Reg: {registration_method}",
    ]
    box_w = 168
    box_h = 18 * len(lines) + 12
    cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), (20, 20, 20), -1)
    cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), (90, 90, 90), 1)
    for i, txt in enumerate(lines):
        cv2.putText(canvas, txt, (16, 28 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

    try:
        cv2.imwrite(out_path, canvas)
        return out_path
    except Exception as e:
        print(f"  QC overlay: could not write {out_path}: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# DEPRECATED / UNUSED — legacy MNN "co-expression" driver (not called)
# ──────────────────────────────────────────────────────────────────────────────

def run_coloc(
    layer_geojsons: dict,
    layer_order: list,
    reg_results: dict,
    max_distance_um: float,
    pixel_size_um: float,
) -> dict:
    """
    DEPRECATED / UNUSED — legacy mutual-nearest-neighbour matcher.

    Kept only for reference. MNN pairing implies single-cell co-expression, which
    serial sections cannot establish; the pipeline uses run_spatial_association()
    (cross-type Ripley's K) instead. Its internal "coexpression"/"CD8+TIM3+"
    naming is intentionally left untouched as a frozen legacy artifact — do not
    reuse it.

    Full (legacy) co-expression analysis for N markers.

    Args:
        layer_geojsons:  {"CD8": "/path/cd8_detections.geojson", "TIM3": "..."}
        layer_order:     ["CD8", "TIM3"]   — first entry is the reference layer
        reg_results:     {"TIM3": <reg_result_dict from registration.py>}
        max_distance_um: matching threshold in microns
        pixel_size_um:   pixel size of the reference image (µm/px)

    Returns:
        {
          "per_marker": {
            "CD8":  {"positive": 340},
            "TIM3": {"positive": 210},
          },
          "coexpression": {
            "CD8+TIM3+": {"count": 145, "matches": [...]}
          }
        }

    Adding a third marker later is just appending it to layer_order and
    providing its geojson + reg_result — no other code changes needed.
    """
    from oasis.common.registration import transform_centroids

    ref_marker = layer_order[0]
    per_marker = {}

    # Load positive centroids for every marker
    for marker in layer_order:
        path = layer_geojsons.get(marker)
        if path:
            cents, _ = load_positive_centroids(path)
        else:
            cents = np.empty((0, 2), dtype=np.float32)
        per_marker[marker] = {"positive": len(cents), "centroids": cents}
        print(f"  {marker}: {len(cents)} positive cells loaded")

    # Register non-reference layers into reference space
    registered = {ref_marker: per_marker[ref_marker]["centroids"]}
    for marker in layer_order[1:]:
        raw = per_marker[marker]["centroids"]
        if marker in reg_results and len(raw) > 0:
            registered[marker] = transform_centroids(raw, reg_results[marker])
            print(f"  {marker}: {len(raw)} centroids registered onto {ref_marker} space")
        else:
            registered[marker] = raw

    # Pairwise mutual-NN matching across all marker combinations
    coexpression = {}
    for i in range(len(layer_order) - 1):
        for j in range(i + 1, len(layer_order)):
            m_a, m_b = layer_order[i], layer_order[j]
            key      = f"{m_a}+{m_b}+"
            matches  = match_layers(
                registered[m_a], registered[m_b],
                max_distance_um, pixel_size_um,
            )
            null = spatial_permutation_null(
                registered[m_a], registered[m_b], len(matches),
                max_distance_um, pixel_size_um,
            )
            coexpression[key] = {
                "count":   len(matches),
                "matches": matches,
                **null,
            }
            print(f"  Co-expression {key}: {len(matches)} matched cells  "
                  f"(null {null['null_mean']:.1f}±{null['null_std']:.1f}, "
                  f"z={null['z_score']}, p={null['p_value']})")

    return {
        "per_marker":   {k: {"positive": v["positive"]} for k, v in per_marker.items()},
        "coexpression": coexpression,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cross-type spatial association pipeline (Ripley's K / g(r))
# ──────────────────────────────────────────────────────────────────────────────

def _build_analysis_window(layer_order, registered, reg_results, pixel_size_um,
                           ref_image_path, layer_images, certified_roi_polygon):
    """Build the cross-section analysis window used by BOTH the bandwidth pre-flight and
    the statistic, so they measure on the identical support:

        window = A_tissue ∩ B_tissue(registered) ∩ certification_ROI

    Regions present in only one section (folds/tears) cannot inform cross-section
    analysis, so they are excluded from area normalization, observed points, and null
    sampling alike. A drawn Certification ROI further restricts the window. Falls back to
    a bounding box (mask_method="bbox", conservative null) when no tissue mask is found.

    Returns (window|None, mask_method, area_px, overlap_iou, overlap_frac_a).
    """
    from oasis.spatial.spatial_stats import (estimate_tissue_polygon, transform_polygon,
                               intersection_window, bounding_box_area)
    mov_marker = layer_order[1] if len(layer_order) > 1 else None

    _, poly_a = estimate_tissue_polygon(ref_image_path, pixel_size_um)
    window, mask_method = None, "otsu_intersection"
    overlap_iou = overlap_frac_a = None
    if poly_a is not None:
        poly_b_in_a = None
        b_img = layer_images.get(mov_marker) if mov_marker else None
        if b_img:
            _, poly_b_native = estimate_tissue_polygon(b_img, pixel_size_um)
            if poly_b_native is not None:
                reg = reg_results.get(mov_marker)
                poly_b_in_a = (transform_polygon(poly_b_native, reg)
                               if reg else poly_b_native)
        if poly_b_in_a is not None:
            w, a_inter, iou, frac = intersection_window(poly_a, poly_b_in_a)
            if w is not None and a_inter > 0:
                window, overlap_iou, overlap_frac_a = w, iou, frac
                print(f"  Tissue mask: A∩B intersection "
                      f"{a_inter * pixel_size_um**2:.0f} µm² "
                      f"(IoU {iou:.2f}, {frac*100:.0f}% of A's tissue)")
        if window is None:
            window, mask_method = poly_a, "otsu_a_only"
            print("  Tissue mask: A-only Otsu (B mask unavailable or empty "
                  "intersection) — cross-section overlap NOT enforced")

    # LOCALLY_CERTIFIED / drawn Certification ROI: permit the statistic only inside the
    # trusted region, never across the whole field.
    if certified_roi_polygon:
        try:
            from shapely.geometry import Polygon
            roi = Polygon(certified_roi_polygon)
            if not roi.is_valid or roi.is_empty:
                raise ValueError("empty or invalid polygon")
            window = window.intersection(roi) if window is not None else roi
            if window.is_empty or window.area <= 0:
                raise ValueError("certified ROI does not overlap the tissue window")
            mask_method += "_certified_roi"
            print(f"  Certification ROI applied "
                  f"({window.area * pixel_size_um**2:.0f} µm²)")
        except Exception as e:
            raise ValueError(f"Invalid certified ROI polygon: {e}")

    area_px = window.area if window is not None else None
    if not area_px or area_px <= 0:
        all_pts = [registered[m] for m in layer_order]
        area_px = bounding_box_area(
            all_pts[0] if all_pts else np.empty((0, 2)),
            np.vstack([p for p in all_pts[1:]]) if len(all_pts) > 1
            else np.empty((0, 2)),
        )
        window, mask_method = None, "bbox"
        print(f"  Tissue mask: bounding-box fallback "
              f"({area_px:.0f} px² — null is biased/conservative without a mask)")
    return window, mask_method, area_px, overlap_iou, overlap_frac_a


def precheck_bandwidth_within_window(registered, layer_order, pixel_size_um, window,
                                     bandwidth_um=None, support=None):
    """Per-image validity of the 75 µm reweight bandwidth, measured WITHIN the analysis
    window (NOT a universal image property — the same image under a different ROI can
    give a different verdict).

    ARCHITECTURE is a property of the TISSUE, not the marker (Q3 / ihc.md §15.7): the
    75 µm reweight is size-controlled when the tissue's characteristic length ℓ̂ is
    coarser than the band. So ℓ̂ is estimated from the marker-independent ALL-CELL
    support field (`support`) when available — which is always dense — instead of
    requiring each sparse marker to self-estimate it. (Falls back to the per-marker
    positives only when no support is supplied, preserving the older behaviour.)

    POWER (whether a marker has enough positives to support a claim) is a SEPARATE
    axis from architecture, so an asymmetric field — one marker rich, the other
    sparse — is no longer force-failed. Per-marker power:
        adequate  n ≥ 30            — full test
        sparse    5 ≤ n < 30        — test still runs on the all-cell support null
                                       (does not need the sparse marker's own
                                       intensity); result carries an underpowered flag
        absent    n < 5             — no spatial arrangement to test → abundance/absence
                                       finding, NOT a failed test

    `worst_status`:
      • ok / caution                       → reweighted 75 µm primary
      • dense_tissue_bandwidth_invalid      → dense morphology-conditioned null
      • underpowered_sparse_marker          → sparse marker present → support null + flag
      • marker_absent                       → a marker has <5 positives → absence finding
      • architecture_not_estimable          → tissue scale not estimable → fail closed

    Registration certification is a SEPARATE concern and is never affected by this check.
    """
    from oasis.spatial.spatial_stats import (estimate_architecture_scale, architecture_scale_verdict,
                               filter_points_in_polygon, _REWEIGHT_BANDWIDTH_UM)
    bw = float(bandwidth_um if bandwidth_um is not None else _REWEIGHT_BANDWIDTH_UM)

    # Tissue architecture from the marker-independent all-cell support (preferred).
    ell_tissue = None
    n_support = 0
    if support is not None and len(support):
        sup = np.asarray(support, float).reshape(-1, 2)
        if window is not None:
            sup, _ = filter_points_in_polygon(sup, window)
        n_support = int(len(sup))
        ell_tissue = estimate_architecture_scale(sup, pixel_size_um, tissue_polygon=window)

    per_image = {}
    for marker in layer_order:
        pts = np.asarray(registered.get(marker, np.empty((0, 2))), float).reshape(-1, 2)
        if window is not None and len(pts):
            pts, _excl = filter_points_in_polygon(pts, window)
        n_pts = int(len(pts))
        # Architecture: prefer the tissue (support) estimate; else the marker's own.
        ell = ell_tissue if ell_tissue is not None else estimate_architecture_scale(
            pts, pixel_size_um, tissue_polygon=window)
        v = architecture_scale_verdict(ell, bandwidth_um=bw)
        status = v.get("status")
        power = ("adequate" if n_pts >= 30
                 else "sparse" if n_pts >= _SPARSE_MIN_POSITIVE else "absent")
        arch_src = "all_cell_support" if ell_tissue is not None else "marker_positives"
        if power == "absent":
            status_reason = (f"only {n_pts} positive cell(s) inside the certified analysis "
                             f"window — too few to test this marker's spatial arrangement")
        elif power == "sparse":
            status_reason = (f"sparse marker (n={n_pts}, <30) — tested on the all-cell "
                             f"support null with reduced power; architecture ℓ̂ from "
                             f"{arch_src}")
        elif status == "dense_tissue":
            status_reason = (
                f"tissue architecture scale is below the {bw:.0f} µm bandwidth "
                f"(ℓ̂={v.get('scale_um')} µm, from {arch_src}), so this is dense/fine "
                f"tissue rather than a valid 75 µm-null field")
        elif status == "caution":
            status_reason = (
                f"architecture scale is between {bw:.0f} µm and {2*bw:.0f} µm "
                f"(from {arch_src}); the 75 µm null is usable but near its boundary")
        elif status == "unknown":
            status_reason = (f"architecture scale was not estimable from the "
                             f"{arch_src} inside the certified analysis window")
        else:
            status_reason = (f"architecture scale is at least {2*bw:.0f} µm "
                             f"(from {arch_src}); the 75 µm null is size-controlled here")
        per_image[marker] = {"scale_um": v.get("scale_um"), "status": status,
                             "ok": v.get("ok"), "n": n_pts, "power": power,
                             "arch_source": arch_src,
                             "min_ok_scale_um": v.get("min_ok_scale_um"),
                             "reason": status_reason}

    absent_markers = [m for m, e in per_image.items() if e.get("power") == "absent"]
    sparse_markers = [m for m, e in per_image.items() if e.get("power") == "sparse"]
    unknown_markers = [
        m for m, e in per_image.items()
        if e.get("status") == "unknown" and e.get("power") != "absent"]
    dense_markers = [
        m for m, e in per_image.items()
        if e.get("status") in ("dense_tissue", "unreliable") and e.get("power") == "adequate"]
    caution_markers = [m for m, e in per_image.items()
                       if e.get("status") == "caution" and e.get("power") == "adequate"]

    if absent_markers:
        worst = "marker_absent"
    elif sparse_markers:
        worst = "underpowered_sparse_marker"
    elif unknown_markers:
        worst = "architecture_not_estimable"
    elif dense_markers:
        worst = "dense_tissue_bandwidth_invalid"
    elif caution_markers:
        worst = "caution"
    else:
        worst = "ok"
    valid = worst in ("ok", "caution")
    reason = {
        "ok": f"tissue architecture is coarser than {bw:.0f} µm — the reweighted "
              f"primary null is size-controlled within this window.",
        "caution": f"architecture is only marginally coarser than {bw:.0f} µm — treat "
                   f"the co-infiltration verdict with care.",
        "dense_tissue_bandwidth_invalid": (
            f"fine/dense tissue architecture is at/inside the {bw:.0f} µm interaction "
            f"band — the 75 µm reweighted null is not the right primary here; OASIS "
            f"will use the dense morphology-conditioned null if its gates pass."),
        "underpowered_sparse_marker": (
            f"one marker is sparse ({', '.join(sparse_markers)}: <30 positives) while the "
            f"other is well-populated — the cross-type test runs on the marker-independent "
            f"all-cell support null (which does not need the sparse marker's own "
            f"intensity), so segregation/association is still reported, but the result is "
            f"UNDERPOWERED and flagged as such."),
        "marker_absent": (
            f"a marker has <{_SPARSE_MIN_POSITIVE} positive cells in this window "
            f"({', '.join(absent_markers)}) — there is no spatial arrangement to test. "
            f"This is an ABUNDANCE/absence finding (report the count), not a failed test."),
        "architecture_not_estimable": (
            "the tissue architecture scale could not be estimated inside the certified "
            "analysis window — fail closed rather than guessing a null model."),
    }[worst]
    return {"bandwidth_um": bw, "window_scope": "certified_analysis_window",
            "per_image": per_image, "worst_status": worst, "valid": bool(valid),
            "reason": reason, "issue_type": worst,
            "tissue_scale_um": (round(float(ell_tissue), 1) if ell_tissue is not None else None),
            "tissue_scale_source": ("all_cell_support" if ell_tissue is not None
                                    else "marker_positives"),
            "n_support": n_support,
            "absent_markers": absent_markers,
            "sparse_markers": sparse_markers,
            "underpowered_markers": sparse_markers,   # back-compat alias
            "dense_markers": dense_markers,
            "unknown_markers": unknown_markers}


def load_detection_centroids_csv(csv_path: str, pixel_size_um: float) -> np.ndarray:
    """
    Load all-cell detection centroids from QuPath's tab-delimited detection export.
    QuPath writes centroids in microns; convert them back to reference-image pixels
    for the spatial statistic.
    """
    import csv
    pts = []
    if not csv_path:
        return np.empty((0, 2), dtype=np.float64)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                x = float(row.get("Centroid X µm", ""))
                y = float(row.get("Centroid Y µm", ""))
                pts.append((x / pixel_size_um, y / pixel_size_um))
            except Exception:
                continue
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _build_precheck_null_plan(bandwidth_precheck, registered, layer_order, window,
                              pixel_size_um, morphology_support_csv,
                              dense_auto_null, landmark_certified,
                              dense_min_positive, dense_min_support):
    """Decide, at pre-flight time, WHICH primary null a full run would use for this
    pair and expose it so the UI can name the null per pair BEFORE the expensive run.

    Three outcomes, matching the full run's selection logic (mirrors the association
    loop's dense_info gating):
      • reweighted_75um  — the 75 µm bandwidth is size-controlled in this window;
                           the calibrated intensity-reweighted primary null is used.
      • dense_morphology — the 75 µm check fails specifically because architecture is
                           dense/fine, and the dense fallback gates pass;
                           the dense morphology-conditioned null (all-cell support +
                           2 µm jitter, 10–30 µm) is used.
      • none (fail-closed)— sparse/underpowered, not-estimable, or dense fallback gates
                           fail; no robust primary null → run withheld.
    """
    from oasis.spatial.spatial_stats import filter_points_in_polygon
    valid = bool(bandwidth_precheck.get("valid"))
    worst = bandwidth_precheck.get("worst_status")
    plan = {"primary_null": None, "primary_null_key": None,
            "primary_null_label": None, "fail_closed": False,
            "reason": "", "dense": {}}

    if valid:
        plan.update({
            "primary_null": "reweighted_75um",
            "primary_null_key": "reweighted",
            "primary_null_label": "Primary null — 75 µm intensity-reweighted "
                                  "inhomogeneous cross-K",
            "reason": "Tissue architecture is coarser than the 75 µm bandwidth "
                      "within the analysis window, so the calibrated primary null "
                      "is size-controlled here.",
        })
        return plan

    if worst == "marker_absent":
        plan.update({
            "primary_null": "none",
            "fail_closed": True,
            "absence": True,
            "primary_null_label": "Absence finding — a marker is (near-)absent",
            "reason": "Not a spatial test: a marker has fewer than "
                      f"{_SPARSE_MIN_POSITIVE} positive cells in this window "
                      + "(" + ", ".join(bandwidth_precheck.get("absent_markers", [])) + "). "
                      "There is no spatial arrangement to test — report this as an "
                      "abundance/absence result, not as 'no association'.",
        })
        return plan

    if worst == "architecture_not_estimable":
        plan.update({
            "primary_null": "none",
            "fail_closed": True,
            "primary_null_label": "Fail-closed — architecture not estimable",
            "reason": "Not tested: the positive-cell architecture scale could not be "
                      "estimated inside the certified analysis window, so OASIS will "
                      "not guess a primary null.",
        })
        return plan

    if not dense_auto_null:
        plan.update({
            "primary_null": "none", "fail_closed": True,
            "reason": "Dense/fine tissue invalidates the 75 µm primary here and the "
                      "dense fallback is disabled — the run is fail-closed for this pair.",
        })
        return plan

    ref_m = layer_order[0]
    mov_m = layer_order[1] if len(layer_order) > 1 else None
    p_a = registered.get(ref_m, np.empty((0, 2), dtype=np.float64))
    p_b = (registered.get(mov_m, np.empty((0, 2), dtype=np.float64))
           if mov_m else np.empty((0, 2), dtype=np.float64))
    if window is not None:
        p_a, _ = filter_points_in_polygon(p_a, window)
        p_b, _ = filter_points_in_polygon(p_b, window)
    support = np.empty((0, 2), dtype=np.float64)
    if morphology_support_csv:
        try:
            support = load_detection_centroids_csv(morphology_support_csv, pixel_size_um)
            if window is not None and len(support):
                support, _ = filter_points_in_polygon(support, window)
        except Exception:
            support = np.empty((0, 2), dtype=np.float64)

    # The dense/support null is the vehicle for BOTH dense tissue AND a sparse marker.
    # For the sparse case the marker-count floor is relaxed to _SPARSE_MIN_POSITIVE
    # (the support null does not need the sparse marker's own intensity), and the plan
    # is flagged underpowered. For dense tissue the strict ≥dense_min_positive holds.
    is_sparse = worst == "underpowered_sparse_marker"
    pos_floor = _SPARSE_MIN_POSITIVE if is_sparse else int(dense_min_positive)
    gates = {
        "landmark_certified": bool(landmark_certified),
        "analysis_window": window is not None,
        "support_csv": bool(morphology_support_csv),
        "min_positive_a": int(len(p_a)) >= pos_floor,
        "min_positive_b": int(len(p_b)) >= pos_floor,
        "min_support": int(len(support)) >= int(dense_min_support),
    }
    failed = [k for k, ok in gates.items() if not ok]
    plan["dense"] = {
        **gates,
        "n_positive_a": int(len(p_a)), "n_positive_b": int(len(p_b)),
        "n_support": int(len(support)),
        "min_positive_required": pos_floor,
        "min_support_required": int(dense_min_support),
        "failed_gates": failed,
        "underpowered": bool(is_sparse),
    }
    if not failed:
        plan["underpowered"] = bool(is_sparse)
        if is_sparse:
            plan.update({
                "primary_null": "dense_morphology",
                "primary_null_key": "dense_morphology",
                "primary_null_label": "Underpowered — all-cell support null (sparse marker)",
                "reason": "One marker is sparse, so the cross-type test runs on the "
                          "marker-independent all-cell support null (which does not need "
                          "the sparse marker's own intensity). Segregation/association is "
                          "reported but the result is UNDERPOWERED — interpret with the "
                          "cell counts in view.",
            })
        else:
            plan.update({
                "primary_null": "dense_morphology",
                "primary_null_key": "dense_morphology",
                "primary_null_label": "Dense-tissue morphological null — dense "
                                      "morphology-conditioned cross-K (all-cell support "
                                      "+ 2 µm jitter, 10–30 µm band)",
                "reason": "The 75 µm bandwidth is not size-controlled here (dense/fine "
                          "architecture), but the dense-fallback gates pass, so the full "
                          "run will use the dense morphology-conditioned primary null.",
            })
    else:
        plan.update({
            "primary_null": "none", "fail_closed": True,
            "primary_null_label": "Fail-closed — support null unavailable",
            "reason": ("A sparse marker" if is_sparse else "Dense/fine tissue")
                      + " needs the all-cell support null here, but it is unavailable "
                      "(gate(s) failed: " + ", ".join(failed) + ") — fail-closed for this pair.",
        })
    return plan


def run_spatial_association(
    layer_geojsons: dict,
    layer_order: list,
    reg_results: dict,
    pixel_size_um: float,
    ref_image_path: str,
    max_radius_um: float = 100.0,
    radius_step_um: float = 2.0,
    n_perm: int = N_PERMUTATIONS,
    layer_images: dict = None,
    certified_roi_polygon=None,
    precheck_only: bool = False,
    morphology_support_csv: str = None,
    dense_auto_null: bool = True,
    landmark_certified: bool = False,
    dense_min_positive: int = 30,
    dense_min_support: int = 500,
    registration_radius_floor_um: float = None,
) -> dict:
    """
    Population-level cross-type spatial association for N markers.

    Replaces the mutual-nearest-neighbour matching of run_coloc() with the
    cross-type Ripley's K / pair-correlation analysis (spatial_stats.py), which
    makes no single-cell pairing claim. Reuses the existing positive-centroid
    loader and registration transform; the reference (first) marker defines the
    coordinate space.

    Two upgrades over the original single-null / single-image-mask version:
      • CALIBRATED PRIMARY null + CSR baseline (cross_k_all_nulls, default
        nulls=("reweighted","homogeneous")): the size-controlled intensity-reweighted
        inhomogeneous cross-K drives the verdict; homogeneous CSR is a diagnostic
        shared-preference flag only. (The earlier homogeneous+inhomogeneous+toroidal
        "three null" design was RETIRED as anti-conservative — see ihc.md §15.3;
        those nulls remain computable for the calibration scripts but never gate.)
      • A∩B INTERSECTION tissue window: the analysis window is the intersection of
        A's tissue mask and B's tissue mask (registered into A space), since regions
        present in only one section cannot inform cross-section analysis.

    Args:
        layer_geojsons:  {"CD8": "/path/cd8.geojson", "TIM3": "..."}
        layer_order:     ["CD8", "TIM3"]  — first entry is the reference layer
        reg_results:     {"TIM3": <reg_result_dict>}
        pixel_size_um:   µm/px of the reference image
        ref_image_path:  reference image (for A's tissue-mask estimation)
        max_radius_um:   largest evaluation radius (µm)
        radius_step_um:  radius step (µm)
        n_perm:          Monte-Carlo permutations per null
        layer_images:    {"CD8": pathA, "TIM3": pathB} — needed for B's tissue mask
                         and the A∩B intersection window

    Returns:
        {
          "per_marker": {"CD8": {"positive": N}, "TIM3": {"positive": M}},
          "association": { "CD8__TIM3": <cross_k_all_nulls result + n_a/n_b/mask> },
          "tissue_area_um2": float,                 # intersection-window area
          "tissue_mask_method": "otsu_intersection" | "otsu_a_only" | "bbox",
          "intersection_overlap_iou": float | None,
          "_registered": {marker: Nx2 array},       # for overlays; stripped from JSON
        }
    """
    from oasis.common.registration import transform_centroids
    from oasis.spatial.spatial_stats import (
        cross_k_all_nulls, estimate_tissue_polygon, transform_polygon,
        intersection_window, filter_points_in_polygon, bounding_box_area,
        _DENSE_DCLF_RMIN_UM, _DENSE_DCLF_RMAX_UM, _DENSE_MORPHOLOGY_JITTER_UM,
    )

    layer_images = layer_images or {}
    ref_marker = layer_order[0]
    mov_marker = layer_order[1] if len(layer_order) > 1 else None

    # Load positive centroids for every marker
    per_marker, centroids = {}, {}
    for marker in layer_order:
        path = layer_geojsons.get(marker)
        if path:
            cents, _ = load_positive_centroids(path)
        else:
            cents = np.empty((0, 2), dtype=np.float32)
        per_marker[marker] = {"positive": len(cents)}
        centroids[marker]  = cents
        print(f"  {marker}: {len(cents)} positive cells loaded")

    # Register non-reference layers into reference space.
    # Every transform is checked to be a SIMILARITY before any cell is moved. Cross-K
    # reads a radius r as a physical distance, which only holds if the transform
    # preserves distances up to one global scale; and the "registration error cannot
    # manufacture association" result (validate_radius_floor.py) holds only for a
    # low-DOF, cell-blind transform. A shear or non-rigid warp would silently violate
    # both, so this fails closed rather than trusting the caller.
    from oasis.spatial.serial_registration import assert_distance_preserving
    registered = {ref_marker: centroids[ref_marker]}
    for marker in layer_order[1:]:
        raw = centroids[marker]
        if reg_results.get(marker) and len(raw) > 0:
            assert_distance_preserving(reg_results[marker]["matrix"],
                                       name=f"{marker}→{ref_marker} registration")
            registered[marker] = transform_centroids(raw, reg_results[marker])
            print(f"  {marker}: {len(raw)} centroids registered onto {ref_marker} space")
        else:
            registered[marker] = raw

    # ── A∩B intersection tissue window (∩ certification ROI) ──────────────────
    window, mask_method, area_px, overlap_iou, overlap_frac_a = _build_analysis_window(
        layer_order, registered, reg_results, pixel_size_um,
        ref_image_path, layer_images, certified_roi_polygon)

    # ── Pre-flight: is the 75 µm reweight bandwidth valid WITHIN this window? ──
    # Architecture ℓ̂ is measured on the marker-INDEPENDENT all-cell support (Q3), so a
    # sparse marker no longer force-fails the pair; per-marker positive counts are graded
    # separately for power. Load the support once, up front, and reuse it downstream.
    _support_precheck = np.empty((0, 2), dtype=np.float64)
    if morphology_support_csv:
        try:
            _support_precheck = load_detection_centroids_csv(
                morphology_support_csv, pixel_size_um)
        except Exception:
            _support_precheck = np.empty((0, 2), dtype=np.float64)
    bandwidth_precheck = precheck_bandwidth_within_window(
        registered, layer_order, pixel_size_um, window, support=_support_precheck)
    _pc = bandwidth_precheck.get("per_image", {})
    print(f"  Bandwidth 75 µm pre-flight (within analysis window): "
          f"worst={bandwidth_precheck.get('worst_status')} "
          f"valid={bandwidth_precheck.get('valid')}")
    for _m in layer_order:
        _e = _pc.get(_m)
        if _e:
            print(f"     {_m}: ℓ̂={_e.get('scale_um')} µm status={_e.get('status')} "
                  f"(n={_e.get('n')})")

    # Pre-flight-only mode (the UI "Validate 75 µm bandwidth" button): return the
    # bandwidth verdict without the expensive Monte-Carlo cross-K loop. Segmentation
    # already ran, so a subsequent full run can reuse the GeoJSONs.
    if precheck_only:
        null_plan = _build_precheck_null_plan(
            bandwidth_precheck, registered, layer_order, window, pixel_size_um,
            morphology_support_csv, dense_auto_null, landmark_certified,
            dense_min_positive, dense_min_support)
        print(f"  Null plan (pre-flight): primary_null={null_plan.get('primary_null')} "
              f"fail_closed={null_plan.get('fail_closed')}")
        return {
            "per_marker":               per_marker,
            "association":              {},
            "tissue_area_um2":          (float(area_px) * pixel_size_um ** 2
                                         if area_px else None),
            "tissue_mask_method":       mask_method,
            "intersection_overlap_iou": overlap_iou,
            "intersection_overlap_frac_a": overlap_frac_a,
            "bandwidth_precheck":       bandwidth_precheck,
            "null_plan":                null_plan,
            "_registered":              registered,
        }

    morphology_support_full = np.empty((0, 2), dtype=np.float64)
    morphology_support_source = None
    if morphology_support_csv:
        try:
            morphology_support_full = load_detection_centroids_csv(
                morphology_support_csv, pixel_size_um)
            morphology_support_source = morphology_support_csv
            print(f"  Dense-null morphology support: "
                  f"{len(morphology_support_full)} all reference-section cells "
                  f"loaded from {morphology_support_csv}")
        except Exception as e:
            morphology_support_full = np.empty((0, 2), dtype=np.float64)
            morphology_support_source = morphology_support_csv
            print(f"  Dense-null morphology support unavailable: {e}")

    # Evaluation radii: 0 → max_radius_um in radius_step_um steps (µm → px)
    radii_um = np.arange(0.0, max_radius_um + radius_step_um, radius_step_um)
    radii_px = radii_um / pixel_size_um

    association = {}
    for i in range(len(layer_order) - 1):
        for j in range(i + 1, len(layer_order)):
            m_a, m_b = layer_order[i], layer_order[j]
            # Neutral key (e.g. "CD8__TIM3") — deliberately NOT "CD8+TIM3+",
            # which would read as a double-positive (co-expressing) cell pool.
            key = f"{m_a}__{m_b}"
            p_a_full, p_b_full = registered[m_a], registered[m_b]

            # Bound the observed points to the analysis window; exclude (and log)
            # any point outside the A∩B intersection.
            if window is not None:
                p_a, n_a_excl = filter_points_in_polygon(p_a_full, window)
                p_b, n_b_excl = filter_points_in_polygon(p_b_full, window)
                if n_a_excl or n_b_excl:
                    print(f"  {key}: excluded {n_a_excl} {m_a} + {n_b_excl} {m_b} "
                          f"point(s) outside the intersection window")
            else:
                p_a, p_b, n_a_excl, n_b_excl = p_a_full, p_b_full, 0, 0

            # A (near-)absent marker has no spatial arrangement to test — report it as an
            # abundance/absence finding, not "no association" (Q3 / ihc.md §15.7).
            if int(len(p_a)) < _SPARSE_MIN_POSITIVE or int(len(p_b)) < _SPARSE_MIN_POSITIVE:
                absent = [m for m, n in ((m_a, len(p_a)), (m_b, len(p_b)))
                          if int(n) < _SPARSE_MIN_POSITIVE]
                print(f"  {key}: ABSENCE — {', '.join(absent)} has <{_SPARSE_MIN_POSITIVE} "
                      f"positive cells in window; no cross-type spatial test (abundance finding)")
                association[key] = {
                    "error": "marker_absent", "absence": True,
                    "absent_markers": absent,
                    "n_a": int(len(p_a)), "n_b": int(len(p_b)),
                }
                continue

            _pf_worst = bandwidth_precheck.get("worst_status")
            _is_sparse = _pf_worst == "underpowered_sparse_marker"
            dense_info = {
                "requested": bool(
                    dense_auto_null and _pf_worst in (
                        "dense_tissue_bandwidth_invalid", "underpowered_sparse_marker")
                ),
                "underpowered": bool(_is_sparse),
                "selected": False,
                "status": ("not_needed" if bandwidth_precheck.get("valid")
                           else "not_requested"),
                "reason": "",
                "gates": {},
                "method": "image_derived_all_reference_nuclei_support_jitter",
                "support_source": morphology_support_source,
                "jitter_um": _DENSE_MORPHOLOGY_JITTER_UM,
                "dclf_band_um": [_DENSE_DCLF_RMIN_UM, _DENSE_DCLF_RMAX_UM],
                "validation_ids": [
                    "public_codex_dense_null",
                    "dense_null_image_morphology",
                    "dense_null_real_ll477",
                ],
            }
            if (not bandwidth_precheck.get("valid")
                    and not dense_info["requested"]):
                dense_info["reason"] = bandwidth_precheck.get("reason") or (
                    "75 µm bandwidth was not validated, but this is not a dense-tissue "
                    "case eligible for the dense morphology-conditioned fallback.")
            dense_support = np.empty((0, 2), dtype=np.float64)
            nulls = ("reweighted", "homogeneous")
            dclf_rmin_um, dclf_rmax_um = 10.0, 50.0
            if dense_info["requested"]:
                # Relax the per-marker count floor for the sparse case: the support null
                # does not need the sparse marker's own intensity, so ≥_SPARSE_MIN_POSITIVE
                # is enough to run it (flagged underpowered). Dense tissue keeps ≥30.
                pos_floor = _SPARSE_MIN_POSITIVE if _is_sparse else int(dense_min_positive)
                gates = {
                    "landmark_certified": bool(landmark_certified),
                    "analysis_window": window is not None,
                    "support_csv": bool(morphology_support_csv),
                    "min_positive_a": int(len(p_a)) >= pos_floor,
                    "min_positive_b": int(len(p_b)) >= pos_floor,
                }
                if window is not None and len(morphology_support_full):
                    dense_support, _support_excl = filter_points_in_polygon(
                        morphology_support_full, window)
                elif len(morphology_support_full):
                    dense_support = morphology_support_full
                gates["min_support"] = int(len(dense_support)) >= int(dense_min_support)
                dense_info["gates"] = {
                    **gates,
                    "n_positive_a": int(len(p_a)),
                    "n_positive_b": int(len(p_b)),
                    "n_support": int(len(dense_support)),
                    "min_positive_required": pos_floor,
                    "min_support_required": int(dense_min_support),
                }
                failed = [k for k, ok in gates.items() if not ok]
                if not failed:
                    dense_info["selected"] = True
                    dense_info["status"] = "selected"
                    nulls = ("dense_morphology", "homogeneous")
                    dclf_rmin_um, dclf_rmax_um = _DENSE_DCLF_RMIN_UM, _DENSE_DCLF_RMAX_UM
                    if _is_sparse:
                        dense_info["reason"] = (
                            "A marker is sparse, so the cross-type test runs on the "
                            "marker-independent all-cell support null (2 µm jitter, "
                            "10–30 µm band). Result is UNDERPOWERED and flagged as such.")
                        print(f"  {key}: sparse marker → all-cell support null "
                              f"(support={len(dense_support)}, UNDERPOWERED)")
                    else:
                        dense_info["reason"] = (
                            "75 µm bandwidth pre-flight failed, so OASIS switched to "
                            "the dense morphology-conditioned primary null "
                            "(all-cell support + 2 µm jitter, 10–30 µm DCLF band).")
                        print(f"  {key}: 75 µm pre-flight failed → switching primary null "
                              f"to dense morphology-conditioned "
                              f"(support={len(dense_support)}, band=10–30 µm)")
                else:
                    dense_info["status"] = "unavailable"
                    dense_info["reason"] = (
                        ("A sparse marker needs" if _is_sparse
                         else "75 µm bandwidth pre-flight failed and the dense fallback needs")
                        + " the all-cell support null, but gate(s) failed: "
                        + ", ".join(failed))
                    print(f"  {key}: support null unavailable — {dense_info['reason']}")

            # Smallest inter-cell distance this pair's registration error can resolve.
            # This is a REPORTING boundary, not a gate on the test: validation shows the
            # DCLF test stays correctly sized under registration error, and narrowing the
            # band to [floor, rmax] only costs power (validate_radius_floor.py). So the
            # band is left alone and the floor is recorded, so the UI can mark the curve
            # below it as unmeasurable rather than as evidence of no association.
            floor = registration_radius_floor_um
            radius_floor = {
                "floor_um": floor,
                "dclf_rmin_um": dclf_rmin_um,
                "dclf_rmax_um": dclf_rmax_um,
                "band_clipped": False,      # deliberately: clipping costs power, adds nothing
                "curve_interpretable_from_um": max(float(floor or 0.0), 0.0) or None,
                "contact_scale_resolved": bool(floor is not None and floor <= 20.0),
            }
            if floor is not None and floor >= max_radius_um:
                # The error exceeds every radius evaluated: nothing on the curve can be
                # read, so there is no result to report. Fail closed for this pair.
                radius_floor["reason"] = (
                    f"registration error leaves no interpretable radius below the "
                    f"{max_radius_um:.0f} µm maximum evaluated (floor {floor:.1f} µm)")
                print(f"  {key}: BLOCKED — radius floor {floor:.1f} µm ≥ max radius "
                      f"{max_radius_um:.0f} µm; no interpretable radii remain")
                association[key] = {
                    "error": "no_interpretable_radius_band",
                    "radius_floor": radius_floor,
                    "n_a": int(len(p_a)), "n_b": int(len(p_b)),
                }
                continue
            if floor is not None and floor > dclf_rmin_um:
                radius_floor["reason"] = (
                    f"registration error resolves distances only above {floor:.1f} µm; "
                    f"the curve below that is unmeasurable, not null. The DCLF test still "
                    f"runs over its usual {dclf_rmin_um:.0f}–{dclf_rmax_um:.0f} µm band "
                    f"(correctly sized under this error; narrowing it would only lose power).")
                print(f"  {key}: curve interpretable from {floor:.1f} µm "
                      f"(cell-cell contact scale NOT resolved); DCLF band unchanged at "
                      f"{dclf_rmin_um:.0f}–{dclf_rmax_um:.0f} µm")

            res = cross_k_all_nulls(
                p_a, p_b, radii_px, area_px, pixel_size_um,
                n_perm=n_perm, seed=_NULL_SEED, tissue_polygon=window,
                dclf_rmin_um=dclf_rmin_um, dclf_rmax_um=dclf_rmax_um,
                nulls=nulls, morphology_support=dense_support,
                registration_radius_floor_um=floor,
            )
            res["radius_floor"] = radius_floor
            res["n_a"] = int(len(p_a))
            res["n_b"] = int(len(p_b))
            res["n_a_excluded"] = int(n_a_excl)
            res["n_b_excluded"] = int(n_b_excl)
            res["tissue_mask_method"] = mask_method
            res["intersection_overlap_iou"] = overlap_iou
            res["intersection_overlap_frac_a"] = overlap_frac_a
            res["primary_null_selection"] = {
                "primary_null": res.get("primary_null"),
                "bandwidth_75um_valid": bool(bandwidth_precheck.get("valid")),
                "auto_switch_enabled": bool(dense_auto_null),
                "dense_fallback": dense_info,
            }

            # Architecture-scale guard (audit A6 / ihc.md §15.5): the reweighted
            # primary is size-controlled only when tissue architecture is coarser than
            # the bandwidth. Measure ℓ̂ per marker (worst case) and flag any 'robust'
            # verdict whose architecture scale is too fine to trust — turning the
            # disclosed assumption into a measured guard. Calibrated by
            # validation/validate_architecture_scale.py.
            try:
                from oasis.spatial.spatial_stats import (estimate_architecture_scale,
                                            architecture_scale_verdict)
                ell_a = estimate_architecture_scale(p_a, pixel_size_um, tissue_polygon=window)
                ell_b = estimate_architecture_scale(p_b, pixel_size_um, tissue_polygon=window)
                ells = [e for e in (ell_a, ell_b) if e is not None]
                arch = architecture_scale_verdict(min(ells) if ells else None)
                arch["scale_per_marker_um"] = {
                    m_a: (round(ell_a, 1) if ell_a is not None else None),
                    m_b: (round(ell_b, 1) if ell_b is not None else None)}
                res["architecture_scale"] = arch
                rob = res.get("robustness") or {}
                if (res.get("primary_null") == "reweighted"
                        and rob.get("verdict") == "robust"
                        and arch.get("ok") is False):
                    rob["architecture_caution"] = True
                    rob["architecture_note"] = (
                        f"Architecture scale ℓ̂≈{arch['scale_um']}µm is below the "
                        f"{arch['min_ok_scale_um']:.0f}µm needed for a size-controlled "
                        f"reweighted test (status={arch['status']}); this 'robust' "
                        f"verdict may be anti-conservative — treat with caution.")
                    res["robustness"] = rob
            except Exception as _arch_e:
                res["architecture_scale"] = {"status": "error", "error": str(_arch_e)}

            association[key] = res

            g = res.get("global", {})
            rob = res.get("robustness", {})
            print(f"  Association {key}: n_a={len(p_a)} n_b={len(p_b)}  "
                  f"primary({res.get('primary_null')}) "
                  f"{'SIGNIFICANT '+str(g.get('direction')) if g.get('significant') else 'n.s.'} "
                  f"(p={g.get('global_p_dclf')})  ROBUSTNESS={rob.get('verdict')}")
            print(f"     {rob.get('summary')}")

    return {
        "per_marker":               per_marker,
        "association":              association,
        "tissue_area_um2":          float(area_px) * pixel_size_um ** 2,
        "tissue_mask_method":       mask_method,
        "intersection_overlap_iou": overlap_iou,
        "intersection_overlap_frac_a": overlap_frac_a,
        "bandwidth_precheck":       bandwidth_precheck,
        "_registered":              registered,
    }
