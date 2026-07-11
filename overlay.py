"""
overlay.py
Generates cell boundary overlays from QuPath GeoJSON + original image.
Uses actual cell polygon boundaries exported from QuPath.
"""

import os
import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Shared cell-boundary drawing (used by the quantification overlay AND the
# spatial-association segmentation overlays — single implementation, identical
# line weights / colour convention)
# ──────────────────────────────────────────────────────────────────────────────

def _classification_is_positive(props: dict) -> bool:
    """True when a GeoJSON detection's classification name contains 'positive'."""
    classification = props.get("classification", {})
    if isinstance(classification, dict):
        cls_name = classification.get("name", "Negative")
    else:
        cls_name = str(classification) if classification else "Negative"
    return "positive" in cls_name.lower()


def _geometry_rings(geometry: dict) -> list:
    """Flatten a (Multi)Polygon GeoJSON geometry into a list of coordinate rings."""
    gtype  = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        return list(coords)
    if gtype == "MultiPolygon":
        rings = []
        for polygon in coords:
            rings.extend(polygon)
        return rings
    return []


def _draw_cell_boundaries(
    img_bgr, features, downsample, pos_color, neg_color,
    line_thickness, show_negative,
    fill_positive=False, fill_alpha=0.40,
    ring_key=None, ring_thickness=2,
):
    """
    Draw every cell's polygon boundary onto img_bgr (BGR) from QuPath GeoJSON.

    This is the exact boundary-drawing logic used by the quantification overlay
    (positive cells in pos_color, negatives in neg_color, polylines at
    line_thickness). Optional extras used only by the spatial segmentation
    overlays — defaults leave quantification behaviour byte-identical:
      • fill_positive : translucently fill positive cells so they stand out
      • ring_key      : also draw an expanded cytoplasm-ring polygon stored under
                        properties[ring_key] (the membrane measurement region)

    pos_color / neg_color are RGB tuples. Returns (pos_count, neg_count, skipped).
    """
    import cv2

    def _bgr(rgb):
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    pos_bgr, neg_bgr = _bgr(pos_color), _bgr(neg_color)

    # ── Pass 1 (optional): translucent fill of positive cells (kept crisp by
    # blending BEFORE any outlines are drawn) ────────────────────────────────
    if fill_positive:
        fill_layer = img_bgr.copy()
        for feature in features:
            props = feature.get("properties", {})
            if not _classification_is_positive(props):
                continue
            for ring in _geometry_rings(feature.get("geometry", {})):
                try:
                    pts = np.array([[int(x / downsample), int(y / downsample)]
                                    for x, y in ring], dtype=np.int32)
                    cv2.fillPoly(fill_layer, [pts], pos_bgr)
                except Exception:
                    continue
        cv2.addWeighted(fill_layer, fill_alpha, img_bgr, 1.0 - fill_alpha, 0,
                        dst=img_bgr)

    # ── Pass 2: nucleus outlines (+ optional cytoplasm ring) ─────────────────
    pos_count = neg_count = skipped = 0
    for feature in features:
        props       = feature.get("properties", {})
        is_positive = _classification_is_positive(props)
        if not is_positive and not show_negative:
            continue
        color_bgr = pos_bgr if is_positive else neg_bgr
        rings = _geometry_rings(feature.get("geometry", {}))
        if not rings:
            skipped += 1
            continue
        try:
            for ring in rings:
                pts = np.array([[int(x / downsample), int(y / downsample)]
                                for x, y in ring], dtype=np.int32)
                cv2.polylines(img_bgr, [pts], isClosed=True,
                              color=color_bgr, thickness=line_thickness)
            # Expanded cytoplasm ring = the actual measurement region (membrane)
            if ring_key:
                ring_coords = props.get(ring_key)
                if ring_coords:
                    pts = np.array([[int(x / downsample), int(y / downsample)]
                                    for x, y in ring_coords], dtype=np.int32)
                    cv2.polylines(img_bgr, [pts], isClosed=True,
                                  color=color_bgr, thickness=ring_thickness)
            if is_positive:
                pos_count += 1
            else:
                neg_count += 1
        except Exception:
            skipped += 1
            continue
    return pos_count, neg_count, skipped


def generate_overlay(
    image_path: str,
    geojson_path: str,
    output_path: str,
    pixel_size_um: float = 0.5,
    downsample: float = 1.0,
    pos_color: tuple = (255, 50, 50),    # bright red — high contrast on H-DAB
    neg_color: tuple = (50, 205, 50),    # lime green — high contrast on blue/purple nuclei
    line_thickness: int = 2,             # thicker for visibility at downsample
    show_negative: bool = True,
) -> str:
    """
    Draw cell boundary overlays on original image using QuPath GeoJSON.

    Parameters
    ----------
    image_path      : path to original image
    geojson_path    : path to QuPath GeoJSON export
    output_path     : where to save overlay PNG
    pixel_size_um   : microns per pixel (for coordinate conversion)
    downsample      : output image downsample factor (2 = half size)
    pos_color       : RGB color for positive cells
    neg_color       : RGB color for negative cells
    line_thickness  : outline thickness in pixels
    show_negative   : draw negative cell outlines too

    Returns
    -------
    output_path on success, None on failure
    """
    try:
        import cv2
    except ImportError:
        print("  Installing opencv-python...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
        import cv2

    from PIL import Image

    if not os.path.exists(image_path):
        print(f"  ERROR: Image not found: {image_path}")
        return None

    if not os.path.exists(geojson_path):
        print(f"  ERROR: GeoJSON not found: {geojson_path}")
        return None

    # Load original image
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)

    # Downsample if needed
    if downsample != 1.0:
        new_w = int(img.shape[1] / downsample)
        new_h = int(img.shape[0] / downsample)
        img = cv2.resize(img, (new_w, new_h))

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load GeoJSON
    with open(geojson_path) as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        print("  WARNING: No features in GeoJSON")
        return None

    pos_count, neg_count, skipped = _draw_cell_boundaries(
        img_bgr, features, downsample, pos_color, neg_color,
        line_thickness, show_negative)

    print(f"  Overlay: {pos_count} positive + {neg_count} negative cells drawn ({skipped} skipped)")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
    print(f"  Overlay saved: {output_path}")
    return output_path


def generate_overlays_for_batch(
    batch_metrics: list,
    input_dir: str,
    output_dir: str,
    pixel_size_um: float = 0.5,
    downsample: float = 1.0,
):
    """
    Generate overlays for all images in a batch.
    Looks for GeoJSON files matching each image.
    """
    results = []

    for metrics in batch_metrics:
        # Image_Name is like "LL477_CD8_x10_1.tif - LL477_CD8_x10_1.tif #1"
        # We want just the filename without extension for searching
        raw_name = metrics["Image_Name"].split(" - ")[0]  # "LL477_CD8_x10_1.tif"
        image_name = os.path.splitext(raw_name)[0]  # "LL477_CD8_x10_1"

        # Try original extension first, then common ones
        extensions = [os.path.splitext(raw_name)[1]] + [".tif", ".tiff", ".png", ".jpg"]
        img_path = None
        for ext in extensions:
            candidate = os.path.join(input_dir, image_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  WARNING: Could not find image for {image_name}")
            continue

        import glob
        # QuPath adds full image name + #1 to filename, use glob to find it
        matches = glob.glob(os.path.join(output_dir, f"{image_name}*_detections.geojson"))
        if matches:
            geojson_path = matches[0]
        else:
            print(f"  WARNING: No GeoJSON found for {image_name}")
            continue

        overlay_path = os.path.join(output_dir, image_name + "_overlay.png")

        result = generate_overlay(
            image_path=img_path,
            geojson_path=geojson_path,
            output_path=overlay_path,
            pixel_size_um=metrics.get("Pixel_Size_um", pixel_size_um),
            downsample=downsample,
        )
        if result:
            results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Spatial association visualizations (cross-type Ripley's K)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_stats_box(canvas, lines, box_w=230):
    """Dark stats box (top-left), OpenCV. Mutates canvas in place."""
    import cv2
    box_h = 18 * len(lines) + 12
    cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), (20, 20, 20), -1)
    cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), (90, 90, 90), 1)
    for i, txt in enumerate(lines):
        cv2.putText(canvas, txt, (16, 28 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _vivid_color(rgb: tuple) -> tuple:
    """
    Brighten/saturate an RGB colour (push saturation and value up while keeping
    the hue) so boundaries stand out more clearly against stained tissue. Used
    for the TIM-3 (image B) overlay; image A keeps its standard colours.
    """
    import colorsys
    r, g, b = (max(0.0, min(1.0, c / 255.0)) for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1.0, s * 1.25 + 0.15)        # more saturated
    v = min(1.0, v + 0.25)               # brighter
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def generate_segmentation_overlay(
    image_path: str,
    geojson_path: str,
    output_path: str,
    stain_name: str,
    pos_color: tuple,
    neg_color: tuple = (50, 205, 50),
    line_thickness: int = 2,
    draw_cyto_ring: bool = False,
    ring_key: str = "cyto_polygon",
    ring_thickness: int = 2,
    expansion_um: float = None,
    vivid: bool = False,
    max_side: int = 1600,
) -> str:
    """
    Segmentation overlay for a spatial-association image, in the same style as the
    quantification tab: every detected cell's BOUNDARY drawn on the original image
    via the shared `_draw_cell_boundaries` logic (negatives = neg_color outline,
    positives = pos_color, translucently filled so they stand out).

    For a membrane marker measured in the cytoplasmic ring (TIM-3), set
    draw_cyto_ring=True: the nucleus boundary is drawn thin and the expanded
    cytoplasm-ring polygon (stored under properties[ring_key] by
    cell_expansion.py) is drawn on top, so the viewer sees the actual measurement
    compartment. Classification (pos/neg) reflects whatever compartment QuPath /
    the cytoplasm step wrote into the GeoJSON.

    vivid=True brightens/saturates both boundary colours (used for the TIM-3
    image so its green negatives and blue positives stand out more clearly);
    image A leaves it False to keep its standard colours.

    Stats box: stain name, total / positive counts, positivity %, (+expansion µm).
    Returns output_path on success, "" on failure.
    """
    try:
        import cv2
    except ImportError:
        print("  Segmentation overlay: opencv-python not installed")
        return ""
    from PIL import Image

    if not os.path.exists(image_path) or not os.path.exists(geojson_path):
        print(f"  Segmentation overlay: missing image/geojson for {stain_name}")
        return ""

    # Load full-resolution image (PIL, mirroring the quantification overlay),
    # downsampling so the long side fits max_side to keep the PNG manageable.
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception:
        from registration import _load_rgb_thumbnail
        thumb, _ = _load_rgb_thumbnail(image_path, max_side)
        if thumb is None:
            print(f"  Segmentation overlay: could not load {stain_name} image")
            return ""
        img = np.ascontiguousarray(thumb[..., :3])

    h0, w0 = img.shape[:2]
    downsample = max(1.0, max(h0, w0) / float(max_side))
    if downsample > 1.0:
        img = cv2.resize(img, (int(w0 / downsample), int(h0 / downsample)))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with open(geojson_path) as f:
        features = json.load(f).get("features", [])

    # Brighter, more saturated boundaries for the TIM-3 overlay (image A keeps
    # its standard colours since vivid defaults to False).
    if vivid:
        pos_color = _vivid_color(pos_color)
        neg_color = _vivid_color(neg_color)

    pos_count, neg_count, skipped = _draw_cell_boundaries(
        img_bgr, features, downsample, pos_color, neg_color,
        line_thickness, show_negative=True,
        fill_positive=True,
        ring_key=ring_key if draw_cyto_ring else None,
        ring_thickness=ring_thickness,
    )

    total = pos_count + neg_count
    pct   = (100.0 * pos_count / total) if total else 0.0
    lines = [
        f"{stain_name}",
        f"Total cells: {total}",
        f"Positive:    {pos_count}",
        f"Positivity:  {pct:.1f}%",
    ]
    if draw_cyto_ring and expansion_um is not None:
        lines.append(f"Expansion:   {expansion_um:.1f} um")
        lines.append("Outer ring = cytoplasm")
    _draw_stats_box(img_bgr, lines)

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img_bgr)
        print(f"  Segmentation overlay ({stain_name}): {pos_count}+ / {total} "
              f"cells -> {output_path}")
        return output_path
    except Exception as e:
        print(f"  Segmentation overlay: could not write {output_path}: {e}")
        return ""


def generate_consolidated_density(
    ref_image_path: str,
    points_a: np.ndarray,
    points_b: np.ndarray,
    pixel_size_um: float,
    assoc: dict,
    out_path: str,
    label_a: str = "CD8+",
    label_b: str = "TIM-3+",
    bandwidth_um: float = 30.0,
    max_side: int = 1400,
    roi_polygon=None,
) -> str:
    """
    Consolidated spatial-result image: a dual-channel density heatmap on the
    registered (CD8) coordinate space — the visual companion to the Ripley's K
    curve.

      • 2D Gaussian-smoothed density of CD8+ positions  → GREEN intensity
      • 2D Gaussian-smoothed density of TIM-3+ positions → BLUE intensity
      • blended additively over a faded grayscale of the CD8 tissue, so where
        both populations are dense the green+blue glow reads as CYAN (overlap),
        while separate green / blue glows mark spatially independent populations.

    A legend (green = label_a density, blue = label_b density, cyan = overlap) and
    a stats box with the key Ripley's K result (significant / not + peak L-r, p)
    are drawn on. OpenCV + numpy + scipy only. Returns out_path or "".
    """
    try:
        import cv2
    except ImportError:
        print("  Consolidated density: opencv-python not installed")
        return ""
    from registration import _load_rgb_thumbnail
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("  Consolidated density: scipy not installed")
        return ""

    pa = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)

    rgb, scale = _load_rgb_thumbnail(ref_image_path, max_side)
    if rgb is not None:
        bgr  = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        scale = float(scale)
    else:
        allp = np.vstack([pa, pb]) if len(pa) or len(pb) else np.zeros((1, 2))
        mx, my = allp.max(axis=0)
        scale = min(max_side / max(float(mx), float(my), 1.0), 1.0)
        H, W = int(my * scale) + 20, int(mx * scale) + 20
        H, W = max(H, 50), max(W, 50)
        gray = np.full((H, W), 60, np.uint8)

    # Faded grayscale tissue backdrop for anatomical context
    bg = (gray.astype(np.float64) * 0.30)
    canvas = np.repeat(bg[..., None], 3, axis=2)            # BGR float

    sigma = max(bandwidth_um / max(pixel_size_um, 1e-6) * scale, 1.5)

    def _density(pts):
        d = np.zeros((H, W), np.float64)
        if len(pts):
            xi = np.clip((pts[:, 0] * scale).astype(int), 0, W - 1)
            yi = np.clip((pts[:, 1] * scale).astype(int), 0, H - 1)
            np.add.at(d, (yi, xi), 1.0)
            d = gaussian_filter(d, sigma)
        m = np.percentile(d, 99.5) if d.max() > 0 else 1.0
        return np.clip(d / (m if m > 0 else 1.0), 0.0, 1.0)

    da = _density(pa)      # CD8+  → green
    db = _density(pb)      # TIM-3+ → blue

    glow = np.zeros((H, W, 3), np.float64)                  # BGR
    glow[..., 1] = da * 255.0                               # green channel (A)
    glow[..., 0] = db * 255.0                               # blue channel  (B)
    out = np.clip(canvas + glow, 0, 255).astype(np.uint8)   # additive glow

    # ── Certified analysis ROI burn-in ───────────────────────────────────────
    # When statistics were restricted to a region (LOCALLY_CERTIFIED hull or an
    # operator-drawn Certification ROI, in full-res reference coords), dim outside it
    # and draw a bright contour so the paper figure visibly reads "restricted here".
    roi_drawn = False
    if roi_polygon is not None and len(roi_polygon) >= 3:
        poly = np.array([[int(round(x * scale)), int(round(y * scale))]
                         for x, y in roi_polygon], np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        outside = m == 0
        out[outside] = (out[outside].astype(np.float64) * 0.42).astype(np.uint8)
        cv2.polylines(out, [poly], isClosed=True, color=(0, 255, 255),
                      thickness=2, lineType=cv2.LINE_AA)
        ti = int(poly[:, 1].argmin())
        tx = int(np.clip(poly[ti, 0] - 10, 6, W - 320))
        ty = int(np.clip(poly[ti, 1] - 8, 16, H - 8))
        lbl = "Certified analysis ROI - statistics restricted here"
        cv2.putText(out, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        roi_drawn = True

    # ── Legend (bottom-left) ─────────────────────────────────────────────────
    legend = [((0, 220, 0),   f"{label_a} density"),
              ((255, 90, 0),  f"{label_b} density"),
              ((220, 220, 0), "overlap")]
    if roi_drawn:
        legend.append(((0, 255, 255), "certified ROI"))
    lx, ly = 12, H - 18 - 20 * len(legend)
    cv2.rectangle(out, (lx - 6, ly - 6), (lx + 210, ly + 20 * len(legend) + 4),
                  (20, 20, 20), -1)
    cv2.rectangle(out, (lx - 6, ly - 6), (lx + 210, ly + 20 * len(legend) + 4),
                  (90, 90, 90), 1)
    for i, (col, txt) in enumerate(legend):
        yy = ly + 8 + i * 20
        cv2.rectangle(out, (lx, yy - 9), (lx + 18, yy + 5), col, -1)
        cv2.putText(out, txt, (lx + 26, yy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Stats box (top-left): the key Ripley's K result ──────────────────────
    g   = (assoc or {}).get("global", {}) or {}
    sig = bool(g.get("significant"))
    pr  = g.get("peak_r_um")
    pp  = g.get("peak_p_value")
    lines = [
        "Cross-type Ripley's K",
        "ASSOCIATED" if sig else "no sig. association",
    ]
    if pr is not None:
        lines.append(f"peak L-r @ r={pr:.0f} um")
    if pp is not None:
        lines.append(f"p = {pp:.4f}")
    _draw_stats_box(out, lines, box_w=210)

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, out)
        print(f"  Consolidated density map -> {out_path}")
        return out_path
    except Exception as e:
        print(f"  Consolidated density: could not write {out_path}: {e}")
        return ""


def generate_association_plot(
    assoc: dict,
    out_path: str,
    label_a: str = "CD8+",
    label_b: str = "TIM-3+",
) -> str:
    """
    The cross-type association figure (matplotlib): L_ab(r) - r vs distance,
    with the 95% Monte-Carlo null envelope shaded and the independence line at 0.

    Where the observed curve rises above the envelope, the populations are
    associated (co-clustered) at that scale; below the envelope indicates
    segregation. This is the publication statistical figure.

    Returns out_path on success, "" on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Association plot: installing matplotlib…")
        import subprocess, sys
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"],
                           check=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"  Association plot: matplotlib unavailable: {e}")
            return ""

    r   = np.asarray(assoc.get("radii_um", []), dtype=float)
    obs = np.asarray(assoc.get("L_minus_r", []), dtype=float)
    if r.size == 0 or obs.size == 0:
        print("  Association plot: no curve data")
        return ""

    g = assoc.get("global", {}) or {}
    nulls = assoc.get("nulls") or {}

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=140)

    # Envelope of the calibrated PRIMARY. Only the primary is overlaid; the
    # homogeneous-CSR diagnostic is reported in the result table, not drawn on this
    # axis when it is not the primary.
    null_styles = {
        "reweighted":  ("#2563eb", "Reweighted inhomogeneous cross-K (calibrated)"),
        "dense_morphology": ("#7c3aed", "Dense morphology-conditioned cross-K"),
    }
    primary_name = assoc.get("primary_null") or "reweighted"
    drew_envelope = False
    if nulls:
        nd = nulls.get(primary_name)
        styles = [(primary_name, *null_styles.get(
            primary_name, ("#2563eb", str(primary_name))))]
        if nd:
            iter_styles = styles
        else:
            iter_styles = [(nm, col, lab) for nm, (col, lab) in null_styles.items()]
        for nm, col, lab in iter_styles:
            nd = nulls.get(nm)
            if not nd:
                continue
            lo = np.asarray(nd.get("null_lower_L", []), dtype=float)
            hi = np.asarray(nd.get("null_upper_L", []), dtype=float)
            if lo.size == r.size and hi.size == r.size:
                gd = nd.get("global", {}) or {}
                sig = gd.get("significant")
                p   = gd.get("global_p_dclf")
                ax.fill_between(r, lo, hi, color=col, alpha=0.16)
                ax.plot(r, hi, color=col, lw=0.8, alpha=0.55)
                ax.plot(r, lo, color=col, lw=0.8, alpha=0.55,
                        label=f"{lab}: {('sig '+str(gd.get('direction'))) if sig else 'n.s.'} (p={p})")
                drew_envelope = True
    else:
        lo = np.asarray(assoc.get("null_lower_L", []), dtype=float)
        hi = np.asarray(assoc.get("null_upper_L", []), dtype=float)
        if lo.size == r.size and hi.size == r.size:
            ax.fill_between(r, lo, hi, color="#9ca3af", alpha=0.30,
                            label=f"95% null envelope (n={assoc.get('n_perm', 0)})")
            drew_envelope = True

    ax.axhline(0.0, color="#374151", lw=1.0, ls="--", label="Independence (L−r = 0)")
    ax.plot(r, obs, color="#111827", lw=2.2, label="Observed L−r")

    # Annotate the robustness-across-nulls verdict (the headline).
    rob = assoc.get("robustness") or {}
    verdict = rob.get("verdict")
    if verdict:
        vcol = {"robust": "#16a34a", "csr_only": "#dc2626",
                "none": "#6b7280", "mixed": "#d97706"}.get(verdict, "#6b7280")
        rmin = g.get("dclf_rmin_um", 10.0)
        rmax = g.get("dclf_rmax_um", 50.0)
        txt = f"Robustness: {verdict.upper()}\ntested r = {rmin:.0f}–{rmax:.0f} µm"
        ax.text(0.975, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=vcol, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=vcol, alpha=0.92))

    ax.set_xlabel("Distance r (µm)")
    dense_primary = primary_name == "dense_morphology"
    ax.set_ylabel("L$_{ab}$(r) − r   (µm%s)" %
                  (", dense morphology-conditioned" if dense_primary else ", intensity-reweighted"))
    ax.set_title(f"Cross-type spatial association: {label_a} ↔ {label_b}\n"
                 f"{'dense morphology-conditioned' if dense_primary else 'reweighted inhomogeneous'} "
                 f"cross-K vs calibrated null", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"  Association plot: could not write {out_path}: {e}")
        plt.close(fig)
        return ""


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python overlay.py <image_path> <geojson_path> <output_path>")
        sys.exit(1)
    generate_overlay(sys.argv[1], sys.argv[2], sys.argv[3])
