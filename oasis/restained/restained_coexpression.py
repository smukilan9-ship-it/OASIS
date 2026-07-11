"""
Same-section restained co-expression analysis.

This module is deliberately separate from the serial-section spatial pipeline.
It segments one hematoxylin reference, applies those unchanged nucleus polygons
to two already-corresponding AEC restains, and reports per-cell co-expression.
No registration or registration certification is performed here: inputs must be
pre-registered images of the same physical section and must have equal pixel
dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from oasis.quant.cell_expansion import (_feature_polygon, _halfplane_clip, _load_rgb_full,
                            _mask_stats)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs", ".ndpi"}
H_VECTOR = np.asarray([0.650, 0.704, 0.286], dtype=np.float64)
AEC_VECTOR = np.asarray([0.274, 0.680, 0.680], dtype=np.float64)
BACKGROUND = np.asarray([255.0, 255.0, 255.0], dtype=np.float64)
PREPROCESS_PERCENTILES = (1.0, 99.0)
PREPROCESS_TARGET_MAX_OD = 1.2


def _normalise_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _stain_matrix():
    hematoxylin = _normalise_vector(H_VECTOR)
    aec = _normalise_vector(AEC_VECTOR)
    residual = _normalise_vector(np.cross(hematoxylin, aec))
    return np.asarray([hematoxylin, aec, residual], dtype=np.float64)


def stain_channels(rgb):
    """Return H and AEC optical-density channels using fixed H/AEC vectors."""
    pixels = np.asarray(rgb, dtype=np.float64)[..., :3]
    od = -np.log10((pixels + 1.0) / BACKGROUND.reshape(1, 1, 3))
    concentrations = od @ np.linalg.inv(_stain_matrix())
    return (concentrations[..., 0].astype(np.float32),
            concentrations[..., 1].astype(np.float32))


def preprocess_hematoxylin(image_path, output_path):
    """Create a fixed, reproducible H-only contrast-normalised RGB image.

    The 1st and 99th percentiles are computed only over positive H optical
    density. The stretched H channel is reconstructed with the fixed
    hematoxylin vector. There are intentionally no per-image tuning controls.
    """
    rgb = _load_rgb_full(str(image_path))
    h_od, _ = stain_channels(rgb)
    positive = h_od[np.isfinite(h_od) & (h_od > 0)]
    if positive.size < 16:
        raise ValueError("Hematoxylin preprocessing found too few positive-OD pixels")
    low, high = np.percentile(positive, PREPROCESS_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Hematoxylin preprocessing could not determine a valid OD range")
    scaled = np.clip((h_od - low) / (high - low), 0.0, 1.0)
    reconstructed_od = (scaled[..., None] * PREPROCESS_TARGET_MAX_OD
                        * _normalise_vector(H_VECTOR).reshape(1, 1, 3))
    processed = np.clip(BACKGROUND.reshape(1, 1, 3)
                        * np.power(10.0, -reconstructed_od), 0, 255).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(processed, mode="RGB").save(output_path)
    return {
        "method": "fixed_H_AEC_colour_deconvolution_and_robust_H_stretch",
        "percentiles": list(PREPROCESS_PERCENTILES),
        "positive_h_od_low": float(low),
        "positive_h_od_high": float(high),
        "target_max_od": PREPROCESS_TARGET_MAX_OD,
        "hematoxylin_vector_rgb": H_VECTOR.tolist(),
        "aec_vector_rgb": AEC_VECTOR.tolist(),
        "output": str(output_path),
    }


def discover_bundles(folder, h_token="_Hematoxylin", marker_a_token="_CD8",
                     marker_b_token="_FoxP3", reference_mask_folder=None):
    """Discover complete same-section bundles by case-insensitive stem suffix."""
    folder = Path(os.path.expanduser(str(folder)))
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    token_map = {"hematoxylin": h_token, "marker_a": marker_a_token,
                 "marker_b": marker_b_token}
    found = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem_lower = path.stem.lower()
        for role, token in token_map.items():
            token_lower = str(token).lower()
            if token_lower and stem_lower.endswith(token_lower):
                base = path.stem[:-len(token)]
                found.setdefault(base.lower(), {"sample_id": base})[role] = str(path)
                break

    mask_index = {}
    if reference_mask_folder:
        mask_dir = Path(os.path.expanduser(str(reference_mask_folder)))
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Reference-mask folder does not exist: {mask_dir}")
        mask_index = {p.stem.lower(): str(p) for p in mask_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    complete, incomplete = [], []
    for key, item in sorted(found.items()):
        missing = [role for role in token_map if role not in item]
        if missing:
            incomplete.append({"sample_id": item["sample_id"], "missing": missing})
            continue
        item["reference_mask"] = mask_index.get(key)
        complete.append(item)
    return complete, incomplete


def _image_dimensions(path):
    with Image.open(path) as image:
        return tuple(image.size)


def validate_bundle_dimensions(bundle):
    paths = {"hematoxylin": bundle["hematoxylin"],
             "marker_a": bundle["marker_a"], "marker_b": bundle["marker_b"]}
    if bundle.get("reference_mask"):
        paths["reference_mask"] = bundle["reference_mask"]
    dimensions = {key: _image_dimensions(value) for key, value in paths.items()}
    unique = set(dimensions.values())
    if len(unique) != 1:
        detail = ", ".join(f"{key}={size[0]}x{size[1]}" for key, size in dimensions.items())
        raise ValueError("Same-section bundle dimension mismatch; registration is not run in "
                         f"this workflow ({detail})")
    return {key: list(value) for key, value in dimensions.items()}


def _load_geojson(path):
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise ValueError(f"Invalid detection GeoJSON: {path}")
    # QuPath exports can include the selected parent annotation beside its
    # child detections. It is an ROI, not a nucleus, and must never enter cell
    # counts or pixel masks. Features with no objectType remain supported for
    # generic/synthetic GeoJSON.
    data["features"] = [feature for feature in data["features"]
                        if str((feature.get("properties") or {}).get(
                            "objectType", "detection")).lower() == "detection"]
    return data


def _geometry_compartments(features, pixel_size_um, expansion_um):
    """Return fixed nuclei and Voronoi-clipped outer rings, aligned to features."""
    from scipy.spatial import cKDTree

    nuclei = []
    for feature in features:
        polygon = _feature_polygon(feature)
        try:
            polygon = polygon.buffer(0) if polygon is not None else None
            if polygon is not None and polygon.is_empty:
                polygon = None
        except Exception:
            polygon = None
        nuclei.append(polygon)

    valid = [index for index, polygon in enumerate(nuclei) if polygon is not None]
    centroids = (np.asarray([[nuclei[i].centroid.x, nuclei[i].centroid.y] for i in valid])
                 if valid else np.empty((0, 2), dtype=float))
    position = {feature_index: centroid_index for centroid_index, feature_index in enumerate(valid)}
    tree = cKDTree(centroids) if len(centroids) else None
    expansion_px = float(expansion_um) / float(pixel_size_um)
    rings = [None] * len(nuclei)
    cells = [None] * len(nuclei)

    for index, nucleus in enumerate(nuclei):
        if nucleus is None:
            continue
        centroid_index = position[index]
        centre = centroids[centroid_index]
        expanded = nucleus.buffer(expansion_px)
        min_x, min_y, max_x, max_y = expanded.bounds
        radius = max(np.hypot(x - centre[0], y - centre[1])
                     for x in (min_x, max_x) for y in (min_y, max_y))
        clipped = expanded
        if tree is not None:
            for neighbour in tree.query_ball_point(centre, 2.0 * radius):
                if neighbour == centroid_index:
                    continue
                clipped = _halfplane_clip(clipped, centre, centroids[neighbour])
                if clipped.is_empty:
                    break
        cells[index] = clipped
        rings[index] = clipped.difference(nucleus) if not clipped.is_empty else None
    return nuclei, rings, cells


def _measure_geometries(channel, geometries):
    height, width = channel.shape
    values = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            values.append(None)
            continue
        min_x, min_y, max_x, max_y = geometry.bounds
        x0, y0 = max(int(math.floor(min_x)), 0), max(int(math.floor(min_y)), 0)
        x1, y1 = min(int(math.ceil(max_x)), width), min(int(math.ceil(max_y)), height)
        if x1 <= x0 or y1 <= y0:
            values.append(None)
            continue
        stats, = _mask_stats([geometry], channel, x0, y0, x1 - x0, y1 - y0)
        values.append(stats["mean"])
    return values


def measure_restained_markers(marker_a_path, marker_b_path, features, pixel_size_um,
                              expansion_um=2.0, compartment_a="ring",
                              compartment_b="nucleus"):
    nuclei, rings, cells = _geometry_compartments(features, pixel_size_um, expansion_um)
    _, aec_a = stain_channels(_load_rgb_full(marker_a_path))
    _, aec_b = stain_channels(_load_rgb_full(marker_b_path))
    geometry_a = rings if compartment_a == "ring" else nuclei
    geometry_b = rings if compartment_b == "ring" else nuclei
    values_a = _measure_geometries(aec_a, geometry_a)
    values_b = _measure_geometries(aec_b, geometry_b)
    return values_a, values_b, nuclei, rings, cells


def summarize_coexpression(positive_a, positive_b):
    from scipy.stats import fisher_exact

    a = np.asarray(positive_a, dtype=bool)
    b = np.asarray(positive_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Marker classifications are not aligned")
    total = int(a.size)
    double = int(np.sum(a & b))
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    a_positive, b_positive = double + a_only, double + b_only
    expected = (a_positive * b_positive / total) if total else None
    enrichment = (double / expected) if expected and expected > 0 else None
    table = [[double, a_only], [b_only, neither]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided") if total else (None, None)
    denominator = math.sqrt(a_positive * (total - a_positive)
                            * b_positive * (total - b_positive)) if total else 0.0
    phi = ((double * neither - a_only * b_only) / denominator) if denominator else None
    pct = lambda count: (100.0 * count / total) if total else 0.0
    return {
        "total_cells": total,
        "marker_a_positive": a_positive,
        "marker_b_positive": b_positive,
        "marker_a_only": a_only,
        "marker_b_only": b_only,
        "double_positive": double,
        "double_negative": neither,
        "marker_a_only_pct": pct(a_only),
        "marker_b_only_pct": pct(b_only),
        "double_positive_pct": pct(double),
        "double_negative_pct": pct(neither),
        "expected_double_positive_independence": expected,
        "double_positive_enrichment": enrichment,
        "fisher_odds_ratio": float(odds_ratio) if odds_ratio is not None else None,
        "fisher_p_value": float(p_value) if p_value is not None else None,
        "phi_coefficient": float(phi) if phi is not None else None,
        "contingency_table": table,
    }


def _draw_geometry(draw, geometry, fill, outline):
    if geometry is None or geometry.is_empty:
        return
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        points = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(points) >= 3:
            draw.polygon(points, fill=fill)
            draw.line(points + [points[0]], fill=outline, width=2)


def create_overlay(reference_path, nuclei, positive_a, positive_b, output_path):
    base = Image.open(reference_path).convert("RGB").convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    palette = {
        (False, False): ((110, 110, 110, 55), (80, 80, 80, 210)),
        (True, False): ((255, 0, 40, 125), (255, 0, 40, 255)),
        (False, True): ((0, 105, 255, 125), (0, 85, 255, 255)),
        (True, True): ((255, 0, 230, 145), (230, 0, 210, 255)),
    }
    for geometry, a_pos, b_pos in zip(nuclei, positive_a, positive_b):
        fill, outline = palette[(bool(a_pos), bool(b_pos))]
        _draw_geometry(draw, geometry, fill, outline)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, layer).convert("RGB").save(output_path)
    return str(output_path)


def validate_segmentation_reference(reference_mask_path, nuclei, pixel_size_um,
                                    tolerance_um=5.0):
    """Compare detections with blue-instance/green-boundary dataset masks."""
    import cv2
    from validation.validate_segmentation import detection_and_classification

    mask_rgb = np.asarray(Image.open(reference_mask_path).convert("RGB"))
    blue = ((mask_rgb[..., 2] > 127) & (mask_rgb[..., 0] < 127)).astype(np.uint8)
    reference_binary = ((mask_rgb[..., 1] > 127) | (mask_rgb[..., 2] > 127)).astype(np.uint8)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(blue, connectivity=8)
    gt_xy = np.asarray([centroids[i] for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] > 0],
                       dtype=float).reshape(-1, 2)
    pred_xy = np.asarray([[p.centroid.x, p.centroid.y] for p in nuclei if p is not None],
                         dtype=float).reshape(-1, 2)
    detection = detection_and_classification(
        gt_xy, [False] * len(gt_xy), pred_xy, [False] * len(pred_xy),
        tol_px=float(tolerance_um) / float(pixel_size_um))["detection"]

    pred_binary = np.zeros(reference_binary.shape, dtype=np.uint8)
    for polygon in nuclei:
        if polygon is None:
            continue
        polygons = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
        for part in polygons:
            exterior = np.rint(np.asarray(part.exterior.coords, dtype=float)).astype(np.int32)
            if len(exterior) >= 3:
                cv2.fillPoly(pred_binary, [exterior], 1)
    intersection = int(np.sum((pred_binary > 0) & (reference_binary > 0)))
    pred_area, gt_area = int(pred_binary.sum()), int(reference_binary.sum())
    union = pred_area + gt_area - intersection
    detection.update({
        "tolerance_um": float(tolerance_um),
        "pixel_dice": (2.0 * intersection / (pred_area + gt_area)) if pred_area + gt_area else 1.0,
        "pixel_iou": (intersection / union) if union else 1.0,
        "reference_mask_format": "blue_nuclear_interiors_green_boundaries",
    })
    return detection


def _write_cell_outputs(geojson, output_dir, sample_id, label_a, label_b,
                        values_a, values_b, positive_a, positive_b, nuclei,
                        compartment_a, compartment_b):
    categories = []
    rows = []
    for index, (feature, value_a, value_b, pos_a, pos_b, nucleus) in enumerate(zip(
            geojson["features"], values_a, values_b, positive_a, positive_b, nuclei), start=1):
        measurable = (value_a is not None and value_b is not None
                      and np.isfinite(value_a) and np.isfinite(value_b))
        category = ("UNMEASURED" if not measurable else
                    f"{label_a}+/{label_b}+" if pos_a and pos_b else
                    f"{label_a}+/{label_b}-" if pos_a else
                    f"{label_a}-/{label_b}+" if pos_b else f"{label_a}-/{label_b}-")
        categories.append(category)
        properties = feature.setdefault("properties", {})
        measurements = properties.get("measurements")
        if not isinstance(measurements, dict):
            measurements = {}
        measurements.update({
            f"{label_a}: AEC OD mean ({compartment_a})": value_a,
            f"{label_b}: AEC OD mean ({compartment_b})": value_b,
        })
        properties["measurements"] = measurements
        properties["classification"] = {"name": category}
        centroid = [nucleus.centroid.x, nucleus.centroid.y] if nucleus is not None else [None, None]
        rows.append({
            "cell_id": index, "x_px": centroid[0], "y_px": centroid[1],
            "nucleus_area_px2": nucleus.area if nucleus is not None else None,
            f"{label_a}_aec_od_{compartment_a}": value_a,
            f"{label_b}_aec_od_{compartment_b}": value_b,
            f"{label_a}_positive": bool(pos_a) if measurable else None,
            f"{label_b}_positive": bool(pos_b) if measurable else None,
            "coexpression_class": category,
        })

    output_dir = Path(output_dir)
    geojson_path = output_dir / f"{sample_id}_restained_detections.geojson"
    csv_path = output_dir / f"{sample_id}_restained_cells.csv"
    with open(geojson_path, "w") as handle:
        json.dump(geojson, handle)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["cell_id"])
        writer.writeheader()
        writer.writerows(rows)
    return str(geojson_path), str(csv_path), categories


def _segment_reference(reference_path, output_dir, config):
    from run_pipeline import run_single_image

    segmentation_config = {
        "qupath_binary": os.path.expanduser(config["qupath_binary"]),
        "instanseg_model": os.path.expanduser(config["instanseg_model"]),
        "device": config.get("device", "mps"),
        "instanseg_threads": int(config.get("instanseg_threads", 4)),
        "tile_dims": 512,
        "timeout_seconds": int(config.get("timeout_seconds", 1800)),
        "mode": "automated",
        "stain_type": "hdab",
        "dab_threshold": 999.0,
        "output_dir": str(output_dir),
        "dashboard_dir": str(output_dir),
        "default_pixel_size": float(config["pixel_size_um"]),
        "pixel_size_mode": "global",
        "_pixel_size_from_ui": True,
        "export_geojson": True,
        "generate_overlays": False,
    }
    groovy_path = str(Path(output_dir) / "restained_segmentation.groovy")
    summary_path = run_single_image(str(reference_path), segmentation_config, groovy_path)
    if not summary_path:
        raise RuntimeError("Existing QuPath/InstanSeg segmentation step did not produce results")
    stem = Path(reference_path).stem
    geojson_path = Path(output_dir) / f"{stem}_detections.geojson"
    if not geojson_path.exists():
        raise RuntimeError(f"Existing segmentation step did not export GeoJSON: {geojson_path}")
    return str(geojson_path), str(summary_path)


def structural_correspondence_diagnostic(bundle, max_side=512):
    """
    DIAGNOSTIC ONLY (audit B4): zero-lag normalized cross-correlation of the
    HEMATOXYLIN channel shared by all three same-section captures. Equal image
    dimensions do NOT prove shared cell coordinates — §21.6 showed tile
    `Case2_S3_1_1` produced an extreme false double-positive purely from grossly
    non-corresponding content that happened to have equal dimensions.

    This value is REPORTED to the operator to inform manual certification; it is NOT
    an automatic pass/fail and applies NO tuned cutoff (per the honesty rule). A low
    value flags possible non-correspondence; the operator decides.
    Returns {min_corr, pairwise{...}, note} or {error}.
    """
    try:
        def hema(path):
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                s = min(max_side / max(w, h), 1.0)
                if s < 1.0:
                    im = im.resize((max(int(w * s), 1), max(int(h * s), 1)))
                return stain_channels(np.asarray(im, dtype=np.float64))[0]
        chans = {"hematoxylin": hema(bundle["hematoxylin"]),
                 "marker_a": hema(bundle["marker_a"]),
                 "marker_b": hema(bundle["marker_b"])}

        def ncc(a, b):
            a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(a @ b / d) if d > 0 else 0.0
        keys = list(chans)
        pair = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair[f"{keys[i]}__{keys[j]}"] = round(ncc(chans[keys[i]], chans[keys[j]]), 4)
        return {"min_corr": min(pair.values()), "pairwise": pair,
                "note": ("zero-lag hematoxylin NCC across the 3 same-section captures; "
                         "low values flag possible non-correspondence — operator judgement, "
                         "no auto cutoff")}
    except Exception as exc:
        return {"error": str(exc)}


def run_bundle(bundle, config, output_dir, progress=None):
    progress = progress or (lambda pct, message: None)
    sample_id = bundle.get("sample_id") or Path(bundle["hematoxylin"]).stem
    sample_dir = Path(output_dir) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    dimensions = validate_bundle_dimensions(bundle)
    progress(8, f"{sample_id}: dimensions verified; no registration run")

    # ── B4: correspondence certification gate (FAIL-CLOSED) ──────────────────────
    # Equal dimensions are NOT sufficient (§21.6). Co-expression statistics run ONLY
    # after correspondence is MANUALLY certified by the operator. Default = blocked.
    corr_diag = structural_correspondence_diagnostic(bundle)
    certified = bool(config.get("correspondence_certified")
                     or bundle.get("correspondence_certified"))
    if not certified:
        progress(100, f"{sample_id}: BLOCKED — correspondence not certified")
        blocked = {
            "sample_id": sample_id,
            "workflow": "same_physical_section_restained_coexpression",
            "validity": {
                "software_execution": "ok",
                "segmentation_valid": "not_run",
                "marker_threshold_valid": "not_run",
                "biological_validity": "blocked_uncertified",
            },
            "correspondence": {
                "certified": False, "method": "manual_required", "status": "BLOCKED",
                "diagnostic": corr_diag,
                "reason": ("Equal dimensions do NOT verify shared cell coordinates "
                           "(§21.6: Case2_S3_1_1 gave a false double-positive from "
                           "non-corresponding tissue with equal dims). Co-expression is "
                           "fail-closed until correspondence is MANUALLY certified — "
                           "set correspondence_certified=true after visually/landmark-"
                           "verifying the same-section overlay."),
            },
            "dimensions": dimensions, "inputs": bundle, "coexpression": None,
        }
        result_path = sample_dir / f"{sample_id}_restained_result.json"
        with open(result_path, "w") as handle:
            json.dump(blocked, handle, indent=2)
        blocked["artifacts"] = {"result_json": str(result_path), "output_dir": str(sample_dir)}
        return blocked

    preprocessing_enabled = bool(config.get("preprocess_hematoxylin", True))
    if preprocessing_enabled:
        segmentation_image = sample_dir / f"{sample_id}_H_preprocessed.png"
        preprocessing = preprocess_hematoxylin(bundle["hematoxylin"], segmentation_image)
    else:
        segmentation_image = Path(bundle["hematoxylin"])
        preprocessing = {"method": "disabled_raw_hematoxylin", "output": str(segmentation_image)}
    progress(22, f"{sample_id}: hematoxylin preprocessing complete")

    if config.get("detections_geojson"):
        detection_geojson = str(Path(config["detections_geojson"]).expanduser())
        segmentation_summary = None
    else:
        detection_geojson, segmentation_summary = _segment_reference(
            segmentation_image, sample_dir, config)
    geojson = _load_geojson(detection_geojson)
    progress(52, f"{sample_id}: {len(geojson['features'])} nuclei segmented once")

    pixel_size = float(config["pixel_size_um"])
    expansion = float(config.get("cell_expansion_um", 2.0))
    compartment_a = config.get("compartment_a", "ring")
    compartment_b = config.get("compartment_b", "nucleus")
    if compartment_a not in {"nucleus", "ring"} or compartment_b not in {"nucleus", "ring"}:
        raise ValueError("Marker compartments must be 'nucleus' or 'ring'")
    values_a, values_b, nuclei, rings, cells = measure_restained_markers(
        bundle["marker_a"], bundle["marker_b"], geojson["features"], pixel_size,
        expansion, compartment_a, compartment_b)
    threshold_a = float(config["threshold_a"])
    threshold_b = float(config["threshold_b"])
    positive_a = [value is not None and value >= threshold_a for value in values_a]
    positive_b = [value is not None and value >= threshold_b for value in values_b]
    measurable = [value_a is not None and value_b is not None
                  and np.isfinite(value_a) and np.isfinite(value_b)
                  for value_a, value_b in zip(values_a, values_b)]
    progress(72, f"{sample_id}: AEC measured in shared cell coordinates")

    label_a = str(config.get("label_a", "CD8"))
    label_b = str(config.get("label_b", "FOXP3"))
    detections_path, cells_csv, _ = _write_cell_outputs(
        geojson, sample_dir, sample_id, label_a, label_b, values_a, values_b,
        positive_a, positive_b, nuclei, compartment_a, compartment_b)
    overlay_path = create_overlay(
        bundle["hematoxylin"], nuclei, positive_a, positive_b,
        sample_dir / f"{sample_id}_coexpression_overlay.png")
    summary = summarize_coexpression(
        [value for value, valid in zip(positive_a, measurable) if valid],
        [value for value, valid in zip(positive_b, measurable) if valid])
    summary["detected_cells"] = len(geojson["features"])
    summary["unmeasured_cells"] = int(len(measurable) - sum(measurable))
    validation = None
    if bundle.get("reference_mask"):
        validation = validate_segmentation_reference(
            bundle["reference_mask"], nuclei, pixel_size,
            float(config.get("validation_tolerance_um", 5.0)))
    progress(90, f"{sample_id}: co-expression summary and validation complete")

    result = {
        "sample_id": sample_id,
        "workflow": "same_physical_section_restained_coexpression",
        "registration": {
            "performed": False,
            "reason": "same-section inputs are required to be pre-registered",
            "dimension_check": "passed",
        },
        "correspondence": {
            "certified": True, "method": "manual_operator_certification",
            "status": "CERTIFIED_MANUAL", "diagnostic": corr_diag,
            "note": ("Operator manually certified shared cell coordinates. Equal "
                     "dimensions alone do not verify this (§21.6); the diagnostic NCC "
                     "is advisory only."),
        },
        "validity": {
            "software_execution": "ok",
            "segmentation_valid": ("reference_checked" if bundle.get("reference_mask")
                                   else "no_ground_truth"),
            "marker_threshold_valid": "operator_supplied_AEC_thresholds_unvalidated",
            "biological_validity": "exploratory_same_section_coexpression",
        },
        "dimensions": dimensions,
        "inputs": bundle,
        "pixel_size_um": pixel_size,
        "segmentation": {
            "engine": "existing_QuPath_InstanSeg_brightfield_nuclei",
            "reference_image": str(segmentation_image),
            "source_geojson": detection_geojson,
            "summary": segmentation_summary,
            "n_detections": len(geojson["features"]),
            "preprocessing": preprocessing,
            "ground_truth_validation": validation,
        },
        "markers": {
            "a": {"label": label_a, "image": bundle["marker_a"],
                  "chromogen": "AEC", "threshold_od": threshold_a,
                  "compartment": compartment_a},
            "b": {"label": label_b, "image": bundle["marker_b"],
                  "chromogen": "AEC", "threshold_od": threshold_b,
                  "compartment": compartment_b},
            "cell_expansion_um": expansion,
            "stain_vectors": {"hematoxylin": H_VECTOR.tolist(), "aec": AEC_VECTOR.tolist()},
        },
        "coexpression": summary,
        "artifacts": {"cells_csv": cells_csv, "detections_geojson": detections_path,
                      "overlay": overlay_path, "output_dir": str(sample_dir)},
    }
    result_path = sample_dir / f"{sample_id}_restained_result.json"
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    result["artifacts"]["result_json"] = str(result_path)
    progress(100, f"{sample_id}: complete")
    return result


def _bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=float)
    running = 1.0
    for rank_index in range(values.size - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * values.size / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def run_config(config, progress=None):
    progress = progress or (lambda pct, message: print(f"RESTAINED_PROGRESS {pct} {message}"))
    required = ["pixel_size_um", "threshold_a", "threshold_b", "output_dir"]
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise ValueError("Missing required restained setting(s): " + ", ".join(missing))
    if float(config["pixel_size_um"]) <= 0:
        raise ValueError("Pixel size must be greater than zero")
    if float(config["threshold_a"]) < 0 or float(config["threshold_b"]) < 0:
        raise ValueError("AEC OD thresholds must be zero or greater")

    if config.get("mode", "single") == "batch":
        bundles, incomplete = discover_bundles(
            config["input_folder"], config.get("hematoxylin_token", "_Hematoxylin"),
            config.get("marker_a_token", "_CD8"), config.get("marker_b_token", "_FoxP3"),
            config.get("reference_mask_folder"))
    else:
        bundles = [{
            "sample_id": config.get("sample_id") or Path(config["hematoxylin_image"]).stem,
            "hematoxylin": os.path.expanduser(config["hematoxylin_image"]),
            "marker_a": os.path.expanduser(config["marker_a_image"]),
            "marker_b": os.path.expanduser(config["marker_b_image"]),
            "reference_mask": os.path.expanduser(config["reference_mask"])
                              if config.get("reference_mask") else None,
        }]
        incomplete = []
    if not bundles:
        raise ValueError("No complete hematoxylin/marker-A/marker-B bundles were found")

    base_output = Path(os.path.expanduser(config["output_dir"]))
    run_output = base_output / ("restained_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, bundle in enumerate(bundles):
        start, span = 100.0 * index / len(bundles), 100.0 / len(bundles)
        result = run_bundle(bundle, config, run_output,
                            lambda pct, msg, s=start, w=span: progress(int(s + w * pct / 100), msg))
        results.append(result)

    # Only certified bundles with computed statistics enter FDR + aggregate; blocked
    # (correspondence-uncertified) bundles contribute nothing (fail-closed, B4).
    scored = [r for r in results if r.get("coexpression")]
    n_blocked = len(results) - len(scored)
    q_values = _bh_adjust([r["coexpression"]["fisher_p_value"] for r in scored])
    for result, q_value in zip(scored, q_values):
        result["coexpression"]["fisher_q_value_bh"] = q_value
    for result in results:
        with open(result["artifacts"]["result_json"], "w") as handle:
            json.dump(result, handle, indent=2)

    aggregate_a = sum(r["coexpression"]["marker_a_only"] for r in scored)
    aggregate_b = sum(r["coexpression"]["marker_b_only"] for r in scored)
    aggregate_double = sum(r["coexpression"]["double_positive"] for r in scored)
    aggregate_neither = sum(r["coexpression"]["double_negative"] for r in scored)
    aggregate = summarize_coexpression(
        [True] * (aggregate_a + aggregate_double) + [False] * (aggregate_b + aggregate_neither),
        [False] * aggregate_a + [True] * aggregate_double + [True] * aggregate_b
        + [False] * aggregate_neither)
    combined = {
        "workflow": "same_physical_section_restained_coexpression",
        "created": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(run_output),
        "n_samples": len(results),
        "n_scored": len(scored),
        "n_blocked_uncertified": n_blocked,
        "incomplete_bundles": incomplete,
        "cohort_coexpression": aggregate,
        "fdr_method": "Benjamini-Hochberg across per-tile Fisher tests (certified tiles only)",
        "results": results,
    }
    combined_path = run_output / "restained_coexpression_results.json"
    with open(combined_path, "w") as handle:
        json.dump(combined, handle, indent=2)
    combined["result_json"] = str(combined_path)
    progress(100, "Restained co-expression run complete")
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description="Same-section restained co-expression")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    with open(args.config) as handle:
        config = yaml.safe_load(handle) or {}
    result = run_config(config)
    print("RESTAINED_RESULT=" + result["result_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
