"""
coloc.py
Cross-section marker co-expression matching.

Takes DAB-positive cell centroid lists from two (or more) IHC serial sections,
applies registration transforms, and finds co-expressing cells using
mutual nearest-neighbour matching within a configurable distance threshold.

Completely marker-agnostic: marker names are passed as strings, results are
keyed by those names, and the matching loop scales to N markers.
"""

import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON loaders
# ──────────────────────────────────────────────────────────────────────────────

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
            ring = np.array(coords[0])
            xy   = ring.mean(axis=0)[:2].tolist()
        elif gtype == "MultiPolygon" and coords:
            ring = np.array(coords[0][0])
            xy   = ring.mean(axis=0)[:2].tolist()
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
# Full co-expression pipeline (N markers)
# ──────────────────────────────────────────────────────────────────────────────

def run_coloc(
    layer_geojsons: dict,
    layer_order: list,
    reg_results: dict,
    max_distance_um: float,
    pixel_size_um: float,
) -> dict:
    """
    Full co-expression analysis for N markers.

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
    from registration import transform_centroids

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
            coexpression[key] = {"count": len(matches), "matches": matches}
            print(f"  Co-expression {key}: {len(matches)} matched cells")

    return {
        "per_marker":   {k: {"positive": v["positive"]} for k, v in per_marker.items()},
        "coexpression": coexpression,
    }
