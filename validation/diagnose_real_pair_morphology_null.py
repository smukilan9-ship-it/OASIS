"""
diagnose_real_pair_morphology_null.py

Real-pair diagnostic for a dense-tissue, marker-independent morphology-conditioned
spatial null. This is validation tooling only; it does not change the production
OASIS pipeline.

The script reuses a completed OASIS spatial result bundle:
  * positive-cell GeoJSONs from the bundle,
  * all-cell detection CSVs from the bundle,
  * the certified landmark transform from the bundle,
  * the same A∩B tissue-intersection window construction as OASIS.

Then it asks whether redrawing population B from reference-section morphology
fields changes the dense-tissue verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration import _load_rgb_thumbnail, extract_hematoxylin, transform_centroids  # noqa: E402
from spatial import load_positive_centroids  # noqa: E402
from spatial_stats import (  # noqa: E402
    _k_from_counts,
    _null_summary_from_k,
    _pair_counts,
    estimate_tissue_polygon,
    filter_points_in_polygon,
    intersection_window,
    transform_polygon,
)


BANDS = [(5.0, 20.0), (10.0, 30.0), (10.0, 50.0)]


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _find_one(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Could not find {label}")
    return paths[0]


def _resolve_image(filename: str, roots: list[Path]) -> Path:
    for root in roots:
        hits = list(root.rglob(filename))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not resolve raw image {filename!r} under {roots}")


def _summary_pixel_size(pair_dir: Path, filename: str) -> float:
    hits = list(pair_dir.glob(f"{filename}*summary.json"))
    if not hits:
        raise FileNotFoundError(f"Could not find summary JSON for {filename}")
    return float(_read_json(hits[0])["pixel_size_um"])


def _load_detection_centroids_px(csv_path: Path, pixel_size_um: float) -> np.ndarray:
    points = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                x_um = float(row["Centroid X µm"])
                y_um = float(row["Centroid Y µm"])
            except Exception:
                continue
            points.append((x_um / pixel_size_um, y_um / pixel_size_um))
    return np.asarray(points, dtype=np.float64)


def _robust01(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    vals = arr[mask] if mask is not None else arr.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if not np.isfinite(hi) or hi <= lo:
        hi = float(vals.max()) if vals.size else 1.0
        lo = float(vals.min()) if vals.size else 0.0
    out = (arr.astype(np.float64) - lo) / max(hi - lo, 1e-9)
    return np.clip(out, 0.0, 1.0)


def _window_mask(window, shape: tuple[int, int], scale: float):
    import shapely

    h, w = shape
    yy, xx = np.indices((h, w))
    x_full = (xx.astype(np.float64) + 0.5) / scale
    y_full = (yy.astype(np.float64) + 0.5) / scale
    mask = shapely.contains_xy(window, x_full, y_full)
    return mask, x_full, y_full


def _pmf_from_field(field: np.ndarray, mask: np.ndarray, floor_frac: float = 0.02) -> np.ndarray:
    weights = np.asarray(field, dtype=np.float64).copy()
    weights[~mask] = 0.0
    inside = weights[mask]
    baseline = floor_frac * max(float(np.mean(inside[inside > 0])) if np.any(inside > 0) else 1.0, 1e-9)
    weights[mask] = weights[mask] + baseline
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        weights = mask.astype(np.float64)
        total = float(weights.sum())
    return weights / total


def _sample_from_pmf(
    pmf: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    window,
    n: int,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    import shapely

    flat = pmf.ravel()
    out = []
    max_iter = 20
    half = 0.5 / scale
    while len(out) < n and max_iter > 0:
        need = n - len(out)
        idx = rng.choice(flat.size, size=max(need * 2, need), p=flat)
        iy, ix = np.unravel_index(idx, pmf.shape)
        xs = x_full[iy, ix] + rng.uniform(-half, half, len(idx))
        ys = y_full[iy, ix] + rng.uniform(-half, half, len(idx))
        keep = shapely.contains_xy(window, xs, ys)
        out.extend(np.column_stack([xs[keep], ys[keep]]).tolist())
        max_iter -= 1
    if len(out) < n:
        idx = rng.choice(flat.size, size=n - len(out), p=flat)
        iy, ix = np.unravel_index(idx, pmf.shape)
        out.extend(np.column_stack([x_full[iy, ix], y_full[iy, ix]]).tolist())
    return np.asarray(out[:n], dtype=np.float64)


def _morphology_cross_k(
    points_a: np.ndarray,
    points_b: np.ndarray,
    area_px: float,
    pixel_size_um: float,
    pmf: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    window,
    scale: float,
    n_perm: int,
    seed: int,
    rmin_um: float,
    rmax_um: float,
) -> dict:
    radii_um = np.arange(0.0, 100.0 + 2.0, 2.0)
    radii_px = radii_um / pixel_size_um
    n_a, n_b = len(points_a), len(points_b)
    obs_counts = _pair_counts(points_a, points_b, radii_px)
    obs_k_px = _k_from_counts(obs_counts, float(area_px), n_a, n_b)
    obs_lmr_px = np.sqrt(np.clip(obs_k_px, 0.0, None) / np.pi) - radii_px

    rng = np.random.default_rng(seed)
    tree_a = cKDTree(points_a)
    norm = float(area_px) / (n_a * n_b)
    null_k = np.empty((n_perm, len(radii_px)), dtype=np.float64)
    for i in range(n_perm):
        b_star = _sample_from_pmf(pmf, x_full, y_full, window, n_b, rng, scale)
        counts = tree_a.count_neighbors(cKDTree(b_star), radii_px)
        null_k[i] = norm * np.asarray(counts, dtype=np.float64)

    summary = _null_summary_from_k(
        radii_px,
        obs_k_px,
        obs_lmr_px,
        null_k,
        pixel_size_um,
        n_perm,
        rmin_um,
        rmax_um,
    )
    return {
        "global": summary["global"],
        "n_perm": int(n_perm),
        "radii_um": radii_um.tolist(),
        "L_minus_r": (obs_lmr_px * pixel_size_um).tolist(),
    }


def _build_fields(ref_image: Path, all_ref_points_px: np.ndarray, window, ref_px: float, max_side: int):
    rgb, scale = _load_rgb_thumbnail(str(ref_image), max_side=max_side)
    if rgb is None:
        raise RuntimeError(f"Could not load reference image thumbnail: {ref_image}")

    h8 = extract_hematoxylin(rgb)
    mask, x_full, y_full = _window_mask(window, h8.shape, scale)
    h_field = _robust01(gaussian_filter(h8.astype(np.float64), sigma=1.0), mask)

    density = np.zeros(h8.shape, dtype=np.float64)
    xs = np.clip((all_ref_points_px[:, 0] * scale).astype(int), 0, h8.shape[1] - 1)
    ys = np.clip((all_ref_points_px[:, 1] * scale).astype(int), 0, h8.shape[0] - 1)
    np.add.at(density, (ys, xs), 1.0)

    thumb_um_per_px = ref_px / scale
    fields = {"hematoxylin": h_field}
    for sigma_um in (8.0, 12.0, 20.0, 35.0):
        sm = gaussian_filter(density, sigma=max(sigma_um / thumb_um_per_px, 0.1))
        fields[f"all_cell_density_{sigma_um:g}um"] = _robust01(sm, mask)

    n12 = fields["all_cell_density_12um"]
    fields["hematoxylin_plus_cell_density_12um"] = _robust01(0.5 * h_field + 0.5 * n12, mask)
    fields["hematoxylin_times_cell_density_12um"] = _robust01(h_field * n12, mask)
    return fields, mask, x_full, y_full, scale


def run(args: argparse.Namespace) -> dict:
    result_dir = Path(args.result_dir).expanduser().resolve()
    if list(result_dir.glob("*_spatial_association.json")):
        pair_dir = result_dir
    else:
        pair_dir = _find_one([p for p in result_dir.iterdir() if p.is_dir()], "pair output directory")
    result_json = _find_one(list(pair_dir.glob("*_spatial_association.json")), "pair spatial JSON")
    result = _read_json(result_json)

    filename_a = result["filename_a"]
    filename_b = result["filename_b"]
    ref_px = float(result["pixel_size_ref_um"])
    mov_px = _summary_pixel_size(pair_dir, filename_b)

    roots = [Path(p).expanduser().resolve() for p in args.asset_roots]
    ref_image = Path(args.ref_image).expanduser().resolve() if args.ref_image else _resolve_image(filename_a, roots)
    mov_image = Path(args.mov_image).expanduser().resolve() if args.mov_image else _resolve_image(filename_b, roots)

    cert = result["certification"]
    reg_result = {
        "matrix": np.asarray(cert["matrix"], dtype=np.float32),
        "scale_ref": 1.0,
        "scale_mov": 1.0,
        "method": "landmark",
        "success": True,
    }

    geo_a = _find_one(list(pair_dir.glob(f"{filename_a}*detections.geojson")), "reference GeoJSON")
    geo_b = _find_one(list(pair_dir.glob(f"{filename_b}*detections.geojson")), "moving GeoJSON")
    csv_a = _find_one(list(pair_dir.glob(f"{filename_a}*detections.csv")), "reference detection CSV")

    pos_a, _ = load_positive_centroids(str(geo_a))
    pos_b_raw, _ = load_positive_centroids(str(geo_b))
    pos_b = transform_centroids(pos_b_raw, reg_result)

    _, poly_a = estimate_tissue_polygon(str(ref_image), ref_px)
    _, poly_b = estimate_tissue_polygon(str(mov_image), mov_px)
    if poly_a is None or poly_b is None:
        raise RuntimeError("Could not estimate both tissue polygons")
    window, area_px, overlap_iou, overlap_frac_a = intersection_window(poly_a, transform_polygon(poly_b, reg_result))
    if window is None:
        raise RuntimeError("A∩B tissue window is empty")

    pos_a_win, excl_a = filter_points_in_polygon(pos_a, window)
    pos_b_win, excl_b = filter_points_in_polygon(pos_b, window)
    all_ref_points_px = _load_detection_centroids_px(csv_a, ref_px)
    all_ref_win, _ = filter_points_in_polygon(all_ref_points_px, window)

    fields, mask, x_full, y_full, scale = _build_fields(
        ref_image, all_ref_points_px, window, ref_px, args.thumbnail_max_side)

    existing = result["spatial_association"]["association"]["CD8__TIM-3"]
    outputs = {
        "input": {
            "result_dir": str(result_dir),
            "pair_dir": str(pair_dir),
            "result_json": str(result_json),
            "ref_image": str(ref_image),
            "mov_image": str(mov_image),
            "filename_a": filename_a,
            "filename_b": filename_b,
            "ref_pixel_size_um": ref_px,
            "mov_pixel_size_um": mov_px,
            "certification": {
                "status": cert.get("status"),
                "tre_median_um": cert.get("tre_median_um"),
                "tre_p90_um": cert.get("tre_p90_um"),
                "n": cert.get("n"),
                "reason": cert.get("reason"),
            },
        },
        "window": {
            "area_um2": float(area_px * ref_px * ref_px),
            "overlap_iou": float(overlap_iou),
            "overlap_frac_a": float(overlap_frac_a),
            "thumbnail_scale": float(scale),
            "thumbnail_pixels_inside_window": int(mask.sum()),
        },
        "points": {
            "positive_a_total": int(len(pos_a)),
            "positive_b_total": int(len(pos_b_raw)),
            "positive_a_inside_window": int(len(pos_a_win)),
            "positive_b_inside_window": int(len(pos_b_win)),
            "positive_a_excluded": int(excl_a),
            "positive_b_excluded": int(excl_b),
            "all_reference_cells_total": int(len(all_ref_points_px)),
            "all_reference_cells_inside_window": int(len(all_ref_win)),
        },
        "existing_oasis": {
            "robustness": existing.get("robustness"),
            "spatial_validity": result.get("spatial_validity"),
        },
        "morphology_conditioned": {},
    }

    for field_name, field in fields.items():
        pmf = _pmf_from_field(field, mask, floor_frac=args.floor_frac)
        outputs["morphology_conditioned"][field_name] = {}
        for rmin, rmax in BANDS:
            key = f"{rmin:g}-{rmax:g}um"
            outputs["morphology_conditioned"][field_name][key] = _morphology_cross_k(
                pos_a_win,
                pos_b_win,
                area_px,
                ref_px,
                pmf,
                x_full,
                y_full,
                window,
                scale,
                args.nperm,
                args.seed,
                rmin,
                rmax,
            )

    return outputs


def write_report(outputs: dict, out_md: Path) -> None:
    lines = []
    lines.append("# Real Dense-Pair Morphology-Conditioned Null Diagnostic")
    lines.append("")
    inp = outputs["input"]
    pts = outputs["points"]
    win = outputs["window"]
    lines.append(f"- Pair: `{inp['filename_a']}` vs `{inp['filename_b']}`")
    lines.append(f"- Reference pixel size: {inp['ref_pixel_size_um']:.6g} um/px")
    lines.append(f"- Moving pixel size: {inp['mov_pixel_size_um']:.6g} um/px")
    lines.append(f"- Certification: {inp['certification']['status']}, TRE median {inp['certification']['tre_median_um']} um, p90 {inp['certification']['tre_p90_um']} um, n={inp['certification']['n']}")
    lines.append(f"- A∩B window: {win['area_um2']:.1f} um^2, IoU {win['overlap_iou']:.3f}, frac(A) {win['overlap_frac_a']:.3f}")
    lines.append(f"- Positives inside window: A={pts['positive_a_inside_window']} / {pts['positive_a_total']}, B={pts['positive_b_inside_window']} / {pts['positive_b_total']}")
    lines.append(f"- All reference cells inside window: {pts['all_reference_cells_inside_window']} / {pts['all_reference_cells_total']}")
    lines.append("")
    lines.append("## Existing OASIS Verdict")
    robust = outputs["existing_oasis"]["robustness"]
    valid = outputs["existing_oasis"]["spatial_validity"] or {}
    lines.append(f"- Robustness: `{robust.get('verdict')}`")
    lines.append(f"- Reweighted p: {robust.get('per_null_global_p', {}).get('reweighted')}")
    lines.append(f"- Homogeneous CSR p: {robust.get('per_null_global_p', {}).get('homogeneous')}")
    lines.append(f"- 75 um validity: {valid.get('bandwidth_75um_valid')} ({valid.get('worst_status')})")
    lines.append("")
    lines.append("## Morphology-Conditioned Results")
    lines.append("")
    lines.append("| Morphology field | DCLF band (um) | p_DCLF | p_assoc | direction | significant | peak r (um) |")
    lines.append("|---|---:|---:|---:|---|---|---:|")
    for field_name, by_band in outputs["morphology_conditioned"].items():
        for band, res in by_band.items():
            g = res["global"]
            lines.append(
                f"| {field_name} | {band} | {g['global_p_dclf']:.5g} | "
                f"{g['global_p_association']:.5g} | {g['direction']} | "
                f"{g['significant']} | {g['peak_r_um']} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This is a real-image diagnostic, not a production validation. It uses the completed OASIS detections, certified transform, and A∩B tissue window.")
    lines.append("- The null is marker-independent only if the chosen morphology field is genuinely independent of the marker-positive population being tested.")
    lines.append("- Agreement across hematoxylin, all-cell-density, and combined fields would support a stable dense-tissue null. Disagreement means the morphology field definition is a scientific dependency that must be validated before shipping dense mode.")
    lines.append("- A significant morphology-conditioned result does not prove same-cell co-expression or direct contact; it remains a population-level serial-section association after conditioning on the chosen tissue architecture field.")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="/Users/mukilan/Desktop/test1")
    parser.add_argument("--asset-roots", nargs="+", default=["/Users/mukilan/Desktop/assets"])
    parser.add_argument("--ref-image")
    parser.add_argument("--mov-image")
    parser.add_argument("--nperm", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thumbnail-max-side", type=int, default=1024)
    parser.add_argument("--floor-frac", type=float, default=0.02)
    args = parser.parse_args()

    outputs = run(args)
    out_dir = Path(args.result_dir).expanduser().resolve()
    out_json = out_dir / "real_pair_morphology_null_diagnostic.json"
    out_md = out_dir / "real_pair_morphology_null_diagnostic.md"
    out_json.write_text(json.dumps(outputs, indent=2))
    write_report(outputs, out_md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
