"""
validate_dense_null_real_ll477.py

Run the dense morphology-conditioned candidate on completed real LL477 OASIS
serial-section bundles.

This is NOT a calibration harness: LL477 has no known null truth. It is a real-use
demonstration / feasibility test:
  * reuse the completed OASIS detections, certified transform, and A∩B window,
  * reject sparse pairs before testing,
  * build marker-independent morphology support from all reference-section nuclei,
  * run the candidate dense null: B* sampled from all-cell support + 2 um jitter,
    DCLF band 10-30 um.

Dense mode remains unshipped unless this is later wired into production with the
same fail-closed gates and reviewed on real certified ROIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration import transform_centroids  # noqa: E402
from spatial import load_positive_centroids  # noqa: E402
from spatial_stats import (  # noqa: E402
    _k_from_counts,
    _null_summary_from_k,
    _pair_counts,
    cross_k_null,
    estimate_tissue_polygon,
    filter_points_in_polygon,
    intersection_window,
    transform_polygon,
)


HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "dense_null_real_ll477_results.json"
OUT_MD = HERE / "dense_null_real_ll477_report.md"
BANDS = [(10.0, 30.0), (5.0, 20.0)]


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
    raise FileNotFoundError(f"Could not resolve raw image {filename!r}")


def _summary_pixel_size(pair_dir: Path, filename: str) -> float:
    hits = list(pair_dir.glob(f"{filename}*summary.json"))
    if not hits:
        raise FileNotFoundError(f"Could not find summary JSON for {filename}")
    return float(_read_json(hits[0])["pixel_size_um"])


def _load_detection_centroids_px(csv_path: Path, pixel_size_um: float) -> np.ndarray:
    pts = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                pts.append((
                    float(row["Centroid X µm"]) / pixel_size_um,
                    float(row["Centroid Y µm"]) / pixel_size_um,
                ))
            except Exception:
                continue
    return np.asarray(pts, dtype=np.float64)


class SupportJitterSampler:
    def __init__(self, support_px: np.ndarray, window, sigma_px: float):
        self.support_px = np.asarray(support_px, dtype=np.float64).reshape(-1, 2)
        self.window = window
        self.sigma_px = float(sigma_px)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        import shapely

        n = int(n)
        out = []
        batch = max(n * 4, 512)
        while len(out) < n:
            anchors = self.support_px[rng.integers(0, len(self.support_px), size=batch)]
            pts = anchors + rng.normal(0.0, self.sigma_px, size=anchors.shape)
            keep = shapely.contains_xy(self.window, pts[:, 0], pts[:, 1])
            out.extend(pts[keep].tolist())
        return np.asarray(out[:n], dtype=np.float64)


def _support_jitter_cross_k(points_a, points_b, support_px, window, area_px, ref_px,
                            jitter_um, n_perm, seed, rmin_um, rmax_um) -> dict:
    radii_um = np.arange(0.0, 100.0 + 2.0, 2.0)
    radii_px = radii_um / ref_px
    n_a, n_b = len(points_a), len(points_b)
    obs_counts = _pair_counts(points_a, points_b, radii_px)
    obs_k = _k_from_counts(obs_counts, float(area_px), n_a, n_b)
    obs_lmr = np.sqrt(np.clip(obs_k, 0.0, None) / np.pi) - radii_px

    rng = np.random.default_rng(seed)
    sampler = SupportJitterSampler(support_px, window, jitter_um / ref_px)
    tree_a = cKDTree(points_a)
    norm = float(area_px) / (n_a * n_b)
    null_k = np.empty((n_perm, len(radii_px)), dtype=float)
    for i in range(n_perm):
        b_star = sampler.sample(n_b, rng)
        counts = tree_a.count_neighbors(cKDTree(b_star), radii_px)
        null_k[i] = norm * np.asarray(counts, dtype=float)

    return _null_summary_from_k(
        radii_px, obs_k, obs_lmr, null_k, ref_px, n_perm, rmin_um, rmax_um
    )["global"]


def _load_pair(pair_dir: Path, asset_roots: list[Path]) -> dict:
    result_json = _find_one(list(pair_dir.glob("*_spatial_association.json")), "spatial JSON")
    result = _read_json(result_json)
    filename_a = result["filename_a"]
    filename_b = result["filename_b"]
    ref_px = float(result["pixel_size_ref_um"])
    mov_px = _summary_pixel_size(pair_dir, filename_b)
    ref_image = _resolve_image(filename_a, asset_roots)
    mov_image = _resolve_image(filename_b, asset_roots)

    cert = result.get("certification") or {}
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
        raise RuntimeError("Could not estimate tissue polygons")
    window, area_px, overlap_iou, overlap_frac_a = intersection_window(
        poly_a, transform_polygon(poly_b, reg_result))
    if window is None:
        raise RuntimeError("A∩B tissue window is empty")

    pos_a_win, excl_a = filter_points_in_polygon(pos_a, window)
    pos_b_win, excl_b = filter_points_in_polygon(pos_b, window)
    all_ref = _load_detection_centroids_px(csv_a, ref_px)
    all_ref_win, _ = filter_points_in_polygon(all_ref, window)
    assoc = (result.get("spatial_association") or {}).get("association") or {}
    assoc_entry = next(iter(assoc.values()), {})
    return {
        "pair_dir": str(pair_dir),
        "result_json": str(result_json),
        "sample_id": result.get("sample_id") or pair_dir.name,
        "filename_a": filename_a,
        "filename_b": filename_b,
        "ref_pixel_size_um": ref_px,
        "mov_pixel_size_um": mov_px,
        "certification": {
            "status": cert.get("status"),
            "is_certified": cert.get("is_certified"),
            "n": cert.get("n"),
            "tre_median_um": cert.get("tre_median_um"),
            "tre_p90_um": cert.get("tre_p90_um"),
        },
        "window": {
            "area_um2": float(area_px * ref_px * ref_px),
            "overlap_iou": float(overlap_iou),
            "overlap_frac_a": float(overlap_frac_a),
        },
        "counts": {
            "positive_a_total": int(len(pos_a)),
            "positive_b_total": int(len(pos_b_raw)),
            "positive_a_inside_window": int(len(pos_a_win)),
            "positive_b_inside_window": int(len(pos_b_win)),
            "positive_a_excluded": int(excl_a),
            "positive_b_excluded": int(excl_b),
            "all_reference_cells_total": int(len(all_ref)),
            "all_reference_cells_inside_window": int(len(all_ref_win)),
        },
        "_points_a": pos_a_win,
        "_points_b": pos_b_win,
        "_support": all_ref_win,
        "_window": window,
        "_area_px": area_px,
        "existing_oasis": {
            "robustness": assoc_entry.get("robustness"),
            "spatial_validity": result.get("spatial_validity"),
            "global": assoc_entry.get("global"),
        },
    }


def run(args: argparse.Namespace) -> dict:
    root = Path(args.results_root).expanduser().resolve()
    asset_roots = [Path(p).expanduser().resolve() for p in args.asset_roots]
    pair_dirs = sorted([p for p in root.iterdir() if p.is_dir() and list(p.glob("*_spatial_association.json"))])
    records = []
    for pair_dir in pair_dirs:
        row = _load_pair(pair_dir, asset_roots)
        counts = row["counts"]
        if (not row["certification"].get("is_certified")
                or counts["positive_a_inside_window"] < args.min_positive
                or counts["positive_b_inside_window"] < args.min_positive
                or counts["all_reference_cells_inside_window"] < args.min_support):
            row["status"] = "skipped"
            row["skip_reason"] = (
                f"requires certified pair, >= {args.min_positive} positives per marker "
                f"inside window, >= {args.min_support} support cells"
            )
            records.append(_strip_private(row))
            continue

        row["status"] = "tested"
        row["candidate"] = {
            "method": "image_derived_all_reference_nuclei_support_jitter",
            "support_source": "OASIS reference-section all-cell detections CSV",
            "jitter_um": args.jitter_um,
            "n_perm": args.nperm,
            "bands": {},
        }
        for rmin, rmax in BANDS:
            key = f"{rmin:g}-{rmax:g}"
            row["candidate"]["bands"][key] = _support_jitter_cross_k(
                row["_points_a"], row["_points_b"], row["_support"], row["_window"],
                row["_area_px"], row["ref_pixel_size_um"], args.jitter_um,
                args.nperm, args.seed, rmin, rmax)
        records.append(_strip_private(row))

    return {
        "parameters": vars(args),
        "records": records,
        "summary": {
            "tested": sum(1 for r in records if r["status"] == "tested"),
            "skipped": sum(1 for r in records if r["status"] == "skipped"),
            "significant_10_30": sum(
                1 for r in records if r["status"] == "tested"
                and r["candidate"]["bands"]["10-30"]["significant"]
                and r["candidate"]["bands"]["10-30"]["direction"] == "association"
            ),
        },
    }


def _strip_private(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def write_report(result: dict, out_md: Path) -> None:
    lines = [
        "# Real LL477 Dense-Null Candidate Test",
        "",
        "This is a real-use test of the dense candidate on completed OASIS serial-section bundles. It is not statistical calibration because LL477 has no known-null ground truth.",
        "",
        "## Summary",
        "",
        f"- Tested pairs: {result['summary']['tested']}",
        f"- Skipped pairs: {result['summary']['skipped']}",
        f"- Candidate 10-30 um association calls: {result['summary']['significant_10_30']}",
        "",
        "## Pair Results",
        "",
        "| Pair | Status | A+ in window | B+ in window | Support cells | Existing OASIS | Existing p(reweighted/CSR) | Dense candidate 10-30 p | Direction | Significant | Notes |",
        "|---|---|---:|---:|---:|---|---|---:|---|---|---|",
    ]
    for r in result["records"]:
        counts = r["counts"]
        robust = r.get("existing_oasis", {}).get("robustness") or {}
        pmap = robust.get("per_null_global_p") or {}
        if r["status"] == "tested":
            g = r["candidate"]["bands"]["10-30"]
            p = f"{g['global_p_dclf']:.5g}"
            direction = g["direction"]
            sig = str(g["significant"])
            notes = f"peak {g['peak_r_um']} um; TRE {r['certification']['tre_median_um']} um"
        else:
            p, direction, sig = "", "", ""
            notes = r.get("skip_reason", "")
        lines.append(
            f"| {r['sample_id']} | {r['status']} | "
            f"{counts['positive_a_inside_window']} | {counts['positive_b_inside_window']} | "
            f"{counts['all_reference_cells_inside_window']} | {robust.get('verdict')} | "
            f"{pmap.get('reweighted')}/{pmap.get('homogeneous')} | {p} | "
            f"{direction} | {sig} | {notes} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A skipped sparse pair is not evidence against association; it is an insufficient-events QC result.",
        "- A significant result here is a real-use demonstration of the candidate, not a calibrated biological claim.",
        "- Dense mode should still remain fail-closed until this candidate is wired into production with provenance, ROI, sparsity, and architecture gates.",
        "",
    ]
    out_md.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="/Users/mukilan/Desktop/ihc_spatial_results")
    parser.add_argument("--asset-roots", nargs="+", default=["/Users/mukilan/Desktop/assets"])
    parser.add_argument("--nperm", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter-um", type=float, default=2.0)
    parser.add_argument("--min-positive", type=int, default=30)
    parser.add_argument("--min-support", type=int, default=500)
    args = parser.parse_args()

    result = run(args)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    write_report(result, OUT_MD)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
