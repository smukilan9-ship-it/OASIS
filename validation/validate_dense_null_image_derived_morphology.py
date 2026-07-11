"""
validate_dense_null_image_derived_morphology.py

Validate the dense-tissue morphology-conditioned null when lambda_M(x) is
extracted from rendered H-DAB-like pixels, not taken directly from coordinates.

This bridges the gap left by validate_public_codex_dense_null.py:
  1. CODEX coordinates define real dense tissue architecture and known simulated
     marker truth.
  2. We render those all-cell coordinates into hematoxylin nuclei.
  3. We recover a marker-independent morphology field from the rendered image
     by deconvolving hematoxylin and detecting nuclei.
  4. We compare oracle-coordinate morphology vs image-derived morphology vs CSR.

This is still validation tooling, not a production dense null.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial import spatial_stats as ss  # noqa: E402
from oasis.common.registration import extract_hematoxylin  # noqa: E402
from validation.validate_public_codex_dense_null import (  # noqa: E402
    BANDS,
    PIXEL_SIZE_UM,
    _parse_floats,
    _rate_verdict,
    _sample_planted,
    homogeneous_csr_test,
    load_templates,
    summarize_pvalues,
)
from validation.datasets import resolve as dataset_resolve  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "dense_null_image_derived_morphology_results.json"
OUT_MD = HERE / "dense_null_image_derived_morphology_report.md"
OUT_EXAMPLE = HERE / "dense_null_image_derived_morphology_example.png"


@dataclass
class ImageMorphologyTemplate:
    key: str
    source_key: str
    points_um: np.ndarray
    detected_points_um: np.ndarray
    hull: object
    area_um2: float
    bbox: tuple[float, float, float, float]
    pixel_size_um: float
    image_shape: tuple[int, int]
    origin_um: tuple[float, float]
    field_correlation: float
    detection_ratio: float
    median_nn_um: float
    h_image: np.ndarray

    def sample_morphology(self, n: int, rng: np.random.Generator, sigma_um: float) -> np.ndarray:
        import shapely

        support = self.detected_points_um
        if len(support) == 0:
            support = self.points_um
        out = []
        batch = max(int(n) * 3, 256)
        sigma = float(sigma_um)
        while len(out) < n:
            anchors = support[rng.integers(0, len(support), size=batch)]
            pts = anchors + rng.normal(0.0, sigma, size=anchors.shape)
            keep = shapely.contains_xy(self.hull, pts[:, 0], pts[:, 1])
            out.extend(pts[keep].tolist())
        return np.asarray(out[:n], dtype=float)


def _ci95(rate: float, n: int) -> tuple[float, float]:
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / max(n, 1))
    return max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def _render_hdab(
    points_um: np.ndarray,
    hull,
    pixel_size_um: float,
    padding_um: float,
    nucleus_radius_um: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    import cv2

    x0, y0, x1, y1 = hull.bounds
    origin = (float(x0 - padding_um), float(y0 - padding_um))
    w = int(math.ceil((x1 - x0 + 2 * padding_um) / pixel_size_um))
    h = int(math.ceil((y1 - y0 + 2 * padding_um) / pixel_size_um))
    rgb = np.full((h, w, 3), 244, dtype=np.uint8)
    radius = max(int(round(nucleus_radius_um / pixel_size_um)), 1)
    for x_um, y_um in points_um:
        x = int(round((x_um - origin[0]) / pixel_size_um))
        y = int(round((y_um - origin[1]) / pixel_size_um))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(rgb, (x, y), radius, (116, 92, 162), -1, lineType=cv2.LINE_AA)
            cv2.circle(rgb, (x, y), max(radius - 1, 1), (92, 72, 142), -1, lineType=cv2.LINE_AA)
    return rgb, origin


def _detect_nuclei_from_render(
    rgb: np.ndarray,
    origin_um: tuple[float, float],
    pixel_size_um: float,
    min_area_um2: float,
    max_area_um2: float,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    h = extract_hematoxylin(rgb)
    blur = cv2.GaussianBlur(h, (0, 0), 1.0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_lbl, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    min_area_px = min_area_um2 / (pixel_size_um * pixel_size_um)
    max_area_px = max_area_um2 / (pixel_size_um * pixel_size_um)
    for i in range(1, n_lbl):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue
        x_px, y_px = cent[i]
        pts.append((
            origin_um[0] + float(x_px) * pixel_size_um,
            origin_um[1] + float(y_px) * pixel_size_um,
        ))
    return np.asarray(pts, dtype=float), h


def _density_image(points_um: np.ndarray, template, cell_um: float = 4.0, sigma_um: float = 8.0) -> np.ndarray:
    x0, y0, x1, y1 = template.bbox
    nx = max(int(math.ceil((x1 - x0) / cell_um)), 4)
    ny = max(int(math.ceil((y1 - y0) / cell_um)), 4)
    img = np.zeros((ny, nx), dtype=float)
    if len(points_um):
        ix = np.clip(((points_um[:, 0] - x0) / cell_um).astype(int), 0, nx - 1)
        iy = np.clip(((points_um[:, 1] - y0) / cell_um).astype(int), 0, ny - 1)
        np.add.at(img, (iy, ix), 1.0)
    return gaussian_filter(img, sigma=max(sigma_um / cell_um, 0.1))


def _field_correlation(oracle_points: np.ndarray, detected_points: np.ndarray, template) -> float:
    a = _density_image(oracle_points, template)
    b = _density_image(detected_points, template)
    av, bv = a.ravel(), b.ravel()
    if av.std() <= 0 or bv.std() <= 0:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def _median_nn_um(oracle_points: np.ndarray, detected_points: np.ndarray) -> float | None:
    if len(oracle_points) == 0 or len(detected_points) == 0:
        return None
    tree = cKDTree(detected_points)
    d, _ = tree.query(oracle_points, k=1)
    return float(np.median(d))


def image_template_from_spot(template, args) -> ImageMorphologyTemplate:
    rgb, origin = _render_hdab(
        template.points_um,
        template.hull,
        args.render_pixel_size_um,
        args.render_padding_um,
        args.nucleus_radius_um,
    )
    detected, h_img = _detect_nuclei_from_render(
        rgb,
        origin,
        args.render_pixel_size_um,
        args.min_nucleus_area_um2,
        args.max_nucleus_area_um2,
    )
    import shapely
    if len(detected):
        keep = shapely.contains_xy(template.hull, detected[:, 0], detected[:, 1])
        detected = detected[keep]
    corr = _field_correlation(template.points_um, detected, template)
    med_nn = _median_nn_um(template.points_um, detected)
    ratio = float(len(detected) / len(template.points_um)) if len(template.points_um) else 0.0
    return ImageMorphologyTemplate(
        key=f"{template.key}_image_morphology",
        source_key=template.key,
        points_um=template.points_um,
        detected_points_um=detected,
        hull=template.hull,
        area_um2=template.area_um2,
        bbox=template.bbox,
        pixel_size_um=args.render_pixel_size_um,
        image_shape=h_img.shape,
        origin_um=origin,
        field_correlation=corr,
        detection_ratio=ratio,
        median_nn_um=med_nn if med_nn is not None else float("nan"),
        h_image=h_img,
    )


def morphology_conditioned_test_with_sampler(
    sampler_template,
    area_template,
    a: np.ndarray,
    b: np.ndarray,
    null_sigma_um: float,
    n_perm: int,
    seed: int,
) -> dict[str, dict]:
    radii_um = np.arange(0.0, 100.0 + 2.0, 2.0)
    n_a, n_b = len(a), len(b)
    obs_counts = ss._pair_counts(a, b, radii_um)
    obs_k = ss._k_from_counts(obs_counts, area_template.area_um2, n_a, n_b)
    obs_lmr = np.sqrt(np.clip(obs_k, 0.0, None) / np.pi) - radii_um

    rng = np.random.default_rng(seed)
    tree_a = cKDTree(a)
    norm = area_template.area_um2 / (n_a * n_b)
    null_k = np.empty((n_perm, len(radii_um)), dtype=float)
    for i in range(n_perm):
        b_star = sampler_template.sample_morphology(n_b, rng, null_sigma_um)
        null_counts = tree_a.count_neighbors(cKDTree(b_star), radii_um)
        null_k[i] = norm * np.asarray(null_counts, dtype=float)

    out = {}
    for rmin, rmax in BANDS:
        key = f"{rmin:g}-{rmax:g}"
        out[key] = ss._null_summary_from_k(
            radii_um, obs_k, obs_lmr, null_k, 1.0, n_perm, rmin, rmax
        )["global"]
    return out


def _save_example(image_template: ImageMorphologyTemplate, out_path: Path) -> None:
    import cv2

    h = image_template.h_image
    vis = cv2.cvtColor(h, cv2.COLOR_GRAY2RGB)
    origin = image_template.origin_um
    s = image_template.pixel_size_um
    for x_um, y_um in image_template.detected_points_um[:5000]:
        x = int(round((x_um - origin[0]) / s))
        y = int(round((y_um - origin[1]) / s))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 2, (255, 80, 40), 1)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def run(args: argparse.Namespace) -> dict:
    csv_path = Path(args.csv).expanduser() if args.csv else dataset_resolve.resolve("codex_crc")
    if csv_path is None or not Path(csv_path).exists():
        raise FileNotFoundError("CODEX CRC CSV not found. Run dataset acquisition or pass --csv.")

    templates = load_templates(Path(csv_path), args.spot_cap, args.min_cells_per_spot)
    if not templates:
        raise RuntimeError("No CODEX spot templates qualified")

    image_templates = [image_template_from_spot(t, args) for t in templates]
    if args.save_example and image_templates:
        _save_example(image_templates[0], OUT_EXAMPLE)

    rng = np.random.default_rng(args.seed)
    records = []
    for sidx, (oracle, img_tpl) in enumerate(zip(templates, image_templates)):
        for rep in range(args.sims_per_spot):
            for gen_sigma in _parse_floats(args.generator_sigmas_um):
                local_seed = int(rng.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(local_seed)
                a = oracle.sample_morphology(args.n_a, local_rng, gen_sigma)
                b0 = oracle.sample_morphology(args.n_b, local_rng, gen_sigma)

                scenarios = [("h0", None, b0)]
                for jitter in _parse_floats(args.power_jitters_um):
                    bp = _sample_planted(
                        oracle, a, args.n_b, local_rng, gen_sigma,
                        jitter, args.planted_fraction)
                    scenarios.append((f"power_{jitter:g}um", jitter, bp))

                for scenario, _jitter, b in scenarios:
                    if args.include_homogeneous:
                        res = homogeneous_csr_test(oracle, a, b, args.nperm, local_seed + 11)
                        for band, g in res.items():
                            records.append(_record("homogeneous_csr", oracle, img_tpl, scenario, gen_sigma, None, band, g))

                    for null_sigma in _parse_floats(args.null_sigmas_um):
                        oracle_res = morphology_conditioned_test_with_sampler(
                            oracle, oracle, a, b, null_sigma, args.nperm, local_seed + 31)
                        image_res = morphology_conditioned_test_with_sampler(
                            img_tpl, oracle, a, b, null_sigma, args.nperm, local_seed + 61)
                        for band, g in oracle_res.items():
                            records.append(_record("oracle_coordinate_morphology", oracle, img_tpl, scenario, gen_sigma, null_sigma, band, g))
                        for band, g in image_res.items():
                            records.append(_record("image_derived_nuclei_morphology", oracle, img_tpl, scenario, gen_sigma, null_sigma, band, g))
        print(f"  image-derived morphology dense-null validation: {sidx + 1}/{len(templates)} spots")

    return _summarize(args, csv_path, templates, image_templates, records)


def _record(method, oracle, img_tpl, scenario, gen_sigma, null_sigma, band, g):
    return {
        "method": method,
        "spot": oracle.key,
        "scenario": scenario,
        "generator_sigma_um": float(gen_sigma),
        "null_sigma_um": None if null_sigma is None else float(null_sigma),
        "band_um": band,
        "p_dclf": g["global_p_dclf"],
        "p_assoc": g["global_p_association"],
        "direction": g["direction"],
        "significant": bool(g["significant"]),
        "image_field_correlation": img_tpl.field_correlation,
        "image_detection_ratio": img_tpl.detection_ratio,
        "image_median_nn_um": img_tpl.median_nn_um,
    }


def _summarize(args, csv_path, templates, image_templates, records):
    summary = {}
    for method in sorted({r["method"] for r in records}):
        summary[method] = {}
        for band in sorted({r["band_um"] for r in records if r["method"] == method}):
            summary[method][band] = {}
            for null_sigma in sorted({r["null_sigma_um"] for r in records if r["method"] == method},
                                     key=lambda x: -1 if x is None else x):
                key = "none" if null_sigma is None else f"{null_sigma:g}"
                summary[method][band][key] = {}
                for gen_sigma in _parse_floats(args.generator_sigmas_um):
                    vals = [r for r in records if r["method"] == method
                            and r["band_um"] == band
                            and r["null_sigma_um"] == null_sigma
                            and r["generator_sigma_um"] == gen_sigma
                            and r["scenario"] == "h0"]
                    if vals:
                        summary[method][band][key][f"gen_{gen_sigma:g}um_h0"] = summarize_pvalues(
                            [float(r["p_dclf"]) for r in vals])
                for scenario in sorted({r["scenario"] for r in records if r["scenario"].startswith("power_")}):
                    vals = [r for r in records if r["method"] == method
                            and r["band_um"] == band
                            and r["null_sigma_um"] == null_sigma
                            and r["scenario"] == scenario]
                    if vals:
                        summary[method][band][key][scenario] = {
                            "n": len(vals),
                            "association_power": round(float(np.mean([
                                bool(r["significant"]) and r["direction"] == "association"
                                for r in vals
                            ])), 4),
                            "median_p": round(float(np.median([float(r["p_dclf"]) for r in vals])), 4),
                        }

    field_metrics = {
        "spots": len(image_templates),
        "field_correlation_mean": round(float(np.mean([t.field_correlation for t in image_templates])), 4),
        "field_correlation_median": round(float(np.median([t.field_correlation for t in image_templates])), 4),
        "detection_ratio_mean": round(float(np.mean([t.detection_ratio for t in image_templates])), 4),
        "detection_ratio_median": round(float(np.median([t.detection_ratio for t in image_templates])), 4),
        "median_nn_um_median": round(float(np.nanmedian([t.median_nn_um for t in image_templates])), 4),
    }

    decisions = []
    for method in ("oracle_coordinate_morphology", "image_derived_nuclei_morphology"):
        for band, by_sigma in summary.get(method, {}).items():
            for null_sigma, stats in by_sigma.items():
                if null_sigma == "none":
                    continue
                h0_rates = [v["p05"] for k, v in stats.items() if k.endswith("_h0")]
                powers = [v["association_power"] for k, v in stats.items() if k.startswith("power_")]
                worst_h0 = max(h0_rates) if h0_rates else None
                min_power = min(powers) if powers else None
                if worst_h0 is None or min_power is None:
                    decision = "incomplete"
                elif worst_h0 > args.max_type1:
                    decision = "do_not_ship_anti_conservative"
                elif min_power < args.min_power:
                    decision = "do_not_ship_underpowered"
                elif method.startswith("image") and field_metrics["field_correlation_median"] < args.min_field_corr:
                    decision = "do_not_ship_morphology_field_not_recovered"
                else:
                    decision = "passes_screen"
                decisions.append({
                    "method": method,
                    "band_um": band,
                    "null_sigma_um": float(null_sigma),
                    "worst_h0_p05": worst_h0,
                    "min_power": min_power,
                    "decision": decision,
                })

    return {
        "dataset": {
            "name": "Schurch CRC CODEX CellNeighs rendered to H-DAB-like morphology",
            "csv": str(csv_path),
            "coordinate_pixel_size_um": PIXEL_SIZE_UM,
            "spots_used": len(templates),
            "spot_keys": [t.key for t in templates],
        },
        "parameters": vars(args),
        "field_metrics": field_metrics,
        "records": records,
        "summary": summary,
        "decisions": decisions,
        "example_png": str(OUT_EXAMPLE) if args.save_example else None,
    }


def write_report(result: dict, out_md: Path) -> None:
    p = result["parameters"]
    fm = result["field_metrics"]
    lines = [
        "# Dense Null: Image-Derived Morphology Validation",
        "",
        "This validation renders real CODEX cell-coordinate architecture into hematoxylin-like pixels, extracts a morphology field from the rendered image, and compares it with the oracle coordinate morphology field.",
        "",
        "## Storage",
        "",
        "The harness is lean by default: generated images are kept in memory. It writes one JSON report, one Markdown report, and optionally one example PNG.",
        "",
        "## Dataset And Settings",
        "",
        f"- Source table: `{result['dataset']['csv']}`",
        f"- Spots used: {result['dataset']['spots_used']}",
        f"- Render pixel size: {p['render_pixel_size_um']} um/px",
        f"- Simulated marker counts: A={p['n_a']}, B={p['n_b']}",
        f"- Simulations per spot: {p['sims_per_spot']}",
        f"- Permutations per test: {p['nperm']}",
        f"- Null sigmas: `{p['null_sigmas_um']}` um",
        f"- Generator sigmas: `{p['generator_sigmas_um']}` um",
        "",
        "## Morphology-Recovery Metrics",
        "",
        f"- Field correlation mean/median: {fm['field_correlation_mean']} / {fm['field_correlation_median']}",
        f"- Detected/true nuclei ratio mean/median: {fm['detection_ratio_mean']} / {fm['detection_ratio_median']}",
        f"- Median nearest detected nucleus distance: {fm['median_nn_um_median']} um",
        "",
        "## Calibration Table",
        "",
        "| Method | Band (um) | Null sigma | Generator H0 | p<=0.05 | 95% CI | Verdict | Power 5um | Power 12um |",
        "|---|---:|---:|---|---:|---|---|---:|---:|",
    ]
    for method, by_band in result["summary"].items():
        for band, by_sigma in by_band.items():
            for null_sigma, stats in by_sigma.items():
                for h0_key in [k for k in stats if k.endswith("_h0")]:
                    h0 = stats[h0_key]
                    p5 = stats.get("power_5um", {}).get("association_power")
                    p12 = stats.get("power_12um", {}).get("association_power")
                    lines.append(
                        f"| {method} | {band} | {null_sigma} | {h0_key.replace('_h0', '')} | "
                        f"{h0['p05']:.3f} | [{h0['ci95'][0]:.3f}, {h0['ci95'][1]:.3f}] | "
                        f"{h0['verdict']} | "
                        f"{'' if p5 is None else f'{p5:.3f}'} | "
                        f"{'' if p12 is None else f'{p12:.3f}'} |"
                    )
    lines += [
        "",
        "## Decisions",
        "",
        "| Method | Band (um) | Null sigma (um) | Worst H0 p<=0.05 | Min power | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for d in result["decisions"]:
        lines.append(
            f"| {d['method']} | {d['band_um']} | {d['null_sigma_um']:.3g} | "
            f"{d['worst_h0_p05']:.3f} | {d['min_power']:.3f} | {d['decision']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Oracle-coordinate morphology is the statistical upper bound.",
        "- Image-derived morphology is the actual blocker for a dense production null.",
        "- Dense mode should ship only if image-derived morphology controls H0 near 5%, preserves planted-positive power, and recovers the morphology field well enough.",
        "- Passing rendered CODEX is still not final production validation on real LL477 H-DAB serial sections; it is the required bridge before real-pair demonstration.",
        "",
    ]
    out_md.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--spot-cap", type=int, default=30)
    parser.add_argument("--min-cells-per-spot", type=int, default=800)
    parser.add_argument("--sims-per-spot", type=int, default=3)
    parser.add_argument("--nperm", type=int, default=99)
    parser.add_argument("--n-a", type=int, default=75)
    parser.add_argument("--n-b", type=int, default=75)
    parser.add_argument("--generator-sigmas-um", default="2,5,10")
    parser.add_argument("--null-sigmas-um", default="2")
    parser.add_argument("--power-jitters-um", default="5,12")
    parser.add_argument("--planted-fraction", type=float, default=0.75)
    parser.add_argument("--render-pixel-size-um", type=float, default=0.5)
    parser.add_argument("--render-padding-um", type=float, default=20.0)
    parser.add_argument("--nucleus-radius-um", type=float, default=1.8)
    parser.add_argument("--min-nucleus-area-um2", type=float, default=4.0)
    parser.add_argument("--max-nucleus-area-um2", type=float, default=80.0)
    parser.add_argument("--include-homogeneous", action="store_true", default=True)
    parser.add_argument("--no-homogeneous", dest="include_homogeneous", action="store_false")
    parser.add_argument("--save-example", action="store_true", default=True)
    parser.add_argument("--no-save-example", dest="save_example", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-type1", type=float, default=0.07)
    parser.add_argument("--min-power", type=float, default=0.80)
    parser.add_argument("--min-field-corr", type=float, default=0.80)
    args = parser.parse_args()

    result = run(args)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    write_report(result, OUT_MD)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    if args.save_example:
        print(f"Wrote {OUT_EXAMPLE}")


if __name__ == "__main__":
    main()
