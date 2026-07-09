"""
validate_public_codex_dense_null.py

Dense-tissue null calibration on PUBLIC real tissue architectures.

Dataset
-------
Schürch et al. CRC CODEX single-cell table
Mendeley Data 10.17632/mpjzbtfgfr.1

Purpose
-------
Use real dense CRC cell-coordinate fields as marker-independent architecture
templates, then simulate known-truth marker populations on top:

  H0: A and B are independently sampled from the same total-cell architecture.
  H1: B is partly planted near A, with the rest sampled from architecture.

This calibrates candidate dense nulls without assuming any biological marker pair
in the public dataset is a true null. It validates the statistic/null behavior,
not OASIS segmentation or serial-section registration.

No production code is changed by this script.
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
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spatial_stats as ss  # noqa: E402
from validation.datasets import resolve as dataset_resolve  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "public_codex_dense_null_results.json"
OUT_MD = HERE / "public_codex_dense_null_report.md"
PIXEL_SIZE_UM = 0.3775
BANDS = [(5.0, 20.0), (10.0, 30.0)]


@dataclass
class SpotTemplate:
    key: str
    points_um: np.ndarray
    hull: object
    area_um2: float
    bbox: tuple[float, float, float, float]

    def sample_morphology(self, n: int, rng: np.random.Generator, sigma_um: float) -> np.ndarray:
        import shapely

        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=float)
        out = []
        batch = max(n * 3, 256)
        sigma = float(sigma_um)
        while len(out) < n:
            anchors = self.points_um[rng.integers(0, len(self.points_um), size=batch)]
            if sigma > 0:
                pts = anchors + rng.normal(0.0, sigma, size=anchors.shape)
            else:
                pts = anchors.copy()
            keep = shapely.contains_xy(self.hull, pts[:, 0], pts[:, 1])
            out.extend(pts[keep].tolist())
        return np.asarray(out[:n], dtype=float)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        import shapely

        x0, y0, x1, y1 = self.bbox
        out = []
        batch = max(n * 4, 512)
        while len(out) < n:
            pts = np.column_stack([
                rng.uniform(x0, x1, batch),
                rng.uniform(y0, y1, batch),
            ])
            keep = shapely.contains_xy(self.hull, pts[:, 0], pts[:, 1])
            out.extend(pts[keep].tolist())
        return np.asarray(out[:n], dtype=float)


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _ci95(rate: float, n: int) -> tuple[float, float]:
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / max(n, 1))
    return max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def _rate_verdict(rate: float, n: int, lo: float = 0.03, hi: float = 0.07) -> str:
    ci_lo, ci_hi = _ci95(rate, n)
    if lo <= rate <= hi:
        return "pass"
    if ci_hi < lo:
        return "conservative"
    if ci_lo > hi:
        return "anti_conservative"
    return "borderline"


def load_templates(csv_path: Path, spot_cap: int, min_cells: int) -> list[SpotTemplate]:
    from shapely.geometry import MultiPoint

    df = pd.read_csv(
        csv_path,
        usecols=["patients", "spots", "X:X", "Y:Y"],
        low_memory=False,
    )
    rows = []
    for key, gdf in df.groupby(["patients", "spots"]):
        if len(gdf) < min_cells:
            continue
        xy_um = gdf[["X:X", "Y:Y"]].to_numpy(float) * PIXEL_SIZE_UM
        hull = MultiPoint([tuple(p) for p in xy_um]).convex_hull
        if hull.is_empty or hull.area <= 0:
            continue
        rows.append((len(gdf), SpotTemplate(
            key=f"{key[0]}_{key[1]}",
            points_um=xy_um,
            hull=hull,
            area_um2=float(hull.area),
            bbox=tuple(float(x) for x in hull.bounds),
        )))
    rows.sort(key=lambda x: x[0], reverse=True)
    if spot_cap > 0:
        rows = rows[:spot_cap]
    return [r[1] for r in rows]


def _sample_planted(
    template: SpotTemplate,
    a: np.ndarray,
    n_b: int,
    rng: np.random.Generator,
    gen_sigma_um: float,
    attraction_jitter_um: float,
    planted_fraction: float,
) -> np.ndarray:
    import shapely

    m = int(round(n_b * planted_fraction))
    bg_n = max(n_b - m, 0)
    bg = template.sample_morphology(bg_n, rng, gen_sigma_um)
    planted = []
    batch = max(m * 3, 256)
    while len(planted) < m:
        anchors = a[rng.integers(0, len(a), size=batch)]
        pts = anchors + rng.normal(0.0, attraction_jitter_um, size=anchors.shape)
        keep = shapely.contains_xy(template.hull, pts[:, 0], pts[:, 1])
        planted.extend(pts[keep].tolist())
    planted = np.asarray(planted[:m], dtype=float)
    return np.vstack([planted, bg]) if len(bg) else planted


def morphology_conditioned_test(
    template: SpotTemplate,
    a: np.ndarray,
    b: np.ndarray,
    null_sigma_um: float,
    n_perm: int,
    seed: int,
) -> dict[str, dict]:
    radii_um = np.arange(0.0, 100.0 + 2.0, 2.0)
    n_a, n_b = len(a), len(b)
    obs_counts = ss._pair_counts(a, b, radii_um)
    obs_k = ss._k_from_counts(obs_counts, template.area_um2, n_a, n_b)
    obs_lmr = np.sqrt(np.clip(obs_k, 0.0, None) / np.pi) - radii_um

    rng = np.random.default_rng(seed)
    tree_a = cKDTree(a)
    norm = template.area_um2 / (n_a * n_b)
    null_k = np.empty((n_perm, len(radii_um)), dtype=float)
    for i in range(n_perm):
        b_star = template.sample_morphology(n_b, rng, null_sigma_um)
        null_counts = tree_a.count_neighbors(cKDTree(b_star), radii_um)
        null_k[i] = norm * np.asarray(null_counts, dtype=float)

    out = {}
    for rmin, rmax in BANDS:
        key = f"{rmin:g}-{rmax:g}"
        out[key] = ss._null_summary_from_k(
            radii_um,
            obs_k,
            obs_lmr,
            null_k,
            1.0,
            n_perm,
            rmin,
            rmax,
        )["global"]
    return out


def homogeneous_csr_test(
    template: SpotTemplate,
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int,
    seed: int,
) -> dict[str, dict]:
    radii_um = np.arange(0.0, 100.0 + 2.0, 2.0)
    return {
        f"{rmin:g}-{rmax:g}": ss.cross_k_null(
            a, b, radii_um, template.area_um2, 1.0,
            n_perm=n_perm, seed=seed, tissue_polygon=template.hull,
            dclf_rmin_um=rmin, dclf_rmax_um=rmax,
        )["global"]
        for rmin, rmax in BANDS
    }


def summarize_pvalues(pvalues: list[float]) -> dict:
    arr = np.asarray(pvalues, dtype=float)
    n = int(len(arr))
    rate = float(np.mean(arr <= 0.05)) if n else float("nan")
    ci = _ci95(rate, n) if n else (float("nan"), float("nan"))
    return {
        "n": n,
        "p05": round(rate, 4),
        "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "mean_p": round(float(arr.mean()), 4) if n else None,
        "median_p": round(float(np.median(arr)), 4) if n else None,
        "verdict": _rate_verdict(rate, n) if n else "no_data",
    }


def run(args: argparse.Namespace) -> dict:
    csv_path = Path(args.csv).expanduser() if args.csv else dataset_resolve.resolve("codex_crc")
    if csv_path is None or not Path(csv_path).exists():
        raise FileNotFoundError("CODEX CRC CSV not found. Run dataset acquisition or pass --csv.")
    csv_path = Path(csv_path)

    gen_sigmas = _parse_floats(args.generator_sigmas_um)
    null_sigmas = _parse_floats(args.null_sigmas_um)
    templates = load_templates(csv_path, args.spot_cap, args.min_cells_per_spot)
    if not templates:
        raise RuntimeError("No CODEX spot templates qualified")

    rng = np.random.default_rng(args.seed)
    records = []
    for sidx, template in enumerate(templates):
        for rep in range(args.sims_per_spot):
            for gen_sigma in gen_sigmas:
                local_seed = int(rng.integers(0, 2**31 - 1))
                local_rng = np.random.default_rng(local_seed)
                a = template.sample_morphology(args.n_a, local_rng, gen_sigma)
                b0 = template.sample_morphology(args.n_b, local_rng, gen_sigma)

                scenarios = [("h0", None, b0)]
                for jitter in _parse_floats(args.power_jitters_um):
                    bp = _sample_planted(
                        template, a, args.n_b, local_rng, gen_sigma,
                        jitter, args.planted_fraction)
                    scenarios.append((f"power_{jitter:g}um", jitter, bp))

                for scenario, jitter, b in scenarios:
                    if args.include_homogeneous:
                        h = homogeneous_csr_test(
                            template, a, b, args.nperm, local_seed + 17)
                        for band, g in h.items():
                            records.append({
                                "method": "homogeneous_csr",
                                "spot": template.key,
                                "scenario": scenario,
                                "generator_sigma_um": gen_sigma,
                                "null_sigma_um": None,
                                "band_um": band,
                                "p_dclf": g["global_p_dclf"],
                                "p_assoc": g["global_p_association"],
                                "direction": g["direction"],
                                "significant": bool(g["significant"]),
                            })
                    for null_sigma in null_sigmas:
                        res = morphology_conditioned_test(
                            template, a, b, null_sigma, args.nperm,
                            local_seed + int(null_sigma * 100) + 31)
                        for band, g in res.items():
                            records.append({
                                "method": "total_cell_morphology_jitter",
                                "spot": template.key,
                                "scenario": scenario,
                                "generator_sigma_um": gen_sigma,
                                "null_sigma_um": null_sigma,
                                "band_um": band,
                                "p_dclf": g["global_p_dclf"],
                                "p_assoc": g["global_p_association"],
                                "direction": g["direction"],
                                "significant": bool(g["significant"]),
                            })
        print(f"  CODEX dense-null calibration: {sidx + 1}/{len(templates)} spots")

    summary = {}
    methods = sorted({r["method"] for r in records})
    for method in methods:
        summary[method] = {}
        for band in sorted({r["band_um"] for r in records if r["method"] == method}):
            summary[method][band] = {}
            for null_sigma in sorted({r["null_sigma_um"] for r in records
                                      if r["method"] == method}, key=lambda x: -1 if x is None else x):
                key = "none" if null_sigma is None else f"{null_sigma:g}"
                summary[method][band][key] = {}
                for gen_sigma in gen_sigmas:
                    h0 = [r for r in records
                          if r["method"] == method and r["band_um"] == band
                          and r["null_sigma_um"] == null_sigma
                          and r["generator_sigma_um"] == gen_sigma
                          and r["scenario"] == "h0"]
                    if not h0:
                        continue
                    summary[method][band][key][f"gen_{gen_sigma:g}um_h0"] = summarize_pvalues(
                        [float(r["p_dclf"]) for r in h0])
                for scenario in sorted({r["scenario"] for r in records if r["scenario"].startswith("power_")}):
                    vals = [r for r in records
                            if r["method"] == method and r["band_um"] == band
                            and r["null_sigma_um"] == null_sigma
                            and r["scenario"] == scenario]
                    if vals:
                        assoc_power = float(np.mean([
                            bool(r["significant"]) and r["direction"] == "association"
                            for r in vals
                        ]))
                        summary[method][band][key][scenario] = {
                            "n": len(vals),
                            "association_power": round(assoc_power, 4),
                            "median_p": round(float(np.median([float(r["p_dclf"]) for r in vals])), 4),
                        }

    decisions = []
    for band, by_sigma in summary.get("total_cell_morphology_jitter", {}).items():
        for null_sigma, stats in by_sigma.items():
            if null_sigma == "none":
                continue
            h0_rates = [
                v["p05"] for k, v in stats.items()
                if k.endswith("_h0") and isinstance(v, dict)
            ]
            power_vals = [
                v["association_power"] for k, v in stats.items()
                if k.startswith("power_") and isinstance(v, dict)
            ]
            worst_h0 = max(h0_rates) if h0_rates else None
            min_power = min(power_vals) if power_vals else None
            decision = "candidate"
            if worst_h0 is None or min_power is None:
                decision = "incomplete"
            elif worst_h0 > args.max_type1:
                decision = "do_not_ship_anti_conservative"
            elif min_power < args.min_power:
                decision = "do_not_ship_underpowered"
            else:
                decision = "passes_screen_needs_real_image_morphology"
            decisions.append({
                "band_um": band,
                "null_sigma_um": float(null_sigma),
                "worst_h0_p05": worst_h0,
                "min_power": min_power,
                "decision": decision,
            })

    return {
        "dataset": {
            "name": "Schurch CRC CODEX CellNeighs",
            "csv": str(csv_path),
            "pixel_size_um": PIXEL_SIZE_UM,
            "spots_used": len(templates),
            "spot_keys": [t.key for t in templates],
        },
        "parameters": vars(args),
        "records": records,
        "summary": summary,
        "decisions": decisions,
    }


def write_report(result: dict, out_md: Path) -> None:
    ds = result["dataset"]
    p = result["parameters"]
    lines = [
        "# Public CODEX Dense-Null Calibration",
        "",
        "This report calibrates candidate dense-tissue nulls on real CRC CODEX cell-coordinate architecture.",
        "It does not validate OASIS segmentation, serial-section registration, or a production dense mode.",
        "",
        "## Dataset",
        "",
        f"- Source table: `{ds['csv']}`",
        f"- Spots used: {ds['spots_used']}",
        f"- Pixel size used only for coordinate conversion: {ds['pixel_size_um']} um/px",
        f"- Simulated marker counts: A={p['n_a']}, B={p['n_b']}",
        f"- Simulations per spot: {p['sims_per_spot']}",
        f"- Permutations per test: {p['nperm']}",
        f"- Generator sigmas: `{p['generator_sigmas_um']}` um",
        f"- Candidate null sigmas: `{p['null_sigmas_um']}` um",
        f"- Planted fraction: {p['planted_fraction']}",
        "",
        "## Calibration Table",
        "",
        "| Method | Band (um) | Null sigma | Generator H0 | p<=0.05 | 95% CI | Verdict | Power 5um | Power 12um |",
        "|---|---:|---:|---|---:|---|---|---:|---:|",
    ]
    summary = result["summary"]
    for method, by_band in summary.items():
        for band, by_sigma in by_band.items():
            for null_sigma, stats in by_sigma.items():
                h0_keys = [k for k in stats if k.endswith("_h0")]
                for h0_key in h0_keys:
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
        "## Candidate Decisions",
        "",
        "| Band (um) | Null sigma (um) | Worst H0 p<=0.05 | Minimum planted-association power | Decision |",
        "|---:|---:|---:|---:|---|",
    ]
    for d in result["decisions"]:
        lines.append(
            f"| {d['band_um']} | {d['null_sigma_um']:.3g} | "
            f"{d['worst_h0_p05']:.3f} | {d['min_power']:.3f} | {d['decision']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The known-null simulations use real CODEX cell layouts as tissue architecture templates, then independently draw A and B from the same marker-independent total-cell field.",
        "- Homogeneous CSR is expected to over-reject because it ignores dense architecture.",
        "- A morphology-conditioned candidate is only acceptable if every tested generator H0 stays near 5% while planted positives retain useful power.",
        "- Passing this harness would still not be enough to ship dense mode: OASIS also needs real-image morphology extraction validated on H-DAB/hematoxylin-derived fields.",
        "- Therefore this report can promote a candidate to further validation, but it cannot by itself make the dense null production-ready.",
        "",
    ]
    out_md.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--spot-cap", type=int, default=60)
    parser.add_argument("--min-cells-per-spot", type=int, default=800)
    parser.add_argument("--sims-per-spot", type=int, default=5)
    parser.add_argument("--nperm", type=int, default=199)
    parser.add_argument("--n-a", type=int, default=75)
    parser.add_argument("--n-b", type=int, default=75)
    parser.add_argument("--generator-sigmas-um", default="2,5,10")
    parser.add_argument("--null-sigmas-um", default="2")
    parser.add_argument("--power-jitters-um", default="5,12")
    parser.add_argument("--planted-fraction", type=float, default=0.75)
    parser.add_argument("--include-homogeneous", action="store_true", default=True)
    parser.add_argument("--no-homogeneous", dest="include_homogeneous", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-type1", type=float, default=0.07)
    parser.add_argument("--min-power", type=float, default=0.80)
    args = parser.parse_args()

    result = run(args)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    write_report(result, OUT_MD)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
