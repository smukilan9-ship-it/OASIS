"""
validate_compartment_conditioned_null.py

Validation harness for a dense-tissue, serial-section-safe conditional null.

This is NOT production code. It tests the next candidate method after the
35-45 um reweighted-null family failed dense shared-architecture calibration.

Core null
---------
Keep population A fixed. Preserve the observed number of B cells inside each
marker-independent local compartment. For every permutation, redraw B uniformly
inside those same compartments. Then compare the observed cross-type K / L-r
curve to this compartment-conditioned null with the same DCLF global test.

This is serial-section-safe because it never permutes labels between A and B
cells from different physical sections. It randomizes only population B within
architecture-defined regions.

What this harness asks
----------------------
Can local compartment conditioning solve the dense-tissue failure mode?

It tests:

  1. Uniform CSR H0.
  2. Dense shared-architecture H0.
  3. Intermediate shared-architecture H0.
  4. Banded shared-compartment H0.
  5. Planted short-range attraction power.
  6. Planted mid-range attraction power.

Compartment model
-----------------
For this first validation, compartments are regular square tiles. That is the
simplest marker-independent version and mirrors what could be drawn from local
morphology/density strata later. If this simple version passes, the next step is
to replace tiles with H-structure superpixels or tissue-compartment masks.

Outputs:
  validation/compartment_conditioned_null_report.md
  validation/compartment_conditioned_null_results.json

Example pilot:
  .venv/bin/python validation/validate_compartment_conditioned_null.py \
    --sims 40 --nperm 99 --tile-sizes 30,40,50,60 --bands 5-20,10-30

Paper-grade:
  .venv/bin/python validation/validate_compartment_conditioned_null.py \
    --sims 300 --nperm 199 --tile-sizes 30,40,50,60,80 --bands 5-20,10-30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial import spatial_stats as ss  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "compartment_conditioned_null_results.json")
OUT_MD = os.path.join(HERE, "compartment_conditioned_null_report.md")


@dataclass(frozen=True)
class Combo:
    tile_um: float
    rmin_um: float
    rmax_um: float

    @property
    def label(self) -> str:
        return f"tile{self.tile_um:g}_r{self.rmin_um:g}-{self.rmax_um:g}"


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_bands(text: str) -> list[tuple[float, float]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split("-", 1)
        out.append((float(lo), float(hi)))
    return out


def _ci95(rate: float, n: int) -> tuple[float, float]:
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / max(n, 1))
    return max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def _rate_verdict(rate: float, sims: int, lo: float = 0.03, hi: float = 0.07) -> str:
    ci_lo, ci_hi = _ci95(rate, sims)
    if lo <= rate <= hi:
        return "pass"
    if ci_hi < lo:
        return "conservative"
    if ci_lo > hi:
        return "anti_conservative"
    return "borderline"


class PatternFactory:
    def __init__(self, window_um: float, n_points: int, grid: int, sigma_log: float):
        self.window_um = float(window_um)
        self.n_points = int(n_points)
        self.grid = int(grid)
        self.sigma_log = float(sigma_log)

    def _lgcp_pmf(self, corr_len_um: float, rng: np.random.Generator):
        cell = self.window_um / self.grid
        g = rng.standard_normal((self.grid, self.grid))
        g = gaussian_filter(g, sigma=max((corr_len_um / cell) / 2.0, 0.1), mode="wrap")
        g = (g - g.mean()) / (g.std() + 1e-9)
        lam = np.exp(self.sigma_log * g)
        return lam / lam.sum(), cell

    @staticmethod
    def _sample_pmf(pmf: np.ndarray, cell: float, n: int, rng: np.random.Generator):
        idx = rng.choice(pmf.size, size=n, p=pmf.ravel())
        iy, ix = np.unravel_index(idx, pmf.shape)
        return np.column_stack([
            ix * cell + rng.uniform(0, cell, n),
            iy * cell + rng.uniform(0, cell, n),
        ])

    def uniform(self, rng: np.random.Generator, n: int | None = None):
        n = self.n_points if n is None else int(n)
        return rng.uniform(0.0, self.window_um, (n, 2))

    def shared_field_pair(self, rng: np.random.Generator, corr_len_um: float):
        pmf, cell = self._lgcp_pmf(corr_len_um, rng)
        return (
            self._sample_pmf(pmf, cell, self.n_points, rng),
            self._sample_pmf(pmf, cell, self.n_points, rng),
        )

    def banded_shared_pair(self, rng: np.random.Generator, period_um: float, sigma_um: float):
        n = self.n_points
        centers = np.arange(period_um / 2.0, self.window_um, period_um)
        ya = rng.choice(centers, size=n) + rng.normal(0, sigma_um, n)
        yb = rng.choice(centers, size=n) + rng.normal(0, sigma_um, n)
        return (
            np.column_stack([rng.uniform(0, self.window_um, n), np.clip(ya, 0, self.window_um)]),
            np.column_stack([rng.uniform(0, self.window_um, n), np.clip(yb, 0, self.window_um)]),
        )

    def attracted_pair(
        self,
        rng: np.random.Generator,
        corr_len_um: float,
        jitter_um: float,
        planted_fraction: float,
    ):
        pmf, cell = self._lgcp_pmf(corr_len_um, rng)
        a = self._sample_pmf(pmf, cell, self.n_points, rng)
        bg = self._sample_pmf(pmf, cell, self.n_points, rng)
        m = int(round(self.n_points * planted_fraction))
        planted = a[rng.integers(0, len(a), m)] + rng.normal(0, jitter_um, (m, 2))
        b = np.vstack([planted, bg[: self.n_points - m]])
        return a, np.clip(b, 0.0, self.window_um)


def _tile_ids(points: np.ndarray, tile_um: float, window_um: float, n_tiles: int) -> np.ndarray:
    pts = np.asarray(points, float).reshape(-1, 2)
    ix = np.clip((pts[:, 0] / tile_um).astype(int), 0, n_tiles - 1)
    iy = np.clip((pts[:, 1] / tile_um).astype(int), 0, n_tiles - 1)
    return iy * n_tiles + ix


def _sample_b_conditioned(
    b_observed: np.ndarray,
    tile_um: float,
    window_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Redraw B uniformly inside each tile while preserving B tile counts."""
    n_tiles = int(math.ceil(window_um / tile_um))
    ids = _tile_ids(b_observed, tile_um, window_um, n_tiles)
    counts = np.bincount(ids, minlength=n_tiles * n_tiles)
    pieces = []
    for tid, n in enumerate(counts):
        if n <= 0:
            continue
        iy, ix = divmod(tid, n_tiles)
        x0, y0 = ix * tile_um, iy * tile_um
        x1, y1 = min(x0 + tile_um, window_um), min(y0 + tile_um, window_um)
        pieces.append(np.column_stack([
            rng.uniform(x0, x1, int(n)),
            rng.uniform(y0, y1, int(n)),
        ]))
    if not pieces:
        return np.empty((0, 2), float)
    return np.vstack(pieces)


def _pcf_from_k(k_px: np.ndarray, radii_px: np.ndarray) -> np.ndarray:
    # Same definition as spatial_stats._pcf_from_k, kept local to avoid relying on
    # every private helper.
    r = np.asarray(radii_px, dtype=float)
    k = np.asarray(k_px, dtype=float)
    if len(r) < 2:
        return np.full_like(r, np.nan)
    dk = np.gradient(k, r, edge_order=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = dk / (2.0 * np.pi * r)
    g[~np.isfinite(g)] = np.nan
    return g


def compartment_conditioned_cross_k_test(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radii_um: np.ndarray,
    window_um: float,
    tile_um: float,
    n_perm: int,
    seed: int,
    dclf_rmin_um: float,
    dclf_rmax_um: float,
) -> dict:
    """Cross-K with a B-within-compartment conditional null."""
    a = np.asarray(points_a, float).reshape(-1, 2)
    b = np.asarray(points_b, float).reshape(-1, 2)
    radii = np.asarray(radii_um, float)
    area = float(window_um * window_um)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        raise ValueError("empty point set")

    counts = ss._pair_counts(a, b, radii)
    k_obs = ss._k_from_counts(counts, area, n_a, n_b)
    l_obs = np.sqrt(np.clip(k_obs, 0.0, None) / np.pi) - radii

    rng = np.random.default_rng(int(seed))
    tree_a = cKDTree(a)
    norm = area / (n_a * n_b)
    null_k = np.empty((n_perm, len(radii)), dtype=float)
    for i in range(n_perm):
        b_star = _sample_b_conditioned(b, tile_um, window_um, rng)
        null_counts = tree_a.count_neighbors(cKDTree(b_star), radii)
        null_k[i] = norm * np.asarray(null_counts, dtype=float)

    summ = ss._null_summary_from_k(
        radii,
        k_obs,
        l_obs,
        null_k,
        1.0,
        n_perm,
        dclf_rmin_um,
        dclf_rmax_um,
    )
    return {
        "radii_um": radii.tolist(),
        "K_observed": k_obs.tolist(),
        "g_observed": [None if not np.isfinite(v) else float(v)
                       for v in _pcf_from_k(k_obs, radii)],
        "L_minus_r": l_obs.tolist(),
        "tile_um": float(tile_um),
        "method": "compartment_conditioned_tile_null",
        "description": (
            "B is redrawn uniformly within marker-independent square compartments "
            "while preserving the observed B count in each compartment."
        ),
        **summ,
    }


def _run_one(a, b, combo: Combo, args, seed: int):
    res = compartment_conditioned_cross_k_test(
        a, b,
        radii_um=np.arange(0.0, args.radii_max + args.radii_step, args.radii_step),
        window_um=args.window_um,
        tile_um=combo.tile_um,
        n_perm=args.nperm,
        seed=seed,
        dclf_rmin_um=combo.rmin_um,
        dclf_rmax_um=combo.rmax_um,
    )
    g = res["global"]
    return float(g["global_p_dclf"]), bool(g["significant"]), str(g.get("direction") or "none")


def _size_rate(combo: Combo, args, factory: PatternFactory, seed: int,
               draw: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]):
    rng = np.random.default_rng(seed)
    ps = []
    assoc = seg = 0
    for i in range(args.sims):
        a, b = draw(rng)
        p, sig, direction = _run_one(a, b, combo, args, seed * 1000 + i)
        ps.append(p)
        if sig and direction == "association":
            assoc += 1
        elif sig and direction == "segregation":
            seg += 1
    arr = np.asarray(ps, float)
    rate = float(np.mean(arr <= 0.05))
    return {
        "p05": round(rate, 4),
        "mean_p": round(float(arr.mean()), 4),
        "median_p": round(float(np.median(arr)), 4),
        "assoc_rate": round(assoc / args.sims, 4),
        "seg_rate": round(seg / args.sims, 4),
        "verdict": _rate_verdict(rate, args.sims),
        "ci95": [round(x, 4) for x in _ci95(rate, args.sims)],
    }


def _power_rate(combo: Combo, args, factory: PatternFactory, seed: int, jitter_um: float):
    rng = np.random.default_rng(seed)
    assoc = sig_any = 0
    ps = []
    for i in range(args.sims):
        a, b = factory.attracted_pair(
            rng,
            corr_len_um=args.intermediate_corr_um,
            jitter_um=jitter_um,
            planted_fraction=args.planted_fraction,
        )
        p, sig, direction = _run_one(a, b, combo, args, seed * 1000 + i)
        ps.append(p)
        if sig:
            sig_any += 1
        if sig and direction == "association":
            assoc += 1
    rate = assoc / args.sims
    return {
        "association_power": round(rate, 4),
        "any_significant": round(sig_any / args.sims, 4),
        "mean_p": round(float(np.mean(ps)), 4),
        "verdict": "pass" if rate >= 0.80 else "weak",
        "ci95": [round(x, 4) for x in _ci95(rate, args.sims)],
    }


def _candidate_verdict(row):
    size_keys = ["uniform_h0", "shared_dense_h0", "shared_intermediate_h0", "shared_banded_h0"]
    sizes = [row[k]["verdict"] for k in size_keys]
    powers_ok = (row["power_short"]["association_power"] >= 0.80
                 and row["power_mid"]["association_power"] >= 0.80)
    if all(x == "pass" for x in sizes) and powers_ok:
        return "SHIP_CANDIDATE"
    if any(x == "anti_conservative" for x in sizes):
        return "REJECT_ANTI_CONSERVATIVE"
    if not powers_ok:
        return "REJECT_WEAK_POWER"
    return "BORDERLINE_NEEDS_MORE_SIM"


def _score(row):
    size_keys = ["uniform_h0", "shared_dense_h0", "shared_intermediate_h0", "shared_banded_h0"]
    s = sum(row[k]["verdict"] in ("pass", "borderline") for k in size_keys)
    s += int(row["power_short"]["association_power"] >= 0.80)
    s += int(row["power_mid"]["association_power"] >= 0.80)
    return s


def run_screen(args):
    factory = PatternFactory(args.window_um, args.n_points, args.grid, args.sigma_log)
    combos = [Combo(t, lo, hi) for t in args.tile_sizes for lo, hi in args.bands]
    rows = []
    for idx, combo in enumerate(combos):
        print(f"\n[{idx + 1}/{len(combos)}] {combo.label}", flush=True)
        row = {"combo": {"tile_um": combo.tile_um,
                         "dclf_rmin_um": combo.rmin_um,
                         "dclf_rmax_um": combo.rmax_um}}
        row["uniform_h0"] = _size_rate(
            combo, args, factory, 101 + idx,
            lambda rng: (factory.uniform(rng), factory.uniform(rng)))
        print(f"  uniform H0 p05={row['uniform_h0']['p05']:.3f} "
              f"{row['uniform_h0']['verdict']}", flush=True)
        row["shared_dense_h0"] = _size_rate(
            combo, args, factory, 201 + idx,
            lambda rng: factory.shared_field_pair(rng, args.dense_corr_um))
        print(f"  shared dense H0 p05={row['shared_dense_h0']['p05']:.3f} "
              f"{row['shared_dense_h0']['verdict']}", flush=True)
        row["shared_intermediate_h0"] = _size_rate(
            combo, args, factory, 301 + idx,
            lambda rng: factory.shared_field_pair(rng, args.intermediate_corr_um))
        print(f"  shared intermediate H0 p05={row['shared_intermediate_h0']['p05']:.3f} "
              f"{row['shared_intermediate_h0']['verdict']}", flush=True)
        row["shared_banded_h0"] = _size_rate(
            combo, args, factory, 401 + idx,
            lambda rng: factory.banded_shared_pair(
                rng, args.band_period_um, args.band_sigma_um))
        print(f"  shared banded H0 p05={row['shared_banded_h0']['p05']:.3f} "
              f"{row['shared_banded_h0']['verdict']}", flush=True)
        row["power_short"] = _power_rate(combo, args, factory, 501 + idx, args.short_jitter_um)
        row["power_mid"] = _power_rate(combo, args, factory, 601 + idx, args.mid_jitter_um)
        print(f"  power @{args.short_jitter_um:g}um="
              f"{row['power_short']['association_power']:.3f}", flush=True)
        print(f"  power @{args.mid_jitter_um:g}um="
              f"{row['power_mid']['association_power']:.3f}", flush=True)
        row["score"] = _score(row)
        row["candidate_verdict"] = _candidate_verdict(row)
        print(f"  candidate verdict: {row['candidate_verdict']} score={row['score']}",
              flush=True)
        rows.append(row)
    return rows


def make_report(args, rows):
    ranked = sorted(rows, key=lambda r: (-r["score"], r["combo"]["tile_um"],
                                        r["combo"]["dclf_rmax_um"]))
    lines = []
    lines.append("# Compartment-Conditioned Dense Null Report")
    lines.append("")
    lines.append("This report was generated by `validation/validate_compartment_conditioned_null.py`.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Simulations per regime: `{args.sims}`")
    lines.append(f"- Permutations per test: `{args.nperm}`")
    lines.append(f"- Points per population: `{args.n_points}`")
    lines.append(f"- Window: `{args.window_um} x {args.window_um} um`")
    lines.append(f"- Tile sizes: `{args.tile_sizes}`")
    lines.append(f"- DCLF bands: `{args.bands}`")
    lines.append(f"- Short/mid attraction: `{args.short_jitter_um}`, `{args.mid_jitter_um} um`")
    lines.append(f"- Planted fraction: `{args.planted_fraction}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| tile (um) | DCLF band (um) | uniform H0 | dense shared H0 | intermediate shared H0 | banded H0 | power short | power mid | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in ranked:
        c = row["combo"]
        lines.append(
            f"| {c['tile_um']:g} | {c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} "
            f"| {row['uniform_h0']['p05']:.3f} ({row['uniform_h0']['verdict']}) "
            f"| {row['shared_dense_h0']['p05']:.3f} ({row['shared_dense_h0']['verdict']}) "
            f"| {row['shared_intermediate_h0']['p05']:.3f} ({row['shared_intermediate_h0']['verdict']}) "
            f"| {row['shared_banded_h0']['p05']:.3f} ({row['shared_banded_h0']['verdict']}) "
            f"| {row['power_short']['association_power']:.3f} "
            f"| {row['power_mid']['association_power']:.3f} "
            f"| {row['candidate_verdict']} |"
        )
    lines.append("")
    if ranked:
        best = ranked[0]
        c = best["combo"]
        lines.append("## Best Candidate")
        lines.append("")
        lines.append(
            f"`tile={c['tile_um']:g} um`, DCLF `{c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} um`, "
            f"verdict `{best['candidate_verdict']}`, score `{best['score']}`."
        )
        lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("")
    lines.append("- Size target: P(p <= 0.05) in [0.03, 0.07].")
    lines.append("- Power target: >=0.80 for both planted attractions.")
    lines.append("- Anti-conservative dense/shared H0 rejects the candidate.")
    lines.append("- Passing here would still require real-data controls before production.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--nperm", type=int, default=99)
    ap.add_argument("--tile-sizes", default="30,40,50,60,80")
    ap.add_argument("--bands", default="5-20,10-30")
    ap.add_argument("--radii-max", type=float, default=60.0)
    ap.add_argument("--radii-step", type=float, default=4.0)
    ap.add_argument("--window-um", type=float, default=800.0)
    ap.add_argument("--n-points", type=int, default=260)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--sigma-log", type=float, default=1.15)
    ap.add_argument("--dense-corr-um", type=float, default=45.0)
    ap.add_argument("--intermediate-corr-um", type=float, default=75.0)
    ap.add_argument("--band-period-um", type=float, default=90.0)
    ap.add_argument("--band-sigma-um", type=float, default=18.0)
    ap.add_argument("--short-jitter-um", type=float, default=5.0)
    ap.add_argument("--mid-jitter-um", type=float, default=12.0)
    ap.add_argument("--planted-fraction", type=float, default=1.0)
    args = ap.parse_args(argv)
    args.tile_sizes = _parse_csv_floats(args.tile_sizes)
    args.bands = _parse_bands(args.bands)

    print("=" * 88)
    print("COMPARTMENT-CONDITIONED DENSE NULL VALIDATION")
    print("=" * 88)
    print(f"sims={args.sims} nperm={args.nperm} tiles={args.tile_sizes} bands={args.bands}",
          flush=True)
    rows = run_screen(args)
    payload = {
        "config": vars(args),
        "screen": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    report = make_report(args, rows)
    with open(OUT_MD, "w") as f:
        f.write(report + "\n")
    print("\nWROTE")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    ranked = sorted(rows, key=lambda r: (-r["score"], r["combo"]["tile_um"],
                                        r["combo"]["dclf_rmax_um"]))
    if ranked:
        best = ranked[0]
        c = best["combo"]
        print(f"\nBEST: tile={c['tile_um']:g} band={c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} "
              f"{best['candidate_verdict']} score={best['score']}")
        return 0 if best["candidate_verdict"] in ("SHIP_CANDIDATE", "BORDERLINE_NEEDS_MORE_SIM") else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
