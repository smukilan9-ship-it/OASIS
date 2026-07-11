"""
validate_morphology_conditioned_null.py

Dense-tissue validation harness for an EXTERNAL morphology-conditioned null.

This is the serial-section-safe alternative after:

  1. the 35-45 um reweighted null stayed anti-conservative in dense tissue, and
  2. naive square-tile compartment conditioning also stayed anti-conservative.

Core idea
---------
Do not estimate architecture from A or B marker-positive cells. Instead, use a
marker-independent morphology / tissue-architecture field lambda_M(x). Keep A
fixed and sample B* from lambda_M(x). Then compare observed cross-K to that null.

In a real OASIS implementation lambda_M would need to come from hematoxylin,
total-cell density, tissue compartments, or certified morphology channels. In this
validation harness we start with the oracle morphology field used to generate the
synthetic tissue. That is deliberately a best-case test: it asks whether the idea
can work statistically before we spend effort estimating lambda_M from images.

Outputs:
  validation/morphology_conditioned_null_report.md
  validation/morphology_conditioned_null_results.json
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
OUT_JSON = os.path.join(HERE, "morphology_conditioned_null_results.json")
OUT_MD = os.path.join(HERE, "morphology_conditioned_null_report.md")


@dataclass(frozen=True)
class Combo:
    morph_smooth_um: float
    rmin_um: float
    rmax_um: float

    @property
    def label(self) -> str:
        return f"morphsmooth{self.morph_smooth_um:g}_r{self.rmin_um:g}-{self.rmax_um:g}"


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


class MorphologyFactory:
    def __init__(self, window_um: float, n_points: int, grid: int, sigma_log: float):
        self.window_um = float(window_um)
        self.n_points = int(n_points)
        self.grid = int(grid)
        self.sigma_log = float(sigma_log)
        self.cell_um = self.window_um / self.grid

    def uniform_pmf(self):
        return np.full((self.grid, self.grid), 1.0 / (self.grid * self.grid))

    def lgcp_pmf(self, corr_len_um: float, rng: np.random.Generator):
        g = rng.standard_normal((self.grid, self.grid))
        g = gaussian_filter(g, sigma=max((corr_len_um / self.cell_um) / 2.0, 0.1),
                            mode="wrap")
        g = (g - g.mean()) / (g.std() + 1e-9)
        lam = np.exp(self.sigma_log * g)
        return lam / lam.sum()

    def banded_pmf(self, period_um: float, sigma_um: float):
        yy, xx = np.indices((self.grid, self.grid))
        y_um = (yy + 0.5) * self.cell_um
        centers = np.arange(period_um / 2.0, self.window_um, period_um)
        dens = np.zeros_like(y_um, dtype=float)
        for c in centers:
            dens += np.exp(-0.5 * ((y_um - c) / sigma_um) ** 2)
        dens = np.maximum(dens, 1e-9)
        return dens / dens.sum()

    def smooth_pmf(self, pmf: np.ndarray, smooth_um: float):
        if smooth_um <= 0:
            out = np.asarray(pmf, float).copy()
        else:
            out = gaussian_filter(np.asarray(pmf, float),
                                  sigma=max(smooth_um / self.cell_um, 0.1),
                                  mode="wrap")
        out = np.maximum(out, 1e-12)
        return out / out.sum()

    def sample(self, pmf: np.ndarray, rng: np.random.Generator, n: int | None = None):
        n = self.n_points if n is None else int(n)
        idx = rng.choice(pmf.size, size=n, p=pmf.ravel())
        iy, ix = np.unravel_index(idx, pmf.shape)
        return np.column_stack([
            ix * self.cell_um + rng.uniform(0, self.cell_um, n),
            iy * self.cell_um + rng.uniform(0, self.cell_um, n),
        ])

    def shared_pair(self, rng: np.random.Generator, pmf: np.ndarray):
        return self.sample(pmf, rng), self.sample(pmf, rng)

    def attracted_pair(self, rng: np.random.Generator, pmf: np.ndarray,
                       jitter_um: float, planted_fraction: float):
        a = self.sample(pmf, rng)
        bg = self.sample(pmf, rng)
        m = int(round(self.n_points * planted_fraction))
        planted = a[rng.integers(0, len(a), m)] + rng.normal(0, jitter_um, (m, 2))
        b = np.vstack([planted, bg[: self.n_points - m]])
        return a, np.clip(b, 0.0, self.window_um)


def _sample_from_morphology(factory: MorphologyFactory, pmf: np.ndarray,
                            rng: np.random.Generator, n: int):
    return factory.sample(pmf, rng, n=n)


def morphology_conditioned_cross_k_test(
    points_a: np.ndarray,
    points_b: np.ndarray,
    morph_pmf: np.ndarray,
    factory: MorphologyFactory,
    radii_um: np.ndarray,
    n_perm: int,
    seed: int,
    dclf_rmin_um: float,
    dclf_rmax_um: float,
) -> dict:
    a = np.asarray(points_a, float).reshape(-1, 2)
    b = np.asarray(points_b, float).reshape(-1, 2)
    radii = np.asarray(radii_um, float)
    area = float(factory.window_um * factory.window_um)
    n_a, n_b = len(a), len(b)
    counts = ss._pair_counts(a, b, radii)
    k_obs = ss._k_from_counts(counts, area, n_a, n_b)
    l_obs = np.sqrt(np.clip(k_obs, 0.0, None) / np.pi) - radii

    rng = np.random.default_rng(int(seed))
    tree_a = cKDTree(a)
    norm = area / (n_a * n_b)
    null_k = np.empty((n_perm, len(radii)), dtype=float)
    for i in range(n_perm):
        b_star = _sample_from_morphology(factory, morph_pmf, rng, n_b)
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
        "L_minus_r": l_obs.tolist(),
        "method": "external_morphology_conditioned_cross_k",
        "description": (
            "B is redrawn from a marker-independent morphology intensity field, "
            "not from B's own marker intensity and not by label permutation."
        ),
        **summ,
    }


def _run_one(a, b, pmf_null, combo: Combo, args, factory, seed):
    res = morphology_conditioned_cross_k_test(
        a, b, pmf_null, factory,
        radii_um=np.arange(0.0, args.radii_max + args.radii_step, args.radii_step),
        n_perm=args.nperm,
        seed=seed,
        dclf_rmin_um=combo.rmin_um,
        dclf_rmax_um=combo.rmax_um,
    )
    g = res["global"]
    return float(g["global_p_dclf"]), bool(g["significant"]), str(g.get("direction") or "none")


def _size_rate(combo, args, factory, seed,
               draw: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray, np.ndarray]]):
    rng = np.random.default_rng(seed)
    ps = []
    assoc = seg = 0
    for i in range(args.sims):
        a, b, pmf = draw(rng)
        pmf_null = factory.smooth_pmf(pmf, combo.morph_smooth_um)
        p, sig, direction = _run_one(a, b, pmf_null, combo, args, factory, seed * 1000 + i)
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


def _power_rate(combo, args, factory, seed, jitter_um):
    rng = np.random.default_rng(seed)
    assoc = sig_any = 0
    ps = []
    for i in range(args.sims):
        pmf = factory.lgcp_pmf(args.intermediate_corr_um, rng)
        a, b = factory.attracted_pair(rng, pmf, jitter_um, args.planted_fraction)
        pmf_null = factory.smooth_pmf(pmf, combo.morph_smooth_um)
        p, sig, direction = _run_one(a, b, pmf_null, combo, args, factory, seed * 1000 + i)
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
    factory = MorphologyFactory(args.window_um, args.n_points, args.grid, args.sigma_log)
    combos = [Combo(s, lo, hi) for s in args.morph_smooth_um for lo, hi in args.bands]
    rows = []
    for idx, combo in enumerate(combos):
        print(f"\n[{idx + 1}/{len(combos)}] {combo.label}", flush=True)
        row = {"combo": {"morph_smooth_um": combo.morph_smooth_um,
                         "dclf_rmin_um": combo.rmin_um,
                         "dclf_rmax_um": combo.rmax_um}}
        row["uniform_h0"] = _size_rate(
            combo, args, factory, 101 + idx,
            lambda rng: (
                factory.sample(factory.uniform_pmf(), rng),
                factory.sample(factory.uniform_pmf(), rng),
                factory.uniform_pmf(),
            ))
        print(f"  uniform H0 p05={row['uniform_h0']['p05']:.3f} "
              f"{row['uniform_h0']['verdict']}", flush=True)
        row["shared_dense_h0"] = _size_rate(
            combo, args, factory, 201 + idx,
            lambda rng: _shared_lgcp(factory, rng, args.dense_corr_um))
        print(f"  shared dense H0 p05={row['shared_dense_h0']['p05']:.3f} "
              f"{row['shared_dense_h0']['verdict']}", flush=True)
        row["shared_intermediate_h0"] = _size_rate(
            combo, args, factory, 301 + idx,
            lambda rng: _shared_lgcp(factory, rng, args.intermediate_corr_um))
        print(f"  shared intermediate H0 p05={row['shared_intermediate_h0']['p05']:.3f} "
              f"{row['shared_intermediate_h0']['verdict']}", flush=True)
        row["shared_banded_h0"] = _size_rate(
            combo, args, factory, 401 + idx,
            lambda rng: _shared_banded(factory, rng, args.band_period_um, args.band_sigma_um))
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


def _shared_lgcp(factory, rng, corr_um):
    pmf = factory.lgcp_pmf(corr_um, rng)
    a, b = factory.shared_pair(rng, pmf)
    return a, b, pmf


def _shared_banded(factory, rng, period_um, sigma_um):
    pmf = factory.banded_pmf(period_um, sigma_um)
    a, b = factory.shared_pair(rng, pmf)
    return a, b, pmf


def make_report(args, rows):
    ranked = sorted(rows, key=lambda r: (-r["score"], r["combo"]["morph_smooth_um"],
                                        r["combo"]["dclf_rmax_um"]))
    lines = []
    lines.append("# Morphology-Conditioned Dense Null Report")
    lines.append("")
    lines.append("This report was generated by `validation/validate_morphology_conditioned_null.py`.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Simulations per regime: `{args.sims}`")
    lines.append(f"- Permutations per test: `{args.nperm}`")
    lines.append(f"- Morphology smoothing candidates: `{args.morph_smooth_um}`")
    lines.append(f"- DCLF bands: `{args.bands}`")
    lines.append(f"- Points per population: `{args.n_points}`")
    lines.append(f"- Window: `{args.window_um} x {args.window_um} um`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| morph smooth (um) | DCLF band (um) | uniform H0 | dense shared H0 | intermediate shared H0 | banded H0 | power short | power mid | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in ranked:
        c = row["combo"]
        lines.append(
            f"| {c['morph_smooth_um']:g} | {c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} "
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
            f"`morph_smooth={c['morph_smooth_um']:g} um`, "
            f"DCLF `{c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} um`, "
            f"verdict `{best['candidate_verdict']}`, score `{best['score']}`."
        )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This is an oracle morphology test. Passing means the conditional-null idea is statistically achievable if OASIS can estimate a marker-independent morphology field well enough from real images. It does not by itself validate an H-derived field.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--nperm", type=int, default=99)
    ap.add_argument("--morph-smooth-um", default="0,10,20,30")
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
    args.morph_smooth_um = _parse_csv_floats(args.morph_smooth_um)
    args.bands = _parse_bands(args.bands)

    print("=" * 88)
    print("MORPHOLOGY-CONDITIONED DENSE NULL VALIDATION")
    print("=" * 88)
    print(f"sims={args.sims} nperm={args.nperm} smooth={args.morph_smooth_um} "
          f"bands={args.bands}", flush=True)
    rows = run_screen(args)
    payload = {"config": vars(args), "screen": rows}
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    report = make_report(args, rows)
    with open(OUT_MD, "w") as f:
        f.write(report + "\n")
    print("\nWROTE")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    ranked = sorted(rows, key=lambda r: (-r["score"], r["combo"]["morph_smooth_um"],
                                        r["combo"]["dclf_rmax_um"]))
    if ranked:
        best = ranked[0]
        c = best["combo"]
        print(f"\nBEST: smooth={c['morph_smooth_um']:g} band="
              f"{c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} "
              f"{best['candidate_verdict']} score={best['score']}")
        return 0 if best["candidate_verdict"] in ("SHIP_CANDIDATE", "BORDERLINE_NEEDS_MORE_SIM") else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
