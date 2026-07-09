"""
validate_dense_reweighted_null.py

Screen and calibrate a DENSE-TISSUE primary null for OASIS spatial association.

This does NOT change production behavior. It is a validation harness for deciding
whether a smaller reweighting bandwidth, e.g. 35-45 um, can become a second
paper-grade primary null for dense tissues.

Why a new harness is needed
---------------------------
The shipped primary null uses an intensity-reweighted inhomogeneous cross-K with
h=75 um and a 10-50 um DCLF band. That is only defensible when the intensity
surface represents tissue architecture coarser than the interaction band.

For dense immune tissue, a smaller intensity bandwidth may be needed. But if
h=35-45 um while the DCLF test still asks about 10-50 um, the intensity surface
overlaps the biological scale being tested. This harness therefore sweeps both:

  * reweighting bandwidth h
  * DCLF radius band

and evaluates each combination under:

  1. uniform CSR H0
  2. dense shared-architecture H0
  3. intermediate shared-architecture H0
  4. short-range attraction power
  5. mid-range attraction power

It also estimates the architecture-scale gate for selected combinations by
varying the simulated architecture correlation scale and recording the measured
ell_hat produced by spatial_stats.estimate_architecture_scale.

Outputs:
  validation/dense_reweighted_null_report.md
  validation/dense_reweighted_null_results.json

Example quick run:
  python validation/validate_dense_reweighted_null.py --sims 8 --nperm 29

Example stronger run:
  python validation/validate_dense_reweighted_null.py --sims 100 --nperm 199

For paper-grade final calibration, use >=300 simulations and >=199 permutations.
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
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatial_stats as ss  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "dense_reweighted_null_results.json")
OUT_MD = os.path.join(HERE, "dense_reweighted_null_report.md")


@dataclass(frozen=True)
class Combo:
    bandwidth_um: float
    rmin_um: float
    rmax_um: float

    @property
    def label(self) -> str:
        return f"h{self.bandwidth_um:g}_r{self.rmin_um:g}-{self.rmax_um:g}"


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_bands(text: str) -> list[tuple[float, float]]:
    bands = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Band must be like 10-30, got {item!r}")
        lo, hi = item.split("-", 1)
        bands.append((float(lo), float(hi)))
    return bands


def _ci95(rate: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / n)
    return max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def _rate_verdict(rate: float, sims: int, lo: float, hi: float) -> str:
    # Keep the hard target visible, but account for Monte-Carlo noise in small
    # screening runs so pilots are not overinterpreted.
    ci_lo, ci_hi = _ci95(rate, sims)
    if lo <= rate <= hi:
        return "pass"
    if ci_hi < lo:
        return "conservative"
    if ci_lo > hi:
        return "anti_conservative"
    return "borderline"


class DensePatternFactory:
    """Synthetic point-pattern generator in micron units (pixel_size = 1)."""

    def __init__(
        self,
        window_um: float,
        n_points: int,
        grid: int,
        sigma_log: float,
    ) -> None:
        self.window_um = float(window_um)
        self.n_points = int(n_points)
        self.grid = int(grid)
        self.sigma_log = float(sigma_log)
        self.window = box(0, 0, self.window_um, self.window_um)

    def _lgcp_pmf(self, corr_len_um: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Log-Gaussian Cox-like intensity field, normalized to a cell PMF."""
        cell = self.window_um / self.grid
        g = rng.standard_normal((self.grid, self.grid))
        # Gaussian smoothing sigma is half the requested correlation knob. This
        # mirrors the existing architecture-scale validation and makes the knob
        # monotonic rather than a literal e-folding range.
        g = gaussian_filter(g, sigma=max((corr_len_um / cell) / 2.0, 0.1), mode="wrap")
        g = (g - g.mean()) / (g.std() + 1e-9)
        lam = np.exp(self.sigma_log * g)
        return lam / lam.sum(), cell

    @staticmethod
    def _sample_from_pmf(
        pmf: np.ndarray,
        cell: float,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        idx = rng.choice(pmf.size, size=n, p=pmf.ravel())
        iy, ix = np.unravel_index(idx, pmf.shape)
        return np.column_stack([
            ix * cell + rng.uniform(0, cell, n),
            iy * cell + rng.uniform(0, cell, n),
        ])

    def uniform(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        n = self.n_points if n is None else int(n)
        return rng.uniform(0.0, self.window_um, (n, 2))

    def shared_field_pair(
        self,
        rng: np.random.Generator,
        corr_len_um: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pmf, cell = self._lgcp_pmf(corr_len_um, rng)
        a = self._sample_from_pmf(pmf, cell, self.n_points, rng)
        b = self._sample_from_pmf(pmf, cell, self.n_points, rng)
        return a, b

    def banded_shared_pair(
        self,
        rng: np.random.Generator,
        band_period_um: float,
        band_sigma_um: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sharp compartment-like shared preference: independent draws from bands."""
        n = self.n_points
        centers = np.arange(band_period_um / 2.0, self.window_um, band_period_um)
        cy_a = rng.choice(centers, size=n)
        cy_b = rng.choice(centers, size=n)
        a = np.column_stack([
            rng.uniform(0, self.window_um, n),
            np.clip(cy_a + rng.normal(0, band_sigma_um, n), 0, self.window_um),
        ])
        b = np.column_stack([
            rng.uniform(0, self.window_um, n),
            np.clip(cy_b + rng.normal(0, band_sigma_um, n), 0, self.window_um),
        ])
        return a, b

    def attracted_pair(
        self,
        rng: np.random.Generator,
        base: str,
        corr_len_um: float,
        jitter_um: float,
        planted_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if base == "uniform":
            a = self.uniform(rng)
            bg = self.uniform(rng)
        else:
            pmf, cell = self._lgcp_pmf(corr_len_um, rng)
            a = self._sample_from_pmf(pmf, cell, self.n_points, rng)
            bg = self._sample_from_pmf(pmf, cell, self.n_points, rng)
        m = int(round(self.n_points * planted_fraction))
        planted = a[rng.integers(0, len(a), m)] + rng.normal(0.0, jitter_um, (m, 2))
        b = np.vstack([planted, bg[: self.n_points - m]])
        return a, np.clip(b, 0.0, self.window_um)


def _run_test(
    a: np.ndarray,
    b: np.ndarray,
    combo: Combo,
    nperm: int,
    seed: int,
    window,
    radii_um: np.ndarray,
) -> tuple[float, bool, str]:
    res = ss.cross_k_inhom_reweighted_test(
        a,
        b,
        radii_um,
        window.area,
        1.0,
        n_perm=nperm,
        seed=seed,
        tissue_polygon=window,
        bandwidth_um=combo.bandwidth_um,
        dclf_rmin_um=combo.rmin_um,
        dclf_rmax_um=combo.rmax_um,
    )
    g = res.get("global") or {}
    return (
        float(g.get("global_p_dclf", 1.0)),
        bool(g.get("significant")),
        str(g.get("direction") or "none"),
    )


def _size_rate(
    combo: Combo,
    sims: int,
    nperm: int,
    seed: int,
    factory: DensePatternFactory,
    draw_pair: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    radii_um: np.ndarray,
) -> dict:
    rng = np.random.default_rng(seed)
    ps = []
    assoc = seg = 0
    ell_a = []
    ell_b = []
    for i in range(sims):
        a, b = draw_pair(rng)
        p, sig, direction = _run_test(
            a, b, combo, nperm, seed=seed * 1000 + i,
            window=factory.window, radii_um=radii_um)
        ps.append(p)
        if sig and direction == "association":
            assoc += 1
        elif sig and direction == "segregation":
            seg += 1
        ea = ss.estimate_architecture_scale(a, 1.0, tissue_polygon=factory.window)
        eb = ss.estimate_architecture_scale(b, 1.0, tissue_polygon=factory.window)
        if ea is not None:
            ell_a.append(float(ea))
        if eb is not None:
            ell_b.append(float(eb))
    arr = np.asarray(ps, float)
    rate = float(np.mean(arr <= 0.05))
    return {
        "p05": round(rate, 4),
        "mean_p": round(float(arr.mean()), 4),
        "median_p": round(float(np.median(arr)), 4),
        "assoc_rate": round(assoc / sims, 4),
        "seg_rate": round(seg / sims, 4),
        "verdict": _rate_verdict(rate, sims, 0.03, 0.07),
        "ci95": [round(x, 4) for x in _ci95(rate, sims)],
        "ell_hat_a_median_um": round(float(np.median(ell_a)), 1) if ell_a else None,
        "ell_hat_b_median_um": round(float(np.median(ell_b)), 1) if ell_b else None,
        "n": sims,
    }


def _power_rate(
    combo: Combo,
    sims: int,
    nperm: int,
    seed: int,
    factory: DensePatternFactory,
    jitter_um: float,
    radii_um: np.ndarray,
    base_corr_um: float,
    planted_fraction: float,
) -> dict:
    rng = np.random.default_rng(seed)
    assoc = sig_any = 0
    ps = []
    for i in range(sims):
        a, b = factory.attracted_pair(
            rng,
            base="shared",
            corr_len_um=base_corr_um,
            jitter_um=jitter_um,
            planted_fraction=planted_fraction,
        )
        p, sig, direction = _run_test(
            a, b, combo, nperm, seed=seed * 1000 + i,
            window=factory.window, radii_um=radii_um)
        ps.append(p)
        if sig:
            sig_any += 1
        if sig and direction == "association":
            assoc += 1
    rate = assoc / sims
    return {
        "association_power": round(rate, 4),
        "any_significant": round(sig_any / sims, 4),
        "mean_p": round(float(np.mean(ps)), 4),
        "verdict": "pass" if rate >= 0.80 else "weak",
        "ci95": [round(x, 4) for x in _ci95(rate, sims)],
        "n": sims,
    }


def _screen_combos(args, combos: list[Combo], factory: DensePatternFactory) -> list[dict]:
    radii_um = np.arange(0.0, args.radii_max + args.radii_step, args.radii_step)
    rows = []
    for idx, combo in enumerate(combos):
        print(f"\n[{idx + 1}/{len(combos)}] {combo.label}")
        row = {
            "combo": {
                "bandwidth_um": combo.bandwidth_um,
                "dclf_rmin_um": combo.rmin_um,
                "dclf_rmax_um": combo.rmax_um,
            }
        }

        row["uniform_h0"] = _size_rate(
            combo, args.sims, args.nperm, 101 + idx, factory,
            lambda rng: (factory.uniform(rng), factory.uniform(rng)), radii_um)
        print(f"  uniform H0 p05={row['uniform_h0']['p05']:.3f} "
              f"{row['uniform_h0']['verdict']}")

        row["shared_dense_h0"] = _size_rate(
            combo, args.sims, args.nperm, 201 + idx, factory,
            lambda rng: factory.shared_field_pair(rng, args.dense_corr_um), radii_um)
        print(f"  shared dense H0 p05={row['shared_dense_h0']['p05']:.3f} "
              f"{row['shared_dense_h0']['verdict']} "
              f"ell~{row['shared_dense_h0']['ell_hat_a_median_um']}")

        row["shared_intermediate_h0"] = _size_rate(
            combo, args.sims, args.nperm, 301 + idx, factory,
            lambda rng: factory.shared_field_pair(rng, args.intermediate_corr_um), radii_um)
        print(f"  shared interm H0 p05={row['shared_intermediate_h0']['p05']:.3f} "
              f"{row['shared_intermediate_h0']['verdict']} "
              f"ell~{row['shared_intermediate_h0']['ell_hat_a_median_um']}")

        row["shared_banded_h0"] = _size_rate(
            combo, args.sims, args.nperm, 401 + idx, factory,
            lambda rng: factory.banded_shared_pair(
                rng, args.band_period_um, args.band_sigma_um), radii_um)
        print(f"  shared bands H0 p05={row['shared_banded_h0']['p05']:.3f} "
              f"{row['shared_banded_h0']['verdict']} "
              f"ell~{row['shared_banded_h0']['ell_hat_a_median_um']}")

        row["power_short"] = _power_rate(
            combo, args.sims, args.nperm, 501 + idx, factory,
            args.short_jitter_um, radii_um, args.intermediate_corr_um,
            args.planted_fraction)
        print(f"  power @{args.short_jitter_um:g}um="
              f"{row['power_short']['association_power']:.3f}")

        row["power_mid"] = _power_rate(
            combo, args.sims, args.nperm, 601 + idx, factory,
            args.mid_jitter_um, radii_um, args.intermediate_corr_um,
            args.planted_fraction)
        print(f"  power @{args.mid_jitter_um:g}um="
              f"{row['power_mid']['association_power']:.3f}")

        row["ship_score"] = _ship_score(row)
        row["ship_verdict"] = _ship_verdict(row)
        print(f"  candidate verdict: {row['ship_verdict']} score={row['ship_score']}")
        rows.append(row)
    return rows


def _ship_score(row: dict) -> int:
    size_keys = [
        "uniform_h0",
        "shared_dense_h0",
        "shared_intermediate_h0",
        "shared_banded_h0",
    ]
    score = 0
    for key in size_keys:
        if row[key]["verdict"] in ("pass", "borderline"):
            score += 1
    if row["power_short"]["association_power"] >= 0.80:
        score += 1
    if row["power_mid"]["association_power"] >= 0.80:
        score += 1
    return score


def _ship_verdict(row: dict) -> str:
    size_keys = [
        "uniform_h0",
        "shared_dense_h0",
        "shared_intermediate_h0",
        "shared_banded_h0",
    ]
    size_statuses = [row[k]["verdict"] for k in size_keys]
    powers_ok = (
        row["power_short"]["association_power"] >= 0.80
        and row["power_mid"]["association_power"] >= 0.80
    )
    if all(s == "pass" for s in size_statuses) and powers_ok:
        return "SHIP_CANDIDATE"
    if any(s == "anti_conservative" for s in size_statuses):
        return "REJECT_ANTI_CONSERVATIVE"
    if not powers_ok:
        return "REJECT_WEAK_POWER"
    return "BORDERLINE_NEEDS_MORE_SIM"


def _architecture_gate(
    args,
    selected: list[Combo],
    factory: DensePatternFactory,
) -> list[dict]:
    radii_um = np.arange(0.0, args.radii_max + args.radii_step, args.radii_step)
    rows = []
    for combo in selected:
        print(f"\nArchitecture gate sweep for {combo.label}")
        combo_rows = []
        for k, corr in enumerate(args.arch_corrs):
            fp = power = 0
            ell = []
            rng = np.random.default_rng(7000 + k + int(combo.bandwidth_um * 10))
            for i in range(args.arch_sims):
                a, b = factory.shared_field_pair(rng, corr)
                p, sig, direction = _run_test(
                    a, b, combo, args.nperm, seed=800000 + k * 1000 + i,
                    window=factory.window, radii_um=radii_um)
                if p <= 0.05:
                    fp += 1
                ea = ss.estimate_architecture_scale(a, 1.0, tissue_polygon=factory.window)
                if ea is not None:
                    ell.append(float(ea))

                a2, b2 = factory.attracted_pair(
                    rng,
                    base="shared",
                    corr_len_um=corr,
                    jitter_um=args.short_jitter_um,
                    planted_fraction=args.planted_fraction,
                )
                _, sig2, dir2 = _run_test(
                    a2, b2, combo, args.nperm, seed=900000 + k * 1000 + i,
                    window=factory.window, radii_um=radii_um)
                if sig2 and dir2 == "association":
                    power += 1
            r = {
                "combo": combo.label,
                "bandwidth_um": combo.bandwidth_um,
                "dclf_rmin_um": combo.rmin_um,
                "dclf_rmax_um": combo.rmax_um,
                "architecture_corr_knob_um": corr,
                "ell_hat_median_um": round(float(np.median(ell)), 1) if ell else None,
                "size_type_I": round(fp / args.arch_sims, 4),
                "power_short": round(power / args.arch_sims, 4),
                "n": args.arch_sims,
            }
            print(f"  corr={corr:5.1f} ell={r['ell_hat_median_um']} "
                  f"typeI={r['size_type_I']:.3f} power={r['power_short']:.3f}")
            combo_rows.append(r)
        rows.extend(combo_rows)
    return rows


def _select_for_architecture(rows: list[dict], max_n: int) -> list[Combo]:
    ranked = sorted(rows, key=lambda r: (-r["ship_score"], r["combo"]["bandwidth_um"],
                                        r["combo"]["dclf_rmax_um"]))
    selected = []
    seen = set()
    for row in ranked:
        c = row["combo"]
        key = (c["bandwidth_um"], c["dclf_rmin_um"], c["dclf_rmax_um"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(Combo(*key))
        if len(selected) >= max_n:
            break
    return selected


def _make_report(args, rows: list[dict], arch_rows: list[dict]) -> str:
    ranked = sorted(rows, key=lambda r: (-r["ship_score"], r["combo"]["bandwidth_um"],
                                        r["combo"]["dclf_rmax_um"]))
    top = ranked[0] if ranked else None
    lines = []
    lines.append("# Dense-Tissue Reweighted Null Calibration Report")
    lines.append("")
    lines.append("This report was generated by `validation/validate_dense_reweighted_null.py`.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Simulations per screen cell: `{args.sims}`")
    lines.append(f"- Architecture-gate simulations per knob: `{args.arch_sims}`")
    lines.append(f"- Permutations per DCLF test: `{args.nperm}`")
    lines.append(f"- Points per population: `{args.n_points}`")
    lines.append(f"- Window: `{args.window_um} x {args.window_um} um`")
    lines.append(f"- Dense shared-field knob: `{args.dense_corr_um} um`")
    lines.append(f"- Intermediate shared-field knob: `{args.intermediate_corr_um} um`")
    lines.append(f"- Banded compartment period/sigma: `{args.band_period_um}/{args.band_sigma_um} um`")
    lines.append(f"- Power jitters: `{args.short_jitter_um}` and `{args.mid_jitter_um} um`")
    lines.append("")
    lines.append("Paper-grade calibration should use at least `--sims 300 --nperm 199`; "
                 "smaller runs are screening evidence only.")
    lines.append("")

    lines.append("## Screened Combinations")
    lines.append("")
    lines.append("| h (um) | DCLF band (um) | uniform H0 | dense shared H0 | intermed shared H0 | banded H0 | power short | power mid | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in ranked:
        c = r["combo"]
        lines.append(
            f"| {c['bandwidth_um']:g} | {c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} "
            f"| {r['uniform_h0']['p05']:.3f} ({r['uniform_h0']['verdict']}) "
            f"| {r['shared_dense_h0']['p05']:.3f} ({r['shared_dense_h0']['verdict']}) "
            f"| {r['shared_intermediate_h0']['p05']:.3f} ({r['shared_intermediate_h0']['verdict']}) "
            f"| {r['shared_banded_h0']['p05']:.3f} ({r['shared_banded_h0']['verdict']}) "
            f"| {r['power_short']['association_power']:.3f} "
            f"| {r['power_mid']['association_power']:.3f} "
            f"| {r['ship_verdict']} |"
        )
    lines.append("")

    if top:
        c = top["combo"]
        lines.append("## Current Best Candidate")
        lines.append("")
        lines.append(
            f"The highest-scoring screened candidate is `h={c['bandwidth_um']:g} um`, "
            f"DCLF `{c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g} um`, with verdict "
            f"`{top['ship_verdict']}` and score `{top['ship_score']}`."
        )
        lines.append("")
        if top["ship_verdict"] == "SHIP_CANDIDATE":
            lines.append("This is a candidate for a dense-tissue primary null, pending "
                         "paper-grade simulation counts and external estimator checks.")
        elif top["ship_verdict"] == "BORDERLINE_NEEDS_MORE_SIM":
            lines.append("This is not yet shippable; the pilot confidence intervals are too "
                         "wide or one size regime is borderline.")
        elif top["ship_verdict"] == "REJECT_WEAK_POWER":
            lines.append("The current best candidate is not useful as a primary null because "
                         "it loses power for planted attraction.")
        else:
            lines.append("The current best candidate is not shippable because at least one "
                         "true-null regime is anti-conservative.")
        lines.append("")

    lines.append("## Architecture Gate Sweep")
    lines.append("")
    if not arch_rows:
        lines.append("Architecture gate sweep was skipped.")
    else:
        lines.append("| candidate | corr knob (um) | ell_hat median (um) | type-I | short power |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in arch_rows:
            lines.append(
                f"| {r['combo']} | {r['architecture_corr_knob_um']:g} "
                f"| {r['ell_hat_median_um']} | {r['size_type_I']:.3f} "
                f"| {r['power_short']:.3f} |"
            )
        lines.append("")
        lines.append("Interpretation: derive the dense-tissue validity gate from the "
                     "smallest measured `ell_hat` where type-I error is controlled "
                     "while power remains acceptable. Do not assume the old `2 x h` "
                     "rule until this table supports it at paper-grade simulation counts.")
    lines.append("")

    lines.append("## Decision Rules Used By This Harness")
    lines.append("")
    lines.append("- Size target: `P(p <= 0.05)` should be in `[0.03, 0.07]`.")
    lines.append("- Short and mid attraction power target: `>= 0.80`.")
    lines.append("- A candidate is rejected if any H0 regime is clearly anti-conservative.")
    lines.append("- Borderline pilot results require more simulations, not optimism.")
    lines.append("")

    lines.append("## Required Before Production Use")
    lines.append("")
    lines.append("1. Re-run the selected candidate(s) with `--sims 300 --nperm 199` or stronger.")
    lines.append("2. Repeat Python-vs-spatstat estimator equivalence at the selected dense bandwidth.")
    lines.append("3. Repeat the edge-correction cancellation check at the selected dense bandwidth/band.")
    lines.append("4. Validate on real dense multiplex/CODEX coordinates with negative and positive controls.")
    lines.append("5. Add a separate runtime preset only after the simulation-derived architecture gate is fixed.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=12,
                    help="screening simulations per combo/regime")
    ap.add_argument("--arch-sims", type=int, default=8,
                    help="simulations per architecture knob for gate sweep")
    ap.add_argument("--nperm", type=int, default=39,
                    help="Monte-Carlo permutations per DCLF test")
    ap.add_argument("--bandwidths", default="35,40,45,50,75")
    ap.add_argument("--bands", default="6-25,8-30,10-30,10-40,10-50")
    ap.add_argument("--radii-max", type=float, default=80.0)
    ap.add_argument("--radii-step", type=float, default=4.0)
    ap.add_argument("--window-um", type=float, default=800.0)
    ap.add_argument("--n-points", type=int, default=260)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--sigma-log", type=float, default=1.15)
    ap.add_argument("--dense-corr-um", type=float, default=45.0)
    ap.add_argument("--intermediate-corr-um", type=float, default=75.0)
    ap.add_argument("--band-period-um", type=float, default=90.0)
    ap.add_argument("--band-sigma-um", type=float, default=18.0)
    ap.add_argument("--short-jitter-um", type=float, default=8.0)
    ap.add_argument("--mid-jitter-um", type=float, default=18.0)
    ap.add_argument("--planted-fraction", type=float, default=0.50)
    ap.add_argument("--arch-corrs", default="25,35,45,60,80,100,130,160")
    ap.add_argument("--arch-top-n", type=int, default=2)
    ap.add_argument("--skip-architecture", action="store_true")
    args = ap.parse_args(argv)

    args.bandwidths = _parse_csv_floats(args.bandwidths)
    args.bands = _parse_bands(args.bands)
    args.arch_corrs = _parse_csv_floats(args.arch_corrs)

    factory = DensePatternFactory(
        window_um=args.window_um,
        n_points=args.n_points,
        grid=args.grid,
        sigma_log=args.sigma_log,
    )
    combos = [Combo(bw, lo, hi) for bw in args.bandwidths for lo, hi in args.bands]

    print("=" * 88)
    print("DENSE-TISSUE REWEIGHTED NULL CALIBRATION HARNESS")
    print("=" * 88)
    print(f"screen sims={args.sims} arch_sims={args.arch_sims} nperm={args.nperm}")
    print(f"bandwidths={args.bandwidths}")
    print(f"bands={args.bands}")
    print(f"window={args.window_um}um n={args.n_points} points/pop")

    rows = _screen_combos(args, combos, factory)
    selected = [] if args.skip_architecture else _select_for_architecture(rows, args.arch_top_n)
    arch_rows = [] if args.skip_architecture else _architecture_gate(args, selected, factory)

    payload = {
        "config": {
            "sims": args.sims,
            "arch_sims": args.arch_sims,
            "nperm": args.nperm,
            "bandwidths": args.bandwidths,
            "bands": args.bands,
            "radii_max": args.radii_max,
            "radii_step": args.radii_step,
            "window_um": args.window_um,
            "n_points": args.n_points,
            "grid": args.grid,
            "sigma_log": args.sigma_log,
            "dense_corr_um": args.dense_corr_um,
            "intermediate_corr_um": args.intermediate_corr_um,
            "band_period_um": args.band_period_um,
            "band_sigma_um": args.band_sigma_um,
            "short_jitter_um": args.short_jitter_um,
            "mid_jitter_um": args.mid_jitter_um,
            "planted_fraction": args.planted_fraction,
            "arch_corrs": args.arch_corrs,
        },
        "screen": rows,
        "architecture_gate": arch_rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    report = _make_report(args, rows, arch_rows)
    with open(OUT_MD, "w") as f:
        f.write(report)

    print("\n" + "=" * 88)
    print("WROTE")
    print("=" * 88)
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")

    ranked = sorted(rows, key=lambda r: (-r["ship_score"], r["combo"]["bandwidth_um"],
                                        r["combo"]["dclf_rmax_um"]))
    best = ranked[0] if ranked else None
    if best:
        c = best["combo"]
        print("\nBEST SCREENED CANDIDATE")
        print(f"  h={c['bandwidth_um']:g}um band={c['dclf_rmin_um']:g}-{c['dclf_rmax_um']:g}um "
              f"verdict={best['ship_verdict']} score={best['ship_score']}")
        return 0 if best["ship_verdict"] in ("SHIP_CANDIDATE", "BORDERLINE_NEEDS_MORE_SIM") else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
