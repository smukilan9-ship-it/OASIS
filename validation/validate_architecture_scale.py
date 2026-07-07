"""
validate_architecture_scale.py — operating characteristics of the reweighted primary
null as a function of tissue architecture scale, and validation of the architecture-
scale estimator that gates it (audit A6 / ihc.md §15.5).

WHY. The reweighted inhomogeneous cross-K (bandwidth 75 µm) assumes tissue
architecture is COARSER than the reweighting bandwidth. When architecture varies
inside the 10–50 µm interaction band, the intensity reweighting cannot separate
shared compartment preference from cell-scale engagement and the test becomes
anti-conservative (false 'robust'). This turns that disclosed assumption into a
measured, calibrated guard.

WHAT. A Monte-Carlo size/power study. For a sweep of architecture scales we simulate:
  - NULL  (shared preference, NO engagement): A and B are independent inhomogeneous
    Poisson draws from the SAME log-Gaussian intensity field. A correct test rejects
    at ~alpha; inflation here is the anti-conservative failure.
  - ALT   (genuine engagement): B is planted within `engage_um` of A cells. Rejection
    here is power.
We record, per setting, the reweighted test's rejection rate AND the estimated
architecture scale ℓ̂ (validation/… estimate_architecture_scale), so the operating
characteristics are plotted against the SAME quantity the runtime gate measures.

PASS. (1) size controlled (≤ size_tol) once ℓ̂ ≥ bandwidth; (2) the anti-conservative
regime is demonstrated at small ℓ̂ (proves the guard is necessary); (3) power is high
in the valid regime; (4) ℓ̂ increases monotonically with the architecture knob.

Reference-free, no external data. Long-running Monte-Carlo; scale via CLI:
  python validation/validate_architecture_scale.py [--sims N] [--nperm M]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatial_stats as ss

BANDWIDTH_UM = ss._REWEIGHT_BANDWIDTH_UM      # 75
BAND_MAX_UM = ss._DCLF_RMAX_UM               # 50
W = 800.0                                    # square window side, µm (pixel_size = 1)
N = 350                                      # points per pattern
RADII = np.arange(0.0, 101.0, 4.0)
ARCH_KNOB_UM = [30.0, 50.0, 75.0, 110.0, 160.0, 240.0, 360.0]   # field correlation length
ENGAGE_UM = 18.0                             # planted attraction scale (in the band)


def _field(corr_len_um, rng, grid=128, sigma_log=1.1):
    """A log-Gaussian intensity field over the window with the given correlation
    length; returns a normalized pmf over grid cells + the cell size (µm)."""
    cell = W / grid
    g = rng.standard_normal((grid, grid))
    g = gaussian_filter(g, sigma=(corr_len_um / cell) / 2.0, mode="wrap")
    g = (g - g.mean()) / (g.std() + 1e-9)
    lam = np.exp(sigma_log * g)
    return (lam / lam.sum()), cell


def _sample(pmf, cell, n, rng):
    idx = rng.choice(pmf.size, size=n, p=pmf.ravel())
    iy, ix = np.unravel_index(idx, pmf.shape)
    return np.column_stack([ix * cell + rng.uniform(0, cell, n),
                            iy * cell + rng.uniform(0, cell, n)])


def _significant(a, b, seed, n_perm):
    o = ss.cross_k_inhom_reweighted_test(a, b, RADII, W * W, 1.0, n_perm=n_perm,
                                         seed=seed, bandwidth_um=BANDWIDTH_UM)
    return bool((o.get("global") or {}).get("significant"))


def run(sims, n_perm):
    rows = []
    for k, corr in enumerate(ARCH_KNOB_UM):
        fp = power = 0
        ell_hat = []
        for i in range(sims):
            rng = np.random.default_rng(1000 * k + i)
            pmf, cell = _field(corr, rng)
            # NULL: shared field, independent A & B
            a = _sample(pmf, cell, N, rng)
            b = _sample(pmf, cell, N, rng)
            if _significant(a, b, seed=i, n_perm=n_perm):
                fp += 1
            ell = ss.estimate_architecture_scale(a, 1.0, bbox=(0, 0, W, W))
            if ell is not None:
                ell_hat.append(ell)
            # ALT: engagement — B planted near A within the interaction band
            a2 = _sample(pmf, cell, N, rng)
            m = N // 2
            base = a2[rng.integers(0, N, m)] + rng.normal(0, ENGAGE_UM, (m, 2))
            bg = _sample(pmf, cell, N - m, rng)
            b2 = np.clip(np.vstack([base, bg]), 0, W)
            if _significant(a2, b2, seed=1000 + i, n_perm=n_perm):
                power += 1
        rows.append({
            "arch_knob_um": corr,
            "ell_hat_um": round(float(np.median(ell_hat)), 1) if ell_hat else None,
            "size_type_I": round(fp / sims, 3),
            "power": round(power / sims, 3),
        })
        r = rows[-1]
        print(f"  knob={corr:5.0f}µm  ℓ̂(median)={r['ell_hat_um']:>6}µm  "
              f"type-I={r['size_type_I']:.3f}  power={r['power']:.3f}")
    return rows


def _figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ell = [r["ell_hat_um"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ell, [r["size_type_I"] for r in rows], "o-", color="#dc2626", label="type-I (null)")
    ax[0].plot(ell, [r["power"] for r in rows], "s-", color="#2563eb", label="power (engaged)")
    ax[0].axhline(0.05, ls=":", c="#64748b", label="α = 0.05")
    ax[0].axvline(BANDWIDTH_UM, ls="--", c="#16a34a", label=f"bandwidth {BANDWIDTH_UM:.0f}µm")
    ax[0].axvspan(0, BAND_MAX_UM, color="#fee2e2", alpha=.5)
    ax[0].set_xlabel("estimated architecture scale ℓ̂ (µm)")
    ax[0].set_ylabel("rejection rate"); ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title("Operating characteristics vs architecture scale"); ax[0].legend(fontsize=8)
    ax[1].plot([r["arch_knob_um"] for r in rows], ell, "o-", color="#111827")
    ax[1].plot([0, max(ARCH_KNOB_UM)], [0, max(ARCH_KNOB_UM)], ":", c="#94a3b8")
    ax[1].set_xlabel("architecture knob (field correlation length, µm)")
    ax[1].set_ylabel("estimated ℓ̂ (µm)"); ax[1].set_title("Estimator recovery (monotonic)")
    fig.tight_layout()
    out = os.path.join(os.environ.get("OASIS_REPORT_DIR", "."), "architecture_scale.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--nperm", type=int, default=79)
    args = ap.parse_args(argv)

    print(f"\nArchitecture-scale operating characteristics "
          f"(bandwidth {BANDWIDTH_UM:.0f}µm, band≤{BAND_MAX_UM:.0f}µm, "
          f"sims={args.sims}, n_perm={args.nperm})\n")
    rows = run(args.sims, args.nperm)

    # ── verdict ──────────────────────────────────────────────────────────────
    # The study's PRODUCT is the calibration curve + the DERIVED validity threshold
    # ℓ*_valid (smallest ℓ̂ with type-I ≤ size_tol). We do not assume the threshold;
    # we derive it and confirm the guard's shape: anti-conservative below, controlled
    # + powered above, estimator monotonic.
    SIZE_TOL = 0.075
    ell_series = [r["ell_hat_um"] for r in rows if r["ell_hat_um"]]
    monotonic = all(x <= y + 1e-6 for x, y in zip(ell_series, ell_series[1:]))
    controlled = [r for r in rows if r["ell_hat_um"] and r["size_type_I"] <= SIZE_TOL]
    ell_valid = min((r["ell_hat_um"] for r in controlled), default=None)
    inflation_shown = any(r["size_type_I"] > 0.10 for r in rows
                          if r["ell_hat_um"] and r["ell_hat_um"] < BANDWIDTH_UM)
    above = [r for r in rows if ell_valid and r["ell_hat_um"] and r["ell_hat_um"] >= ell_valid]
    power_ok = (np.mean([r["power"] for r in above]) >= 0.60) if above else False
    # sanity: the shipped gate (1.5×bw) should sit at/above the derived threshold
    gate_conservative = (ell_valid is not None and
                         ell_valid <= ss._ARCH_MIN_SCALE_FACTOR * BANDWIDTH_UM + 1e-6)

    fig = _figure(rows)
    metrics = {
        "derived_validity_threshold_um": ell_valid,
        "shipped_gate_min_ok_um": ss._ARCH_MIN_SCALE_FACTOR * BANDWIDTH_UM,
        "gate_at_or_above_derived_threshold": gate_conservative,
        "anti_conservative_shown_below_bandwidth": inflation_shown,
        "power_above_threshold": round(float(np.mean([r["power"] for r in above])), 3) if above else None,
        "estimator_monotonic": monotonic,
        "bandwidth_um": BANDWIDTH_UM, "size_tol": SIZE_TOL,
        "table": rows,
    }
    print("\n##METRICS## " + json.dumps(metrics))
    if fig:
        print(f"  figure: {fig}")

    ok = bool(monotonic and inflation_shown and ell_valid is not None
              and power_ok and gate_conservative)
    print(f"\n  derived validity threshold ℓ*≈{ell_valid} µm · shipped gate "
          f"{ss._ARCH_MIN_SCALE_FACTOR}×bw={ss._ARCH_MIN_SCALE_FACTOR*BANDWIDTH_UM:.0f}µm "
          f"(≥ threshold: {gate_conservative}) · anti-conservative below bw: "
          f"{inflation_shown} · power above: {power_ok} · monotonic: {monotonic}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
