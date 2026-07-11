"""
validate_reweighted_null.py — 3-regime proof for the redesigned PRIMARY test
(spatial_stats.cross_k_inhom_reweighted_test), the intensity-reweighted
inhomogeneous cross-K with per-simulation intensity re-estimation (ihc.md §15).

A null is shippable ONLY if it passes ALL THREE:
  1. SHARED-PREFERENCE H0  (the regime the old nulls failed): A,B independent draws
     from the SAME coarse field. PASS = P(p<=0.05) in ~0.03–0.07.
  2. UNIFORM-CSR SANITY     (true independence, no preference): must ALSO be ~0.05
     — proves the null isn't just globally conservative.
  3. POWER / POSITIVE CONTROL: genuine short-range cross-attraction must still be
     detected at high power. A null that passes 1+2 by never firing is a FAILURE.

We sweep the intensity bandwidth (µm) because the bandwidth must sit ABOVE the
interaction band (10–50 µm) but capture the architecture; the calibration picks it,
we do not assume it.

Config via env: NREAL (default 300), NPERM (default 199),
                BANDWIInhDTHS (csv µm, default "50,75,100,150").
Output: printed + saved to validation/reweighted_null_output.txt
"""

import os
import sys
import numpy as np
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial.spatial_stats import cross_k_inhom_reweighted_test   # noqa: E402

NREAL = int(os.environ.get("NREAL", "300"))
NPERM = int(os.environ.get("NPERM", "199"))
BANDWIDTHS = [float(x) for x in os.environ.get("BANDWIDTHS", "50,75,100,150").split(",")]
PIX   = 1.0                                   # 1 µm/px → band 10–50 µm == 10–50 px
RADII = np.arange(0.0, 100.0, 4.0)
WIN   = 960.0
N_PTS = 250
SIGMA = 70.0
CENTERS = np.array([(cx, cy) for cx in (160, 480, 800)
                    for cy in (160, 480, 800)], dtype=float)
WINBOX = box(0, 0, WIN, WIN)
ATTRACT_JITTER = 7.0                          # short-range attraction scale (px)

_log = []
def log(s=""):
    print(s); _log.append(s)


def draw_shared(rng, n):
    idx = rng.integers(0, len(CENTERS), n)
    return np.clip(CENTERS[idx] + rng.normal(0, SIGMA, (n, 2)), 1, WIN - 1)


def draw_uniform(rng, n):
    return rng.uniform(0, WIN, (n, 2))


def p_of(A, B, bw):
    res = cross_k_inhom_reweighted_test(A, B, RADII, WINBOX.area, PIX,
                                        n_perm=NPERM, seed=0, tissue_polygon=WINBOX,
                                        bandwidth_um=bw)
    g = res["global"]
    return g["global_p_dclf"], g["significant"], g["direction"]


def regime_size(draw_fn, bw, seed):
    """Fraction of realizations rejecting (P(p<=.05)) under an H0 regime."""
    rng = np.random.default_rng(seed)
    ps = []
    for i in range(NREAL):
        A = draw_fn(rng, N_PTS)
        B = draw_fn(rng, N_PTS)        # independent of A → H0 true
        p, _, _ = p_of(A, B, bw)
        ps.append(p)
    ps = np.asarray(ps)
    return float(np.mean(ps <= 0.05)), float(ps.mean())


def regime_power(bw, seed, jitter, base="shared"):
    """Power: A from a regime, B = A's neighbourhood + jitter (true attraction at
    scale `jitter` px). Fraction called significant ASSOCIATION. We test both a
    SHORT-range (≪ band) and a MID-range (inside the band) attraction, because a
    bandwidth that absorbs mid-band structure would still pass short-range power."""
    rng = np.random.default_rng(seed)
    n_assoc = 0
    for i in range(NREAL):
        A = (draw_shared(rng, N_PTS) if base == "shared" else draw_uniform(rng, N_PTS))
        pick = A[rng.integers(0, len(A), N_PTS)]
        B = np.clip(pick + rng.normal(0, jitter, (N_PTS, 2)), 1, WIN - 1)
        p, sig, direction = p_of(A, B, bw)
        if sig and direction == "association":
            n_assoc += 1
    return float(n_assoc / NREAL)


def ci95(rate, n):
    se = (rate * (1 - rate) / max(n, 1)) ** 0.5
    return rate - 1.96 * se, rate + 1.96 * se


def main():
    log("=" * 80)
    log("REWEIGHTED INHOMOGENEOUS CROSS-K — 3-REGIME CALIBRATION + POWER")
    log("=" * 80)
    log(f"  NREAL={NREAL}  NPERM={NPERM}  bandwidths={BANDWIDTHS} µm  "
        f"field σ={SIGMA:.0f}px  band 10–50 µm")
    log(f"  PASS = shared-pref size ∈[0.03,0.07]  AND  uniform size ∈[0.03,0.07]  "
        f"AND  power ≥ 0.80\n")

    log(f"{'bw(µm)':>7} {'shared P05':>11} {'uniform P05':>12} "
        f"{'pow@7px':>9} {'pow@25px':>9}   verdict")
    log("-" * 80)
    results = {}
    for bw in BANDWIDTHS:
        s_shared, m_shared = regime_size(draw_shared, bw, seed=1001)
        s_unif,  m_unif    = regime_size(draw_uniform, bw, seed=2002)
        pw_short = regime_power(bw, seed=3003, jitter=7.0,  base="shared")
        pw_mid   = regime_power(bw, seed=4004, jitter=25.0, base="shared")
        cal_ok = (0.03 <= s_shared <= 0.07) and (0.03 <= s_unif <= 0.07)
        pw_ok  = pw_short >= 0.80 and pw_mid >= 0.80
        verdict = ("SHIP" if (cal_ok and pw_ok)
                   else "calibrated-but-weak-power" if cal_ok
                   else "shared-leak" if s_shared > 0.07
                   else "uniform-leak" if s_unif > 0.07
                   else "conservative" if s_shared < 0.03
                   else "fail")
        results[bw] = dict(shared=s_shared, uniform=s_unif,
                           pw_short=pw_short, pw_mid=pw_mid, verdict=verdict)
        log(f"{bw:>7.0f} {s_shared:>11.3f} {s_unif:>12.3f} {pw_short:>9.3f} "
            f"{pw_mid:>9.3f}   {verdict}")
    log("-" * 80)

    log("\nDETAIL (95% CI on the rates):")
    for bw, r in results.items():
        cs = ci95(r["shared"], NREAL); cu = ci95(r["uniform"], NREAL)
        log(f"  bw={bw:.0f}µm  shared {r['shared']:.3f} [{cs[0]:.3f},{cs[1]:.3f}]  "
            f"uniform {r['uniform']:.3f} [{cu[0]:.3f},{cu[1]:.3f}]  "
            f"pow@7px {r['pw_short']:.3f}  pow@25px {r['pw_mid']:.3f}")

    ship = [bw for bw, r in results.items() if r["verdict"] == "SHIP"]
    log("\n" + "=" * 80)
    log("VERDICT")
    log("=" * 80)
    if ship:
        best = min(ship, key=lambda bw: abs(results[bw]["shared"] - 0.05))
        log(f"  SHIPPABLE bandwidth(s): {ship} µm.")
        log(f"  Recommended: {best:.0f} µm (shared size "
            f"{results[best]['shared']:.3f}, uniform {results[best]['uniform']:.3f}, "
            f"power@7px {results[best]['pw_short']:.3f}, "
            f"power@25px {results[best]['pw_mid']:.3f}).")
        log("  → The reweighted inhomogeneous cross-K is calibrated under shared "
            "preference AND uniform independence AND retains power. It replaces the "
            "anti-conservative resampling-Kinhom + toroidal gate.")
    else:
        log("  ⚠ NO bandwidth passes all three regimes. Do NOT ship this null as-is.")
        log("  Report the failure plainly (ihc.md §15) — the honest fallback is a "
            "covariate-conditioned null requiring architecture segmentation we do "
            "not yet have.")

    with open(os.path.join(os.path.dirname(__file__), "reweighted_null_output.txt"),
              "w") as f:
        f.write("\n".join(_log) + "\n")
    log(f"\n  (saved to validation/reweighted_null_output.txt)")
    return 0 if ship else 2


if __name__ == "__main__":
    sys.exit(main())
