"""
validate_edge_correction.py — does EDGE CORRECTION change the reweighted cross-K
null's calibration? (Settles "why no edge correction"; ihc.md §17.)

Hypothesis: observed and null K are both computed with NO edge correction, so a
systematic boundary undercount hits BOTH sides of the DCLF rank test and cancels →
uncorrected is safe. Test: re-run the §15 three-regime calibration with a proper
translation correction applied IDENTICALLY to observed and null K, and compare.

The translation (Ripley) correction for a rectangular window [0,Lx]×[0,Ly] is exact:
a pair separated by (dx,dy) gets weight e = |W| / ((Lx−dx)(Ly−dy)). We first VALIDATE
this against spatstat's correction="translate" (reusing the §16 R bridge), then use
it in the calibration. Because the correction goes through the SAME function for
observed and null, it is applied symmetrically by construction.

Config via env: NREAL (500), NPERM (199). Output: validation/edge_correction_output.txt
"""

import os
import sys
import csv
import shutil
import tempfile
import subprocess
import numpy as np
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from spatial_stats import (_loo_kernel_intensity, _build_intensity_grid,      # noqa: E402
                           _draw_n_from_grid, _null_summary_from_k,
                           _cross_k_inhom_weighted, _DCLF_RMIN_UM, _DCLF_RMAX_UM)

NREAL = int(os.environ.get("NREAL", "500"))
NPERM = int(os.environ.get("NPERM", "199"))
PIX   = 1.0
RADII = np.arange(0.0, 100.0, 4.0)
WIN   = 960.0
WINBOX = box(0, 0, WIN, WIN)
N_PTS = 250
SIGMA = 70.0
BW    = 75.0
CENTERS = np.array([(cx, cy) for cx in (160, 480, 800) for cy in (160, 480, 800)],
                   dtype=float)
R_SCRIPT = os.path.join(HERE, "spatstat_crossval.R")
OUT = os.path.join(HERE, "edge_correction_output.txt")

_log = []
def log(s=""):
    print(s); _log.append(s)


# ──────────────────────────────────────────────────────────────────────────────
# Edge-corrected reweighted cross-K (HARNESS ONLY — production is untouched)
# ──────────────────────────────────────────────────────────────────────────────

def _cross_k_weighted_corr(A, B, inv_a, inv_b, radii_px, area_px, Lx, Ly, correction):
    """Reweighted cross-K with optional translation edge correction on a rectangle.
    correction='none' reproduces the production estimator exactly."""
    from scipy.spatial import cKDTree
    n_r = len(radii_px)
    if len(A) == 0 or len(B) == 0:
        return np.zeros(n_r)
    ta, tb = cKDTree(A), cKDTree(B)
    rmax = float(np.max(radii_px))
    sdm = ta.sparse_distance_matrix(tb, rmax, output_type="coo_matrix")
    d = sdm.data
    if d.size == 0:
        return np.zeros(n_r)
    ai, bj = sdm.row, sdm.col
    w = inv_a[ai] * inv_b[bj]
    if correction == "translate":
        dx = np.abs(A[ai, 0] - B[bj, 0])
        dy = np.abs(A[ai, 1] - B[bj, 1])
        ov = np.clip(Lx - dx, 1e-9, None) * np.clip(Ly - dy, 1e-9, None)
        w = w * (Lx * Ly) / ov
    order = np.argsort(d, kind="mergesort")
    d_s = d[order]
    cumw = np.concatenate([[0.0], np.cumsum(w[order])])
    idx = np.searchsorted(d_s, radii_px, side="right")
    return cumw[idx] / float(area_px)


def reweighted_test_corr(A, B, correction, n_perm=NPERM, seed=0):
    """Mirror of cross_k_inhom_reweighted_test, with `correction` applied to BOTH
    observed and null K via the same function. Returns the DCLF global dict."""
    area = WINBOX.area
    h = BW
    gB = _build_intensity_grid(B, WINBOX, WINBOX.bounds, h)
    inv_a = 1.0 / _loo_kernel_intensity(A, h, area)
    inv_b = 1.0 / _loo_kernel_intensity(B, h, area)
    k_obs = _cross_k_weighted_corr(A, B, inv_a, inv_b, RADII, area, WIN, WIN, correction)
    l_obs = np.sqrt(np.clip(k_obs, 0.0, None) / np.pi) - RADII
    rng = np.random.default_rng(int(seed))
    null_k = np.empty((n_perm, len(RADII)))
    for k in range(n_perm):
        b_star = _draw_n_from_grid(gB, len(B), rng)
        inv_bs = 1.0 / _loo_kernel_intensity(b_star, h, area)
        null_k[k] = _cross_k_weighted_corr(A, b_star, inv_a, inv_bs, RADII, area,
                                           WIN, WIN, correction)
    summ = _null_summary_from_k(RADII, k_obs, l_obs, null_k, PIX, n_perm,
                                _DCLF_RMIN_UM, _DCLF_RMAX_UM)
    return summ["global"]


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — validate our translation correction against spatstat (rigor)
# ──────────────────────────────────────────────────────────────────────────────

def validate_translate_vs_spatstat():
    if shutil.which("Rscript") is None:
        log("  (spatstat check skipped — Rscript not found)")
        return None
    rng = np.random.default_rng(2)
    cen = rng.uniform(150, WIN - 150, (5, 2))
    A = np.clip(cen[rng.integers(0, 5, 220)] + rng.normal(0, 40, (220, 2)), 1, WIN - 1)
    B = np.clip(cen[rng.integers(0, 5, 200)] + rng.normal(0, 40, (200, 2)), 1, WIN - 1)
    area = WIN * WIN
    inv_a = 1.0 / _loo_kernel_intensity(A, BW, area)
    inv_b = 1.0 / _loo_kernel_intensity(B, BW, area)
    k_mine = _cross_k_weighted_corr(A, B, inv_a, inv_b, RADII, area, WIN, WIN, "translate")

    d = tempfile.mkdtemp(prefix="edge_")
    poly = np.array([[0.0, 0.0], [WIN, 0.0], [WIN, WIN], [0.0, WIN]])
    def w(fn, rows, header):
        with open(os.path.join(d, fn), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(header); wr.writerows(rows)
    w("points.csv", [("A", f"{x:.6f}", f"{y:.6f}") for x, y in A]
                    + [("B", f"{x:.6f}", f"{y:.6f}") for x, y in B], ["type", "x", "y"])
    w("window.csv", [(f"{x:.6f}", f"{y:.6f}") for x, y in poly], ["x", "y"])
    w("params.csv", [(f"{BW:.10f}", f"{area:.6f}")], ["h", "area"])
    w("rgrid.csv", [(f"{ri:.10f}",) for ri in RADII], ["r"])
    w("lambdaA.csv", [(f"{1.0/v:.12e}",) for v in inv_a], ["lambda"])
    w("lambdaB.csv", [(f"{1.0/v:.12e}",) for v in inv_b], ["lambda"])
    proc = subprocess.run(["Rscript", R_SCRIPT, d], capture_output=True, text=True)
    res = None
    p = os.path.join(d, "r_K_translate.csv")
    if proc.returncode == 0 and os.path.exists(p):
        with open(p) as f:
            k_r = np.array([float(row["K"]) for row in csv.DictReader(f)])
        valid = (k_mine > 0) & (k_r > 0)
        rel = np.abs(k_mine[valid] - k_r[valid]) / np.abs(k_r[valid])
        res = float(rel.max())
    shutil.rmtree(d, ignore_errors=True)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# 3-regime calibration under each correction
# ──────────────────────────────────────────────────────────────────────────────

def draw_shared(rng, n):
    return np.clip(CENTERS[rng.integers(0, len(CENTERS), n)]
                   + rng.normal(0, SIGMA, (n, 2)), 1, WIN - 1)

def draw_uniform(rng, n):
    return rng.uniform(0, WIN, (n, 2))

def size_rate(draw, correction, seed):
    rng = np.random.default_rng(seed)
    p = np.array([reweighted_test_corr(draw(rng, N_PTS), draw(rng, N_PTS),
                                       correction)["global_p_dclf"]
                  for _ in range(NREAL)])
    return float(np.mean(p <= 0.05))

def power_rate(correction, seed, jitter, base=draw_shared):
    rng = np.random.default_rng(seed)
    n = 0
    for _ in range(NREAL):
        A = base(rng, N_PTS)
        B = np.clip(A[rng.integers(0, len(A), N_PTS)] + rng.normal(0, jitter, (N_PTS, 2)),
                    1, WIN - 1)
        g = reweighted_test_corr(A, B, correction)
        n += g["significant"] and g["direction"] == "association"
    return float(n / NREAL)

def ci(rate):
    se = (rate * (1 - rate) / NREAL) ** 0.5
    return rate - 1.96 * se, rate + 1.96 * se


def main():
    log("=" * 78)
    log("EDGE CORRECTION vs reweighted cross-K calibration (bw=75µm + LOO)")
    log("=" * 78)
    log(f"  NREAL={NREAL}  NPERM={NPERM}  window {WIN:.0f}px  field σ={SIGMA:.0f}px")

    chk = validate_translate_vs_spatstat()
    if chk is not None:
        log(f"\n  translation-correction implementation vs spatstat 'translate': "
            f"max rel diff {chk:.2e}  -> {'MATCH' if chk < 1e-6 else 'CHECK'}")

    log(f"\n{'correction':12s} {'shared P05':>11} {'uniform P05':>12} "
        f"{'pow@7px':>9} {'pow@25px':>9}")
    log("-" * 78)
    rows = {}
    for corr in ("none", "translate"):
        s_sh = size_rate(draw_shared, corr, 1001)
        s_un = size_rate(draw_uniform, corr, 2002)
        p7   = power_rate(corr, 3003, 7.0)
        p25  = power_rate(corr, 4004, 25.0)
        rows[corr] = (s_sh, s_un, p7, p25)
        log(f"{corr:12s} {s_sh:>11.3f} {s_un:>12.3f} {p7:>9.3f} {p25:>9.3f}")
    log("-" * 78)
    for corr, (s_sh, s_un, p7, p25) in rows.items():
        csh, cun = ci(s_sh), ci(s_un)
        log(f"  {corr:10s}: shared {s_sh:.3f} [{csh[0]:.3f},{csh[1]:.3f}]  "
            f"uniform {s_un:.3f} [{cun[0]:.3f},{cun[1]:.3f}]")

    # Decision
    n_sh, n_un, n_p7, n_p25 = rows["none"]
    t_sh, t_un, t_p7, t_p25 = rows["translate"]
    # Noise band: 3·SE with a 0.03 floor (so identical rates, incl. exactly 0/1
    # where SE=0, correctly read as "within noise"). Paired design (same seeds).
    se = lambda r: (r * (1 - r) / NREAL) ** 0.5
    band = lambda r: max(3 * se(r), 0.03)
    within_noise = (abs(n_sh - t_sh) <= band(n_sh) and abs(n_un - t_un) <= band(n_un)
                    and abs(n_p7 - t_p7) <= 0.05 and abs(n_p25 - t_p25) <= 0.05)
    log("\n" + "=" * 78); log("DECISION"); log("=" * 78)
    if within_noise:
        log("  Calibration AND power are UNCHANGED (within noise) between uncorrected")
        log("  and translation-corrected. The boundary undercount cancels in the DCLF")
        log("  rank test exactly as hypothesized (it hits observed and null K alike).")
        log("  → UNCORRECTED IS JUSTIFIED. Keep production as-is; this is the evidence")
        log("    that retires the edge-correction reviewer objection.")
    else:
        better = (abs(t_sh - 0.05) + abs(t_un - 0.05)) < (abs(n_sh - 0.05) + abs(n_un - 0.05))
        if better and t_p7 >= 0.80 and t_p25 >= 0.80:
            log("  Translation correction is BETTER calibrated while keeping power.")
            log("  → ADOPT translation correction in production (wire through, re-run suite).")
        else:
            log("  Correction changes calibration but is NOT better (or loses power).")
            log("  → UNCORRECTED is the better choice; report why.")

    with open(OUT, "w") as f:
        f.write("\n".join(_log) + "\n")
    log(f"\n  (saved to {os.path.relpath(OUT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
