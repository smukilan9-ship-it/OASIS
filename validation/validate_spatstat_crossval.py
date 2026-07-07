"""
validate_spatstat_crossval.py — cross-validate our custom intensity-reweighted
inhomogeneous cross-K (spatial_stats) against spatstat's reference implementation.

READ-COMPARE-REPORT. No production code is modified. We feed BYTE-IDENTICAL inputs
to both tools (everything in PIXELS, pixel_size=1, so no unit mismatch) and compare:

  Stage A — INTENSITY: our `_loo_kernel_intensity` vs spatstat
            density.ppp(sigma=h, kernel="gaussian", leaveoneout=TRUE, edge=FALSE).
  Stage B — ESTIMATOR (the GATE): our `_cross_k_inhom_weighted` K(r)/L(r) vs
            spatstat Kcross.inhom(..., lambdaI, lambdaJ, correction="none",
            normpower=0), fed the IDENTICAL Python lambda so the estimator is
            isolated from the intensity. We also report border/translate/isotropic
            (ours is "none") so the matching correction is explicit.
  Stage C — TEST: noted, not numerically forced — our null is a per-simulation
            re-estimation bootstrap, spatstat's envelope is a different procedure;
            a difference there is NOT an estimator bug (reported, not gated).

Cases: CSR, clustered, the shared-preference calibration field (synthetic, known
answer) AND one REAL point pattern (a Schürch CODEX spot: real cells, real
coordinates, hull window) if the dataset CSV is present.

TOLERANCES (stated up front, applied in Step 5):
  Stage A: max relative |Δλ| over UNfloored points < 1e-3 (≈1e-6 expected; only the
           4h Gaussian-tail cutoff differs).
  Stage B (GATE): max relative |ΔL(r)| over the r-grid (where K>0) < 1e-3. A
           constant scale factor, if present, is reported as a definitional
           (normalization) difference, not a silent pass.

Output: printed + validation/spatstat_crossval_output.txt
Requires: R + spatstat.explore (the script reports a BLOCKER if unavailable).
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from spatial_stats import _loo_kernel_intensity, _cross_k_inhom_weighted  # noqa: E402

R_SCRIPT = os.path.join(HERE, "spatstat_crossval.R")
OUT = os.path.join(HERE, "spatstat_crossval_output.txt")
CSV = os.path.join(HERE, "CRC_clusters_neighborhoods_markers.csv")

_log = []
def log(s=""):
    print(s); _log.append(s)


# ──────────────────────────────────────────────────────────────────────────────
# Case construction (everything in pixels)
# ──────────────────────────────────────────────────────────────────────────────

def _box_poly(W):
    # anticlockwise, first vertex NOT repeated (spatstat owin convention)
    return np.array([[0.0, 0.0], [W, 0.0], [W, W], [0.0, W]])


def synthetic_cases():
    W = 1000.0
    h = 75.0
    r = np.arange(0.0, 100.0, 4.0)
    poly = _box_poly(W)
    cases = {}

    rng = np.random.default_rng(1)
    A = rng.uniform(0, W, (220, 2)); B = rng.uniform(0, W, (180, 2))
    cases["CSR"] = dict(A=A, B=B, poly=poly, area=W * W, h=h, r=r)

    rng = np.random.default_rng(2)
    cen = rng.uniform(150, W - 150, (5, 2))
    A = np.clip(cen[rng.integers(0, 5, 220)] + rng.normal(0, 40, (220, 2)), 1, W - 1)
    B = np.clip(cen[rng.integers(0, 5, 200)] + rng.normal(0, 40, (200, 2)), 1, W - 1)
    cases["clustered_shared"] = dict(A=A, B=B, poly=poly, area=W * W, h=h, r=r)

    # the calibration shared-preference field (periodic grid, sigma 70)
    W2 = 960.0
    centers = np.array([(cx, cy) for cx in (160, 480, 800) for cy in (160, 480, 800)],
                       dtype=float)
    rng = np.random.default_rng(3)
    def field(n):
        return np.clip(centers[rng.integers(0, len(centers), n)]
                       + rng.normal(0, 70, (n, 2)), 1, W2 - 1)
    cases["calibration_field"] = dict(A=field(250), B=field(250),
                                      poly=_box_poly(W2), area=W2 * W2, h=h, r=r)
    return cases


def real_case():
    """One REAL point pattern: a Schürch CODEX spot (CD8 = A, CD4 = B), hull window.
    Real cells/coordinates/window — the estimator is source-agnostic, so this is a
    valid real-data estimator check (it is NOT a CD8/TIM-3 serial pair; no committed
    image pairs exist). Pixel size 0.3775 µm/px → bandwidth 75 µm = 198.7 px."""
    if not os.path.exists(CSV):
        return None
    try:
        import pandas as pd
        from shapely.geometry import MultiPoint
    except Exception:
        return None
    CD8 = {"CD8+ T cells"}
    CD4 = {"CD4+ T cells CD45RO+", "CD4+ T cells", "CD4+ T cells GATA3+"}
    df = pd.read_csv(CSV, usecols=["patients", "spots", "ClusterName", "X:X", "Y:Y"],
                     low_memory=False)
    best = None
    for key, g in df.groupby(["patients", "spots"]):
        na = g["ClusterName"].isin(CD8).sum(); nb = g["ClusterName"].isin(CD4).sum()
        if na >= 40 and nb >= 40 and (best is None or min(na, nb) > best[0]):
            best = (min(na, nb), key)
    if best is None:
        return None
    g = df.groupby(["patients", "spots"]).get_group(best[1])
    A = g[g["ClusterName"].isin(CD8)][["X:X", "Y:Y"]].to_numpy(float)
    B = g[g["ClusterName"].isin(CD4)][["X:X", "Y:Y"]].to_numpy(float)
    allxy = g[["X:X", "Y:Y"]].to_numpy(float)
    hull = MultiPoint([tuple(p) for p in allxy]).convex_hull
    poly = np.asarray(hull.exterior.coords)[:-1]   # drop repeated last vertex
    # ensure anticlockwise (positive shoelace area)
    x, y = poly[:, 0], poly[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        poly = poly[::-1]
    h = 75.0 / 0.3775
    r = np.arange(0.0, 100.0, 4.0) / 0.3775
    return {"real_schurch_CD8_CD4": dict(A=A, B=B, poly=poly, area=float(hull.area),
                                         h=h, r=r)}


# ──────────────────────────────────────────────────────────────────────────────
# Run one case through both tools
# ──────────────────────────────────────────────────────────────────────────────

def run_case(name, c):
    d = tempfile.mkdtemp(prefix="xval_")
    A, B, poly, area, h, r = c["A"], c["B"], c["poly"], c["area"], c["h"], c["r"]
    # Buffer the window outward by a negligible epsilon and keep only points
    # strictly inside it, so Python and spatstat's ppp retain the IDENTICAL points
    # (spatstat drops boundary points; buffering moves any boundary vertex interior).
    # The buffer (~1e-4 of the window scale) changes area negligibly and is applied
    # identically to both tools, so the comparison is unaffected.
    try:
        import shapely
        from shapely.geometry import Polygon
        eps = 1e-4 * float(np.sqrt(area))
        pbuf = Polygon(poly).buffer(eps)
        insA = shapely.contains_xy(pbuf, A[:, 0], A[:, 1])
        insB = shapely.contains_xy(pbuf, B[:, 0], B[:, 1])
        A, B = A[insA], B[insB]
        poly = np.asarray(pbuf.exterior.coords)[:-1]
        x, y = poly[:, 0], poly[:, 1]
        if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
            poly = poly[::-1]
        area = float(pbuf.area)
    except Exception:
        pass
    win_area = area

    # Python intensity (the PRODUCTION function, with floor) and K
    lamA = _loo_kernel_intensity(A, h, win_area)
    lamB = _loo_kernel_intensity(B, h, win_area)
    k_py = _cross_k_inhom_weighted(A, B, 1.0 / lamA, 1.0 / lamB, r, area)

    # export byte-identical inputs
    import csv
    def w(fn, rows, header):
        with open(os.path.join(d, fn), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(header); wr.writerows(rows)
    w("points.csv", [("A", f"{x:.6f}", f"{y:.6f}") for x, y in A]
                    + [("B", f"{x:.6f}", f"{y:.6f}") for x, y in B], ["type", "x", "y"])
    w("window.csv", [(f"{x:.6f}", f"{y:.6f}") for x, y in poly], ["x", "y"])
    w("params.csv", [(f"{h:.10f}", f"{area:.6f}")], ["h", "area"])
    w("rgrid.csv", [(f"{ri:.10f}",) for ri in r], ["r"])
    w("lambdaA.csv", [(f"{v:.12e}",) for v in lamA], ["lambda"])
    w("lambdaB.csv", [(f"{v:.12e}",) for v in lamB], ["lambda"])

    proc = subprocess.run(["Rscript", R_SCRIPT, d], capture_output=True, text=True)
    log(proc.stdout.rstrip())
    if proc.returncode != 0:
        log(f"  R FAILED (rc={proc.returncode}): {proc.stderr.strip()[:500]}")
        shutil.rmtree(d, ignore_errors=True)
        return None

    def rdcol(fn, col):
        import csv as _csv
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return np.array([float(row[col]) for row in _csv.DictReader(f)])

    r_lamA = rdcol("r_lambdaA.csv", "lambda")
    r_lamB = rdcol("r_lambdaB.csv", "lambda")
    res = {"name": name, "n_a": len(A), "n_b": len(B), "h": h}

    # ── Stage A — intensity ──
    floorA = 0.02 * (len(A) / win_area); floorB = 0.02 * (len(B) / win_area)
    if r_lamA is not None:
        unf = lamA > floorA * 1.0000001
        relA = np.abs(lamA - r_lamA) / np.maximum(np.abs(r_lamA), 1e-30)
        res["A_max_rel_all"] = float(relA.max())
        res["A_max_rel_unfloored"] = float(relA[unf].max()) if unf.any() else 0.0
        res["A_frac_floored"] = float((~unf).mean())
    if r_lamB is not None:
        unf = lamB > floorB * 1.0000001
        relB = np.abs(lamB - r_lamB) / np.maximum(np.abs(r_lamB), 1e-30)
        res["B_max_rel_unfloored"] = float(relB[unf].max()) if unf.any() else 0.0

    # ── Stage B — estimator (K/L), correction="none" is ours ──
    L_py = np.sqrt(np.clip(k_py, 0, None) / np.pi) - r
    stageB = {}
    for corr in ("none", "border", "translate", "isotropic"):
        kr = rdcol(f"r_K_{corr}.csv", "K")
        if kr is None:
            continue
        valid = (k_py > 0) & np.isfinite(kr) & (kr > 0)
        if not valid.any():
            continue
        relK = np.abs(k_py[valid] - kr[valid]) / np.maximum(np.abs(kr[valid]), 1e-30)
        L_r = np.sqrt(np.clip(kr, 0, None) / np.pi) - r
        relL = np.abs(L_py[valid] - L_r[valid]) / np.maximum(np.abs(L_r[valid]) + r[valid], 1e-9)
        ratio = float(np.median(k_py[valid] / kr[valid]))
        stageB[corr] = {"K_max_rel": float(relK.max()), "K_mean_rel": float(relK.mean()),
                        "L_max_rel": float(relL.max()), "median_K_ratio": ratio}
    res["stageB"] = stageB
    shutil.rmtree(d, ignore_errors=True)
    return res


def report_case(res):
    log(f"\n{'─'*72}\nCASE: {res['name']}  (n_a={res['n_a']} n_b={res['n_b']} h={res['h']:.2f}px)")
    if "A_max_rel_unfloored" in res:
        log(f"  Stage A intensity λ̂: max rel diff (unfloored) "
            f"A={res['A_max_rel_unfloored']:.2e}  B={res.get('B_max_rel_unfloored', float('nan')):.2e}"
            f"   (floored {100*res.get('A_frac_floored',0):.0f}% of A, all-points rel {res.get('A_max_rel_all',float('nan')):.2e})")
    sb = res.get("stageB", {})
    if "none" in sb:
        b = sb["none"]
        log(f"  Stage B estimator (correction=none = OURS): "
            f"K max rel {b['K_max_rel']:.2e}  mean {b['K_mean_rel']:.2e}  "
            f"L max rel {b['L_max_rel']:.2e}  median K ratio {b['median_K_ratio']:.6f}")
    for corr in ("border", "translate", "isotropic"):
        if corr in sb:
            log(f"     [{corr:9s}] K max rel {sb[corr]['K_max_rel']:.2e} "
                f"(expected to differ — edge-corrected, not ours)")


def main():
    if shutil.which("Rscript") is None:
        log("BLOCKER: Rscript not on PATH — install R (brew install r) + "
            "spatstat.explore. Cannot cross-validate.")
        return 3
    # spatstat availability probe
    probe = subprocess.run(["Rscript", "-e",
                            'if(!suppressMessages(require(spatstat.explore))) quit(status=3)'],
                           capture_output=True, text=True)
    if probe.returncode == 3:
        log("BLOCKER: spatstat.explore not installed in R. Run: "
            'Rscript -e \'install.packages("spatstat.explore")\'')
        return 3

    log("=" * 72)
    log("CROSS-VALIDATION vs spatstat — intensity-reweighted inhomogeneous cross-K")
    log("=" * 72)
    log("  Inputs identical (pixels); Stage B feeds spatstat the SAME Python λ̂.")
    log("  GATE = Stage B correction='none' L(r) max rel diff < 1e-3.")

    cases = synthetic_cases()
    rc = real_case()
    if rc:
        cases.update(rc)
    else:
        log("  (real Schürch case skipped — dataset CSV not present)")

    results = []
    for name, c in cases.items():
        res = run_case(name, c)
        if res:
            results.append(res); report_case(res)

    # ── Verdict (Stage B gate on correction=none) ──
    log("\n" + "=" * 72); log("VERDICT (Stage B estimator gate)"); log("=" * 72)
    gate, worst = True, 0.0
    for res in results:
        b = res.get("stageB", {}).get("none")
        if not b:
            log(f"  {res['name']}: no 'none' K from spatstat — cannot gate"); gate = False; continue
        passed = b["L_max_rel"] < 1e-3
        worst = max(worst, b["L_max_rel"])
        log(f"  {res['name']:26s}: L max rel {b['L_max_rel']:.2e}  "
            f"K ratio {b['median_K_ratio']:.6f}  -> {'MATCH' if passed else 'DIVERGE'}")
        gate = gate and passed
    log("")
    if gate and results:
        log("  PASS — our reweighted cross-K estimator matches spatstat within "
            "tolerance on\n  synthetic AND real inputs. Recommendation: KEEP the "
            "Python core; cite spatstat\n  as cross-validation in the paper.")
    else:
        log("  DIVERGENCE — see per-case numbers. Diagnose layer (window/intensity/"
            "edge/r-grid).\n  If a constant K ratio != 1.0, it is a normalization "
            "definition difference, not a\n  pairing bug. Only an unreconcilable "
            "estimator diff justifies switching to spatstat.")
    with open(OUT, "w") as f:
        f.write("\n".join(_log) + "\n")
    log(f"\n  (saved to {os.path.relpath(OUT)})")
    return 0 if (gate and results) else 2


if __name__ == "__main__":
    sys.exit(main())
