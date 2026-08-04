#!/usr/bin/env python
"""
validate_residual_origin.py — what IS the 4.07 um, now that it is not the matcher?

BACKGROUND. research/registration.md s 11 (Phase A) measured LoFTR's localisation error
against a known warp and got 0.22 um. The shipped estimator said 0.199 um; it was right. So
the 4.073 um sigma_fit on LL477_CD8_x10_1 <-> Tim3_x10_1 is real, and the question becomes
what produces it. Three hypotheses were recorded; this script tests all three.

WHY IT MATTERS, precisely. cell_error_budget = sqrt(TRE_pred^2 + deformation^2), and
deformation is whatever sigma_fit^2 - 2*FLE^2 leaves over. The machinery is CORRECT if the
FLE it is given is the true random localisation error of a correspondence. The estimator only
ever measures the MATCHER's share, because it perturbs the image. Two serial sections are
~4 um apart in the block, so the "same" vessel wall is a different cross-section and the true
partner of a point is genuinely undefined by roughly a nuclear diameter. That ambiguity is
random fiducial localisation error in the Fitzpatrick-West sense and belongs in FLE -- but no
image perturbation can produce it.

    residuals spatially CORRELATED -> a real displacement field -> belongs in `deformation`
                                      -> the gate is right to refuse this pair.
    residuals spatially WHITE      -> random per-correspondence error -> belongs in `FLE`
                                      -> the gate double-counts: a similarity fitted from 59
                                         unbiased noisy points averages the noise away
                                         (tre_pred_p90 was 0.091 um), and charging cells the
                                         full per-match scatter is wrong.

TWO EXPERIMENTS.

  1. tol_um SWEEP. All three correspondence filters admit disagreement up to `tol_um`, whose
     default is 4.0 um -- and the observed median residual is 4.14 um. That is what a
     TRUNCATED distribution looks like: the filters cap the pairwise displacement difference
     between neighbours, so a population whose true scatter exceeded tol_um would be culled
     back to roughly tol_um and its survivors' median would land there whatever the tissue
     does. If the median tracks tol_um, the 4.14 um is an artefact of what we admit.
     Read `n` alongside it: tightening tol_um also SELECTS, and a residual that falls because
     the population shrank is not the same finding.

  2. VARIOGRAM + MORAN'S I on the residual vectors. The semivariogram of a 2-D residual field
     separates the two terms directly:

         gamma(h) = 0.5 * mean over pairs at lag h of ||e_i - e_j||^2
         gamma(0) = NUGGET  = the spatially random part   -> this is 2*FLE_combined^2
         sill - nugget      = the spatially structured part -> this is deformation

     Pure white noise has gamma flat at the sill (nugget/sill = 1). A smooth deformation field
     has gamma -> 0 as h -> 0 (nugget/sill -> 0). Moran's I on the same points, with a
     permutation null, gives the significance.

     CAVEAT, stated because it biases one way: residuals are measured AFTER a similarity fit,
     so the fit has already absorbed any global linear component. That makes this test
     CONSERVATIVE for detecting deformation at long lags, and unaffected at short lags -- and
     the nugget is a short-lag quantity.

WHAT MAKES THIS SCRIPT FAIL. Only the reproduction check: it must recover the recorded
n = 59, fit_residual 4.144 um, landmark_noise 4.073 um on the disputed ROI. If it does not,
it is not looking at the thing under dispute and nothing below means anything. The two
experiments are measurements and are reported, never asserted.

Run:
    python validation/validate_residual_origin.py
    python validation/validate_residual_origin.py --ref A.tif --mov B.tif --pixel-size 0.7519

Writes validation/residual_origin_results.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REF = os.path.expanduser("~/Desktop/cd8_input/LL477_CD8_x10_1.tif")
DEFAULT_MOV = os.path.expanduser("~/Desktop/tim3 input/LL477_Tim3_x10_1.tif")
DEFAULT_PX = 0.7519

# research/registration.md s 9 — the ROI the whole dispute is about.
ROI_HALF = 0.45
WORK_MAX_DIM = 800
SHIPPED_TOL_UM = 4.0

# What the disputed run recorded. Reproduction is checked against these.
EXPECTED = {"n": 59, "fit_residual_um": 4.144, "landmark_noise_um": 4.073}

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "residual_origin_results.json")

_RAYLEIGH_MEDIAN = np.sqrt(2.0 * np.log(2.0))
_FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)
    return bool(cond)


# ---------------------------------------------------------------------------------------
def load_pair(ref_path, mov_path, px):
    """Exactly what api.certify_local_roi_multi does: 1920 px thumbnails, one provisional."""
    from oasis.common.registration import _load_rgb_thumbnail
    from oasis.spatial import serial_registration as sr

    ref_rgb, ref_scale = _load_rgb_thumbnail(ref_path, max_side=1920)
    mov_rgb, _ = _load_rgb_thumbnail(mov_path, max_side=1920)
    if ref_rgb is None or mov_rgb is None:
        raise SystemExit(f"could not load {ref_path} or {mov_path}")
    px_t = px / max(ref_scale, 1e-9)
    M_t = np.asarray(sr.register_similarity(ref_rgb, mov_rgb, px_t)["matrix"], float)
    H, W = ref_rgb.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    hw, hh = ROI_HALF * W, ROI_HALF * H
    roi = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                    [cx + hw, cy + hh], [cx - hw, cy + hh]], float)
    return ref_rgb, mov_rgb, px_t, M_t, roi, (W, H)


def certify(ref_rgb, mov_rgb, roi, px_t, M_t, tol_um, fle_fast=True):
    from oasis.spatial import loftr_matcher as lm
    return lm.certify_local_roi(ref_rgb, mov_rgb, roi, px_t, provisional_matrix=M_t,
                                tol_um=tol_um, work_max_dim=WORK_MAX_DIM,
                                return_correspondences=True, fle_fast=fle_fast)


def residuals_um(cert, px_t):
    """Per-correspondence residual VECTORS in um, at the reference points.

    local_matrix is moving -> reference (the codebase convention; the other direction gives
    ~148 um of nonsense, which is how this was got wrong once already).
    """
    from oasis.spatial.serial_registration import _apply_affine
    ref = np.asarray(cert["corr_ref"], float)
    mov = np.asarray(cert["corr_mov"], float)
    M = np.asarray(cert["local_matrix"], float)
    e = (_apply_affine(mov, M) - ref) * float(px_t)      # um, at ref positions
    return ref, e


# ---------------------------------------------------------------------------------------
# Experiment 2: spatial structure
# ---------------------------------------------------------------------------------------
def variogram(pos_um, e, n_bins=8, min_pairs=15):
    """Empirical semivariogram of a 2-D vector field.

    gamma(h) = 0.5 * mean ||e_i - e_j||^2 over pairs whose separation falls in bin h.
    For white noise with per-axis variance s^2 this is flat at 2*s^2 (the sill); for a smooth
    field it falls towards 0 at short lag. Returns bins plus a nugget extrapolated to h=0
    from the short-lag bins.
    """
    n = len(e)
    iu = np.triu_indices(n, 1)
    d = np.linalg.norm(pos_um[iu[0]] - pos_um[iu[1]], axis=1)
    g = 0.5 * ((e[iu[0]] - e[iu[1]]) ** 2).sum(axis=1)
    # equal-count bins: an equal-width binning puts almost nothing in the short-lag bins,
    # which is exactly where the answer lives.
    order = np.argsort(d)
    d, g = d[order], g[order]
    per = max(len(d) // n_bins, min_pairs)
    bins = []
    for s in range(0, len(d) - per // 2, per):
        sl = slice(s, min(s + per, len(d)))
        if sl.stop - sl.start < min_pairs:
            break
        bins.append({"h_um": round(float(d[sl].mean()), 2),
                     "gamma": round(float(g[sl].mean()), 4),
                     "n_pairs": int(sl.stop - sl.start)})
    sill = float(0.5 * ((e - e.mean(axis=0)) ** 2).sum(axis=1).mean() * 2.0)
    nugget = None
    if len(bins) >= 3:
        # linear extrapolation of the three shortest-lag bins back to h = 0
        hs = np.array([b["h_um"] for b in bins[:3]])
        gs = np.array([b["gamma"] for b in bins[:3]])
        slope, intercept = np.polyfit(hs, gs, 1)
        nugget = float(max(intercept, 0.0))
    return {"bins": bins, "sill": round(sill, 4),
            "nugget": None if nugget is None else round(nugget, 4),
            "nugget_over_sill": None if not sill else
                                (None if nugget is None else round(nugget / sill, 3))}


def morans_i(pos_um, v, k=6, n_perm=999, seed=0):
    """Moran's I of a scalar on k-nearest-neighbour, row-standardised weights.

    Permutation null rather than the analytic one: n is 59 and the analytic normality of I is
    an asymptotic result. Under no spatial autocorrelation E[I] = -1/(n-1).
    """
    from scipy.spatial import cKDTree
    n = len(v)
    if n < k + 2:
        return None
    _, idx = cKDTree(pos_um).query(pos_um, k=k + 1)
    idx = idx[:, 1:]                                     # drop self
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), idx.ravel()] = 1.0 / k

    def _I(x):
        z = x - x.mean()
        denom = (z ** 2).sum()
        if denom <= 0:
            return 0.0
        return float(n / W.sum() * (z @ W @ z) / denom)

    obs = _I(np.asarray(v, float))
    rng = np.random.default_rng(seed)
    null = np.array([_I(rng.permutation(v)) for _ in range(int(n_perm))])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"I": round(obs, 4), "expected_under_null": round(-1.0 / (n - 1), 4),
            "p_permutation": round(p, 4), "k": k,
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4)}


def spatial_structure(pos_px, e, px_t):
    pos = np.asarray(pos_px, float) * float(px_t)        # um
    out = {"n": int(len(e))}
    out["variogram"] = variogram(pos, e)
    out["morans_i_dx"] = morans_i(pos, e[:, 0])
    out["morans_i_dy"] = morans_i(pos, e[:, 1])
    # The decomposition the gate actually needs, read straight off the variogram.
    vg = out["variogram"]
    if vg["nugget"] is not None and vg["sill"]:
        nug, sill = vg["nugget"], vg["sill"]
        fle_comb = float(np.sqrt(max(nug, 0.0) / 2.0))
        out["decomposition"] = {
            "fle_combined_um": round(fle_comb, 4),
            "fle_um": round(fle_comb / np.sqrt(2.0), 4),
            "deformation_um": round(float(np.sqrt(max(sill - nug, 0.0) / 2.0)), 4),
            "random_frac_of_variance": round(float(min(nug / sill, 1.0)), 3),
        }
    return out


# ---------------------------------------------------------------------------------------
def run(ref_path, mov_path, px, tols, out_json=OUT_JSON):
    print("=" * 78)
    print("What IS the residual? - tol_um sweep + spatial structure")
    print("=" * 78)
    print(f"  ref : {ref_path}")
    print(f"  mov : {mov_path}")
    print(f"  px  : {px} um/px")

    for p in (ref_path, mov_path):
        if not os.path.exists(p):
            print(f"\nERROR: not found: {p}")
            return 2

    ref_rgb, mov_rgb, px_t, M_t, roi, (W, H) = load_pair(ref_path, mov_path, px)
    span = (roi[:, 0].max() - roi[:, 0].min()) * px_t
    print(f"  ROI : {span:.0f} um across, thumbnail {W}x{H}, px_t {px_t:.4f}")

    print("\n0. Reproduce the disputed certification (shipped settings)")
    base = certify(ref_rgb, mov_rgb, roi, px_t, M_t, SHIPPED_TOL_UM, fle_fast=False)
    got = {"n": base.get("n_correspondences"),
           "fit_residual_um": base.get("fit_residual_um"),
           "landmark_noise_um": base.get("landmark_noise_um")}
    print(f"      verdict {base.get('verdict')}  n {got['n']}  "
          f"fit_residual {got['fit_residual_um']}  landmark_noise {got['landmark_noise_um']}  "
          f"FLE {base.get('fle_um_loftr')}")
    ok = (got["n"] == EXPECTED["n"]
          and abs((got["fit_residual_um"] or 0) - EXPECTED["fit_residual_um"]) < 0.05
          and abs((got["landmark_noise_um"] or 0) - EXPECTED["landmark_noise_um"]) < 0.05)
    check("reproduces the recorded n=59 / 4.144 / 4.073", ok,
          f"got n={got['n']} {got['fit_residual_um']} {got['landmark_noise_um']}")

    # ── 1. tol_um sweep ────────────────────────────────────────────────────────────────
    print("\n1. tol_um sweep - does the residual track what the filters admit?")
    print("   The raw LoFTR passes are content-cached and tol is applied after them, so only")
    print("   the first row pays for inference.")
    print(f"   {'tol_um':>7} {'n':>5} {'med um':>7} {'p90 um':>7} {'sigma_fit':>9} "
          f"{'med/tol':>7} {'verdict':>18}")
    sweep = []
    for t in tols:
        c = certify(ref_rgb, mov_rgb, roi, px_t, M_t, t, fle_fast=True)
        row = {"tol_um": t, "verdict": c.get("verdict"),
               "n": c.get("n_correspondences"),
               "fit_residual_um": c.get("fit_residual_um"),
               "landmark_noise_um": c.get("landmark_noise_um"),
               "funnel": c.get("loftr_funnel")}
        if c.get("corr_ref"):
            _, e = residuals_um(c, px_t)
            mag = np.linalg.norm(e, axis=1)
            row.update(res_med_um=round(float(np.median(mag)), 4),
                       res_p90_um=round(float(np.percentile(mag, 90)), 4),
                       res_max_um=round(float(mag.max()), 4),
                       med_over_tol=round(float(np.median(mag)) / t, 3))
            print(f"   {t:>7.1f} {row['n']:>5} {row['res_med_um']:>7.3f} "
                  f"{row['res_p90_um']:>7.3f} {row['landmark_noise_um']:>9.3f} "
                  f"{row['med_over_tol']:>7.3f} {str(row['verdict']):>18}")
        else:
            print(f"   {t:>7.1f} {'--':>5}  {c.get('msg') or c.get('verdict')}")
        sweep.append(row)

    ok_rows = [r for r in sweep if r.get("res_med_um")]
    verdict_sweep = None
    if len(ok_rows) >= 3:
        tt = np.array([r["tol_um"] for r in ok_rows], float)
        mm = np.array([r["res_med_um"] for r in ok_rows], float)
        nn = np.array([r["n"] for r in ok_rows], float)
        r_tol = float(np.corrcoef(tt, mm)[0, 1])
        # A truncated population has median proportional to the tolerance, so med/tol is
        # flat; a tissue-set residual has a median that stops rising, so med/tol falls.
        ratio = mm / tt
        verdict_sweep = {
            "pearson_r_median_vs_tol": round(r_tol, 3),
            "med_over_tol_range": [round(float(ratio.min()), 3), round(float(ratio.max()), 3)],
            "median_um_range": [round(float(mm.min()), 3), round(float(mm.max()), 3)],
            "n_range": [int(nn.min()), int(nn.max())],
            "reading": ("residual TRACKS the filter tolerance - the 4.14 um is set by what "
                        "we admit, not by the tissue"
                        if r_tol > 0.8 and ratio.max() / max(ratio.min(), 1e-9) < 2.0 else
                        "residual does NOT simply track the tolerance - it is a property of "
                        "the pair, not of the filter"),
        }
        print(f"\n   r(median, tol) = {verdict_sweep['pearson_r_median_vs_tol']}, "
              f"median spans {verdict_sweep['median_um_range']} um "
              f"while n spans {verdict_sweep['n_range']}")
        print(f"   -> {verdict_sweep['reading']}")

    # ── 2. spatial structure at the shipped tolerance ──────────────────────────────────
    print("\n2. Spatial structure of the residual field (shipped tol_um = 4.0)")
    struct = None
    if base.get("corr_ref"):
        pos, e = residuals_um(base, px_t)
        struct = spatial_structure(pos, e, px_t)
        vg = struct["variogram"]
        print(f"   {'lag um':>8} {'gamma':>9} {'pairs':>6}")
        for b in vg["bins"]:
            print(f"   {b['h_um']:>8.1f} {b['gamma']:>9.3f} {b['n_pairs']:>6}")
        print(f"   sill {vg['sill']:.3f}   nugget {vg['nugget']}   "
              f"nugget/sill {vg['nugget_over_sill']}")
        for ax in ("dx", "dy"):
            m = struct[f"morans_i_{ax}"]
            if m:
                print(f"   Moran's I ({ax}): {m['I']:+.4f}  "
                      f"(null {m['expected_under_null']:+.4f}, p = {m['p_permutation']})")
        d = struct.get("decomposition")
        if d:
            print(f"\n   Variogram decomposition of sigma_fit:")
            print(f"     random (nugget)     -> FLE {d['fle_um']} um "
                  f"(combined {d['fle_combined_um']})")
            print(f"     structured          -> deformation {d['deformation_um']} um")
            print(f"     random share of variance: {d['random_frac_of_variance']:.0%}")

    _interpret(struct, verdict_sweep, base)

    payload = {"ref": ref_path, "mov": mov_path, "pixel_size_um": px,
               "px_thumbnail_um": px_t, "roi_span_um": round(float(span), 1),
               "reproduction": {"expected": EXPECTED, "got": got,
                                "verdict": base.get("verdict"),
                                "fle_um_loftr": base.get("fle_um_loftr"),
                                "cell_error_p90_um": base.get("cell_error_p90_um"),
                                "tre_pred_p90_um": base.get("tre_pred_p90_um")},
               "tol_sweep": sweep, "tol_sweep_reading": verdict_sweep,
               "spatial_structure": struct, "controls_failed": list(_FAILS)}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {out_json}")
    return 1 if _FAILS else 0


def _interpret(struct, sweep_reading, base):
    print("\n" + "=" * 78)
    print("WHAT THIS SAYS ABOUT THE GATE")
    print("=" * 78)
    if not struct or not struct.get("decomposition"):
        print("  no residual field to read - nothing concluded.")
        return
    d = struct["decomposition"]
    frac = d["random_frac_of_variance"]
    mx = max((struct[k] or {}).get("p_permutation", 1.0) for k in
             ("morans_i_dx", "morans_i_dy"))
    mn = min((struct[k] or {}).get("p_permutation", 1.0) for k in
             ("morans_i_dx", "morans_i_dy"))
    print(f"  random share of the residual variance : {frac:.0%}")
    print(f"  Moran's I permutation p              : {mn} .. {mx}")
    if frac > 0.8 and mn > 0.05:
        print("\n  The residual field is WHITE. It is random per-correspondence error, which")
        print("  is fiducial localisation error in the Fitzpatrick-West sense and belongs in")
        print("  the FLE term, not in `deformation`. The similarity fitted from these points")
        print(f"  averages that noise away (tre_pred_p90 = {base.get('tre_pred_p90_um')} um),")
        print("  so charging a cell the full per-match scatter DOUBLE-COUNTS it.")
        print(f"  Implied FLE for the budget: {d['fle_um']} um, not the matcher's 0.2.")
    elif frac < 0.5 and mn < 0.05:
        print("\n  The residual field is STRUCTURED - a real displacement field. It belongs")
        print("  in `deformation` exactly as the gate assumes, and the gate is RIGHT to")
        print("  refuse this pair at a 5 um cell-error claim. The FLE is not the defect and")
        print("  neither is the attribution; s 4's threshold-statistic question is then the")
        print("  only live one.")
    else:
        print("\n  MIXED: neither purely white nor purely structured. Both terms are real,")
        print("  and the budget needs the variogram split above rather than an FLE measured")
        print("  from image perturbation alone. Read the per-lag table before deciding.")
    if sweep_reading:
        print(f"\n  Filter tolerance: {sweep_reading['reading']}")


def main():
    ap = argparse.ArgumentParser(description="Origin of the certification residual")
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--mov", default=DEFAULT_MOV)
    ap.add_argument("--pixel-size", type=float, default=DEFAULT_PX)
    ap.add_argument("--tols", type=float, nargs="+",
                    default=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    code = run(a.ref, a.mov, a.pixel_size, a.tols, out_json=a.out)
    print("\n" + "=" * 78)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} check(s) FAILED: {', '.join(_FAILS)}")
        print("The reproduction did not match; nothing above is about the disputed ROI.")
    elif code == 0:
        print("RESULT: reproduction PASSED - the measurements above are of the disputed ROI.")
    sys.exit(code)


if __name__ == "__main__":
    main()
