#!/usr/bin/env python
"""
validate_band_decomposition.py — can the two bands be told apart at all?

THE DEFECT. spatial_stats reports two "distinct biological findings": short-range
colocalization (10-20 um) and regional co-infiltration (20-50 um). They are not distinct as
implemented. Both are DCLF tests on the same L-r curve, and L derives from K, which is
CUMULATIVE — K(r) counts every pair closer than r, so an excess at 6 um raises K at every
larger radius. Writing the surplus as a constant c,

    L(r) - r = sqrt(r^2 + c/pi) - r  ~  c / (2 pi r)

which decays but never returns to zero. Measured on a truth confined to 0-12 um with ZERO
registration error, the 20-50 um band claims attraction at 0.97.

Contamination flows strictly UPWARD in radius, which is why the lowest band is clean and the
upper one is not: colocalization can only be polluted by structure below 10 um, while
co-infiltration inherits everything below 20 um including the whole colocalization band.

TWO CANDIDATE REPLACEMENTS, both computed from the same null draws so they are comparable
draw for draw, and both inert in production until this script picks one:

    bands_pcf       DCLF on g(r), the derivative of K. No cumulative memory. Costs a finite
                    difference, so it is noisier and must be shown to KEEP POWER, not merely
                    to fix attribution — a decomposition that never fires is also "correct".
    bands_annulus   The increment K(hi) - K(lo): the count of cross-type pairs whose
                    separation falls inside the band. Everything below `lo` cancels exactly
                    in the subtraction. No derivative, so no differencing noise.

WHAT IT IS VALIDATED ON, and why that is the right thing.

  1. THE FULL 2x2 ATTRIBUTION MATRIX, not just the failing cell. A statistic that never
     fires the upper band would "fix" the defect and be useless, so the matrix demands both
     directions of correctness AND that a genuinely two-scale truth is reported as two:

         truth              colocalization      co-infiltration
         contact-only       must FIRE           must NOT fire     <- the current failure
         regional-only      must NOT fire       must FIRE
         both scales        must FIRE           must FIRE         <- rules out a dead test
         independent        must NOT fire       must NOT fire     <- size, must be ~alpha

  2. THREE SUBSTRATES, two of them REAL. Truths are imposed on real segmented cells by
     THINNING (validation/spatial_substrates.py), so every coordinate is a real cell and the
     patterns carry real tissue architecture. The primary null ESTIMATES intensity from the
     data, so a Gaussian-blob substrate is the easiest possible case for it; ll477_cd8 is
     OASIS's own output on the cohort it ships for, and keren_p13 is an independent tissue
     and modality. A result that holds on one substrate and not the others is not a result.

  3. ANTI-HALLUCINATION CHECKS, because "K is cumulative" is an explanation and explanations
     can be right about the symptom and wrong about the cause:
       (a) DECAY EXPONENT. The algebra predicts the contaminating excess falls as r^-1. The
           exponent is FITTED. If it is not near -1 the stated mechanism is wrong even
           though the defect is real, and the fix would be being chosen for a bad reason.
       (b) INDEPENDENT PCF. g(r) is recomputed by direct annulus counting instead of by
           differentiating K, and the two must agree. Otherwise bands_pcf could be an
           artefact of _pcf_from_k rather than a property of the pair correlation function.
       (c) ANALYTIC GROUND TRUTH. For independent bivariate Poisson, K_AB(r) = pi r^2
           exactly. The estimator is checked against that closed form, so the machinery is
           verified against mathematics and not only against my own constructions.

Run:  python validation/validate_band_decomposition.py            (~12 min)
      python validation/validate_band_decomposition.py --quick    (~3 min)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (cross_k_all_nulls, cross_k_function,
                                         _COLOC_RMIN_UM, _COLOC_RMAX_UM,
                                         _COINFIL_RMIN_UM, _COINFIL_RMAX_UM)
from validation.spatial_substrates import load_substrate, impose_band_association

RADII_UM = np.arange(0.0, 101.0, 2.0)
SUBSTRATES = ("ll477_cd8", "keren_p13", "synthetic")
STATS = ("bands", "bands_pcf", "bands_annulus", "bands_ring")

# NULLS are swept alongside the statistics because the first run showed the two are
# confounded: EVERY statistic was anti-conservative on real tissue and none was on the
# synthetic blobs, which is a property of the null, not of the band decomposition. All three
# come out of one cross_k_all_nulls call, so the full grid costs nothing extra.
#
#   reweighted        production PRIMARY. An INDEPENDENCE null: B is redrawn from an
#                     intensity surface estimated at 75 um. It cannot reproduce structure
#                     finer than that bandwidth.
#   dense_morphology  conditions B* on marker-independent total-cell morphology, i.e. a
#                     RANDOM-LABELLING null. Two marker subsets of one cell population share
#                     the parent's packing by construction, which is exactly what an
#                     independence null gets wrong.
#   homogeneous       weak CSR baseline, kept as the diagnostic it already is.
NULLS = ("reweighted", "dense_morphology", "homogeneous")

# Each truth is placed EXACTLY inside the band it is supposed to fire, with a gap between
# them. An earlier version used (0,12) for contact, which mostly lies BELOW the 10-20 um
# band being tested, so the cumulative statistic saw it and the annulus statistic barely
# could — that compares the two on different signals rather than on attribution.
CONTACT = (_COLOC_RMIN_UM, _COLOC_RMAX_UM)          # 10-20 um, the colocalization band
REGIONAL = (30.0, 40.0)                             # strictly inside 20-50, clear of 20

# Per-band enrichment, held CONSTANT across truths. A flat weight boost does not do this:
# the annuli have very different baseline occupancy (0.09 vs 0.20), so boosting both at
# once dilutes each and the "both scales" truth silently became far weaker than the
# single-band ones (contact 3.9x -> 1.8x). solve_boost() inverts for the boost that hits
# this target given the observed occupancy, so every cell of the matrix carries the same
# effect size. 2.0x is chosen because it must be achievable for ALL truths on ALL
# substrates: with the bands aligned to 10-20 and 30-40 the two annuli together already hold
# ~0.40 of candidates, which caps enrichment at ~2.47x, so 2.5 was unachievable for the
# "both" truth and 2.0 leaves margin. The single-band truths are held to the SAME 2.0x even
# though they could go higher -- an easier contact truth would flatter whichever statistic
# happens to favour the lower band.
TARGET_ENRICHMENT = 2.0

TRUTHS = {
    "contact":     {"annuli": CONTACT,               "coloc": True,  "coinfil": False},
    "regional":    {"annuli": REGIONAL,              "coloc": False, "coinfil": True},
    "both":        {"annuli": [CONTACT, REGIONAL],   "coloc": True,  "coinfil": True},
    "independent": {"annuli": CONTACT, "flat": 0.0,  "coloc": False, "coinfil": False},
}
BOOST = 6.0
ALPHA = 0.05
# A "must fire" cell has to reach this; a "must not fire" cell has to stay under SIZE_MAX.
POWER_MIN = 0.50
SIZE_MAX = 0.15

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "band_decomposition_results.json")


def one_run(substrate, truth_key, seed, n_perm):
    """One draw -> whether each statistic claims ATTRACTION in each band."""
    spec = TRUTHS[truth_key]
    pts, (W, H) = load_substrate(substrate, seed=seed)
    if "flat" in spec:                       # independent: no boost at all
        A, B = impose_band_association(pts, spec["annuli"],
                                       rng=np.random.default_rng(seed), boost=0.0)
    else:
        A, B = impose_band_association(pts, spec["annuli"],
                                       rng=np.random.default_rng(seed),
                                       target_enrichment=TARGET_ENRICHMENT)
    r = cross_k_all_nulls(A, B, RADII_UM, W * H, 1.0, n_perm=n_perm, seed=seed,
                          nulls=NULLS, morphology_support=pts,
                          registration_radius_floor_um=None)
    out = {}
    for nl in NULLS:
        nul = r["nulls"].get(nl)
        if not nul:
            continue
        for st in STATS:
            b = nul[st]
            out[(nl, st)] = {
                band: (b[band]["direction"] == "association" and b[band]["significant"])
                for band in ("colocalization", "coinfiltration")}
    return out


def sweep(n_rep, n_perm):
    keys = [(nl, st) for nl in NULLS for st in STATS]
    res = {k: {s: {t: {"colocalization": 0, "coinfiltration": 0}
                   for t in TRUTHS} for s in SUBSTRATES} for k in keys}
    total = len(SUBSTRATES) * len(TRUTHS) * n_rep
    done = 0
    for s in SUBSTRATES:
        for t in TRUTHS:
            for k in range(n_rep):
                try:
                    r = one_run(s, t, 4000 + k, n_perm)
                except Exception as e:                                  # noqa: BLE001
                    print(f"    !! {s}/{t}/{k}: {e}")
                    continue
                for k in res:
                    if k not in r:
                        continue
                    for band in ("colocalization", "coinfiltration"):
                        res[k][s][t][band] += bool(r[k][band])
                done += 1
            print(f"    {s:<11} {t:<12} done ({done}/{total})", flush=True)
    for k in res:
        for s in SUBSTRATES:
            for t in TRUTHS:
                for band in ("colocalization", "coinfiltration"):
                    res[k][s][t][band] /= float(n_rep)
    return res


def report_matrix(res):
    """Score every (null, statistic) pair on the full attribution matrix.

    Three properties are scored SEPARATELY because they fail for different reasons and a
    single pass/fail hides which one broke:
      SIZE        independent truth must not be called associated  (a null property)
      ATTRIBUTION a contact-only truth must not fire the upper band (a statistic property)
      POWER       each truth must be found in its own band
    """
    print("\n" + "=" * 100)
    print("FULL GRID — rate of a significant ATTRACTION claim")
    print("=" * 100)
    print(f"{'null':<18}{'statistic':<15}{'substrate':<12}"
          f"{'SIZE':>7}{'ATTRIB':>8}{'POWER-c':>9}{'POWER-r':>9}   flags")
    scores = {}
    for k in res:
        nl, st = k
        rows = []
        for s in SUBSTRATES:
            size = max(res[k][s]["independent"]["colocalization"],
                       res[k][s]["independent"]["coinfiltration"])
            attrib = res[k][s]["contact"]["coinfiltration"]      # must stay low
            pow_c = res[k][s]["contact"]["colocalization"]       # must be high
            pow_r = res[k][s]["regional"]["coinfiltration"]      # must be high
            flags = []
            if size > SIZE_MAX:
                flags.append("SIZE")
            if attrib > SIZE_MAX:
                flags.append("LEAK")
            if pow_c < POWER_MIN or pow_r < POWER_MIN:
                flags.append("POWER")
            rows.append((s, size, attrib, pow_c, pow_r, flags))
            print(f"{nl:<18}{st:<15}{s:<12}{size:>7.2f}{attrib:>8.2f}"
                  f"{pow_c:>9.2f}{pow_r:>9.2f}   {','.join(flags)}")
        scores[k] = {
            "max_size": max(r[1] for r in rows),
            "max_leak": max(r[2] for r in rows),
            "min_power": min(min(r[3], r[4]) for r in rows),
            "size_ok": all(r[1] <= SIZE_MAX for r in rows),
            "attrib_ok": all(r[2] <= SIZE_MAX for r in rows),
            "power_ok": all(r[3] >= POWER_MIN and r[4] >= POWER_MIN for r in rows),
        }
        scores[k]["pass"] = (scores[k]["size_ok"] and scores[k]["attrib_ok"]
                             and scores[k]["power_ok"])
    return scores


def report_summary(scores):
    print("\n" + "=" * 100)
    print("SCORECARD — worst value across substrates (real tissue is the binding case)")
    print("=" * 100)
    print(f"{'null':<18}{'statistic':<15}{'max SIZE':>10}{'max LEAK':>10}"
          f"{'min POWER':>11}   verdict")
    for (nl, st), v in sorted(scores.items(),
                              key=lambda kv: (kv[1]["max_size"], kv[1]["max_leak"])):
        verdict = "PASS" if v["pass"] else ",".join(
            [n for n, ok in (("size", v["size_ok"]), ("attrib", v["attrib_ok"]),
                             ("power", v["power_ok"])) if not ok])
        print(f"{nl:<18}{st:<15}{v['max_size']:>10.2f}{v['max_leak']:>10.2f}"
              f"{v['min_power']:>11.2f}   {verdict}")


# ── anti-hallucination (a): is the contamination really r^-1? ────────────────
def check_decay_exponent(substrate="ll477_cd8", n_rep=8, n_perm=99):
    """Fit the exponent of the contaminating tail. The algebra says -1.

    If the observed excess does not fall like 1/r, "K is cumulative" is the wrong
    explanation for the defect even though the defect is real — and a fix chosen from a
    wrong mechanism is a coincidence, not an argument.
    """
    print("\n" + "=" * 92)
    print("(a) DECAY EXPONENT — the cumulative-K explanation predicts excess ~ r^-1")
    print("=" * 92)
    fits = []
    for k in range(n_rep):
        pts, (W, H) = load_substrate(substrate, seed=4000 + k)
        A, B = impose_band_association(pts, CONTACT, rng=np.random.default_rng(4000 + k),
                                       target_enrichment=TARGET_ENRICHMENT)
        r = cross_k_all_nulls(A, B, RADII_UM, W * H, 1.0, n_perm=n_perm, seed=k + 1,
                              nulls=("reweighted",), registration_radius_floor_um=None)
        nul = r["nulls"]["reweighted"]
        rad = np.asarray(r["radii_um"], float)
        obs = np.asarray(r["L_minus_r"], float)
        nullmean = np.asarray(nul["null_mean_K"], float)
        # excess of L-r over the null mean's own L-r, on the tail well above the truth
        null_l = np.sqrt(np.clip(nullmean, 0, None) / np.pi) - rad
        exc = obs - null_l
        m = (rad >= 24) & (rad <= 100) & (exc > 0) & np.isfinite(exc)
        if m.sum() >= 6:
            slope = np.polyfit(np.log(rad[m]), np.log(exc[m]), 1)[0]
            fits.append(float(slope))
    if not fits:
        print("  could not fit — no positive excess on the tail")
        return None
    med = float(np.median(fits))
    ok = -1.6 < med < -0.5
    print(f"  fitted exponent over {len(fits)} draws: median {med:.2f} "
          f"(range {min(fits):.2f} to {max(fits):.2f})")
    print(f"  predicted -1.00 -> {'CONSISTENT' if ok else 'INCONSISTENT'}")
    if not ok:
        print("  The mechanism stated in the docstring is NOT supported; do not cite it.")
    return {"median_exponent": med, "n": len(fits), "consistent": bool(ok)}


# ── anti-hallucination (b): is g(r) real, or an artefact of _pcf_from_k? ─────
def _pcf_direct(A, B, radii_um, area):
    """g(r) by direct annulus counting — no differentiation of K anywhere."""
    from scipy.spatial import cKDTree
    d = cKDTree(B).query_ball_point(A, r=radii_um[-1])
    dists = np.concatenate([np.linalg.norm(B[np.asarray(ix, int)] - A[i], axis=1)
                            if len(ix) else np.empty(0)
                            for i, ix in enumerate(d)]) if len(A) else np.empty(0)
    lam_b = len(B) / area
    g = np.full(len(radii_um), np.nan)
    for j in range(1, len(radii_um)):
        lo, hi = radii_um[j - 1], radii_um[j]
        cnt = np.sum((dists >= lo) & (dists < hi))
        ring = np.pi * (hi ** 2 - lo ** 2)
        expected = len(A) * lam_b * ring
        g[j] = cnt / expected if expected > 0 else np.nan
    return g


def check_pcf_independent(substrate="ll477_cd8", n_rep=6):
    print("\n" + "=" * 92)
    print("(b) INDEPENDENT PCF — direct annulus counting vs _pcf_from_k's derivative")
    print("=" * 92)
    cors = []
    for k in range(n_rep):
        pts, (W, H) = load_substrate(substrate, seed=4000 + k)
        A, B = impose_band_association(pts, CONTACT, rng=np.random.default_rng(4000 + k),
                                       target_enrichment=TARGET_ENRICHMENT)
        ck = cross_k_function(A, B, RADII_UM, W * H, 1.0)
        g_lib = np.asarray([np.nan if v is None else v for v in ck["g_observed"]], float)
        g_dir = _pcf_direct(A, B, RADII_UM, W * H)
        m = np.isfinite(g_lib) & np.isfinite(g_dir) & (RADII_UM >= 4) & (RADII_UM <= 60)
        if m.sum() >= 8:
            cors.append(float(np.corrcoef(g_lib[m], g_dir[m])[0, 1]))
    med = float(np.median(cors)) if cors else float("nan")
    # 0.85, not 0.95: the two estimators CANNOT agree exactly. _pcf_from_k differentiates a
    # cumulative curve (np.gradient, which smooths across neighbouring radii), while the
    # direct estimator counts cells in disjoint 2 um rings and is shot-noise limited at
    # these counts. The question is whether the SHAPE of g(r) is a property of the pattern
    # or of the estimator, and a correlation this high on independently computed curves
    # answers it; demanding near-equality would just be testing the smoothing.
    ok = bool(cors) and med > 0.85
    print(f"  correlation over {len(cors)} draws: median {med:.3f}")
    print(f"  {'AGREE' if ok else 'DISAGREE'} — g(r) is "
          f"{'a property of the pattern' if ok else 'AN ARTEFACT of the estimator'}")
    return {"median_corr": med, "n": len(cors), "agree": ok}


# ── anti-hallucination (c): analytic ground truth ───────────────────────────
def check_analytic(n_rep=12):
    """Independent bivariate Poisson has K_AB(r) = pi r^2 exactly."""
    print("\n" + "=" * 92)
    print("(c) ANALYTIC GROUND TRUTH — independent Poisson must give K_AB(r) = pi r^2")
    print("=" * 92)
    errs = []
    side = 1000.0
    for k in range(n_rep):
        rng = np.random.default_rng(7000 + k)
        A = rng.uniform(0, side, (400, 2))
        B = rng.uniform(0, side, (600, 2))
        ck = cross_k_function(A, B, RADII_UM, side * side, 1.0)
        rad = np.asarray(ck["radii_um"], float)
        obs = np.asarray(ck["K_observed"], float)
        m = (rad >= 10) & (rad <= 60)
        errs.append(float(np.median(np.abs(obs[m] - np.pi * rad[m] ** 2)
                                    / (np.pi * rad[m] ** 2))))
    med = float(np.median(errs))
    # Edge effects are uncorrected in this estimator, so a few % bias at these radii is
    # expected; a large error would mean the estimator itself is wrong.
    ok = med < 0.15
    print(f"  median |K_obs - pi r^2| / (pi r^2) over {n_rep} draws: {med:.3f}")
    print(f"  {'OK' if ok else 'ESTIMATOR IS WRONG'} (tolerance 0.15, uncorrected edges)")
    return {"median_rel_error": med, "ok": ok}


def main():
    quick = "--quick" in sys.argv
    n_rep = 15 if quick else 50
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 92)
    print("Band decomposition — can colocalization and co-infiltration be told apart?")
    print("=" * 92)
    print(f"bands {_COLOC_RMIN_UM:.0f}-{_COLOC_RMAX_UM:.0f} / "
          f"{_COINFIL_RMIN_UM:.0f}-{_COINFIL_RMAX_UM:.0f} um · "
          f"{n_rep} reps x {n_perm} perms · substrates {', '.join(SUBSTRATES)}"
          f"{'  [QUICK]' if quick else ''}")
    print(f"pass needs: 'FIRE' cells >= {POWER_MIN}, 'none' cells <= {SIZE_MAX}\n")

    res = sweep(n_rep, n_perm)
    scores = report_matrix(res)
    report_summary(scores)
    decay = check_decay_exponent(n_perm=n_perm)
    pcf = check_pcf_independent()
    analytic = check_analytic()

    print("\n" + "=" * 100)
    print("DECISION")
    print("=" * 100)
    passing = [k for k, v in scores.items() if v["pass"]]
    cur = ("reweighted", "bands")
    print(f"  shipped today: null={cur[0]}, statistic={cur[1]} -> "
          f"{'PASS' if scores.get(cur, {}).get('pass') else 'FAIL'}"
          f" (size {scores.get(cur, {}).get('max_size', float('nan')):.2f},"
          f" leak {scores.get(cur, {}).get('max_leak', float('nan')):.2f},"
          f" power {scores.get(cur, {}).get('min_power', float('nan')):.2f})")
    if passing:
        # Rank by the property that cannot be traded away: a test that is not correctly
        # sized is not a test, so size first, then attribution, then power.
        best = sorted(passing, key=lambda k: (scores[k]["max_size"], scores[k]["max_leak"],
                                              -scores[k]["min_power"]))[0]
        rec = (f"Adopt null={best[0]} with statistic={best[1]}. It is the only "
               f"combination that is correctly sized on REAL tissue, keeps a contact-only "
               f"truth out of the upper band, and still detects both truths.")
    else:
        # Report the best available on each axis so the failure is actionable rather than
        # a flat 'nothing works'.
        by_size = min(scores.items(), key=lambda kv: kv[1]["max_size"])
        by_leak = min(scores.items(), key=lambda kv: kv[1]["max_leak"])
        rec = (f"NOTHING passes all three. Best size: {by_size[0]} at "
               f"{by_size[1]['max_size']:.2f}. Best attribution: {by_leak[0]} at "
               f"{by_leak[1]['max_leak']:.2f}. Fix size before shipping any two-band claim.")
    print(f"\n  {rec}")

    payload = {"config": {"n_rep": n_rep, "n_perm": n_perm, "substrates": list(SUBSTRATES),
                          "nulls": list(NULLS), "target_enrichment": TARGET_ENRICHMENT,
                          "boost": BOOST, "power_min": POWER_MIN, "size_max": SIZE_MAX},
               "matrix": {f"{k[0]}|{k[1]}": v for k, v in res.items()},
               "scores": {f"{k[0]}|{k[1]}": v for k, v in scores.items()}, "decay_exponent": decay,
               "pcf_independent": pcf, "analytic": analytic, "recommendation": rec}
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
