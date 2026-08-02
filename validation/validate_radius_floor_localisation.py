#!/usr/bin/env python
"""
validate_radius_floor_localisation.py — calibrate _RADIUS_FLOOR_FACTOR against band leakage.

WHY THIS EXISTS, AND WHY validate_radius_floor.py DOES NOT ANSWER IT.
    _RADIUS_FLOOR_FACTOR = 3.0 decides the ONE claim the Spatial Association tab exists to
    make. Contact-scale engagement is reported only when the pair's registration floor
    (factor x TRE) sits below the colocalization band's top, so at 3.0 the tab's implicit
    registration specification is TRE <= 20/3 = 6.67 um. Nothing else in the codebase
    states a required registration accuracy: every matcher decision and every certification
    threshold is aimed at a target this constant defines. It was never derived.

    validate_radius_floor.py measured SIZE (registration error cannot invent a finding, out
    to eps = 20 um) and POWER (0.44 -> 0.34). Both are properties of the SINGLE 10-50 um
    DCLF band. Neither touches the question the floor is actually for, which is
    LOCALISATION: when the pipeline splits that band into short-range colocalization
    (10-20 um) and regional co-infiltration (20-50 um), does registration error move a
    finding from one band into the other? A floor is only needed if it does.

    Two further reasons the earlier harness cannot settle this:
      * it displaces B by IID Gaussian noise, but research/registration.md 11.4b measured
        the real residual and found it 98 % spatially structured (nugget/sill 0.021,
        Moran's I +0.60). A smooth field translates a whole neighbourhood of B coherently
        relative to A, which MOVES the radius at which an excess appears. IID noise mostly
        blurs amplitude. The realistic model is the one that can confuse the bands, so
        calibrating on IID would understate the error.
      * it runs nulls=("homogeneous",); production reports on the reweighted primary.

THE MEASUREMENT.
    Two ground truths on a shared-architecture substrate (both markers prefer the same
    tissue compartments, the hard case the reweighted null is built for):

      CONTACT-ONLY   a fraction of B recruited into an annulus 0-12 um around an A cell.
                     Truth lives inside the colocalization band.
      REGIONAL-ONLY  recruited into an annulus 25-45 um. Truth lives strictly inside the
                     co-infiltration band, with NO contact-scale excess by construction.

    Each is displaced by eps um under both models, clipped to the window exactly as
    run_spatial_association does, and pushed through cross_k_all_nulls with the production
    reweighted primary. The floor is passed as None so nothing is suppressed -- this
    measures the raw leakage in order to SET the floor.

    The decisive number is the REGIONAL-ONLY truth's rate of significant *attraction* in
    the colocalization band. That is a false cell-scale-engagement claim: the tab asserting
    contact-scale proximity for a pair whose only real structure is regional. eps = 0 is
    the control -- it must sit at the nominal 0.05, or the truth construction is
    contaminated and nothing downstream means anything.

CALIBRATION RULE.
    Let eps* be the largest displacement at which the false-engagement rate stays within
    tolerance. Beyond eps* the contact-scale claim must be withheld, i.e. the floor must
    exceed the band top:

        factor x eps* >= _COLOC_RMAX_UM      =>      factor = _COLOC_RMAX_UM / eps*

    factor 3.0 is therefore correct iff leakage sets in at about 6.7 um. Larger eps* means
    the shipped floor is too strict and is withholding valid contact-scale findings (and
    the registration spec is looser than the matcher work assumed); smaller eps* means it
    is too permissive and some shipped engagement claims are not supported.

    An eps* that lands on the top of the sweep is CENSORED and yields only an upper bound
    on the factor, never a point estimate.

A SECOND, LARGER FINDING — see band_independence().
    Running the CONTACT-ONLY truth exposed a defect that has nothing to do with
    registration: at eps = 0, with the truth confined to 0-12 um, the CO-INFILTRATION band
    (20-50 um) claims attraction anyway. Both bands are DCLF tests on the same L-r curve,
    and L derives from K, which is cumulative — an excess at 6 um lifts K(r) at every
    larger r, so L-r never returns inside the envelope. g(r), the derivative, does return.
    The two "distinct biological findings" the decomposition promises are therefore not
    independent in the direction that matters for this tab. Diagnosed here; fixing it
    (decomposing on g rather than L-r) needs its own calibration.

Run:  python validation/validate_radius_floor_localisation.py           (~18 min)
      python validation/validate_radius_floor_localisation.py --quick   (~4 min)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (cross_k_all_nulls, _RADIUS_FLOOR_FACTOR,
                                         _COLOC_RMIN_UM, _COLOC_RMAX_UM,
                                         _COINFIL_RMIN_UM, _COINFIL_RMAX_UM)

SIDE_PX = 2000.0
PIXEL_SIZE_UM = 0.5
SIDE_UM = SIDE_PX * PIXEL_SIZE_UM
AREA_PX = SIDE_PX ** 2
RADII_PX = np.arange(2.0, 101.0, 2.0) / PIXEL_SIZE_UM

EPS_UM = (0.0, 3.0, 5.0, 8.0, 12.0, 20.0)

# Tissue architecture shared by BOTH markers. Compartment scale is deliberately set above
# the reweighted null's 75 um bandwidth so the null absorbs it as intended -- the leakage
# under test must not be confounded with an architecture the primary cannot see.
N_COMPARTMENTS = 12
COMPARTMENT_SD_UM = 180.0
DIFFUSE_FRAC = 0.35

N_A, N_B = 300, 500
RECRUIT_FRAC = 0.30
CONTACT_ANNULUS_UM = (0.0, 12.0)
REGIONAL_ANNULUS_UM = (25.0, 45.0)

# Deformation-field correlation length. A serial-section warp varies over hundreds of
# microns, far above the 10-50 um band, so locally it acts as a coherent translation of B
# relative to A -- which is exactly the mechanism that relocates an excess between bands.
CORR_LENGTH_UM = 200.0
NUGGET_FRAC = 0.021          # measured on the real pair (registration.md 11.4b)

# Tolerance on the false-engagement rate before the contact claim must be withheld.
# Nominal alpha is 0.05; 0.10 allows one doubling before the floor has to intervene.
LEAK_TOLERANCE = 0.10

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "radius_floor_localisation_results.json")


# ──────────────────────────────────────────────────────────────────────────────
# Substrate and truths
# ──────────────────────────────────────────────────────────────────────────────

def _compartment_draw(rng, centres, sd_px, n):
    """Points from shared tissue compartments plus a diffuse fraction."""
    n_diffuse = int(DIFFUSE_FRAC * n)
    diffuse = rng.uniform(0, SIDE_PX, (n_diffuse, 2))
    idx = rng.integers(0, len(centres), n - n_diffuse)
    clustered = centres[idx] + rng.normal(0, sd_px, (n - n_diffuse, 2))
    return np.vstack([diffuse, clustered])


def _annulus_offsets(rng, n, lo_um, hi_um):
    """Uniform-by-area offsets in an annulus, so the radial excess is flat in r."""
    theta = rng.uniform(0, 2 * np.pi, n)
    lo, hi = lo_um / PIXEL_SIZE_UM, hi_um / PIXEL_SIZE_UM
    r = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def make_truth(seed, annulus_um):
    """A/B sharing tissue architecture, with B additionally recruited into `annulus_um`
    around A. The annulus, not a Gaussian, is what keeps the truth confined to ONE band."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, SIDE_PX, (N_COMPARTMENTS, 2))
    sd_px = COMPARTMENT_SD_UM / PIXEL_SIZE_UM

    a = _compartment_draw(rng, centres, sd_px, N_A)
    a = a[(a > 0).all(1) & (a < SIDE_PX).all(1)]

    n_rec = int(RECRUIT_FRAC * N_B)
    anchors = a[rng.integers(0, len(a), n_rec)]
    recruited = anchors + _annulus_offsets(rng, n_rec, *annulus_um)
    background = _compartment_draw(rng, centres, sd_px, N_B - n_rec)

    b = np.vstack([recruited, background])
    return a, b[(b > 0).all(1) & (b < SIDE_PX).all(1)]


# ──────────────────────────────────────────────────────────────────────────────
# Displacement models
# ──────────────────────────────────────────────────────────────────────────────

def _displace_iid(points, eps_um, rng):
    """Independent per-point Gaussian error -- the model validate_radius_floor.py used.
    Kept as the CONTRAST, not as the realistic case: it destroys B's internal structure
    rather than translating it, so it blurs amplitude without relocating the excess."""
    if eps_um <= 0:
        return points
    sigma_px = (eps_um / np.sqrt(2.0)) / PIXEL_SIZE_UM     # RMS |displacement| = eps
    return points + rng.normal(0, sigma_px, points.shape)


def _displace_smooth(points, eps_um, rng, corr_um=CORR_LENGTH_UM):
    """Spatially-correlated deformation, matched to the measured residual field.

    A coarse iid vector field smoothed to `corr_um` gives neighbouring B points nearly the
    same displacement, which is what a real warp does; NUGGET_FRAC of the variance is left
    uncorrelated to match the measured nugget/sill of 0.021. Scaled so the RMS displacement
    MAGNITUDE equals eps_um, since that is how TRE is reported (a distance, not a per-axis
    sigma).
    """
    if eps_um <= 0:
        return points
    g = 64                                                  # field grid resolution
    sigma_cells = (corr_um / PIXEL_SIZE_UM) / (SIDE_PX / g)
    field = rng.normal(0, 1, (g, g, 2))
    for c in range(2):
        field[..., c] = gaussian_filter(field[..., c], sigma_cells, mode="wrap")

    # bilinear sample at each point
    uv = np.clip(points / SIDE_PX, 0, 1 - 1e-9) * (g - 1)
    i0, j0 = np.floor(uv[:, 1]).astype(int), np.floor(uv[:, 0]).astype(int)
    fi, fj = uv[:, 1] - i0, uv[:, 0] - j0
    i1, j1 = np.minimum(i0 + 1, g - 1), np.minimum(j0 + 1, g - 1)
    d = (field[i0, j0] * ((1 - fi) * (1 - fj))[:, None] +
         field[i0, j1] * ((1 - fi) * fj)[:, None] +
         field[i1, j0] * (fi * (1 - fj))[:, None] +
         field[i1, j1] * (fi * fj)[:, None])

    d = d + np.sqrt(NUGGET_FRAC / max(1.0 - NUGGET_FRAC, 1e-9)) * \
        rng.normal(0, d.std() if d.std() > 0 else 1.0, d.shape)
    rms = np.sqrt((d ** 2).sum(1).mean())
    if rms > 0:
        d *= (eps_um / PIXEL_SIZE_UM) / rms
    return points + d


MODELS = {"smooth field": _displace_smooth, "iid noise": _displace_iid}


def _clip(points):
    """Points pushed out of the analysis window are dropped, exactly as
    run_spatial_association's filter_points_in_polygon does. Retaining them while holding
    the area fixed corrupts the density bookkeeping (validate_radius_floor.py, scope note)."""
    return points[(points > 0).all(1) & (points < SIDE_PX).all(1)]


# ──────────────────────────────────────────────────────────────────────────────
# One trial
# ──────────────────────────────────────────────────────────────────────────────

def band_verdicts(a, b, seed, n_perm):
    """Production path, floor disabled so raw leakage is visible."""
    r = cross_k_all_nulls(a, b, RADII_PX, AREA_PX, PIXEL_SIZE_UM,
                          n_perm=n_perm, seed=seed,
                          nulls=("reweighted", "homogeneous"),
                          registration_radius_floor_um=None)
    it = r["interaction"]
    return it["colocalization"]["verdict"], it["coinfiltration"]["verdict"]


def sweep(truth_name, annulus_um, n_rep, n_perm, verbose=True):
    """Rate at which each band claims attraction, per displacement model and eps."""
    out = {}
    for model_name, model in MODELS.items():
        row = {}
        for eps in EPS_UM:
            coloc = coinfil = csr_only = 0
            for k in range(n_rep):
                a, b = make_truth(5000 + k, annulus_um)
                rng = np.random.default_rng(80000 + 977 * k + int(eps * 13))
                bb = _clip(model(b, eps, rng))
                if len(bb) < 20:
                    continue
                cv, iv = band_verdicts(a, bb, k + 1, n_perm)
                coloc += (cv == "attraction")
                coinfil += (iv == "attraction")
                csr_only += (cv == "csr_only")
            row[eps] = {"coloc_attraction": coloc / n_rep,
                        "coinfil_attraction": coinfil / n_rep,
                        "coloc_csr_only": csr_only / n_rep}
            if verbose:
                print(f"    {truth_name:<14} {model_name:<13} eps={eps:>4.0f}  "
                      f"coloc {coloc / n_rep:>5.2f}   coinfil {coinfil / n_rep:>5.2f}",
                      flush=True)
        out[model_name] = row
    return out


def band_independence(n_rep, n_perm):
    """Are the two bands independent claims? Measured on a CONTACT-ONLY truth at eps = 0.

    This is not about registration error at all -- it is a property of the statistic. Both
    bands are DCLF tests on the SAME L-r curve, and L is derived from K, which is
    CUMULATIVE: K(r) counts every pair closer than r, so an excess at 6 um raises K(r) for
    every larger r as well. Writing the surplus as a constant c,

        L(r) - r = sqrt(r^2 + c/pi) - r  ~  c / (2 pi r)

    which decays but never returns to zero. A purely contact-scale association therefore
    keeps L-r above the envelope out to 50 um and fires the co-infiltration band too.

    g(r) is the DERIVATIVE of K and carries no such memory, so comparing the two curves on
    the same data isolates the effect: g collapses back toward its baseline beyond the
    contact band while L-r stays flat.
    """
    radii_report = (6, 10, 16, 20, 30, 40, 50)
    acc = {r: [] for r in radii_report}
    for k in range(n_rep):
        a, b = make_truth(5000 + k, CONTACT_ANNULUS_UM)
        r = cross_k_all_nulls(a, b, RADII_PX, AREA_PX, PIXEL_SIZE_UM,
                              n_perm=n_perm, seed=k + 1, nulls=("reweighted",),
                              registration_radius_floor_um=None)
        rad = np.asarray(r["radii_um"], float)
        lmr = np.asarray(r["L_minus_r"], float)
        hi = np.asarray(r["null_upper_L"], float)
        g = np.asarray([np.nan if v is None else v for v in r["g_observed"]], float)
        for rr in radii_report:
            i = int(np.argmin(np.abs(rad - rr)))
            acc[rr].append((lmr[i], hi[i], g[i]))

    rows = {}
    print(f"\n    {'r (um)':>7} {'obs L-r':>9} {'null hi':>9} {'outside':>8} {'obs g(r)':>9}")
    for rr in radii_report:
        L = float(np.mean([x[0] for x in acc[rr]]))
        H = float(np.mean([x[1] for x in acc[rr]]))
        G = float(np.nanmean([x[2] for x in acc[rr]]))
        rows[rr] = {"L_minus_r": round(L, 2), "null_upper_L": round(H, 2),
                    "outside": bool(L > H), "g": round(G, 2)}
        print(f"    {rr:>7} {L:>9.2f} {H:>9.2f} {('YES' if L > H else 'no'):>8} {G:>9.2f}")
    return rows


def _table(title, res, key, note):
    print(f"\n  {title}")
    print(f"    {'model':<14}" + "".join(f"{'e=' + str(int(e)):>8}" for e in EPS_UM))
    for model_name in MODELS:
        cells = "".join(f"{res[model_name][e][key]:>8.2f}" for e in EPS_UM)
        print(f"    {model_name:<14}{cells}")
    print(f"    {note}")


def main():
    quick = "--quick" in sys.argv
    n_rep = 20 if quick else 60
    n_perm = 99 if quick else 199
    t0 = time.time()

    print("=" * 78)
    print("Radius-floor calibration — does registration error move a finding between bands?")
    print("=" * 78)
    print(f"Field {SIDE_UM:.0f}x{SIDE_UM:.0f} um at {PIXEL_SIZE_UM} um/px · "
          f"{n_rep} repeats x {n_perm} permutations{'  [QUICK]' if quick else ''}")
    print(f"Bands: colocalization {_COLOC_RMIN_UM:.0f}-{_COLOC_RMAX_UM:.0f} um · "
          f"co-infiltration {_COINFIL_RMIN_UM:.0f}-{_COINFIL_RMAX_UM:.0f} um · "
          f"primary null = reweighted")
    print(f"Shipped factor {_RADIUS_FLOOR_FACTOR} implies a registration spec of "
          f"TRE <= {_COLOC_RMAX_UM / _RADIUS_FLOOR_FACTOR:.2f} um\n")

    print("  running REGIONAL-ONLY truth (excess at 25-45 um, none at contact scale)")
    regional = sweep("regional", REGIONAL_ANNULUS_UM, n_rep, n_perm)
    print("\n  running CONTACT-ONLY truth (excess at 0-12 um)")
    contact = sweep("contact", CONTACT_ANNULUS_UM, n_rep, n_perm)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    _table("REGIONAL-ONLY truth → rate of significant ATTRACTION in the "
           "colocalization band", regional, "coloc_attraction",
           "This is the FALSE CELL-SCALE-ENGAGEMENT rate. eps=0 is the control (~0.05).")
    _table("CONTACT-ONLY truth → rate of significant ATTRACTION in the "
           "colocalization band", contact, "coloc_attraction",
           "Correct attribution: the truth is inside this band and should be found here.")
    _table("CONTACT-ONLY truth → rate of significant ATTRACTION in the "
           "co-infiltration band", contact, "coinfil_attraction",
           "Upward leakage: a contact-scale truth reported as a regional one.")

    # ── control ───────────────────────────────────────────────────────────────
    ctrl = {m: regional[m][0.0]["coloc_attraction"] for m in MODELS}
    ctrl_ok = all(v <= LEAK_TOLERANCE for v in ctrl.values())
    print(f"\n  CONTROL at eps=0: " +
          ", ".join(f"{m} {v:.2f}" for m, v in ctrl.items()) +
          ("   OK" if ctrl_ok else "   <-- CONTAMINATED, calibration below is void"))

    # ── calibration ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CALIBRATION")
    print("=" * 78)
    summary = {}
    eps_max = max(EPS_UM)
    for model_name in MODELS:
        rates = regional[model_name]
        eps_star, censored = 0.0, False
        for eps in EPS_UM:
            if rates[eps]["coloc_attraction"] <= LEAK_TOLERANCE:
                eps_star = eps
            else:
                break
        # eps* landing on the top of the sweep is CENSORED, not measured: leakage might
        # begin just beyond it. The implied factor is then an UPPER BOUND on what is
        # needed, never a point estimate — reporting 20/20 = 1.00 as if measured would
        # invent precision the sweep does not have.
        if eps_star >= eps_max:
            censored = True
        factor = (_COLOC_RMAX_UM / eps_star) if eps_star > 0 else float("inf")
        summary[model_name] = {"eps_star_um": eps_star, "implied_factor": factor,
                               "censored_at_sweep_max": censored}
        if eps_star <= 0:
            desc = "leaks at every tested eps  ->  factor > "
            desc += f"{_COLOC_RMAX_UM / EPS_UM[1]:.1f}"
        elif censored:
            desc = (f"no leakage at any tested eps (<= {eps_max:.0f} um)  ->  "
                    f"factor <= {factor:.2f} [CENSORED at sweep max]")
        else:
            desc = (f"leakage begins above eps* = {eps_star:.0f} um  ->  "
                    f"factor = {factor:.2f}")
        print(f"  {model_name:<14} {desc}")

    realistic = summary["smooth field"]
    f, censored = realistic["implied_factor"], realistic["censored_at_sweep_max"]
    print(f"\n  The smooth field is the realistic model (98 % of the measured residual is")
    print(f"  spatially structured), so it sets the answer.")
    if censored:
        verdict = (f"No displacement up to {eps_max:.0f} um moved a regional-only truth "
                   f"into the contact band: the false-engagement rate stayed <= "
                   f"{max(regional['smooth field'][e]['coloc_attraction'] for e in EPS_UM):.2f} "
                   f"against a {LEAK_TOLERANCE:.2f} tolerance. Registration error does not "
                   f"MANUFACTURE a cell-scale claim, so the floor is not needed for the job "
                   f"it was introduced for, and {_RADIUS_FLOOR_FACTOR} is over-strict by at "
                   f"least {_RADIUS_FLOOR_FACTOR / max(f, 1e-9):.0f}x within the tested "
                   f"range. Where leakage begins beyond {eps_max:.0f} um is UNMEASURED.")
    elif f > _RADIUS_FLOOR_FACTOR * 1.25:
        verdict = (f"Leakage sets in EARLIER than {_RADIUS_FLOOR_FACTOR} assumes: the "
                   f"honest factor is {f:.2f}, tightening the spec to TRE <= "
                   f"{realistic['eps_star_um']:.0f} um. Some shipped contact-scale claims "
                   f"are not supported.")
    elif f < _RADIUS_FLOOR_FACTOR * 0.8:
        verdict = (f"{_RADIUS_FLOOR_FACTOR} is too strict: the honest factor is {f:.2f}, "
                   f"loosening the spec to TRE <= {realistic['eps_star_um']:.0f} um. The "
                   f"floor is currently withholding valid contact-scale findings.")
    else:
        verdict = (f"{_RADIUS_FLOOR_FACTOR} is corroborated: the measured factor is "
                   f"{f:.2f}, and the implied spec of TRE <= "
                   f"{realistic['eps_star_um']:.0f} um is what the tab has been assuming.")
    print(f"\n  {verdict}")

    # ── the separate defect this sweep exposed ────────────────────────────────
    print("\n" + "=" * 78)
    print("BAND INDEPENDENCE — a defect the sweep exposed, unrelated to registration")
    print("=" * 78)
    print("  CONTACT-ONLY truth at eps = 0. The truth is confined to 0-12 um, yet the")
    print(f"  co-infiltration band claims attraction at "
          f"{contact['smooth field'][0.0]['coinfil_attraction']:.2f}. Why:")
    indep = band_independence(max(12, n_rep // 4), n_perm)
    g_near = indep[10]["g"]
    g_far = indep[40]["g"]
    far_outside = indep[40]["outside"]
    band_ok = not far_outside
    print(f"\n  g(r) falls {g_near:.1f} -> {g_far:.1f} beyond the contact band, so the "
          f"DENSITY of the\n  excess really is localised. L-r does not follow it: it stays "
          f"above the null\n  envelope out to 50 um because K is CUMULATIVE.")
    print(f"\n  {'PASS' if band_ok else 'FAIL'} — the two bands are "
          f"{'independent claims' if band_ok else 'NOT independent claims'}. "
          f"{'' if band_ok else 'Splitting a cumulative statistic by radius does not separate scales: '}"
          f"{'' if band_ok else 'a contact-scale finding is reported as regional co-infiltration too. '}"
          f"{'' if band_ok else 'The decomposition needs to run on g(r), not L-r — which needs its own '}"
          f"{'' if band_ok else 'calibration before anything changes.'}")

    payload = {"config": {"n_rep": n_rep, "n_perm": n_perm, "eps_um": list(EPS_UM),
                          "corr_length_um": CORR_LENGTH_UM,
                          "leak_tolerance": LEAK_TOLERANCE,
                          "shipped_factor": _RADIUS_FLOOR_FACTOR,
                          "coloc_band_um": [_COLOC_RMIN_UM, _COLOC_RMAX_UM]},
               "regional_only": {m: {str(k): v for k, v in r.items()}
                                 for m, r in regional.items()},
               "contact_only": {m: {str(k): v for k, v in r.items()}
                                for m, r in contact.items()},
               "control_eps0": ctrl, "control_ok": ctrl_ok,
               "calibration": summary, "verdict": verdict,
               "band_independence": {"radii": {str(k): v for k, v in indep.items()},
                                     "bands_independent": band_ok}}
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0 if (ctrl_ok and band_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
