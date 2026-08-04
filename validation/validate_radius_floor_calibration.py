#!/usr/bin/env python
"""
validate_radius_floor_calibration.py — calibrate the two radius constants SEPARATELY.

WHY TWO. One constant, _RADIUS_FLOOR_FACTOR = 3.0, was doing two unrelated jobs:

  (a) INTERPRETATION FLOOR. The smallest inter-cell distance a pair may be QUOTED at. A
      reporting boundary: set it wrong and either a readable radius is withheld, or an
      unmeasurable one is presented as biology.

  (b) ANALYSABILITY GATE. The error beyond which the pair is not analysed AT ALL. In
      _certify_fitzpatrick_west this is the direct `else` of band_ok, so it decides DEFORMED
      vs RADIUS_LIMITED. On the real 10X windows 56 % of them land on this branch, so it
      governs far more output than (a) does.

They are now separate constants (_RADIUS_FLOOR_FACTOR, _ANALYSABILITY_FACTOR) and they need
DIFFERENT evidence, which is the whole point of this script:

  (a) is a LOCALISATION question — at what error can a finding no longer be attributed to
      the radius it is quoted at? Measured as leakage: a truth that lives only at regional
      scale must not produce a contact-scale claim.

  (b) is a POWER question — a pair is worth analysing for exactly as long as a real
      association is still detectable in it. Registration error costs sensitivity, never
      validity (validate_radius_floor.py), so the gate should close when the test goes
      blind, not before.

Both convert to a factor through the shipped arithmetic:

      analysable  <=>  factor * TRE <= max_radius * (1 - band_frac) = 50 um
      contact scale claimable  <=>  factor * TRE <= _COLOC_RMAX_UM  = 20 um

  =>  interpretation factor = 20 / eps_leak*      analysability factor = 50 / eps_power*

WHAT IT RUNS ON, and why.

  * REAL substrates (ll477_cd8, keren_p13) with synthetic kept only as a contrast. The
    earlier calibration used Gaussian blobs whose structure sits at 180 um — above the
    null's 75 um bandwidth, so the null absorbed it and everything looked easy. Real tissue
    has structure below the bandwidth and behaves differently.

  * The CORRECTED statistic. validate_band_decomposition.py established that the shipped
    (reweighted, bands) combination is anti-conservative on real tissue AND leaks a
    contact-scale truth into the upper band. Calibrating a constant against a broken
    statistic would produce a number that is precisely wrong, so this runs on
    dense_morphology + bands_ring.

  * A SMOOTH deformation field, matched to the measured residual (98 % spatially
    structured), with iid noise as the contrast. A smooth field translates a neighbourhood
    of B coherently against A, which is the mechanism that can relocate an excess between
    bands; iid merely blurs.

  * eps out to 80 um. The earlier sweep stopped at 20 and every answer came back CENSORED at
    the sweep maximum, which bounds a factor from above but never measures it.

Run:  python validation/validate_radius_floor_calibration.py           (~45 min)
      python validation/validate_radius_floor_calibration.py --quick   (~8 min)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.spatial.spatial_stats import (cross_k_all_nulls, _COLOC_RMIN_UM,
                                         _COLOC_RMAX_UM, _RADIUS_FLOOR_FACTOR,
                                         _ANALYSABILITY_FACTOR)
from validation.spatial_substrates import load_substrate, impose_band_association
from validation.validate_radius_floor_localisation import _displace_smooth, _displace_iid

RADII_UM = np.arange(0.0, 101.0, 2.0)
EPS_UM = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0)
SUBSTRATES = ("ll477_cd8", "keren_p13")
CONTRAST_SUBSTRATE = "synthetic"

# The band-decomposition validation picked these; calibrating on the shipped pair would
# calibrate against a statistic that is both anti-conservative and leaky.
NULL = "dense_morphology"
STAT = "bands_ring"

CONTACT = (_COLOC_RMIN_UM, _COLOC_RMAX_UM)      # 10-20 um
REGIONAL = (30.0, 40.0)
TARGET_ENRICHMENT = 2.0

MAX_RADIUS_UM = 100.0
BAND_FRAC = 0.5
ANALYSABLE_BUDGET_UM = MAX_RADIUS_UM * (1.0 - BAND_FRAC)     # 50 um

LEAK_TOLERANCE = 0.10          # (a): a false contact claim is a wrong scientific statement
POWER_FLOOR = 0.50             # (b): below this the pair is not worth analysing
PIXEL_SIZE_UM = 1.0

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "radius_floor_calibration_results.json")


def _clip(pts, W, H):
    """Points pushed out of the window are dropped, as run_spatial_association does."""
    return pts[(pts[:, 0] > 0) & (pts[:, 0] < W) & (pts[:, 1] > 0) & (pts[:, 1] < H)]


def _displace(B, eps_um, model, rng, W, H):
    if eps_um <= 0:
        return B
    # The shared displacement helpers work in PIXELS at their module's pixel size; here one
    # pixel is one micron, so scale in and out to keep them exact.
    from validation import validate_radius_floor_localisation as L
    px = L.PIXEL_SIZE_UM
    out = (_displace_smooth if model == "smooth" else _displace_iid)(B / px, eps_um, rng)
    return _clip(out * px, W, H)


def one(substrate, truth, eps_um, model, seed, n_perm):
    pts, (W, H) = load_substrate(substrate, seed=seed)
    if truth == "independent":
        A, B = impose_band_association(pts, CONTACT, rng=np.random.default_rng(seed),
                                       boost=0.0)
    else:
        ann = CONTACT if truth == "contact" else REGIONAL
        A, B = impose_band_association(pts, ann, rng=np.random.default_rng(seed),
                                       target_enrichment=TARGET_ENRICHMENT)
    B = _displace(B, eps_um, model, np.random.default_rng(90000 + seed), W, H)
    if len(B) < 30:
        return None
    r = cross_k_all_nulls(A, B, RADII_UM, W * H, PIXEL_SIZE_UM, n_perm=n_perm, seed=seed,
                          nulls=(NULL,), morphology_support=pts,
                          registration_radius_floor_um=None)
    b = r["nulls"][NULL][STAT]
    hit = lambda k: (b[k]["direction"] == "association" and b[k]["significant"])   # noqa: E731
    return {"coloc": hit("colocalization"), "coinfil": hit("coinfiltration")}


def sweep(substrates, models, n_rep, n_perm):
    res = {}
    for sub in substrates:
        for model in models:
            for truth in ("contact", "regional", "independent"):
                for eps in EPS_UM:
                    c = i = n = 0
                    for k in range(n_rep):
                        out = one(sub, truth, eps, model, 4000 + k, n_perm)
                        if out is None:
                            continue
                        c += out["coloc"]
                        i += out["coinfil"]
                        n += 1
                    res[(sub, model, truth, eps)] = {
                        "coloc": c / max(n, 1), "coinfil": i / max(n, 1), "n": n}
                print(f"    {sub:<11}{model:<8}{truth:<12} done", flush=True)
    return res


def _table(res, sub, model, key, truths, title, note):
    print(f"\n  {title}   [{sub} / {model}]")
    print(f"    {'truth':<13}" + "".join(f"{('e=' + str(int(e))):>8}" for e in EPS_UM))
    for t in truths:
        cells = "".join(f"{res[(sub, model, t, e)][key]:>8.2f}" for e in EPS_UM)
        print(f"    {t:<13}{cells}")
    print(f"    {note}")


def calibrate(res, substrates, model="smooth"):
    """eps* for each constant, taken as the WORST real substrate."""
    print("\n" + "=" * 92)
    print("CALIBRATION")
    print("=" * 92)

    # (a) interpretation floor <- leakage of a regional-only truth into the contact band
    leak_eps = []
    for sub in substrates:
        best = 0.0
        for e in EPS_UM:
            if res[(sub, model, "regional", e)]["coloc"] <= LEAK_TOLERANCE:
                best = e
            else:
                break
        leak_eps.append((sub, best))
    eps_leak = min(e for _, e in leak_eps)
    censored_a = eps_leak >= max(EPS_UM)
    fac_a = _COLOC_RMAX_UM / eps_leak if eps_leak > 0 else float("inf")

    # (b) analysability <- power on a real association
    pow_eps = []
    for sub in substrates:
        best = 0.0
        for e in EPS_UM:
            ok = (res[(sub, model, "contact", e)]["coloc"] >= POWER_FLOOR
                  or res[(sub, model, "regional", e)]["coinfil"] >= POWER_FLOOR)
            if ok:
                best = e
            else:
                break
        pow_eps.append((sub, best))
    eps_pow = min(e for _, e in pow_eps)
    censored_b = eps_pow >= max(EPS_UM)
    fac_b = ANALYSABLE_BUDGET_UM / eps_pow if eps_pow > 0 else float("inf")

    print(f"  (a) INTERPRETATION FLOOR   leakage <= {LEAK_TOLERANCE:.2f} up to "
          f"eps* = {eps_leak:.0f} um" + ("  [CENSORED]" if censored_a else ""))
    print(f"      per substrate: " + ", ".join(f"{s}={e:.0f}" for s, e in leak_eps))
    print(f"      => factor = {_COLOC_RMAX_UM:.0f} / {eps_leak:.0f} = "
          f"{fac_a:.2f}   (shipped {_RADIUS_FLOOR_FACTOR})")
    print()
    print(f"  (b) ANALYSABILITY GATE     power  >= {POWER_FLOOR:.2f} up to "
          f"eps* = {eps_pow:.0f} um" + ("  [CENSORED]" if censored_b else ""))
    print(f"      per substrate: " + ", ".join(f"{s}={e:.0f}" for s, e in pow_eps))
    print(f"      => factor = {ANALYSABLE_BUDGET_UM:.0f} / {eps_pow:.0f} = "
          f"{fac_b:.2f}   (shipped {_ANALYSABILITY_FACTOR})")
    print()
    print(f"      at factor {_ANALYSABILITY_FACTOR}, a pair is analysed only while TRE <= "
          f"{ANALYSABLE_BUDGET_UM / _ANALYSABILITY_FACTOR:.1f} um")
    print(f"      at factor {fac_b:.2f}, that becomes TRE <= {eps_pow:.0f} um")
    return {"interpretation": {"eps_star_um": eps_leak, "factor": fac_a,
                               "censored": bool(censored_a), "per_substrate": dict(leak_eps)},
            "analysability": {"eps_star_um": eps_pow, "factor": fac_b,
                              "censored": bool(censored_b), "per_substrate": dict(pow_eps)}}


def main():
    quick = "--quick" in sys.argv
    n_rep = 12 if quick else 40
    n_perm = 99 if quick else 199
    models = ("smooth",) if quick else ("smooth", "iid")
    subs = list(SUBSTRATES) + ([] if quick else [CONTRAST_SUBSTRATE])
    t0 = time.time()

    print("=" * 92)
    print("Radius-floor calibration — two constants, two criteria")
    print("=" * 92)
    print(f"null={NULL} statistic={STAT} · {n_rep} reps x {n_perm} perms · "
          f"eps up to {max(EPS_UM):.0f} um · substrates {', '.join(subs)}"
          f"{'  [QUICK]' if quick else ''}\n")

    res = sweep(subs, models, n_rep, n_perm)

    for sub in subs:
        for model in models:
            _table(res, sub, model, "coloc", ("regional",),
                   "LEAKAGE — regional-only truth claiming CONTACT scale",
                   f"sets the interpretation floor; tolerance {LEAK_TOLERANCE}")
            _table(res, sub, model, "coloc", ("contact",),
                   "POWER — contact truth found in the contact band",
                   f"sets the analysability gate; floor {POWER_FLOOR}")
            _table(res, sub, model, "coinfil", ("regional",),
                   "POWER — regional truth found in the regional band", "")
            _table(res, sub, model, "coloc", ("independent",),
                   "SIZE — independent truth falsely called associated",
                   "registration error must not INVENT a finding")

    cal = calibrate(res, list(SUBSTRATES), model="smooth")
    payload = {"config": {"n_rep": n_rep, "n_perm": n_perm, "eps_um": list(EPS_UM),
                          "null": NULL, "statistic": STAT, "substrates": subs,
                          "leak_tolerance": LEAK_TOLERANCE, "power_floor": POWER_FLOOR,
                          "shipped_interpretation_factor": _RADIUS_FLOOR_FACTOR,
                          "shipped_analysability_factor": _ANALYSABILITY_FACTOR},
               "grid": {f"{a}|{b}|{c}|{d}": v for (a, b, c, d), v in res.items()},
               "calibration": cal}
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\n  Wrote {OUT_JSON}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
