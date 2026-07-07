"""
validate_primary_null_calibration.py — calibrate the PRODUCTION nulls under a
realistic shared-preference null hypothesis.

WHY THIS IS THE DECISIVE TEST. The production headline ("robust association")
comes from the inhomogeneous (Kinhom) and toroidal nulls, not the homogeneous
CSR baseline. validate_dclf.py only checked CSR calibration. An estimated-
intensity KDE-resampling null can be ANTI-CONSERVATIVE (its DCLF p too small too
often), which would inflate "robust" findings. This script measures, under a null
where there is NO true cross-association, how often each null's DCLF test fires.

NULL HYPOTHESIS (H0) — shared tissue preference, no cross-relationship:
  A and B are each clustered (inhomogeneous) by drawing INDEPENDENTLY from the
  SAME coarse, distributed, ~stationary intensity field (a periodic grid of
  Gaussian compartments, σ coarser than the 10–50 µm test band — the biologically
  realistic regime where CD8 and TIM-3 both pile into the same inflamed
  compartments for their own independent reasons). Under H0 there is no cross-
  association beyond shared preference.

CORRECT CALIBRATION: P(global_p_dclf <= 0.05) ≈ 0.05 for a well-calibrated null.
  • P(p<=0.05) >> 0.05  → ANTI-CONSERVATIVE  → inflates findings  (REPORT LOUDLY)
  • P(p<=0.05) << 0.05  → conservative       → loses power
  • homogeneous CSR is EXPECTED to be strongly anti-conservative here (that is
    exactly its documented weakness, and why it is only a baseline).

We report calibration for homogeneous CSR, the inhomogeneous null at the KDE
bandwidth sweep 0.5× / 1× / 2× (the 1× = 50 µm is the production default), and the
toroidal null. The audit flagged that 2× bandwidth false-positives on shared-
preference cases — this quantifies it.

Config via env: NREAL (default 500), NPERM (default 199).
Output: printed + saved to validation/primary_null_calibration_output.txt
"""

import os
import sys
import numpy as np
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_stats import cross_k_all_nulls   # noqa: E402

NREAL = int(os.environ.get("NREAL", "500"))
NPERM = int(os.environ.get("NPERM", "199"))
PIX   = 1.0                                   # 1 µm/px → DCLF band 10–50 µm = 10–50 px
RADII = np.arange(0.0, 100.0, 4.0)           # 25 radii
WIN   = 960.0                                 # 3 × 320 grid period (stationary field)
N_PTS = 250
SIGMA = 70.0                                  # compartment scale (> 50 µm band)
CENTERS = np.array([(cx, cy) for cx in (160, 480, 800)
                    for cy in (160, 480, 800)], dtype=float)
WINBOX = box(0, 0, WIN, WIN)

OUT = os.path.join(os.path.dirname(__file__), "primary_null_calibration_output.txt")
_log = []
def log(s=""):
    print(s); _log.append(s)


def _draw_from_field(rng, n):
    idx = rng.integers(0, len(CENTERS), n)
    return np.clip(CENTERS[idx] + rng.normal(0, SIGMA, (n, 2)), 1, WIN - 1)


def _draw_uniform(rng, n):
    """Uniform CSR draw — TRUE independence with NO shared preference. This is the
    HARNESS SANITY CONTROL: every null (incl. the structure-preserving ones) MUST
    be ~well-calibrated here. If they are, the anti-conservatism seen under the
    shared-preference H0 is a real property of that regime, not a harness bug."""
    return rng.uniform(0, WIN, (n, 2))


def _verdict(rate):
    se = (0.05 * 0.95 / NREAL) ** 0.5
    if rate > 0.05 + 3 * se:
        return "ANTI-CONSERVATIVE (inflates findings)"
    if rate < 0.05 - 3 * se:
        return "conservative (loses power)"
    return "well-calibrated"


_LABELS = {
    "homogeneous": "homogeneous CSR",
    "inhom_0.5x":  "inhomogeneous 0.5x",
    "inhom_1x":    "inhomogeneous 1x*",   # production default (50 µm)
    "inhom_2x":    "inhomogeneous 2x",
    "toroidal":    "toroidal shift",
}
_KEYS = ["homogeneous", "inhom_0.5x", "inhom_1x", "inhom_2x", "toroidal"]


def run_scenario(title, draw_fn, rng_seed):
    """Run NREAL realizations of one H0 and report per-null DCLF calibration.
    Returns {key: (mean_p, P(p<=.05), P(p<=.10), verdict_str)}."""
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)
    ps = {k: [] for k in _KEYS}
    rng = np.random.default_rng(rng_seed)
    for i in range(NREAL):
        A = draw_fn(rng, N_PTS)
        B = draw_fn(rng, N_PTS)              # independent draw (same regime)
        # This script DOCUMENTS the old nulls' failure → request them explicitly
        # (the production default now ships only the calibrated reweighted primary).
        res = cross_k_all_nulls(A, B, RADII, WINBOX.area, PIX,
                                n_perm=NPERM, seed=0, tissue_polygon=WINBOX,
                                nulls=("homogeneous", "inhomogeneous", "toroidal"))
        nulls = res["nulls"]
        ps["homogeneous"].append(nulls["homogeneous"]["global"]["global_p_dclf"])
        ps["toroidal"].append(nulls["toroidal"]["global"]["global_p_dclf"])
        sens = nulls["inhomogeneous"].get("bandwidth_sensitivity", {})
        ps["inhom_0.5x"].append(sens["0.5x"]["global"]["global_p_dclf"])
        ps["inhom_1x"].append(sens["1x"]["global"]["global_p_dclf"])
        ps["inhom_2x"].append(sens["2x"]["global"]["global_p_dclf"])
        if (i + 1) % max(1, NREAL // 10) == 0:
            print(f"    {title[:28]} ... {i+1}/{NREAL}")

    log(f"\n{'null / bandwidth':22s} {'mean p':>8} {'P(p<=.05)':>10} "
        f"{'P(p<=.10)':>10}   calibration")
    log("-" * 78)
    summary = {}
    for k in _KEYS:
        arr = np.asarray(ps[k], float)
        r05, r10 = float(np.mean(arr <= 0.05)), float(np.mean(arr <= 0.10))
        v = _verdict(r05)
        summary[k] = (float(arr.mean()), r05, r10, v)
        log(f"{_LABELS[k]:22s} {arr.mean():>8.3f} {r05:>10.3f} {r10:>10.3f}   {v}")
    log("-" * 78)
    hist, edges = np.histogram(np.asarray(ps["inhom_1x"], float), bins=10, range=(0, 1))
    log(f"  inhomogeneous-1x p histogram (uniform ⇒ ~{NREAL//10}/bin): "
        + " ".join(f"{hist[j]}" for j in range(10)))
    return summary


def main():
    log("=" * 78)
    log("PRIMARY-NULL CALIBRATION")
    log("=" * 78)
    log(f"  realizations NREAL={NREAL}   permutations NPERM={NPERM}   "
        f"DCLF band 10–50 µm   pixel size {PIX} µm/px")
    log(f"  * inhomogeneous 1x (50 µm) was the FORMER primary — this script documents")
    log(f"    why it (and toroidal) were RETIRED. The production primary is now the")
    log(f"    calibrated reweighted cross-K; see validate_reweighted_null.py + ihc.md §15.")

    # Scenario 1 (the question): shared preference, no cross-association.
    shared = run_scenario(
        "H0-A  SHARED PREFERENCE — A,B independent draws from the SAME coarse "
        "field\n      (σ=70px > 50µm band; the biologically realistic regime)",
        _draw_from_field, rng_seed=20260615)

    # Scenario 2 (harness sanity control): true independence, no preference.
    sanity = run_scenario(
        "H0-B  SANITY CONTROL — A,B uniform CSR (true independence, NO shared "
        "preference)\n      (every null, incl. structure-preserving, MUST be "
        "~well-calibrated here)",
        _draw_uniform, rng_seed=20260616)

    log("\n" + "=" * 78)
    log("INTERPRETATION")
    log("=" * 78)
    log("  Sanity control (H0-B, uniform independence) — proves the harness is fair:")
    for k in ("homogeneous", "inhom_1x", "toroidal"):
        log(f"     {_LABELS[k]:22s} P(p<=.05)={sanity[k][1]:.3f}  -> {sanity[k][3]}")
    sanity_ok = all("well-calibrated" in sanity[k][3]
                    for k in ("homogeneous", "inhom_1x", "toroidal"))
    log(f"     → harness {'FAIR (all nulls ~5% under true independence)' if sanity_ok else 'SUSPECT (a null mis-fires even under uniform CSR)'}")

    log("\n  Shared preference (H0-A, the regime that matters):")
    log(f"     homogeneous CSR        P(p<=.05)={shared['homogeneous'][1]:.3f}  "
        f"-> {shared['homogeneous'][3]} (expected weak baseline)")
    log(f"     PRODUCTION primary 1x  P(p<=.05)={shared['inhom_1x'][1]:.3f}  "
        f"-> {shared['inhom_1x'][3]}")
    log(f"     toroidal               P(p<=.05)={shared['toroidal'][1]:.3f}  "
        f"-> {shared['toroidal'][3]}")
    log(f"     inhomogeneous 0.5x     P(p<=.05)={shared['inhom_0.5x'][1]:.3f}  "
        f"-> {shared['inhom_0.5x'][3]}")
    log(f"     inhomogeneous 2x       P(p<=.05)={shared['inhom_2x'][1]:.3f}  "
        f"-> {shared['inhom_2x'][3]}")

    prim_ok = "well-calibrated" in shared["inhom_1x"][3]
    tor_ok  = "well-calibrated" in shared["toroidal"][3]
    log("")
    if sanity_ok and not (prim_ok and tor_ok):
        log("  ⚠⚠ FINDING: the harness is FAIR (all nulls ~5% under true "
            "independence),\n  yet under the realistic SHARED-PREFERENCE H0 the "
            "structure-preserving nulls\n  (production primary + toroidal) are "
            "ANTI-CONSERVATIVE. This is a real property\n  of the estimated-"
            "intensity KDE-resampling and toroidal-shift tests, NOT a harness\n  "
            "artifact. The production 'robust' verdict therefore does NOT control "
            "the\n  false-positive rate against shared tissue preference — 'robust'"
            " findings are\n  INFLATED and must be reported with this caveat. Do "
            "NOT adjust thresholds to hide it.")
    elif prim_ok and tor_ok:
        log("  VERDICT: structure-preserving nulls well-calibrated under shared "
            "preference → production 'robust' headline is trustworthy on this H0.")
    else:
        log("  ⚠ VERDICT: calibration problem AND the sanity control also misfired"
            " — investigate the harness before drawing conclusions.")

    with open(OUT, "w") as f:
        f.write("\n".join(_log) + "\n")
    log(f"\n  (saved to {os.path.relpath(OUT)})")
    return 0 if (prim_ok and tor_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
