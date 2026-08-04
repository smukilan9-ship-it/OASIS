"""
validate_fw_hyreco.py — the FW bound on the TARGET MODALITY at last.

WHAT THIS CLOSES. Every external calibration of §3.5's cell-error budget so far has been
on ANHIR/CIMA: lung, mammary, kidney, largely cross-stain H&E↔IHC, and §24 showed the
bound is TISSUE-DEPENDENT (under-stating on 9/20 pairs, worst 1.93x). The obvious
question — which side of that envelope does H-DAB fall on? — was unanswerable, because
the tool ships for CD8/TIM-3 and no CD8 serial-section set with expert landmarks was in
hand. HyReCo is that set.

THE DATA. Van Zon, Stathonikos et al., J. Med. Imaging 10(6):067501 (2023); landmarks by
Fraunhofer MEVIS; CC BY-SA 4.0. Nine cases, five stains each (H&E, CD8, CD45, Ki67,
PHH3) on CONSECUTIVE sections, 11-19 manually placed expert landmarks per section.
Restricting to the IHC stains gives 6 stain-pairs per case, 54 pairs, all in the
IHC<->IHC regime §7.1 identified as the one OASIS actually operates in.

TWO PROPERTIES OF THIS DATA VERIFIED BEFORE USE, not assumed (the registry says so
explicitly, and it is right to):

  1. Coordinates are MILLIMETRE world coordinates, not pixels. Checked against the
     BigTIFF metadata for the two cases whose slides are on disk: 0.2430 um/px over
     95601 x 218145 px = 23.2 x 53.0 mm, and every landmark falls inside that. Because
     they are already physical, NO pixel size is needed anywhere below — coordinates are
     converted mm -> um and the pipeline is handed pixel_size_um = 1.0.
  2. Landmark INDEX corresponds across stains. The paired CD8-vs-CD45 offset is 5.1-9.3 mm
     with median ~ max, i.e. a near-rigid offset between the two slides' world origins
     rather than an erratic spread — which is what mismatched indices would produce. The
     similarity fit absorbs a constant offset as translation.

THE LIMITATION, stated up front. HyReCo ships ONE landmark set per section, not two
independent passes, so inter-observer FLE cannot be measured here and the §24 "gold"
two-annotator design is unavailable. This is the §24 PROXY design — fit on a subset,
predict and measure realized error at held-out landmarks — which §24 credentialed against
the gold on six pairs (proxy - gold = +0.07 median, erring SAFE). That +0.07 should be
subtracted when reading these ratios. FLE is supplied rather than measured, so it is
swept, exactly as in §24.5.

Landmark counts are low (11-19), so a strict half-split would leave 5-9 points to fit a
similarity — marginal against §3.5's n>=6. The primary design therefore holds out a small
fraction and rotates; the §24-comparable half-split is reported alongside.

Run:  .venv/bin/python -m validation.validate_fw_hyreco
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from oasis.spatial import serial_registration as sr   # noqa: E402

ROOTS = ["/Volumes/Expansion/oasis_datasets/HyReCo/HyReCo",
         os.path.expanduser("~/oasis_validation_datasets/HyReCo/inputs")]
CASES = ["29", "108", "361", "464", "533", "611", "628", "644", "679"]
IHC = ["CD8", "CD45", "KI67", "PHH3"]      # the IHC<->IHC regime (§7.1)
ALL_STAINS = IHC + ["HE"]
MM_TO_UM = 1000.0
MIN_N = 8


def root():
    for r in ROOTS:
        if os.path.isdir(r):
            return r
    return None


def load(stain, case, base):
    p = os.path.join(base, stain, f"{case}.csv")
    if not os.path.exists(p):
        return None
    d = np.loadtxt(p, delimiter=",")
    if d.ndim != 2 or len(d) < 3:
        return None
    return d[:, :2] * MM_TO_UM            # -> microns; everything downstream uses px=1.0


def calibrate(fit_ref, fit_mov, ev_ref, ev_mov, fle):
    """Identical arithmetic to validate_fw_calibration_extended, at pixel_size_um = 1.0
    because the coordinates are already microns."""
    M = sr._fit_similarity_robust(fit_mov, fit_ref)
    dec = sr.deformation_from_landmarks(fit_ref, fit_mov, M, 1.0, fle, method="robust")
    deform = max(dec["deformation_p90_um"] or 0.0, dec["deformation_rms_um"] or 0.0)
    deform_ub = max(dec["deformation_p90_ub_um"] or 0.0, dec["deformation_rms_ub_um"] or 0.0)
    tre = sr.transform_prediction_error(fit_ref, fle * np.sqrt(2.0), ev_ref)
    if tre is None:
        return None
    ann = 2.0 * fle ** 2
    pred = np.sqrt(tre ** 2 + deform ** 2 + ann)
    pred_ub = np.sqrt(tre ** 2 + deform_ub ** 2 + ann)
    real = np.linalg.norm(sr._apply_affine(ev_mov, M) - ev_ref, axis=1)
    return {"ratio": float(np.percentile(real, 90) / np.percentile(pred, 90)),
            "cov": float(np.mean(real <= pred_ub)),
            "real_p90": float(np.percentile(real, 90)),
            "pred_p90": float(np.percentile(pred, 90))}


def holdout(ref, mov, fle, frac=0.30, n_rep=24, seed=0):
    """Fit on (1-frac), predict and measure at the held-out frac; median over rotations."""
    rng = np.random.default_rng(seed)
    n = len(ref)
    k = max(2, int(round(frac * n)))
    if n - k < 4:
        return None
    rs, cs, rp, pp = [], [], [], []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        ei, fi = idx[:k], idx[k:]
        r = calibrate(ref[fi], mov[fi], ref[ei], mov[ei], fle)
        if r:
            rs.append(r["ratio"]); cs.append(r["cov"])
            rp.append(r["real_p90"]); pp.append(r["pred_p90"])
    if len(rs) < 5:
        return None
    return {"ratio": float(np.median(rs)), "cov": float(np.median(cs)),
            "real_p90": float(np.median(rp)), "pred_p90": float(np.median(pp)),
            "n_lm": int(n), "n_fit": int(n - k), "n_eval": int(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fle-um", type=float, default=14.1,
                    help="supplied FLE (no second annotator in HyReCo); default = §24 gold median")
    ap.add_argument("--include-he", action="store_true",
                    help="also score H&E<->IHC pairs (cross-modal, NOT the target regime)")
    a = ap.parse_args()

    base = root()
    if base is None:
        print("HyReCo not found — is the Expansion volume attached?")
        return 1
    stains = ALL_STAINS if a.include_he else IHC
    print("=" * 84)
    print("FW bound on the TARGET MODALITY — HyReCo CD8/CD45/Ki67/PHH3, consecutive sections")
    print("=" * 84)
    print(f"stains: {', '.join(stains)}   FLE (supplied, swept below) = {a.fle_um:.1f} µm")
    print("design: §24 PROXY (single annotator, held-out landmarks); subtract its +0.07 "
          "credentialed offset\n")

    pts = {(s, c): load(s, c, base) for s in stains for c in CASES}
    rows = []
    print(f"  {'case':6s} {'pair':14s} {'n_lm':>5s} {'fit/ev':>7s} {'real p90':>9s} "
          f"{'pred p90':>9s} {'ratio':>6s} {'cov':>5s}")
    for c in CASES:
        for s1, s2 in itertools.combinations(stains, 2):
            ref, mov = pts.get((s1, c)), pts.get((s2, c))
            if ref is None or mov is None:
                continue
            n = min(len(ref), len(mov))
            if n < MIN_N:
                continue
            r = holdout(ref[:n], mov[:n], a.fle_um)
            if not r:
                continue
            r.update(case=c, pair=f"{s1}->{s2}",
                     regime="ihc" if (s1 != "HE" and s2 != "HE") else "cross")
            rows.append(r)
            print(f"  {c:6s} {r['pair']:14s} {r['n_lm']:5d} "
                  f"{r['n_fit']}/{r['n_eval']:<5d} {r['real_p90']:9.1f} {r['pred_p90']:9.1f} "
                  f"{r['ratio']:6.2f} {100*r['cov']:4.0f}%")
    if not rows:
        print("  no usable pairs")
        return 1

    ihc = [r for r in rows if r["regime"] == "ihc"]
    rt = np.array([r["ratio"] for r in ihc])
    cv = np.array([r["cov"] for r in ihc])
    print(f"\n  IHC<->IHC pairs: n = {len(ihc)} over {len(set(r['case'] for r in ihc))} cases")
    print(f"  ratio  median {np.median(rt):.2f}  IQR {np.percentile(rt,25):.2f}-"
          f"{np.percentile(rt,75):.2f}  max {rt.max():.2f}")
    print(f"  ratio corrected for the proxy's +0.07 credentialed offset: "
          f"median {np.median(rt)-0.07:.2f}")
    print(f"  coverage median {100*np.median(cv):.0f}%   "
          f"{int((cv>=0.85).sum())}/{len(cv)} pairs at >=85%")
    print(f"  pairs UNDER-stating realized error (ratio > 1.15): "
          f"**{int((rt>1.15).sum())}/{len(rt)}**")

    print("\n  FLE sensitivity (the FLE is supplied, so this is not optional):")
    print(f"    {'FLE µm':>8s} {'median':>8s} {'max':>7s} {'>1.15':>8s}")
    for fle in (3.0, 7.0, 14.1, 25.0, 40.0):
        rr = []
        for c in CASES:
            for s1, s2 in itertools.combinations(IHC, 2):
                ref, mov = pts.get((s1, c)), pts.get((s2, c))
                if ref is None or mov is None:
                    continue
                n = min(len(ref), len(mov))
                if n < MIN_N:
                    continue
                r = holdout(ref[:n], mov[:n], fle)
                if r:
                    rr.append(r["ratio"])
        if rr:
            rr = np.array(rr)
            print(f"    {fle:8.1f} {np.median(rr):8.2f} {rr.max():7.2f} "
                  f"{int((rr>1.15).sum()):5d}/{len(rr)}")

    out = os.path.join(_HERE, "fw_hyreco_results.json")
    json.dump({"fle_um": a.fle_um, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {out}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
